import torch
import pandas as pd
import numpy as np
import os
import argparse
import logging
import random
import transformers
from transformers import AutoTokenizer, EarlyStoppingCallback, DataCollatorForSeq2Seq, GenerationConfig
from torch.utils.data import ConcatDataset, DataLoader
from data.MultiTaskDataset import MultiTaskDataset
from data.TestDataset import TestDataset
from data.PretrainDataset import PretrainDataset
from transformers import T5Config
from transformers import T5ForConditionalGeneration
from utils import utils, arguments, initialization, evaluate, generation_trie
from datasets import Dataset, load_metric, interleave_datasets, concatenate_datasets
from rec_trainer import RecTrainer
from accelerate import Accelerator
from transformers.integrations.integration_utils import is_wandb_available

def get_dataset(args):
    # init dataset
    datasets = args.datasets.split(',')
    train_all_datasets = []
    valid_all_datasets = []
    for data in datasets:
        TrainDataset = MultiTaskDataset(args, data, 'train')
        train_all_datasets.append(TrainDataset)
        if args.valid_select > 0:
            ValidDataset = MultiTaskDataset(args, data, 'validation')
            valid_all_datasets.append(ValidDataset)
        
    TrainSet = ConcatDataset(train_all_datasets)
    if args.valid_select > 0:
        ValidSet = ConcatDataset(valid_all_datasets)
    else:
        ValidSet = None

    # load test dataset
    tasks = args.tasks.split(',')
    if len(tasks) > 1:
        logging.warning(f"Only support single task evaluation.\nUsing {tasks[0]} task for evaluation now.")
        logging.warning(f"Only support single dataset evaluation.\nUsing {datasets[0]} dataset for evaluation now.")
    task = tasks[0]
    dataset = datasets[0]
    TestSet = TestDataset(args, dataset, task)
    args.sample_prompt = 0
    ContentSet = PretrainDataset(args, dataset, 'content_to_id,category_to_id,brand_to_id,lowest_price_id,highest_price_id')

    return TrainSet, ValidSet, TestSet, ContentSet

def main(args):
    # Settings  
    accelerator = Accelerator()
    utils.setup_logging(args)
    utils.set_seed(args.seed)

    # decide output dir
    if len(args.datasets.split(',')) > 1:
        folder_name = '_'.join(args.datasets.split(','))
    else:
        folder_name = args.datasets
    output_dir = os.path.join(args.model_dir, folder_name, args.item_indexing, args.backbone)

    # load model
    if 't5' in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementedError  
    model = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)

    metrics = args.metrics.split(',')
    generate_num = max([int(m.split('@')[1]) for m in metrics])

    # get generation config
    generation_config = model.generation_config
    generation_config.max_length = 3 if args.item_indexing == 'independent' else 10
    generation_config.num_beams = generate_num
    generation_config.num_return_sequences = generate_num
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = True

    training_args = transformers.Seq2SeqTrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        logging_strategy="steps" if args.logging_step > 0 else "epoch",
        logging_steps=args.logging_step,
        optim=args.optim,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        evaluation_strategy="epoch" if args.valid_select > 0 else "no",
        per_device_eval_batch_size = args.eval_batch_size,
        dataloader_drop_last=False,
        save_strategy="epoch",
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if args.valid_select > 0 else False,
        group_by_length=False,
        predict_with_generate=True,
        generation_config=generation_config,
        full_determinism=True,
        seed = args.seed,     
        metric_for_best_model=args.eval_metric,
        ddp_find_unused_parameters=False,
        include_inputs_for_metrics=True if args.filter_prediction else False,
    )

    training_args_for_content = transformers.Seq2SeqTrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=10,
        learning_rate=args.lr,
        fp16=False,
        logging_strategy="steps" if args.logging_step > 0 else "epoch",
        logging_steps=args.logging_step,
        optim=args.optim,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        evaluation_strategy="no",
        per_device_eval_batch_size = args.eval_batch_size,
        dataloader_drop_last=False,
        save_strategy="epoch",
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=False,
        group_by_length=False,
        predict_with_generate=False,
        generation_config=generation_config,
        full_determinism=True,
        seed = args.seed,     
        ddp_find_unused_parameters=False,
        include_inputs_for_metrics=True if args.filter_prediction else False,
    )
    
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    def process_func(datapoint):
        encoding = tokenizer(datapoint['input'], max_length=512, truncation=True)
        labels = tokenizer(datapoint['output'], max_length=512, truncation=True)
        encoding['labels'] = labels['input_ids']
        return encoding
    
    # load dataset in main process
    temp_dir = os.path.join(args.data_path, args.datasets, 'processed')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if accelerator.is_main_process:
        logging.info(vars(args))
        train_data, valid_data, test_data, content_data = get_dataset(args)
        if hasattr(test_data, 'new_token'):
            tokenizer.add_tokens(test_data.new_token)
        torch.save(test_data, os.path.join(temp_dir, 'test.pt'))
        if args.train:
            trainSet = Dataset.from_list(train_data)
            contentSet = Dataset.from_list(content_data)
            trainSet = trainSet.map(process_func, batched=False, num_proc=8)
            contentSet = contentSet.map(process_func, batched=False, num_proc=8)
            trainSet.save_to_disk(os.path.join(temp_dir, 'train'))
            contentSet.save_to_disk(os.path.join(temp_dir, 'content'))
            if args.valid_select > 0:
                validSet = Dataset.from_list(valid_data)
                validSet = validSet.map(process_func, batched=False, num_proc=8)
                validSet.save_to_disk(os.path.join(temp_dir, 'valid'))
        testSet = Dataset.from_list(test_data)
        testSet = testSet.map(process_func, batched=False, num_proc=8)
        testSet.save_to_disk(os.path.join(temp_dir, 'test'))
    else:
        accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        if args.train:
            trainSet = Dataset.load_from_disk(os.path.join(temp_dir, 'train'))
            contentSet = Dataset.load_from_disk(os.path.join(temp_dir, 'content'))
            if args.valid_select > 0:
                validSet = Dataset.load_from_disk(os.path.join(temp_dir, 'valid'))
        test_data = torch.load(os.path.join(temp_dir, 'test.pt'))
        if hasattr(test_data, 'new_token'):
            tokenizer.add_tokens(test_data.new_token)
        testSet = Dataset.load_from_disk(os.path.join(temp_dir, 'test'))
    else:
        accelerator.wait_for_everyone()

    # resize token embedding
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    if args.init_new_tokens:
        new_token_embs = torch.load(os.path.join(args.data_path, args.datasets, 'new_token_emb.pt'))
        if args.user_indexing == 'independent':
            user_embs = torch.load(os.path.join(args.data_path, args.datasets, 'user_emb.pt'))
            new_token_embs = torch.cat([new_token_embs, user_embs], dim=0)
        start_idx = tokenizer.get_vocab()[test_data.new_token[0]]
        end_idx = tokenizer.get_vocab()[test_data.new_token[-1]]
        model.shared.weight.data[start_idx:end_idx+1] = new_token_embs

    rec_metrics = load_metric(args.metric_file)
    rec_metrics.set_metrics(args.metrics)
    rec_metrics.set_his_sep(args.his_sep)
    if not args.filter_prediction:
        def compute_metrics(p):
            return rec_metrics.compute(predictions=p.predictions, references=p.label_ids)
    else:
        def compute_metrics(p):
            return rec_metrics.compute(predictions=p.predictions, references=p.label_ids, inputs=p.inputs)
    
    # Post-processing
    def post_processing_fn(tokens):
        tokens[tokens==-100] = 0
        text = tokenizer.batch_decode(
            tokens, skip_special_tokens=True
        )
        return text

    # load trainer 
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        label_pad_token_id=-100,
    )

    # first stage: content understanding tasks
    first_trainer = RecTrainer(
        model=model,
        train_dataset=contentSet,
        eval_dataset=None,
        args=training_args_for_content,
        data_collator=data_collator,
        post_process_fn=None,
        compute_metrics=None,
    )
    # get the model of last epoch for next stage training
    first_trainer.train()
    model = first_trainer.model
    # second stage training
    trainer = RecTrainer(
        model=model,
        train_dataset=trainSet if args.train else None,
        eval_dataset=validSet if args.valid_select > 0 else None,
        args= training_args,
        data_collator = data_collator,
        post_process_fn=post_processing_fn if training_args.predict_with_generate else None,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)] if args.valid_select > 0 else None,
    )
    
    # Train
    if args.train:
        trainer.train()
    else:
        trainer._load_from_checkpoint(args.model_path)
    
    # Test
    if args.use_prefix_allowed_tokens_fn:
        candidates = test_data.all_items
        candidate_trie = generation_trie.Trie(
            [
                [0] + tokenizer.encode(f"{test_data.dataset} item_{candidate}")
                for candidate in candidates
            ]
        )
        prefix_allowed_tokens_fn = generation_trie.prefix_allowed_tokens_fn(candidate_trie)

    results = trainer.predict(
        test_dataset=testSet,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if args.use_prefix_allowed_tokens_fn else None,
        num_beams=generate_num,
        num_return_sequences=generate_num,
        output_scores=True,
        return_dict_in_generate=True,
    )
    trainer.log_metrics('test', results.metrics)

    if accelerator.is_main_process and is_wandb_available():
        prediction_ids = results.predictions.reshape(-1, results.predictions.shape[-1])
        label_ids = results.label_ids
        # convert -100 to 0 for decode
        label_ids[label_ids==-100] = 0
        predict_sentences = tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
        target_sentences = tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        generate_num = len(predict_sentences)//len(test_data)
        # count how many predict items occur in history
        all_inputs = [test_data[i]["input"] for i in range(len(test_data))]
        # extract history, between [] and seperated by args.his_sep
        all_history = [utils.extract_history(input, args.his_sep) for input in all_inputs]  
        all_predictions = [predict_sentences[i*generate_num:(i+1)*generate_num] for i in range(len(test_data))]
        hit_ranks = [evaluate.hit_rank(target_sentences[i], all_predictions[i]) for i in range(len(target_sentences))]
        # count how many predict items occur in history
        hit_count = [utils.hit_count(history, predictions) for history, predictions in zip(all_history, all_predictions)]
        table = pd.DataFrame({
            'input': all_inputs,
            'target': target_sentences,
            'predictions': all_predictions,
            'hit_rank': hit_ranks,
            'repeat_count': hit_count,
        })
        table.to_csv(os.path.join(args.log_dir, 'test_output.csv'))
        print(min(hit_count), max(hit_count), np.mean(hit_count), np.median(hit_count), np.std(hit_count))
        # sample 10 for log
        if args.train:
            import wandb
            wandb.log(results.metrics)
            wandb.config.update(args)
            log_index = random.sample(range(len(target_sentences)), 10)
            inputs = [test_data[i]["input"] for i in log_index]
            targets = [target_sentences[i] for i in log_index]
            predictions = [';'.join(predict_sentences[i*generate_num:(i+1)*generate_num]) for i in log_index]
            hit_ranks = [evaluate.hit_rank(targets[i], predictions[i]) for i in range(len(targets))]
            table = pd.DataFrame({
                'input': inputs,
                'target': targets,
                'predictions': predictions,
                'hit_rank': hit_ranks,
            })
            table = wandb.Table(dataframe=table)
            wandb.log({"sample_predictions": table})
    else:
        return
    
if __name__ == "__main__":
    parser = arguments.get_argparser()
    args, extras = parser.parse_known_args()
    main(args)
    