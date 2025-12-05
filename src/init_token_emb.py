import torch
import os
import logging
from transformers import AutoTokenizer
from data.PretrainDataset import PretrainDataset
from transformers import T5Config
from transformers import T5EncoderModel as P5
from utils import utils, arguments
from datasets import Dataset
from tqdm import tqdm

def get_pretrain_dataset(args):
    # load pretrain dataset
    datasets = args.datasets.split(',')
    tasks = args.tasks.split(',')
    if len(tasks) > 1:
        logging.warning(f"Only support single task evaluation.\nUsing {tasks[0]} task for evaluation now.")
        logging.warning(f"Only support single dataset evaluation.\nUsing {datasets[0]} dataset for evaluation now.")
    task = tasks[0]
    dataset = datasets[0]
    pretrain_data = PretrainDataset(args, dataset, task)
    return pretrain_data

def get_user_sequence(args):
    data_path = args.data_path
    dataset = args.datasets.split(',')[0]
    item_index_file = os.path.join(data_path, dataset, 'item_independent_indexing.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_independent_indexing.txt')
    user_sequence_lines = utils.ReadLineFromFile(reindex_sequence_file)
    item_index = utils.ReadLineFromFile(item_index_file)
    item_index_dict = dict()
    for idx, line in enumerate(item_index):
        parts = line.split()
        if len(parts) < 2:
            continue
        token = parts[1]
        item_index_dict[token] = idx
    processed_sequences = []
    for seq in user_sequence_lines:
        tokens = seq.split()[1:-2]
        indices = []
        for token in tokens:
            if token not in item_index_dict:
                raise KeyError(f"Token {token} not found in item_independent_indexing.txt")
            indices.append(item_index_dict[token])
        processed_sequences.append(indices)
    return processed_sequences

def main(args):    
    utils.setup_logging(args)
    #utils.setup_model_path(args)
    utils.set_seed(args.seed)
    logging.info(vars(args))
    
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    
    pretrain_data = get_pretrain_dataset(args)
    pretrainSet = Dataset.from_list(pretrain_data)

    # load model
    if 't5' in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementedError  
    model = P5.from_pretrained(args.backbone)
    model.to('cuda:0')

    # add additional tokens and resize token embedding
    if hasattr(pretrain_data, 'new_token'):
        tokenizer.add_tokens(pretrain_data.new_token)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    
    def process_func(datapoint):
        if 't5' in args.backbone.lower():
            encoding = tokenizer(datapoint['input'], max_length=512, truncation=True)
            labels = tokenizer(datapoint['output'], max_length=512, truncation=True)
            encoding['labels'] = labels['input_ids']
        else:
            raise NotImplementedError
        return encoding

    pretrainSet = pretrainSet.map(process_func, batched=False)

    embeddings = []
    with torch.no_grad():
        for data in tqdm(pretrainSet):
            input_ids = torch.tensor(data['input_ids']).unsqueeze(0).to('cuda:0')
            outputs = model.forward(input_ids=input_ids)
            last_hidden_state = outputs.last_hidden_state
            embeddings.append(last_hidden_state.squeeze(0).mean(0))
    embeddings = torch.stack(embeddings) # dim = (num_token, emb_dim)
    user_sequence = get_user_sequence(args)
    user_embeddings = [embeddings[torch.tensor(s).to('cuda:0')].mean(0) for s in user_sequence]
    user_embeddings = torch.stack(user_embeddings)

    # save
    torch.save(embeddings, os.path.join(args.data_path, args.datasets, 'new_token_emb.pt'))
    torch.save(user_embeddings, os.path.join(args.data_path, args.datasets, 'user_emb.pt'))

    # temporal token embeddings
    if args.use_time:
        time_bins = args.time_bins
        bins = [int(bin) if bin != 'inf' else float('inf') for bin in time_bins.split(',')]
        data = []
        for left, right in zip(bins[:-1], bins[1:]):
            if right != float('inf'):
                input = f"gap_days = range({left}, {right})"
            else:
                input = f"gap_days >= {left}"
            output = f'<[{left},{right})>'
            data.append({'input': input, 'output': output})
        data = Dataset.from_list(data)
        data = data.map(process_func, batched=False)
        embeddings = []
        with torch.no_grad():
            for data in tqdm(data):
                input_ids = torch.tensor(data['input_ids']).unsqueeze(0).to('cuda:0')
                outputs = model.forward(input_ids=input_ids)
                last_hidden_state = outputs.last_hidden_state
                embeddings.append(last_hidden_state.squeeze(0).mean(0))
        embeddings = torch.stack(embeddings) # dim = (num_token, emb_dim)
        torch.save(embeddings, os.path.join(args.data_path, args.datasets, 'temporal_token_emb.pt'))
        
if __name__ == "__main__":
    parser = arguments.get_argparser()
    args, extras = parser.parse_known_args()
    main(args)
    
