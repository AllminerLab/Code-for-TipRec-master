### Requirements
Python Version: 3.9

```
torch==2.1.0
transformers==4.35.2
accelerate==0.23.0
scikit-learn==1.3.1
datasets==2.14.6
```

### Datasets

Download [P5_data.zip](https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G) and [raw_data.zip](https://drive.google.com/file/d/1uE-_wpGmIiRLxaIy8wItMspOf5xRNF2O/view?usp=sharing) from P5 repository.

Run the following command to unzip the dataset

```bash
unzip P5_data.zip
unzip raw_data.zip
```

Using raw data to generate sequential_time.txt

```bash
cd preprocess
python generate_timestamps_amazon.py
```

### Run

Beauty dataset as example

```
cd scripts
bash beauty_init_tokens.sh
bash beauty_training.sh
```