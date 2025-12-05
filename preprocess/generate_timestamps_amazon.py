# Import necessary libraries
from collections import defaultdict
import json
from tqdm import tqdm

# Add helper functions
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def parse(path):
    import gzip
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def Amazon(dataset_name, rating_score):
    datas = []
    data_file = '../raw_data/reviews_' + dataset_name + '.json.gz'
    for inter in parse(data_file):
        if float(inter['overall']) <= rating_score:
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas

def main(short_data_name):
    if short_data_name == 'beauty':
        full_data_name = 'Beauty'
    elif short_data_name == 'toys':
        full_data_name = 'Toys_and_Games'
    elif short_data_name == 'sports':
        full_data_name = 'Sports_and_Outdoors'
    else:
        raise NotImplementedError
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0
    datas = Amazon(full_data_name+'_5', rating_score=rating_score)
    datamaps = load_json("../data/{}/datamaps.json".format(short_data_name))

    timestamps = {}
    for data in datas:
        user, item, time = data
        user_id = datamaps['user2id'].get(user, None)
        item_id = datamaps['item2id'].get(item, None)
        if user_id is None or item_id is None:
            continue
        user_id = int(user_id)
        item_id = int(item_id)
        if user_id not in timestamps:
            timestamps[user_id] = {}
        timestamps[user_id][item_id] = time

    # load sequential data
    seq_file = '../data/{}/sequential_data.txt'.format(short_data_name)
    user_items = {}
    with open(seq_file, 'r') as f:
        for line in f:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            items = [int(item) for item in items]
            user_items[int(user)] = items

    # get timestamps
    user_timestamps = {}
    for user, items in user_items.items():
        user_timestamps[user] = [timestamps[user][item] for item in items]
    # Save the user_timestamps data
    user_timestamps_file = '../data/{}/sequential_time.txt'.format(short_data_name)
    with open(user_timestamps_file, 'w') as out:
        for user, timestamps in user_timestamps.items():
            out.write(str(user) + ' ' + ' '.join(map(str, timestamps)) + '\n')

if __name__ == '__main__':
    for short_data_name in ['sports', 'beauty', 'toys']:
        main(short_data_name)