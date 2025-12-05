import json
import gzip
from collections import defaultdict
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)



import os

def get_yelp_timestamps(rating_score=0.0):
    datas = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, '../raw_data/reviews_Yelp_5.json.gz')
    
    print(f"Reading from {data_file}...")
    # The file is a JSON list, so we load it all at once
    with gzip.open(data_file, 'r') as f:
        raw_data = json.load(f)
        
    print(f"Processing {len(raw_data)} reviews...")
    for inter in tqdm(raw_data):
        if float(inter['overall']) <= rating_score:
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas

def main():
    short_data_name = 'yelp'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load datamaps
    datamaps_file = os.path.join(base_dir, f"../data/{short_data_name}/datamaps.json")
    print(f"Loading datamaps from {datamaps_file}...")
    datamaps = load_json(datamaps_file)
    
    # Load raw timestamps
    print("Loading raw timestamps...")
    datas = get_yelp_timestamps()
    
    # Map timestamps to user_id and item_id
    print("Mapping timestamps...")
    timestamps = defaultdict(lambda: defaultdict(list))
    for data in tqdm(datas):
        user, item, time = data
        user_id = datamaps['user2id'].get(user, None)
        item_id = datamaps['item2id'].get(item, None)
        
        if user_id is None or item_id is None:
            continue
            
        user_id = int(user_id)
        item_id = int(item_id)
        timestamps[user_id][item_id].append(time)

    # Sort timestamps for each user-item pair to ensure we pop them in order
    for user_id in timestamps:
        for item_id in timestamps[user_id]:
            timestamps[user_id][item_id].sort()

    # Load sequential data to ensure order
    seq_file = os.path.join(base_dir, f'../data/{short_data_name}/sequential_data.txt')
    print(f"Loading sequential data from {seq_file}...")
    user_items = {}
    with open(seq_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            user = int(parts[0])
            if len(parts) > 1:
                items = [int(item) for item in parts[1].split(' ')]
            else:
                items = []
            user_items[user] = items

    # Generate sequential timestamps
    print("Generating sequential timestamps...")
    user_timestamps = {}
    for user, items in user_items.items():
        user_ts_list = []
        for item in items:
            if user in timestamps and item in timestamps[user] and timestamps[user][item]:
                # Pop the first timestamp (earliest) for this item
                # Assuming sequential_data is sorted by time, this should align
                # If sequential_data has duplicates, we need to consume them in order
                user_ts_list.append(timestamps[user][item].pop(0))
            else:
                # This shouldn't happen if data is consistent, but handle gracefully or raise error
                print(f"Warning: Missing timestamp for user {user} item {item}")
                user_ts_list.append(0) 
        user_timestamps[user] = user_ts_list

    # Save the user_timestamps data
    output_file = os.path.join(base_dir, f'../data/{short_data_name}/sequential_time.txt')
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as out:
        for user, ts_list in user_timestamps.items():
            out.write(str(user) + ' ' + ' '.join(map(str, ts_list)) + '\n')
    print("Done.")

if __name__ == '__main__':
    main()
