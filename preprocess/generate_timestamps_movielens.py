from collections import defaultdict
import os


def load_mapping(file_path):
    """Load raw to indexed id mapping from a whitespace separated txt file."""
    mapping = {}
    with open(file_path, "r") as f:
        for line in f:
            raw_id, mapped_id = line.strip().split(" ")
            mapping[raw_id] = int(mapped_id)
    return mapping


def parse_ratings(rating_file, user_map, item_map):
    """Collect timestamps for the retained (user, item) pairs."""
    user_item_time = defaultdict(dict)
    with open(rating_file, "r") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            user_raw, item_raw, _, timestamp = parts
            if user_raw not in user_map or item_raw not in item_map:
                continue
            user_idx = user_map[user_raw]
            item_idx = item_map[item_raw]
            user_item_time[user_idx][item_idx] = int(timestamp)
    return user_item_time


def load_sequences(seq_file):
    """Load re-indexed user sequences."""
    sequences = {}
    with open(seq_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            user = int(parts[0])
            items = [int(item) for item in parts[1:]]
            sequences[user] = items
    return sequences


def write_timestamps(output_path, sequences, timestamps):
    """Write sequential timestamps matched to the sequence file."""
    with open(output_path, "w") as out:
        for user, items in sequences.items():
            if user not in timestamps:
                raise KeyError(f"Missing timestamp entries for user {user}.")
            try:
                time_list = [str(timestamps[user][item]) for item in items]
            except KeyError as err:
                missing_item = err.args[0]
                raise KeyError(f"Missing timestamp for user {user}, item {missing_item}.") from err
            out.write(f"{user} {' '.join(time_list)}\n")


def get_sequence_file(data_dir):
    """Select the sequence file to align with."""
    candidates = ["sequential_data.txt", "user_sequence_sequential_indexing_original.txt"]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No sequence file found in {data_dir}.")


def main(raw_data_name="ml-1m"):
    if raw_data_name != "ml-1m":
        raise NotImplementedError("Only ml-1m is supported.")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(base_dir, "..", "data", "ML1M"))
    rating_file = os.path.normpath(os.path.join(base_dir, "..", "raw_data", "ml-1m", "ratings.dat"))
    user_index_file = os.path.join(data_dir, "user_indexing.txt")
    item_index_file = os.path.join(data_dir, "item_sequential_indexing_original.txt")
    sequence_file = get_sequence_file(data_dir)
    output_file = os.path.join(data_dir, "sequential_time.txt")

    user_map = load_mapping(user_index_file)
    item_map = load_mapping(item_index_file)
    timestamps = parse_ratings(rating_file, user_map, item_map)
    sequences = load_sequences(sequence_file)

    write_timestamps(output_file, sequences, timestamps)


if __name__ == "__main__":
    main()
