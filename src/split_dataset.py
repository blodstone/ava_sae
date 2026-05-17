import logging
import argparse
import json
import random

from pathlib import Path

def main(args):
    with open(args.dataset_path, "r") as f:
        data = [json.loads(line) for line in f]
    paradigm_map = {}
    idx = 0
    for item in data:
        item[0]['global_idx'] = idx
        idx += 1
        item[1]['global_idx'] = idx
        idx += 1
        paradigm = item[0]['id'].split('#')[0]
        if paradigm not in paradigm_map:
            paradigm_map[paradigm] = []
        paradigm_map[paradigm].append(item)
    valid_set = {paradigm: [] for paradigm in paradigm_map.keys()}
    test_set = {paradigm: [] for paradigm in paradigm_map.keys()}

    for paradigm, items in paradigm_map.items():
        shuffled_items = items[:]
        random.shuffle(shuffled_items)
        split_idx = int(len(shuffled_items) * 0.8)
        test_set[paradigm] = shuffled_items[:split_idx]
        valid_set[paradigm] = shuffled_items[split_idx:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "blimp_test.jsonl", "w") as f_test, open(args.output_dir / "blimp_valid.jsonl", "w") as f_valid:
        for paradigm in paradigm_map.keys():
            for item in test_set[paradigm]:
                f_test.write(json.dumps(item) + "\n")
            for item in valid_set[paradigm]:
                f_valid.write(json.dumps(item) + "\n")

if __name__ == '__main__':
    random.seed(42)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()    
    parser.add_argument('--dataset_name', type=str, default='blimp')
    parser.add_argument('--dataset_path', type=Path, default=PROJECT_ROOT / "data" / "input_data" / "blimp_data.jsonl")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "data" / "input_data", help="Directory to save the split datasets.")
    args = parser.parse_args()
    main(args)