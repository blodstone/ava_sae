"""
This script preprocesses the datasets (BLIMP, Pylkkanen-McElree, Warren, Osterhout-Nicol) for Linguistic Constraint Interpretability.
"""
import argparse
import json
import tqdm
import logging
import pandas as pd
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO)
CHOICES = ['blimp', ' pylkkanen-mcelree', 'warren-et-al', 'osterhout-nicol', 'all']

def build_from_blimp_row(row: dict) -> str:
    sentences = [row['sentence_good'], row['sentence_bad']]
    results = []
    for idx, sentence in enumerate(sentences):
        prompt = sentence
        results.append({
            "dataset": "blimp",
            "prompt": prompt,
            "id": row['UID'] + '#' + row['pairID'] + '#' + str(idx),
            "linguistics_term": row['linguistics_term'],
            "label": 'bad' if idx == 1 else 'good'
        })
    return json.dumps(results)

def preprocess_blimp_data(blimp_path: str, output_path: str):
    logging.info(f"Processing BLIMP dataset from {blimp_path}")
    blimp_dir = Path(blimp_path)
    blimp_files = list(blimp_dir.glob("*.jsonl"))
    if len(blimp_files) == 0:
        logging.warning(f"No BLIMP files found in {blimp_path}")
        return
    blimp_data = []
    for file in tqdm.tqdm(blimp_files):
        with file.open() as f:
            blimp =  [json.loads(line) for line in f]
            blimp_data.extend([build_from_blimp_row(row) for row in blimp])
    output_dir = Path(output_path) / "input_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    blimp_output_path = output_dir / "blimp_data.jsonl"
    logging.info(f"Writing BLIMP data to {blimp_output_path}")
    with blimp_output_path.open("w") as f:
        for item in blimp_data:
            f.write(item + "\n")

def preprocess_dataset(df: pd.DataFrame, dataset: str, output_path: str):
    results = []
    i = 0
    for idx, row in df.iterrows():
        if dataset == "pylkkanen-mcelree":
            sent_bad = row['sent_anomaly']
            sent_good = row['sent_control']
        elif dataset == 'warren-et-al':
            sent_bad = row['sent_violation']
            sent_good = row['sent_control']
        elif dataset == 'federmeier-kutas':
            sent_bad = row['incorrect']
            sent_good = row['correct']
        elif dataset == 'osterhout-nicol':
            sent_bad = row['semantic_anomaly']
            sent_good = row['original_sentence']
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        results.append({
            "dataset": dataset,
            "prompt": sent_good,
            "id": str(i) + '#' + str(idx) + '#0',
            "label": 'good'
        })
        results.append({
            "dataset": dataset,
            "prompt": sent_bad,
            "id": str(i) + '#' + str(idx) + '#1',
            "label": 'bad'
        })
        i += 1
    
    output_dir = Path(output_path) / "input_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset}_data.jsonl"
    logging.info(f"Writing {dataset} data to {output_file}")
    with output_file.open("w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

def preprocess_pylkkanen_mcelree_data(pylkkanen_mcelree_path: str, output_path: str):
    logging.info(f"Processing Pylkkanen-McElree dataset from {pylkkanen_mcelree_path}")
    df = pd.read_csv(pylkkanen_mcelree_path, encoding='utf-8')
    preprocess_dataset(df, "pylkkanen-mcelree", output_path)
    # Process Pylkkanen-McElree dataset


def preprocess_warren_data(warren_path: str, output_path: str):
    logging.info(f"Processing Warren dataset from {warren_path}")
    df = pd.read_csv(warren_path, encoding='utf-8')
    preprocess_dataset(df, "warren-et-al", output_path)
    # Process Warren dataset


def preprocess_osterhout_nicol_data(osterhout_nicol_path: str, output_path: str):
    logging.info(f"Processing Osterhout-Nicol dataset from {osterhout_nicol_path}")
    # Process Osterhout-Nicol dataset
    df = pd.read_csv(osterhout_nicol_path, encoding='utf-8')
    preprocess_dataset(df, "osterhout-nicol", output_path)

def preprocess_federmeier_kutas_data(federmeier_kutas_path: str, output_path: str):
    logging.info(f"Processing Federmeier-Kutas dataset from {federmeier_kutas_path}")
    # Process Federmeier-Kutas dataset
    df = pd.read_csv(federmeier_kutas_path, encoding='utf-8')
    preprocess_dataset(df, "federmeier-kutas", output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=CHOICES, required=True)
    parser.add_argument('--blimp_path', type=str, default=PROJECT_ROOT / 'data' / 'blimp' / 'data')
    parser.add_argument('--pylkkanen-mcelree_path', type=str, default=PROJECT_ROOT / 'data' / 'pylkkanen-mcelree.csv')
    parser.add_argument('--warren_path', type=str, default=PROJECT_ROOT / 'data' / 'warren-et-al.csv')
    parser.add_argument('--osterhout-nicol_path', type=str, default=PROJECT_ROOT / 'data' / 'osterhout-nicol.csv')
    parser.add_argument('--output_path', type=str, default=PROJECT_ROOT / 'data')
    args = parser.parse_args()
    if args.dataset == 'blimp':
        preprocess_blimp_data(args.blimp_path, args.output_path)
    elif args.dataset == 'pylkkanen-mcelree':
        preprocess_pylkkanen_mcelree_data(args.pylkkanen_mcelree_path, args.output_path)
    elif args.dataset == 'warren-et-al':
        preprocess_warren_data(args.warren_path, args.output_path)
    elif args.dataset == 'osterhout-nicol':
        preprocess_osterhout_nicol_data(args.osterhout_nicol_path, args.output_path)
    elif args.dataset == 'federmeier-kutas':
        preprocess_federmeier_kutas_data(args.federmeier_kutas_path, args.output_path)  
    elif args.dataset == 'all':
        preprocess_blimp_data(args.blimp_path, args.output_path)
        preprocess_pylkkanen_mcelree_data(args.pylkkanen_mcelree_path, args.output_path)
        preprocess_warren_data(args.warren_path, args.output_path)
        preprocess_osterhout_nicol_data(args.osterhout_nicol_path, args.output_path)