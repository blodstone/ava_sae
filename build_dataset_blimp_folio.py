"""
This file takes the FOLIO and BLIMP datasets, format, stack, and prepares them for SAE training.
"""
#%%
from typing import Tuple
import json
from pathlib import Path
import tqdm
import pandas as pd
import random
from transformers import AutoTokenizer
#%%

def build_from_folio_row(row: dict) -> str:
    tokenizer = AutoTokenizer.from_pretrained("gpt2", device='cuda')
    
    premises_text = row['premises']
    conclusion_text = row['conclusion']
    prefix = (
        "Premises:\n"
        + premises_text
        + "\nConclusion:\n"
    )
    prompt = prefix + conclusion_text + "\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    conclusion_tokens = tokenizer.encode(conclusion_text, add_special_tokens=False)
    conclusion_start_idx = len(prefix_tokens)
    conclusion_end_idx = conclusion_start_idx + len(conclusion_tokens) - 1
    return json.dumps({
        "dataset": "folio",
        "prompt": prompt,
        "id": row['story_id'],
        "conclusion_start_idx": conclusion_start_idx,
        "conclusion_end_idx": conclusion_end_idx,
        "label": row['label']
    })

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


#%%
folio_path = Path("/home/users1/hardy/hardy/datasets/FOLIO/folio_v2_validation.jsonl")
with folio_path.open() as f:
    folio =  [json.loads(line) for line in f]
folio_data = [build_from_folio_row(row) for row in tqdm.tqdm(folio)]
folio_output_path = Path("/home/users1/hardy/hardy/datasets/blimp_folio_sae/folio_data.jsonl")
with folio_output_path.open("w") as f:
    f.write('\n'.join(folio_data))
#%%
blimp_path = Path("/home/users1/hardy/hardy/datasets/blimp/data")
blimp_files = list(blimp_path.glob("*.jsonl"))
blimp_data = []
for file in tqdm.tqdm(blimp_files):
    with file.open() as f:
        blimp =  [json.loads(line) for line in f]
        blimp_data.extend([build_from_blimp_row(row) for row in blimp])
#%%
blimp_output_path = Path("/home/users1/hardy/hardy/datasets/blimp_folio_sae/blimp_data.jsonl")

with blimp_output_path.open('w') as f:
    f.write('\n'.join(blimp_data))