#%%
import torch
import pandas as pd
import json
import re
#%%

with open('/mount/arbeitsdaten66/projekte/multiview/hardy/datasets/FOLIO/folio_v2_validation.jsonl') as f:
    data = [json.loads(line) for line in f]
#%%

data_pd = pd.DataFrame(data)


def build_prompt_from_row(row: dict) -> str:
    # try common keys for premises
    prompt = (
        "Premises " + row['premises-FOL'] + " ")
    prompt += f"Conclusion -> {row['conclusion-FOL']} -> Label: {row['label']}"
    return prompt


# Create a `text` column with the formatted prompt and save to JSONL
if not data_pd.empty:
    data_pd["text"] = data_pd.apply(lambda r: build_prompt_from_row(r.to_dict()), axis=1)

with open('/mount/arbeitsdaten66/projekte/multiview/hardy/datasets/FOLIO/folio_v2_validation_tokenized.jsonl', 'w') as f:
    for _, row in data_pd.iterrows():
        json.dump({"text": row["text"]}, f)
        f.write('\n')
#%%
