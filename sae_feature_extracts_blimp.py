#%%
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoTokenizer
from pathlib import Path
import json
import torch
import tqdm
from collections import defaultdict, Counter
#%%
release = "gpt2-small-res-jb"
sae_id = "blocks.8.hook_resid_pre"
sae = SAE.from_pretrained(release, sae_id, device="cuda")
model = HookedSAETransformer.from_pretrained_no_processing("gpt2-small", device="cuda")
sae.eval()
model.eval()
#%%
query_path = Path("/home/users1/hardy/hardy/datasets/blimp_folio_sae/blimp_data.jsonl")
with open(query_path, 'r') as f:
    data = [json.loads(line) for line in f]
    good_examples = [item[0]['prompt'] for item in data]
    bad_examples = [item[1]['prompt'] for item in data]
    linguistics_terms = [item[0]['linguistics_term'] for item in data]
#%%
batch_size = 32
good_feature_ids = []
good_token_feature_ids = []
bad_feature_ids = []
bad_token_feature_ids = []
for start in tqdm.tqdm(range(0, len(good_examples), batch_size)):
    batch_good_examples = good_examples[start:start + batch_size]
    batch_bad_examples = bad_examples[start:start + batch_size]
    with torch.no_grad():
        _, cache = model.run_with_cache(batch_good_examples, prepend_bos=True)
        batch_feature_acts = sae.encode(cache[sae.cfg.metadata.hook_name])
        for i in range(batch_feature_acts.shape[0]):
            per_token = [
                (batch_feature_acts[i, t] > 0).nonzero(as_tuple=True)[0].cpu() 
                for t in range(1, batch_feature_acts.shape[1])
            ]
            good_token_feature_ids.append(per_token)
            all_features = set()
            for token_features in per_token:
                all_features.update(token_features.tolist())
            per_example_features = torch.tensor(sorted(list(all_features))) if all_features else torch.tensor([])
            good_feature_ids.append(per_example_features)

        del cache
        _, cache = model.run_with_cache(batch_bad_examples, prepend_bos=True)
        batch_feature_acts = sae.encode(cache[sae.cfg.metadata.hook_name])
        for i in range(batch_feature_acts.shape[0]):
            per_token = [
                (batch_feature_acts[i, t] > 0).nonzero(as_tuple=True)[0].cpu() 
                for t in range(1, batch_feature_acts.shape[1])
            ]
            bad_token_feature_ids.append(per_token)
            all_features = set()
            for token_features in per_token:
                all_features.update(token_features.tolist())
            per_example_features = torch.tensor(sorted(list(all_features))) if all_features else torch.tensor([])
            bad_feature_ids.append(per_example_features)
        del cache
#%%
ling_to_feature = defaultdict(list)

for term, gset, bset in tqdm.tqdm(zip(linguistics_terms, good_feature_ids, bad_feature_ids)):
    grammatical_feature_ids = set(gset.tolist()) - set(bset.tolist())
    ungrammatical_feature_ids = set(bset.tolist()) - set(gset.tolist())
    ling_to_feature[term].append((grammatical_feature_ids, ungrammatical_feature_ids))

#%%
ling_to_grammatical_union = {term: set.union(*[fs[0] for fs in feature_sets]) for term, feature_sets in ling_to_feature.items()}
ling_to_ungrammatical_union = {term: set.union(*[fs[1] for fs in feature_sets]) for term, feature_sets in ling_to_feature.items()}
output_path = Path("/home/users1/hardy/hardy/datasets/blimp_folio_sae/blimp_sae_features.jsonl")
with open(output_path, 'w') as f:
    for term in ling_to_grammatical_union.keys():
        record = {
            'linguistics_term': term,
            'grammatical_features': sorted(list(ling_to_grammatical_union[term])),
            'ungrammatical_features': sorted(list(ling_to_ungrammatical_union[term]))
        }
        f.write(json.dumps(record) + '\n')

