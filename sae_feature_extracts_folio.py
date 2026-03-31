#%%
from sae_lens import SAE, HookedSAETransformer
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
query_path = Path("/home/users1/hardy/hardy/datasets/blimp_folio_sae/folio_data.jsonl")
with open(query_path, 'r') as f:
    data = [json.loads(line) for line in f]
    queries = [item['prompt'] for item in data]
    conclusion_start_idxs = [item['conclusion_start_idx'] for item in data]
    conclusion_end_idxs = [item['conclusion_end_idx'] for item in data]
    labels = [item['label'] for item in data]
#%%
batch_size = 16
feature_ids = []

for start in tqdm.tqdm(range(0, len(queries), batch_size)):
    batch_queries = queries[start:start + batch_size]
    batch_c_start_idsx = conclusion_start_idxs[start:start + batch_size]
    batch_c_end_idxs = conclusion_end_idxs[start:start + batch_size]
    with torch.no_grad():
        _, cache = model.run_with_cache(batch_queries, prepend_bos=True)
        batch_feature_acts = sae.encode(cache[sae.cfg.metadata.hook_name])
        # Pool variable-length conclusion spans to fixed-size [n_features] vectors.
        for i, (s, e) in enumerate(zip(batch_c_start_idsx, batch_c_end_idxs)):
            span_acts = batch_feature_acts[i, s:e]
            active_feature_ids = (span_acts > 0).any(dim=0).nonzero(as_tuple=True)[0]
            feature_ids.append(set(active_feature_ids.cpu().tolist()))
        del cache

#%%
# Group feature IDs by label
label_to_feature = defaultdict(list)


for lbl, fset in zip(labels, feature_ids):
    label_to_feature[lbl].append(fset)


# strict overlap: feature IDs present in every example of a label
label_to_strict_overlap = {}


for lbl, fsets in label_to_feature.items():
    if not fsets:
        label_to_strict_overlap[lbl] = set()
        continue

    strict_overlap = set.intersection(*fsets)

    freq = Counter()
    for fset in fsets:
        freq.update(fset)
    min_count = max(1, int(0.5 * len(fsets)))
    loose_overlap = {fid for fid, c in freq.items() if c >= min_count}

    label_to_strict_overlap[lbl] = strict_overlap

    print(f"Label: {lbl}")
    print(f"  num examples: {len(fsets)}")
    print(f"  strict overlap (in all examples): {len(strict_overlap)} features")
    print(f"  top strict feature IDs: {sorted(list(strict_overlap))[:30]}")

# Save strict-overlap feature IDs per label
strict_overlap_out = {
    lbl: sorted(list(fset))
    for lbl, fset in label_to_strict_overlap.items()
}
out_path = Path("label_to_strict_overlap.json")
with open(out_path, "w") as f:
    json.dump(strict_overlap_out, f, indent=2)
print(f"\nSaved strict overlap to {out_path.resolve()}")



