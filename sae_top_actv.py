#%%
from sae_lens import SAE, HookedSAETransformer
from pathlib import Path
from transformers import AutoTokenizer

from collections import Counter
from transformer_lens import utils
import json
import torch
import re
#%%
release = "gpt2-small-res-jb"
sae_id = "blocks.8.hook_resid_pre"
sae = SAE.from_pretrained(release, sae_id, device="cuda")
model = HookedSAETransformer.from_pretrained_no_processing("gpt2-small", device="cuda")

sae.eval()
#%%
query_path = Path("/home/users1/hardy/hardy/datasets/blimp_folio_sae/blimp_folio_data.jsonl")
with open(query_path, 'r') as f:
    data = [json.loads(line) for line in f]
queries = [item['prompt'] for item in data]
answers = [item['label'] for item in data]
sample_queries = [queries[0], queries[-1]]
#%%
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(sample_queries, return_tensors="pt", padding=True).to(model.cfg.device)
last_token_positions = inputs.attention_mask.sum(dim=1) - 1
#%%
with torch.no_grad():
    _, cache = model.run_with_cache(sample_queries, prepend_bos=True)
    feature_acts = sae.encode(cache[sae.cfg.metadata.hook_name])
    sae_out = sae.decode(feature_acts)
    del cache
#%%
# example_prompt = queries[1]
# example_answer = answers[1]
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True, prepend_space_to_answer=False)
#%%
# feature_acts: [batch, seq_len, n_features]
# Only look at the last token position
last_token_acts = feature_acts[torch.arange(feature_acts.shape[0]), last_token_positions]
  # [batch, n_features]

active_per_example = last_token_acts > 0  # [batch, n_features]

# Count how many examples each feature is active in
feature_counts = active_per_example.sum(dim=0)  # [n_features]

# Overlapping features: active in more than one example
overlap_mask = feature_counts > 1
overlap_feature_ids = overlap_mask.nonzero(as_tuple=True)[0]
print(f"Total features active at last token: {(feature_counts > 0).sum().item()}")
print(f"Features active in >1 example (overlapping): {overlap_mask.sum().item()}")
print(f"Overlap feature indices: {overlap_feature_ids.tolist()}")
#%%
# For each overlapping feature, show which examples it fires in and the activation value
print("\n--- Overlapping Feature Details (last token) ---")
for feat_id in overlap_feature_ids:
    feat_id = int(feat_id.item())
    active_in = active_per_example[:, feat_id].nonzero(as_tuple=True)[0].tolist()
    acts = last_token_acts[:, feat_id]
    print(f"Feature {feat_id:5d} | active in examples {active_in} | acts: {[round(v, 4) for v in acts[active_in].tolist()]}")
#%%
# Explain features in human language via Neuronpedia
# gpt2-small-res-jb maps to Neuronpedia model "gpt2-small", layer "8-res-jb"
import requests

# NEURONPEDIA_MODEL = "gpt2-small"
# NEURONPEDIA_LAYER = "8-res-jb"

# def get_feature_description(feature_id: int) -> str:
#     """Fetch the auto-generated description for a feature from Neuronpedia."""
#     url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{NEURONPEDIA_LAYER}/{feature_id}"
#     try:
#         resp = requests.get(url, timeout=10)
#         resp.raise_for_status()
#         data = resp.json()
#         explanations = data.get("explanations", [])
#         if explanations:
#             # Take the highest-scoring explanation
#             best = max(explanations, key=lambda x: x.get("score") or 0)
#             return best.get("description", "no description")
#         return "no explanation available"
#     except Exception as e:
#         return f"error: {e}"

# # Look up descriptions for all overlapping features
# print("\n--- Feature Descriptions (Neuronpedia) ---")
# for feat_id in overlap_feature_ids:
#     feat_id = int(feat_id.item())
#     description = get_feature_description(feat_id)
#     active_in = active_per_example[:, feat_id].nonzero(as_tuple=True)[0].tolist()
#     acts = last_token_acts[:, feat_id]
#     print(f"Feature {feat_id:5d} | examples {active_in} | acts {[round(v,3) for v in acts[active_in].tolist()]}")
#     print(f"           description: {description}")

#%%
# Build your own feature explanations from BLiMP/FOLIO examples.
# Idea: for each feature, gather top-activating prompts and summarize shared patterns.
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "that", "this", "as",
    "from", "if", "then", "than", "but", "not", "do", "does", "did", "have", "has", "had",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their", "our", "your"
}


def _extract_ngrams(text, n):
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def explain_feature_from_dataset(feature_id, all_last_acts, prompts, labels, top_k=30):
    """Return a simple, dataset-grounded explanation for one SAE feature."""
    scores = all_last_acts[:, feature_id]
    positive = (scores > 0).nonzero(as_tuple=True)[0]
    if positive.numel() == 0:
        return {
            "feature_id": feature_id,
            "summary": "No positive activation at the chosen token position in this dataset slice.",
            "top_examples": [],
        }

    pos_scores = scores[positive]
    k = min(top_k, positive.numel())
    top_rel = torch.topk(pos_scores, k=k).indices
    top_idx = positive[top_rel].tolist()

    unigram = Counter()
    bigram = Counter()
    trigram = Counter()
    label_counter = Counter()

    examples = []
    for i in top_idx:
        text = prompts[i]
        label_counter[labels[i]] += 1
        unigram.update(_extract_ngrams(text, 1))
        bigram.update(_extract_ngrams(text, 2))
        trigram.update(_extract_ngrams(text, 3))
        examples.append({"idx": i, "act": float(scores[i]), "label": labels[i], "text": text})

    top_uni = [w for w, _ in unigram.most_common(6)]
    top_bi = [w for w, _ in bigram.most_common(4)]
    top_tri = [w for w, _ in trigram.most_common(3)]
    total = sum(label_counter.values())
    label_desc = ", ".join(
        [f"{lbl}: {cnt}/{total}" for lbl, cnt in label_counter.most_common()]
    )

    summary = (
        f"Top activations co-occur with words {top_uni[:4]}"
        f"; frequent phrases include {top_bi[:2]}"
        f"; label skew among top examples: {label_desc}."
    )

    return {
        "feature_id": feature_id,
        "summary": summary,
        "top_unigrams": top_uni,
        "top_bigrams": top_bi,
        "top_trigrams": top_tri,
        "label_distribution": dict(label_counter),
        "top_examples": examples[:8],
    }

#%%
dataset_last_acts = last_token_acts
#%%
print("\n--- Dataset-Grounded Explanations (BLiMP/FOLIO) ---")
features_to_explain = overlap_feature_ids[:10].tolist() if overlap_feature_ids.numel() else []
for fid in features_to_explain:
    report = explain_feature_from_dataset(fid, dataset_last_acts, queries[:dataset_last_acts.shape[0]], answers[:dataset_last_acts.shape[0]], top_k=40)
    print(f"Feature {fid}: {report['summary']}")
    print(f"  top bigrams: {report.get('top_bigrams', [])[:3]}")
    for ex in report.get("top_examples", [])[:3]:
        print(f"  ex idx={ex['idx']} act={ex['act']:.3f} label={ex['label']} text={ex['text'][:120]}")