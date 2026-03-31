#%%
from pathlib import Path
import json
import csv
from collections import defaultdict

BLIMP_FEATURES_PATH = Path("/home/users1/hardy/hardy/datasets/blimp_folio_sae/blimp_sae_features.jsonl")
FOLIO_FEATURES_PATH = Path("/home/users1/hardy/hardy/project/ava_sae/label_to_strict_overlap.json")
OUT_PATH = Path("/home/users1/hardy/hardy/project/ava_sae/blimp_folio_overlap_counts.json")
OUT_CSV_PATH = Path("/home/users1/hardy/hardy/project/ava_sae/blimp_folio_overlap_counts.csv")


def load_blimp_features(path: Path) -> list[dict]:
	with path.open("r") as f:
		return [json.loads(line) for line in f if line.strip()]


def load_folio_features(path: Path) -> dict[str, set[int]]:
	with path.open("r") as f:
		raw = json.load(f)

	folio = {}
	for label, feature_ids in raw.items():
		folio[str(label)] = set(int(fid) for fid in feature_ids)
	return folio

blimp_rows = load_blimp_features(BLIMP_FEATURES_PATH)
folio_by_label = load_folio_features(FOLIO_FEATURES_PATH)
#%%
grammatical_terms_result = defaultdict(list)
for ling_term in blimp_rows:
    linguistics_term = ling_term['linguistics_term']
    grammatical_features = ling_term['grammatical_features']
    true_folio_label = folio_by_label['True']
    grammatical_folio_overlap = set(grammatical_features) & true_folio_label
    grammatical_terms_result[linguistics_term].append(len(grammatical_folio_overlap))    


#%%
ungrammatical_terms_result = defaultdict(list)
for ling_term in blimp_rows:
    linguistics_term = ling_term['linguistics_term']
    ungrammatical_features = ling_term['ungrammatical_features']
    true_folio_label = folio_by_label['Uncertain']
    ungrammatical_folio_overlap = set(ungrammatical_features) & true_folio_label
    ungrammatical_terms_result[linguistics_term].append(ungrammatical_folio_overlap)
#%%
# Explain features in human language via Neuronpedia
# gpt2-small-res-jb maps to Neuronpedia model "gpt2-small", layer "8-res-jb"
import requests

NEURONPEDIA_MODEL = "gpt2-small"
NEURONPEDIA_LAYER = "8-res-jb"

def get_feature_description(feature_id: int) -> str:
    """Fetch the auto-generated description for a feature from Neuronpedia."""
    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL}/{NEURONPEDIA_LAYER}/{feature_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        explanations = data.get("explanations", [])
        if explanations:
            # Take the highest-scoring explanation
            best = max(explanations, key=lambda x: x.get("score") or 0)
            return best.get("description", "no description")
        return "no explanation available"
    except Exception as e:
        return f"error: {e}"

# Look up descriptions for all overlapping features
print("\n--- Feature Descriptions (Neuronpedia) ---")
for feat_id in {6494, 18691}:
    description = get_feature_description(feat_id)
    print(f"Feature {feat_id:5d} | description: {description}")


#%%

