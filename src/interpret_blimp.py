#%%
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoTokenizer
from pathlib import Path
import json
import csv
import torch
import torch
import gc
import tqdm
from collections import defaultdict, Counter
import random
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained('gpt2', device='cuda')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
release = "gpt2-small-resid-post-v5-128k"
model = HookedSAETransformer.from_pretrained_no_processing("gpt2-small", device="cuda")
model.eval()
#%%
# Prepare the samples
# The idea is to run on the argument structure of the BLIMP dataset, which is a set of minimal pairs of sentences that differ in one grammatical aspect. We will run the model on these sentences and see how the activations differ.
BASE_PATH = Path("/home/users1/hardy/hardy/project/ava_sae/")
blimp_prediction_path = BASE_PATH / "output" / "blimp_predictions.jsonl"
with open(blimp_prediction_path) as f:
    blimp_predictions = [json.loads(line) for line in f]
count = 0
# argument_structure: the ability of different verbs to appear with different types of arguments.
target_phenomena = 'argument_structure'
sampled_data = []
for data in blimp_predictions:
    if data[0]['linguistics_term'] == target_phenomena:
        sampled_data.append(data)
        count += 1
        break
sampled_sentences = []
for data in sampled_data:
    sampled_sentences.append(data[0]['prompt'])
    sampled_sentences.append(data[1]['prompt'])
print(f"Sampled {len(sampled_sentences)} sentences for analysis.")
#%%
def _find_min_changed_token_span(tokens_a, tokens_b):
    prefix_len = 0
    max_prefix = min(len(tokens_a), len(tokens_b))
    while prefix_len < max_prefix and tokens_a[prefix_len] == tokens_b[prefix_len]:
        prefix_len += 1

    suffix_len = 0
    max_suffix = min(len(tokens_a) - prefix_len, len(tokens_b) - prefix_len)
    while (
        suffix_len < max_suffix
        and tokens_a[len(tokens_a) - 1 - suffix_len] == tokens_b[len(tokens_b) - 1 - suffix_len]
    ):
        suffix_len += 1

    span_a = (prefix_len, len(tokens_a) - suffix_len)
    span_b = (prefix_len, len(tokens_b) - suffix_len)
    return span_a, span_b

def build_min_change_token_mask(sampled_sentences, tokenizer, prepend_bos=True):
    if len(sampled_sentences) % 2 != 0:
        raise ValueError("sampled_sentences must contain an even number of entries arranged as minimal pairs.")

    enc = tokenizer(
        sampled_sentences,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"].bool()
    changed_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    changed_token_spans = []

    for pair_start in range(0, len(sampled_sentences), 2):
        tokens_a = input_ids[pair_start][attention_mask[pair_start]].tolist()
        tokens_b = input_ids[pair_start + 1][attention_mask[pair_start + 1]].tolist()
        span_a, span_b = _find_min_changed_token_span(tokens_a, tokens_b)
        changed_token_spans.extend([span_a, span_b])

        changed_token_mask[pair_start, span_a[0]:span_a[1]] = True
        changed_token_mask[pair_start + 1, span_b[0]:span_b[1]] = True

    if prepend_bos:
        bos_column = torch.zeros((changed_token_mask.shape[0], 1), dtype=torch.bool)
        changed_token_mask = torch.cat([bos_column, changed_token_mask], dim=1)

    return changed_token_mask, changed_token_spans, input_ids

changed_token_mask, changed_token_spans, tokenized_input_ids = build_min_change_token_mask(
    sampled_sentences,
    tokenizer,
    prepend_bos=True,
)
print(f"Changed token mask shape: {changed_token_mask.shape}")
#%%
with torch.no_grad():
        _, cache = model.run_with_cache(sampled_sentences, prepend_bos=True)
        # Extract padding mask
        attention_mask = model.tokenizer(sampled_sentences, padding=True, return_tensors="pt")["attention_mask"].to("cuda")
        padding_mask = attention_mask == 0
        print(f"Padding mask shape: {padding_mask.shape}")
#%%
# Calculate the length of each sample (number of non-padding tokens)
sample_lengths = (~padding_mask).sum(dim=1)
print(f"Sample lengths: {sample_lengths}")
#%%
def top_cumsum(active_features, active_features_logit, threshold_ratio=0.9):
    total_sum = sum(active_features_logit)
    threshold = total_sum * threshold_ratio
    cumsum = 0
    top_features = []
    for feat, logit in sorted(zip(active_features, active_features_logit), key=lambda x: x[1], reverse=True):
        cumsum += logit
        top_features.append((feat, logit))
        if cumsum >= threshold:
            break
    return top_features

def compute_random_features(sample_lengths, feature_acts, reference_batch_top_features):
    batch_random_features = []
    for i in range(feature_acts.shape[0]):
        tokens_random_features = []
        for t in range(1, sample_lengths[i]):  # Only iterate over non-padding tokens and skip the first token (BOS)
            active_feature_ids = (feature_acts[i, t] > 0).nonzero(as_tuple=True)[0]
            n_to_sample = len(reference_batch_top_features[i][t - 1])
            if active_feature_ids.numel() == 0 or n_to_sample == 0:
                tokens_random_features.append([])
                continue

            n_to_sample = min(n_to_sample, active_feature_ids.numel())
            random_idx = torch.randperm(active_feature_ids.numel(), device=feature_acts.device)[:n_to_sample]
            random_feature_ids = active_feature_ids[random_idx]
            random_features = [
                (feat.item(), feature_acts[i, t, feat].item())
                for feat in random_feature_ids
            ]
            tokens_random_features.append(random_features)
        batch_random_features.append(tokens_random_features)
    return batch_random_features

def compute_top_features(sample_lengths, top_cumsum, feature_acts):
    batch_active_features = []
    batch_top_features = []
    batch_active_features_logit = []
    for i in range(feature_acts.shape[0]):
        tokens_active_features = []
        tokens_active_features_logit = []
        tokens_top_features = []
        for t in range(1, sample_lengths[i]):  # Only iterate over non-padding tokens and skip the first token (BOS)
            active_features = (feature_acts[i, t] > 0).nonzero(as_tuple=True)[0].cpu().tolist()
            active_features_logit = feature_acts[i, t][active_features].cpu().tolist()
            top_features = top_cumsum(active_features, active_features_logit, threshold_ratio=0.7)

            tokens_active_features.append(active_features)
            tokens_active_features_logit.append(active_features_logit)
            tokens_top_features.append(top_features)
        batch_active_features.append(tokens_active_features)
        batch_top_features.append(tokens_top_features)
        batch_active_features_logit.append(tokens_active_features_logit)
    return batch_top_features, batch_active_features, batch_active_features_logit

def load_sensitivity_top_features(base_path, layer, sample_lengths, feature_acts, top_k=32, active_only=True, eps=1e-8):
    """Load top-k sensitive feature ids for a layer and rebuild batch_top_features.

    Returns:
        batch_top_features: shape [batch][token_without_bos][(feat_id, feat_value)]
        top_feature_ids: list of selected feature ids sorted by descending sensitivity
    """
    sensitivity_path = base_path / "output" / f"sensitivity_Sf_layer_{layer}.csv"
    if not sensitivity_path.exists():
        raise FileNotFoundError(f"Sensitivity file not found: {sensitivity_path}")

    rows = []
    with open(sensitivity_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["feature_id"]), float(row["S_f_bad_minus_good"])))

    if not rows:
        raise ValueError(f"No rows found in sensitivity file: {sensitivity_path}")

    rows.sort(key=lambda x: x[1], reverse=True)
    top_feature_ids = [feature_id for feature_id, _ in rows[:top_k]]

    batch_top_features = []
    for i in range(feature_acts.shape[0]):
        tokens_top_features = []
        for t in range(1, sample_lengths[i]):  # Skip BOS to match ablate token mapping.
            token_features = []
            for feat_id in top_feature_ids:
                feat_val = feature_acts[i, t, feat_id].item()
                if (not active_only) or (feat_val > eps):
                    token_features.append((feat_id, feat_val))
            tokens_top_features.append(token_features)
        batch_top_features.append(tokens_top_features)

    return batch_top_features, top_feature_ids

def ablate(feature_acts, batch_top_features, batch_random_features, block_name, keep_feature='', changed_token_mask=None):
    if keep_feature == 'good' or keep_feature == 'sensitivity':
        batch_top_features = [batch_top_features[0], batch_top_features[0]]
    elif keep_feature == 'bad':
        batch_top_features = [batch_top_features[1], batch_top_features[1]]
    elif keep_feature == 'random':
        batch_top_features = batch_random_features
    # decode back to activations
    sae_out = sae.decode(feature_acts)
    print(f"Decoded SAE output shape: {sae_out.shape}")
    # calculate reconstruction error
    recon_error_vector = cache[block_name] - sae_out
    print(f"Reconstruction error shape: {recon_error_vector.shape}")
    # Mean ablate the top 90% features
    # Calculate the mean of feature_acts
    # Apply padding mask when calculating feature mean
    
    activation_means = feature_acts.mean(dim=(0, 1))  # [n_features]
    print(f"Feature means shape: {activation_means.shape}")
    # Create a copy of feature_acts to modify
    feature_acts_ablated = feature_acts.clone()

    # Ablate top features by replacing with mean, restricted to changed tokens only if mask provided
    
    for i in range(len(batch_top_features)):
        # skip the "good" sentence for now to check if ablation is working
        for t in range(len(batch_top_features[i])):
            token_pos = t + 1  # batch_top_features skips BOS, so map list index t -> token position t+1
            # if changed_token_mask is not None and not changed_token_mask[i, token_pos]:
            #     continue
            for feat, _ in batch_top_features[i][t]:
                feature_acts_ablated[i, token_pos, feat] = activation_means[feat] # just for checking
                # feature_acts_ablated[i, token_pos, feat] = 0 # just for checking
                # break # only good sentence is ablated for now
                # feature_acts_ablated[i, t, feat] = activation_means[i,t]

    print("Applied mean ablation to top features.")
    # Decode the ablated features
    sae_out_ablated = sae.decode(feature_acts_ablated)
    print(f"Decoded ablated SAE output shape: {sae_out_ablated.shape}")
    # Add back error vector to get modified activations
    modified_activations = sae_out_ablated + recon_error_vector
    print(f"Modified activations shape: {modified_activations.shape}")
    return modified_activations

def run_model(padding_mask):
    with torch.no_grad():
        # Compare outputs
        output, _ = model.run_with_cache(sampled_sentences, prepend_bos=True)

        logits_sums = []
        enc = tokenizer(sampled_sentences, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")
        logits = output  # [B, T, V]
        shift_logits = logits[:, :-1, :]      # [B, T, V]: position i predicts token i+1 (BOS at 0 predicts t1)
        shift_labels = input_ids               # [B, T]: real tokens t1..tT (no BOS)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
        token_logp = token_logp * attention_mask  # [B, T]: mask padding, BOS not a label
        sent_loglik = token_logp.sum(dim=1)   # [B]
        avg_sent_loglik = sent_loglik / attention_mask.sum(dim=1)  # [B]: average loglik per token (excluding padding and BOS)
        for i in range(output.shape[0]):
            output_masked = output[i][~attention_mask[i]]
            logits_sum = output_masked.sum().item()
            logits_sums.append(logits_sum)
        return logits_sums, avg_sent_loglik.tolist(), sent_loglik.tolist()

def compare_sentence_loglikelihoods(avg_sent_loglik, model, padding_mask, run_model, sae, ablated_activations):
    model.reset_hooks()  # Clear any hooks from the baseline run
    model.add_hook(sae.cfg.metadata.hook_name, lambda value, hook: ablated_activations)
    _, avg_sent_loglik_ablated, sent_loglik_ablated = run_model(padding_mask)
    results = []
    delta_loglik = avg_sent_loglik[0] - avg_sent_loglik[1]
    delta_ablation_loglik = avg_sent_loglik_ablated[0] - avg_sent_loglik_ablated[1]
    delta_ablation_baseline = delta_ablation_loglik - delta_loglik
    if avg_sent_loglik[0] > avg_sent_loglik[1]:
        results.append(('good', delta_loglik, delta_ablation_loglik, delta_ablation_baseline))
    else:    
        results.append(('bad', delta_loglik, delta_ablation_loglik, delta_ablation_baseline))
    # Run the modified activations through the rest of the model
    
    if avg_sent_loglik_ablated[0] > avg_sent_loglik_ablated[1]:
        results.append(('good', delta_loglik, delta_ablation_loglik, delta_ablation_baseline))
    else:    
        results.append(('bad', delta_loglik, delta_ablation_loglik, delta_ablation_baseline))
    return results
#%%
## This experiment is designed to test the effect of ablating top features vs. random features on the model's ability to distinguish between the "good" and "bad" sentences in the BLIMP minimal pair. We will run this analysis for each layer's SAE and compare the results across layers and ablation types.
saved_top_features = {
    'good': {},
    'bad': {},
    'random': {},
    'sensitivity': {},
}
for mode in ['sensitivity']:
    print(f"\nRunning analysis with keep_feature='{mode}'...")
    all_results = []
    model.reset_hooks()  # Clear any existing hooks
    _, avg_sent_loglik, sent_loglik = run_model(padding_mask)
    for i in tqdm.tqdm(range(12)):
        print(f"Loading SAE for layer {i}...")
        sae_id = f"blocks.{str(i)}.hook_resid_post"
        sae = SAE.from_pretrained(release, sae_id, device="cuda")
        sae.eval()
        print(f"Running model with SAE for layer {i}...")
        feature_acts = sae.encode(cache[sae_id])
        print(f"Layer {i} feature activations shape: {feature_acts.shape}")
        print(f"Computing top features for layer {i}...")
        if mode == 'sensitivity':
            batch_top_features, top_feature_ids = load_sensitivity_top_features(
                BASE_PATH, i, sample_lengths, feature_acts, top_k=32, active_only=True
            )
        else:
            batch_top_features, _, _ = compute_top_features(sample_lengths, top_cumsum, feature_acts)
        batch_random_features = None
        if mode in ['good', 'bad']:
            saved_top_features[mode][i] = batch_top_features  # Save top features for this mode to use in random feature ablation
        elif mode == 'random':
            if i not in saved_top_features['random']:
                saved_top_features['random'][i] = compute_random_features(sample_lengths, feature_acts, batch_top_features)
            batch_random_features = saved_top_features['random'][i]
        # I need to clean this part
        
        print(f"Shape of batch_top_features for layer {i}: {len(batch_top_features)}, {len(batch_top_features[0][0])}, {len(batch_top_features[0][1]) if len(batch_top_features[0]) > 0 else 0}")
        ablated_activations = ablate(feature_acts, batch_top_features, batch_random_features, sae_id, mode, changed_token_mask)
        # ablated_activations = ablate(feature_acts, batch_active_features, sae_id)
        results = compare_sentence_loglikelihoods(avg_sent_loglik, model, padding_mask, run_model, sae, ablated_activations)  # Just run for the first layer for now to check everything is working
        all_results.append((i, results))
        
        # # Release layer-local tensors and SAE before loading the next layer.
        del ablated_activations
        del batch_top_features
        del feature_acts
        del sae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
        
    print("\nFinal Results:")
    for layer, result in all_results:
        print(f"Layer {layer}: {result}")
    csv_path = BASE_PATH / "output" / f"interpret_blimp_results_{mode}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "baseline_pref",
            "baseline_metric",
            "baseline_aux",
            "ablated_pref",
            "ablated_metric",
            "ablated_aux",
            "delta_ablation_baseline",
        ])
        for layer, result in all_results:
            baseline = result[0]
            ablated = result[1]

            baseline_pref = baseline[0] if len(baseline) > 0 else ""
            baseline_metric = baseline[1] if len(baseline) > 1 else ""
            baseline_aux = baseline[2] if len(baseline) > 2 else ""

            ablated_pref = ablated[0] if len(ablated) > 0 else ""
            ablated_metric = ablated[1] if len(ablated) > 1 else ""
            ablated_aux = ablated[2] if len(ablated) > 2 else ""

            # Prefer explicit delta in tuple index 3; fall back to ablated aux when unavailable.
            delta_ablation_baseline = ablated[3] if len(ablated) > 3 else ablated_aux
            writer.writerow([
                layer,
                baseline_pref,
                baseline_metric,
                baseline_aux,
                ablated_pref,
                ablated_metric,
                ablated_aux,
                delta_ablation_baseline,
            ])

    print(f"Saved all_results CSV to: {csv_path}")
#%%
def _feature_ids_from_batch_top_features(batch_top_features):
    ids = set()
    for sent_tokens in batch_top_features:              # sentence index
        for token_feats in sent_tokens:                 # token index (BOS already skipped upstream)
            for feat, _ in token_feats:                 # tuple: (feature_id, activation/logit)
                ids.add(int(feat))
    return ids

def compute_sensitivity_for_layer(feature_acts, sample_lengths, feature_ids=None, good_idx=0, bad_idx=1):
    # feature_acts shape: [2, T, F] in your current setup
    # sample_lengths includes BOS, and your feature lists start at token position 1
    valid_len = int(min(sample_lengths[good_idx].item(), sample_lengths[bad_idx].item()))
    if valid_len <= 1:
        raise ValueError("Not enough non-BOS tokens to compute sensitivity.")

    # token range 1..valid_len-1 (skip BOS)
    diff = feature_acts[bad_idx, 1:valid_len, :] - feature_acts[good_idx, 1:valid_len, :]  # [T_eff, F]
    sf_all = diff.mean(dim=0)  # [F], this is S_f for every feature

    if feature_ids is None:
        # return all features
        return {f: float(sf_all[f].item()) for f in range(sf_all.shape[0])}

    return {f: float(sf_all[f].item()) for f in sorted(feature_ids)}

sensitivity_results = {}  # layer -> {feature_id: S_f}

for layer in range(12):
    sae_id = f"blocks.{layer}.hook_resid_post"
    sae = SAE.from_pretrained(release, sae_id, device="cuda")
    sae.eval()

    with torch.no_grad():
        feature_acts = sae.encode(cache[sae_id])  # [2, T, F]

    # Candidate features from both good and bad sets for this layer
    good_feats_layer = saved_top_features["good"][layer]
    bad_feats_layer = saved_top_features["bad"][layer]
    candidate_ids = _feature_ids_from_batch_top_features(good_feats_layer) | _feature_ids_from_batch_top_features(bad_feats_layer)

    sf_selected = compute_sensitivity_for_layer(
        feature_acts=feature_acts,
        sample_lengths=sample_lengths,
        feature_ids=candidate_ids,
        good_idx=0,
        bad_idx=1,
    )

    sensitivity_results[layer] = sf_selected

    # Optional CSV per layer
    out_csv = BASE_PATH / "output" / f"sensitivity_Sf_layer_{layer}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "feature_id", "S_f_bad_minus_good"])
        for feat_id, sf in sorted(sf_selected.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([layer, feat_id, sf])

    del feature_acts
    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Saved sensitivity results for all layers.")

# %%
