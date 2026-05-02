import random
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE
from functools import partial
import tqdm
from scipy.stats import wilcoxon

from extract_features import calculate_accuracy, calculate_preference_margin, compute_log_likelihood, extract_layer_number, extract_layer_number, get_sae_release_id, load_model, load_sae_model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_grad_enabled(False)




def ablate_feature(sae_acts, hook, feature_id):
    sae_acts[:, :, feature_id] = 0.0
    return sae_acts

def load_positive_feature_ids(phi_path: Path, top_k: int = 10) -> np.ndarray:
    """Return top-k feature indices by descending phi score (positive-phi only)."""
    data = np.load(phi_path)
    sorted_phi_idx = data["sorted_phi_idx"]
    sorted_phi_values = data["sorted_phi_values"]
    pos_mask = sorted_phi_values > 0
    feature_ids = sorted_phi_idx[pos_mask]
    logging.info(
        f"Loaded {len(feature_ids)} positive-phi features "
        f"(out of {len(sorted_phi_idx)} total) from {phi_path}, using top {top_k}"
    )
    return feature_ids[:top_k]

def load_random_feature_ids(phi_path: Path, top_k: int = 10) -> np.ndarray:
    """Return top-k feature indices by descending phi score (nonzero-phi only)."""
    data = np.load(phi_path)
    sorted_phi_idx = data["sorted_phi_idx"]
    sorted_phi_values = data["sorted_phi_values"]
    # any activated feature
    pos_mask = ~(sorted_phi_values == 0)
    feature_ids = sorted_phi_idx[pos_mask]
    random_feature_ids = np.random.choice(feature_ids, size=min(top_k, len(feature_ids)), replace=False)
    logging.info(
        f"Loaded {len(feature_ids)} activated-phi features "
        f"(out of {len(sorted_phi_idx)} total) from {phi_path}, using top {top_k}"
    )
    return random_feature_ids


def save_dataset_evaluation_to_jsonl(output_dir, dataset_name, sentences, 
                                     all_layers_loglikelihoods_base, all_layers_loglikelihoods_abl, all_layers_loglikelihoods_random):
    total_pref_margins_constraint_violation = []
    total_pref_margins_well_formedness = []
    total_control_pref_margins_constraint_violation = []
    total_control_pref_margins_well_formedness = []
    all_pref_margins_constraint_violation = {}
    all_pref_margins_well_formedness = {}
    all_control_pref_margins_constraint_violation = {}
    all_control_pref_margins_well_formedness = {}
    for layer_idx, (loglikelihoods_base, loglikelihoods_abl, loglikelihoods_random) in enumerate(zip(
        all_layers_loglikelihoods_base, all_layers_loglikelihoods_abl, all_layers_loglikelihoods_random
    )):
        # The len of each list of loglikelihoods should be the same as the number of sentences (2x the number of pairs)
        good_avg_loglik_base = loglikelihoods_base[0::2]  # Even indices are "good" sentences
        bad_avg_loglik_base = loglikelihoods_base[1::2]   # Odd indices are "bad" sentences
        good_avg_loglik_abl = loglikelihoods_abl[0::2]  # Even indices are "good" sentences
        bad_avg_loglik_abl = loglikelihoods_abl[1::2]   # Odd indices are "bad" sentences
        good_avg_loglik_random = loglikelihoods_random[0::2]  # Even indices are "good" sentences
        bad_avg_loglik_random = loglikelihoods_random[1::2]   # Odd indices are "bad" sentences

        # If pref_margins_constraint_violation > 0, meaning the ablation made the model ignore the constraint violations more
        pref_margins_constraint_violation = calculate_preference_margin(bad_avg_loglik_abl, bad_avg_loglik_base)
        control_pref_margins_constraint_violation = calculate_preference_margin(bad_avg_loglik_random, bad_avg_loglik_base)
        
        pref_margins_well_formedness = calculate_preference_margin(good_avg_loglik_abl, good_avg_loglik_base)
        control_pref_margins_well_formedness = calculate_preference_margin(good_avg_loglik_random, good_avg_loglik_base)
        all_pref_margins_constraint_violation[layer_idx] = pref_margins_constraint_violation
        all_pref_margins_well_formedness[layer_idx] = pref_margins_well_formedness
        all_control_pref_margins_constraint_violation[layer_idx] = control_pref_margins_constraint_violation
        all_control_pref_margins_well_formedness[layer_idx] = control_pref_margins_well_formedness


        average_pref_margin_constraint_violation = np.mean(pref_margins_constraint_violation)
        average_pref_margin_well_formedness = np.mean(pref_margins_well_formedness)
        total_pref_margins_constraint_violation.append(average_pref_margin_constraint_violation)
        total_pref_margins_well_formedness.append(average_pref_margin_well_formedness)
        average_control_pref_margin_constraint_violation = np.mean(control_pref_margins_constraint_violation)
        average_control_pref_margin_well_formedness = np.mean(control_pref_margins_well_formedness)
        total_control_pref_margins_constraint_violation.append(average_control_pref_margin_constraint_violation)
        total_control_pref_margins_well_formedness.append(average_control_pref_margin_well_formedness)
    max_total_control_pref_margin_constraint_violation = max(total_control_pref_margins_constraint_violation)
    l_star_total_control_pref_margin_constraint_violation = total_control_pref_margins_constraint_violation.index(max_total_control_pref_margin_constraint_violation)
    max_total_control_pref_margin_well_formedness = max(total_control_pref_margins_well_formedness)
    l_star_total_control_pref_margin_well_formedness = total_control_pref_margins_well_formedness.index(max_total_control_pref_margin_well_formedness)
    max_total_pref_margin_constraint_violation = max(total_pref_margins_constraint_violation)
    l_star_total_pref_margin_constraint_violation = total_pref_margins_constraint_violation.index(max_total_pref_margin_constraint_violation)
    max_total_pref_margin_well_formedness = max(total_pref_margins_well_formedness)
    l_star_total_pref_margin_well_formedness = total_pref_margins_well_formedness.index(max_total_pref_margin_well_formedness)

    wilcoxon_stat_constraint, wilcoxon_pvalue_constraint = wilcoxon(
        all_pref_margins_constraint_violation[l_star_total_pref_margin_constraint_violation],
        all_control_pref_margins_constraint_violation[l_star_total_control_pref_margin_constraint_violation],
        alternative="greater",
    )
    wilcoxon_stat_well_formedness, wilcoxon_pvalue_well_formedness = wilcoxon(
        all_pref_margins_well_formedness[l_star_total_pref_margin_well_formedness],
        all_control_pref_margins_well_formedness[l_star_total_control_pref_margin_well_formedness],
        alternative="two-sided",
    )
    constraint_significant = wilcoxon_pvalue_constraint < 0.001/len(all_layers_loglikelihoods_base)  # Bonferroni correction for multiple layers
    well_formedness_significant = wilcoxon_pvalue_well_formedness < 0.001/len(all_layers_loglikelihoods_base)  # Bonferroni correction for multiple layers
    output_path = output_dir / f"{dataset_name}_summary.txt"

    bonferroni_alpha = 0.001 / len(all_layers_loglikelihoods_base)
    with open(output_path, "w") as f:
        f.write(f"Best layer index (constraint violation): {l_star_total_pref_margin_constraint_violation}\n")
        f.write(f"Best layer index (well-formedness): {l_star_total_pref_margin_well_formedness}\n")
        f.write(f"Best control layer index (constraint violation): {l_star_total_control_pref_margin_constraint_violation}\n")
        f.write(f"Best control layer index (well-formedness): {l_star_total_control_pref_margin_well_formedness}\n")
        f.write(f"Max average preference margin constraint violation (best layer): {max_total_pref_margin_constraint_violation:.6f}\n")
        f.write(f"Max average control preference margin constraint violation (best layer): {max_total_control_pref_margin_constraint_violation:.6f}\n")
        f.write(f"Max average preference margin well-formedness (best layer): {max_total_pref_margin_well_formedness:.6f}\n")
        f.write(f"Max average control preference margin well-formedness (best layer): {max_total_control_pref_margin_well_formedness:.6f}\n")
        f.write(f"Wilcoxon signed-rank stat (constraint_violation vs control): {wilcoxon_stat_constraint:.6f}\n")
        f.write(f"Wilcoxon signed-rank p-value (constraint_violation vs control): {wilcoxon_pvalue_constraint:.6e}\n")
        f.write(f"Wilcoxon significant (constraint_violation vs control, alpha={bonferroni_alpha:.6f}): {constraint_significant}\n")
        f.write(f"Wilcoxon signed-rank stat (well_formedness vs control): {wilcoxon_stat_well_formedness:.6f}\n")
        f.write(f"Wilcoxon signed-rank p-value (well_formedness vs control): {wilcoxon_pvalue_well_formedness:.6e}\n")
        f.write(f"Wilcoxon significant (well_formedness vs control, alpha={bonferroni_alpha:.6f}): {well_formedness_significant}\n")
    logging.info(
        f"Wilcoxon signed-rank test ({dataset_name}): stat={wilcoxon_stat_constraint:.6f}, p={wilcoxon_pvalue_constraint:.6e}"
    )

def main(args):
    """
    Scan all *_phi.npz files in --phi_dir, run ablation per layer, and report
    max_l mean_margin_bad to test: max_l (1/|D|) sum_i M_{i,l,bad} > 0.
    """
    model = load_model(args.model_name)
    release, _ = get_sae_release_id(args.model_name)
    phi_files = sorted(args.phi_dir.glob("*.npz"))
    if not phi_files:
        raise FileNotFoundError(f"No *.npz files found in {args.phi_dir}")

    # Process the dataset
    with open(args.dataset_path) as f:
        data = [json.loads(line) for line in f]
    if args.n_pairs is not None:
        data = data[: args.n_pairs]

    sampled_sentences = []
    for dataline in data:
        sampled_sentences.append(dataline[0]["prompt"])
        sampled_sentences.append(dataline[1]["prompt"])
    if type(args.output_dir) == str:
        output_dir = Path(args.output_dir)
    else:
        output_dir = args.output_dir
    output_dir = output_dir / args.model_name.replace("/", "_") / "ablation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = args.batch_size
    all_layers_avg_sent_loglik_abl = []
    all_layers_avg_sent_loglik_base = []
    all_layers_avg_sent_loglik_random = []
    for phi_path in phi_files:
        # Infer sae_id from filename: strip trailing _phi.npz
        sae_id = phi_path.stem
        feature_ids = load_positive_feature_ids(phi_path)
        random_feature_ids = load_random_feature_ids(phi_path)

        sae = load_sae_model(release, sae_id)
        all_avg_sent_loglik_abl = []
        all_avg_sent_loglik_base = []
        all_avg_sent_loglik_random = []
        for i in tqdm.tqdm(range(0, len(sampled_sentences), batch_size)):
            batch_sentences = sampled_sentences[i:i+batch_size]        
            batch_avg_sent_loglik_abl, batch_avg_sent_loglik_base, batch_avg_sent_loglik_random = run_model_with_ablation(
                model, args.model_name, sae, sae_id, feature_ids, random_feature_ids, batch_sentences
            )
            all_avg_sent_loglik_abl.extend(batch_avg_sent_loglik_abl.tolist())
            all_avg_sent_loglik_base.extend(batch_avg_sent_loglik_base.tolist())
            all_avg_sent_loglik_random.extend(batch_avg_sent_loglik_random.tolist())
        all_layers_avg_sent_loglik_abl.append(all_avg_sent_loglik_abl)
        all_layers_avg_sent_loglik_base.append(all_avg_sent_loglik_base)
        all_layers_avg_sent_loglik_random.append(all_avg_sent_loglik_random)
    save_dataset_evaluation_to_jsonl(
        output_dir, args.dataset_name, sampled_sentences,
        all_layers_avg_sent_loglik_abl, all_layers_avg_sent_loglik_base, all_layers_avg_sent_loglik_random
    ) 

def run_model_with_ablation(
    model,
    model_name,
    sae: SAE,
    sae_id: str,
    feature_ids: np.ndarray,
    random_feature_ids: np.ndarray,
    sampled_sentences: list[str],
    prepend_bos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer_number = extract_layer_number(sae_id)
    if 'gpt2' in model_name:
        # GPT 2 specifics
        hook_block_name = f'blocks.{layer_number}.hook_in.hook_sae_acts_post'
    else:
        hook_block_name = f'blocks.{layer_number}.hook_out.hook_sae_acts_post'

    results = []
    input_ids = model.to_tokens(sampled_sentences, prepend_bos=prepend_bos)
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    attention_mask = (input_ids != pad_id).long()
    # ---- ablated pass ------------------------------------------------
    logits_abl = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=True,
        use_error_term=True,
        fwd_hooks=[(hook_block_name, partial(ablate_feature, feature_id=feature_ids))],  # Identity forward hook to capture activations
    )
    # ---- baseline pass ------------------------------------------------
    logits_base, _ = model.run_with_cache(sampled_sentences, prepend_bos=True)
    
    # ---- random pass ------------------------------------------------
    logits_random  = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=True,
        use_error_term=True,
        fwd_hooks=[(hook_block_name, partial(ablate_feature, feature_id=random_feature_ids))],  # Identity forward hook to capture activations
    )

    batch_avg_sent_loglik_abl = compute_log_likelihood(logits_abl, input_ids, attention_mask)
    batch_avg_sent_loglik_base = compute_log_likelihood(logits_base, input_ids, attention_mask)
    batch_avg_sent_loglik_random = compute_log_likelihood(logits_random, input_ids, attention_mask)
    # ---- record per-pair results ----------------------------------------
    return batch_avg_sent_loglik_abl, batch_avg_sent_loglik_base, batch_avg_sent_loglik_random




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(
        description="Ablate positive-phi SAE features and measure log-likelihood impact on BLIMP pairs."
    )
    # Multi-layer mode
    parser.add_argument(
        "--phi_dir", type=Path, default=None,
        help="Directory containing *_phi.npz files (one per layer). Enables multi-layer hypothesis check.",
    )
    # Shared
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m")
    parser.add_argument(
        "--dataset_path", type=Path,
        default=PROJECT_ROOT / "data" / "input_data" / "blimp_data.jsonl",
    )
    parser.add_argument("--dataset_name", type=str, default="blimp", help="Name of the dataset (for output file naming).")
    parser.add_argument('--top_k', type=int, default=10, help="Number of top +/- Phi features to print.")
    parser.add_argument(
        "--output_dir", type=Path,
        default=PROJECT_ROOT / "output" / "features",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--n_pairs", type=int, default=None,
        help="Limit to first N pairs. Defaults to all.",
    )
    args = parser.parse_args()
    main(args)