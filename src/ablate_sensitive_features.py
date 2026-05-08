import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from sae_lens import SAE
from functools import partial
import tqdm
from scipy.stats import wilcoxon

from extract_features import calculate_preference_margin, compute_log_likelihood, calculate_accuracy,extract_layer_number, get_sae_release_id, load_model, load_sae_model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_grad_enabled(False)




def ablate_features(sae_acts, hook, feature_ids):
    sae_acts[:, :, feature_ids] = 0.0
    return sae_acts

def load_pos_neg_feature_ids(phi_path: Path, top_k: int = 10, rank_i: int = 0, positive_only: bool = False) -> np.ndarray:
    """Return top-k feature indices by ascending/descending phi score."""
    data = np.load(phi_path)
    # import pdb;pdb.set_trace()
    
    if positive_only:
        sorted_phi_idx = data["sorted_phi_idx"]
        sorted_phi_values = data["sorted_phi_values"]
        pos_mask = sorted_phi_values > 0
    else:
        sorted_phi_idx = data["sorted_phi_idx"][::-1]
        sorted_phi_values = data["sorted_phi_values"][::-1]
        pos_mask = sorted_phi_values < 0
    feature_ids = sorted_phi_idx[pos_mask]
    logging.info(
        f"Loaded {len(feature_ids)} {'positive' if positive_only else 'negative'}-phi features "
        f"(out of {len(sorted_phi_idx)} total) from {phi_path}, using top {top_k}"
    )
    return feature_ids[rank_i: rank_i + top_k]

def log_feature_sentence_distribution(phi_path: Path, feature_ids: np.ndarray, n_sentences: int | None = None, output_dir: Path | None = None) -> None:
    """For each top feature, log pair-level counts matching the phi non-overlap criterion (BOS excluded)."""
    h5_path = phi_path.with_suffix(".h5")
    if not h5_path.exists():
        logging.warning(f"H5 file not found at {h5_path}, skipping feature distribution logging.")
        return

    feature_id_set = set(int(f) for f in feature_ids)
    # per-feature pair counts: bad_only (phi-eligible), good_only, both
    sent_counts: dict[int, dict[str, int]] = {fid: {"bad_only": 0, "good_only": 0, "both": 0} for fid in feature_id_set}

    with h5py.File(h5_path, "r") as h5f:
        offsets = np.asarray(h5f["offsets"])
        feature_idx_ds = h5f["feature_idx"]
        token_idx_ds = h5f["token_idx"]
        total = int(h5f.attrs.get("n_sentences", len(offsets) - 1))
        if n_sentences is not None:
            total = min(total, n_sentences)
        n_pairs = total // 2  # ensure even

        for pair_i in range(n_pairs):
            good_i, bad_i = pair_i * 2, pair_i * 2 + 1

            gs, ge = int(offsets[good_i]), int(offsets[good_i + 1])
            if ge > gs:
                good_tok = np.asarray(token_idx_ds[gs:ge])
                good_feat = np.asarray(feature_idx_ds[gs:ge])
                good_feat_set = set(int(f) for f in good_feat[good_tok != 0])
            else:
                good_feat_set = set()

            bs, be = int(offsets[bad_i]), int(offsets[bad_i + 1])
            if be > bs:
                bad_tok = np.asarray(token_idx_ds[bs:be])
                bad_feat = np.asarray(feature_idx_ds[bs:be])
                bad_feat_set = set(int(f) for f in bad_feat[bad_tok != 0])
            else:
                bad_feat_set = set()

            for fid in feature_id_set:
                in_good = fid in good_feat_set
                in_bad  = fid in bad_feat_set
                if in_bad and not in_good:
                    sent_counts[fid]["bad_only"] += 1
                elif in_good and not in_bad:
                    sent_counts[fid]["good_only"] += 1
                elif in_good and in_bad:
                    sent_counts[fid]["both"] += 1

    lines = [f"Feature distribution across pairs ({phi_path.stem}, n_pairs={n_pairs}):"]
    lines.append(f"  {'Feature':>10}  {'bad_only':>10}  {'good_only':>10}  {'both':>6}  {'bad_only_%':>11}  {'good_only_%':>12}")
    for fid in feature_ids:
        fid = int(fid)
        bo = sent_counts[fid]["bad_only"]
        go = sent_counts[fid]["good_only"]
        bt = sent_counts[fid]["both"]
        lines.append(f"  {fid:>10}  {bo:>10}  {go:>10}  {bt:>6}  {100.0*bo/n_pairs:>10.1f}%  {100.0*go/n_pairs:>11.1f}%")
    text = "\n".join(lines)
    logging.info(text)
    if output_dir is not None:
        out_path = output_dir / f"{phi_path.stem}_feature_distribution.txt"
        out_path.write_text(text + "\n")


def load_random_feature_ids(phi_path: Path, top_k: int = 10, exclude_feature_ids: np.ndarray | None = None) -> np.ndarray:
    """Return top_k feature indices sampled uniformly at random from activated features, excluding exclude_feature_ids."""
    data = np.load(phi_path)
    sorted_phi_idx = data["sorted_phi_idx"]
    sorted_phi_values = data["sorted_phi_values"]
    # Only sample from features that actually fire on this dataset (non-zero phi)
    activated_ids = sorted_phi_idx[sorted_phi_values != 0]
    if exclude_feature_ids is not None and len(exclude_feature_ids) > 0:
        exclude_set = set(int(f) for f in exclude_feature_ids)
        activated_ids = activated_ids[~np.isin(activated_ids, list(exclude_set))]
    random_feature_ids = np.random.choice(activated_ids, size=min(top_k, len(activated_ids)), replace=False)
    logging.info(
        f"Sampled {len(random_feature_ids)} random control features "
        f"(from {len(activated_ids)} activated eligible out of {len(sorted_phi_idx)} total) from {phi_path}"
    )
    return random_feature_ids


def save_dataset_evaluation_to_jsonl(output_dir, dataset_name, phi_files, sentences, 
                                     all_layers_loglikelihoods_base, all_layers_loglikelihoods_abl, all_layers_loglikelihoods_random):
    total_pref_margins_constraint_violation = []
    total_pref_margins_well_formedness = []
    total_control_pref_margins_constraint_violation = []
    total_control_pref_margins_well_formedness = []
    all_pref_margins_constraint_violation = {}
    all_pref_margins_well_formedness = {}
    all_control_pref_margins_constraint_violation = {}
    all_control_pref_margins_well_formedness = {}
    all_layers_delta_acc_abl_base = []
    all_layers_delta_acc_random_base = []

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
        
        pref_margins_abl = calculate_preference_margin(good_avg_loglik_abl, bad_avg_loglik_abl)
        pref_margins_base = calculate_preference_margin(good_avg_loglik_base, bad_avg_loglik_base)
        pref_margins_random = calculate_preference_margin(good_avg_loglik_random, bad_avg_loglik_random)

        accuracy_abl = calculate_accuracy(pref_margins_abl)
        accuracy_base = calculate_accuracy(pref_margins_base)
        accuracy_random = calculate_accuracy(pref_margins_random)

        delta_acc_abl_base = accuracy_abl - accuracy_base
        delta_acc_random_base = accuracy_random - accuracy_base

        all_pref_margins_constraint_violation[layer_idx] = pref_margins_constraint_violation
        all_pref_margins_well_formedness[layer_idx] = pref_margins_well_formedness
        all_control_pref_margins_constraint_violation[layer_idx] = control_pref_margins_constraint_violation
        all_control_pref_margins_well_formedness[layer_idx] = control_pref_margins_well_formedness
        all_layers_delta_acc_abl_base.append(delta_acc_abl_base)
        all_layers_delta_acc_random_base.append(delta_acc_random_base)

        average_pref_margin_constraint_violation = np.mean(pref_margins_constraint_violation)
        average_pref_margin_well_formedness = np.mean(pref_margins_well_formedness)
        total_pref_margins_constraint_violation.append(average_pref_margin_constraint_violation)
        total_pref_margins_well_formedness.append(average_pref_margin_well_formedness)
        average_control_pref_margin_constraint_violation = np.mean(control_pref_margins_constraint_violation)
        average_control_pref_margin_well_formedness = np.mean(control_pref_margins_well_formedness)
        total_control_pref_margins_constraint_violation.append(average_control_pref_margin_constraint_violation)
        total_control_pref_margins_well_formedness.append(average_control_pref_margin_well_formedness)
    
    max_delta_acc_abl_base = max(all_layers_delta_acc_abl_base)
    l_star_delta_acc_abl_base = all_layers_delta_acc_abl_base.index(max_delta_acc_abl_base)
    
    # "greater": directional hypothesis — ablating bad-sensitive features should raise bad-sentence log-likelihood
    wilcoxon_stat_constraint, wilcoxon_pvalue_constraint = wilcoxon(
        all_pref_margins_constraint_violation[l_star_delta_acc_abl_base],
        all_control_pref_margins_constraint_violation[l_star_delta_acc_abl_base],
        alternative="greater",
    )
    # "two-sided": no directional hypothesis for well-formedness (ablation may help or hurt good sentences)
    wilcoxon_stat_well_formedness, wilcoxon_pvalue_well_formedness = wilcoxon(
        all_pref_margins_well_formedness[l_star_delta_acc_abl_base],
        all_control_pref_margins_well_formedness[l_star_delta_acc_abl_base],
        alternative="two-sided",
    )
    wilcoxon_stat_delta_acc, wilcoxon_pvalue_delta_acc = wilcoxon(
        all_layers_delta_acc_abl_base,
        all_layers_delta_acc_random_base,
        alternative="greater",
    )
    constraint_significant = wilcoxon_pvalue_constraint < 0.001/len(all_layers_loglikelihoods_base)  # Bonferroni correction for multiple layers
    well_formedness_significant = wilcoxon_pvalue_well_formedness < 0.001/len(all_layers_loglikelihoods_base)  # Bonferroni correction for multiple layers
    delta_acc_significant = wilcoxon_pvalue_delta_acc < 0.001/len(all_layers_loglikelihoods_base)  # Bonferroni correction for multiple layers
    output_path = output_dir / f"{dataset_name}_summary.txt"

    bonferroni_alpha = 0.001 / len(all_layers_loglikelihoods_base)
    with open(output_path, "w") as f:
        f.write(f"Best layer (constraint violation): {phi_files[l_star_delta_acc_abl_base].stem}\n")
        f.write(f"Best layer (well-formedness): {phi_files[l_star_delta_acc_abl_base].stem}\n")
        f.write(f"Wilcoxon signed-rank stat (constraint_violation vs control): {wilcoxon_stat_constraint:.6f}\n")
        f.write(f"Wilcoxon signed-rank p-value (constraint_violation vs control): {wilcoxon_pvalue_constraint:.6e}\n")
        f.write(f"Wilcoxon significant (constraint_violation vs control, alpha={bonferroni_alpha:.6f}): {constraint_significant}\n")
        f.write(f"Wilcoxon signed-rank stat (well_formedness vs control): {wilcoxon_stat_well_formedness:.6f}\n")
        f.write(f"Wilcoxon signed-rank p-value (well_formedness vs control): {wilcoxon_pvalue_well_formedness:.6e}\n")
        f.write(f"Wilcoxon significant (well_formedness vs control, alpha={bonferroni_alpha:.6f}): {well_formedness_significant}\n")
        f.write(f"Wilcoxon signed-rank stat (delta_acc_abl_base vs delta_acc_random_base): {wilcoxon_stat_delta_acc:.6f}\n")
        f.write(f"Wilcoxon signed-rank p-value (delta_acc_abl_base vs delta_acc_random_base): {wilcoxon_pvalue_delta_acc:.6e}\n")
        f.write(f"Wilcoxon significant (delta_acc_abl_base vs delta_acc_random_base, alpha={bonferroni_alpha:.6f}): {delta_acc_significant}\n")
        f.write("\n--- Per-layer results ---\n")
        for layer_idx, phi_file in enumerate(phi_files):
            wstat_c, wpval_c = wilcoxon(
                all_pref_margins_constraint_violation[layer_idx],
                all_control_pref_margins_constraint_violation[layer_idx],
                alternative="greater",
            )
            wstat_w, wpval_w = wilcoxon(
                all_pref_margins_well_formedness[layer_idx],
                all_control_pref_margins_well_formedness[layer_idx],
                alternative="two-sided",
            )
            f.write(f"\nLayer: {phi_file.stem}\n")
            f.write(f"  avg pref margin (constraint violation): {total_pref_margins_constraint_violation[layer_idx]:.6f}\n")
            f.write(f"  avg pref margin (well-formedness):      {total_pref_margins_well_formedness[layer_idx]:.6f}\n")
            f.write(f"  avg control margin (constraint):        {total_control_pref_margins_constraint_violation[layer_idx]:.6f}\n")
            f.write(f"  avg control margin (well-formedness):   {total_control_pref_margins_well_formedness[layer_idx]:.6f}\n")
            f.write(f"  Wilcoxon (constraint): stat={wstat_c:.6f}, p={wpval_c:.6e}, sig={wpval_c < bonferroni_alpha}\n")
            f.write(f"  Wilcoxon (well-form.): stat={wstat_w:.6f}, p={wpval_w:.6e}, sig={wpval_w < bonferroni_alpha}\n")
    logging.info(
        f"Wilcoxon signed-rank test ({dataset_name}): stat={wilcoxon_stat_constraint:.6f}, p={wilcoxon_pvalue_constraint:.6e}"
    )
    logging.info(
        f"Wilcoxon signed-rank test ({dataset_name}): stat={wilcoxon_stat_well_formedness:.6f}, p={wilcoxon_pvalue_well_formedness:.6e}"
    )
    logging.info(
        f"Wilcoxon signed-rank test ({dataset_name}): stat={wilcoxon_stat_delta_acc:.6f}, p={wilcoxon_pvalue_delta_acc:.6e}"
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
    if isinstance(args.output_dir, str):
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
        # Infer sae_id from filename
        sae_id = phi_path.stem
        feature_ids = load_pos_neg_feature_ids(phi_path, top_k=args.top_k, positive_only=args.positive_only)
        log_feature_sentence_distribution(phi_path, feature_ids, n_sentences=args.n_pairs * 2 if args.n_pairs else None, output_dir=output_dir)
        random_feature_ids = load_random_feature_ids(phi_path, top_k=args.top_k, exclude_feature_ids=feature_ids)
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
        output_dir, args.dataset_name, phi_files, sampled_sentences,
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

    input_ids = model.to_tokens(sampled_sentences, prepend_bos=prepend_bos)
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    attention_mask = (input_ids != pad_id).long()
    # ---- ablated pass ------------------------------------------------
    logits_abl = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=prepend_bos,
        use_error_term=True,
        fwd_hooks=[(hook_block_name, partial(ablate_features, feature_ids=feature_ids))],
    )
    # ---- baseline pass (SAE active, no features ablated) -------------
    logits_base = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=prepend_bos,
        use_error_term=True,
        fwd_hooks=[],
    )
    # ---- random pass ------------------------------------------------
    logits_random = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=prepend_bos,
        use_error_term=True,
        fwd_hooks=[(hook_block_name, partial(ablate_features, feature_ids=random_feature_ids))],
    )

    batch_avg_sent_loglik_abl = compute_log_likelihood(logits_abl, input_ids, attention_mask)
    batch_avg_sent_loglik_base = compute_log_likelihood(logits_base, input_ids, attention_mask)
    batch_avg_sent_loglik_random = compute_log_likelihood(logits_random, input_ids, attention_mask)
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
    parser.add_argument('--top_k', type=int, default=10, help='Number of top positive-phi features to ablate per layer.')
    parser.add_argument('--rank_i', type=int, default=0, help='Rank index for random feature selection (if using --random_features).')
    parser.add_argument(
        "--output_dir", type=Path,
        default=PROJECT_ROOT / "output" / "features",
    )
    parser.add_argument("--positive_only", action="store_true", help="Only ablate features with positive phi.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--n_pairs", type=int, default=None,
        help="Limit to first N pairs. Defaults to all.",
    )
    args = parser.parse_args()
    main(args)