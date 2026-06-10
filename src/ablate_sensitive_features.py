import random
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

from extract_features import calculate_accuracy, calculate_preference_margin, compute_log_likelihood, extract_layer_number, extract_layer_number, get_sae_release_id, load_model, load_sae_model
from util.blimp_util import make_blimp_prefix_mask


def ablate_top_k_active_phi(
    sae_acts, hook,
    phi_scores: torch.Tensor,
    top_k: int,
    attention_mask: torch.Tensor,
    divergence_indices: torch.Tensor,
    stats_sink: list[dict] | None = None,
):
    """Ablate the top-k φ-ranked features that are active at or after the divergence token.

    sae_acts:          [batch, seq, n_features]
    phi_scores:        [n_features] — higher = more discriminative for this term
    attention_mask:    [batch, seq] — 1 for real tokens, 0 for padding
    divergence_indices:[batch]      — first token position where good/bad sentences differ
    """
    seq_len = sae_acts.shape[1]
    time_steps = torch.arange(seq_len, device=sae_acts.device).unsqueeze(0)  # [1, seq]
    # Temporal mask: True for t >= divergence_index AND real (non-padding) token
    temporal_mask = (time_steps >= divergence_indices.unsqueeze(1)) & attention_mask.bool()  # [batch, seq]
    # Determine active features only within the violation window
    post_violation_acts = sae_acts * temporal_mask.unsqueeze(-1)   # [batch, seq, n_features]
    active = post_violation_acts.abs().sum(dim=1) > 0              # [batch, n_features]
    # Restrict further to features with φ > 0
    masked_phi = phi_scores.unsqueeze(0).expand(sae_acts.shape[0], -1).clone()  # [batch, n_features]
    masked_phi[~active | (masked_phi <= 0)] = float('-inf')
    eligible_counts = (active & (phi_scores.unsqueeze(0) > 0)).sum(dim=1)        # [batch]
    k = min(top_k, sae_acts.shape[2])
    top_vals, top_indices = torch.topk(masked_phi, k=k, dim=1)     # [batch, k]
    valid = top_vals > float('-inf')                                # [batch, k]
    ablated_counts = valid.sum(dim=1)                               # [batch]
    if stats_sink is not None:
        stats_sink.append({
            "batch_size": int(sae_acts.shape[0]),
            "eligible_total": int(eligible_counts.sum().item()),
            "ablated_total": int(ablated_counts.sum().item()),
            "eligible_per_sample": eligible_counts.detach().cpu().tolist(),
            "ablated_per_sample": ablated_counts.detach().cpu().tolist(),
        })
    if not valid.any():
        return sae_acts
    sae_acts_cloned = sae_acts.clone()
    # Build boolean mask [batch, seq, n_features]: True where we zero
    idx_exp      = top_indices.unsqueeze(1).expand(-1, seq_len, -1)           # [batch, seq, k]
    valid_exp    = valid.unsqueeze(1).expand(-1, seq_len, -1)                 # [batch, seq, k]
    temporal_exp = temporal_mask.unsqueeze(-1).expand(-1, -1, k)              # [batch, seq, k]
    feature_mask = torch.zeros_like(sae_acts_cloned, dtype=torch.bool)
    feature_mask.scatter_(2, idx_exp, valid_exp & temporal_exp)
    sae_acts_cloned[feature_mask] = 0.0
    return sae_acts_cloned


def ablate_random_active(
    sae_acts, hook,
    top_k: int,
    attention_mask: torch.Tensor,
    divergence_indices: torch.Tensor,
):
    """Ablate top-k randomly chosen features that are active at or after the divergence token."""
    seq_len = sae_acts.shape[1]
    time_steps = torch.arange(seq_len, device=sae_acts.device).unsqueeze(0)
    temporal_mask = (time_steps >= divergence_indices.unsqueeze(1)) & attention_mask.bool()  # [batch, seq]
    post_violation_acts = sae_acts * temporal_mask.unsqueeze(-1)
    active = post_violation_acts.abs().sum(dim=1) > 0              # [batch, n_features]
    rand_scores = torch.rand(sae_acts.shape[0], sae_acts.shape[2], device=sae_acts.device)
    rand_scores[~active] = float('-inf')
    k = min(top_k, sae_acts.shape[2])
    top_vals, chosen = torch.topk(rand_scores, k=k, dim=1)         # [batch, k]
    valid = top_vals > float('-inf')
    if not valid.any():
        return sae_acts
    sae_acts_cloned = sae_acts.clone()
    idx_exp      = chosen.unsqueeze(1).expand(-1, seq_len, -1)
    valid_exp    = valid.unsqueeze(1).expand(-1, seq_len, -1)
    temporal_exp = temporal_mask.unsqueeze(-1).expand(-1, -1, k)
    feature_mask = torch.zeros_like(sae_acts_cloned, dtype=torch.bool)
    feature_mask.scatter_(2, idx_exp, valid_exp & temporal_exp)
    sae_acts_cloned[feature_mask] = 0.0
    return sae_acts_cloned

def load_positive_feature_ids(
    phi_path: Path, top_k: int = 10, rank_i: int = 0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return top-k positive-φ feature indices and values per linguistic term.

    Filters to φ > 0 before selecting top-k starting at rank_i.
    Returns a tuple (feature_ids, feature_values) where each element is a list
    of 1D arrays (one per term). Arrays may be shorter than top_k if fewer
    positive-φ features exist for that term.
    """
    data = np.load(phi_path)
    sorted_phi_idx = data["sorted_phi_idx"]        # (n_terms, n_features)
    sorted_phi_values = data["sorted_phi_values"]  # (n_terms, n_features)
    n_terms = sorted_phi_idx.shape[0]
    feature_ids_list: list[np.ndarray] = []
    feature_values_list: list[np.ndarray] = []
    for i in range(n_terms):
        pos_mask = sorted_phi_values[i] > 0
        pos_idx = sorted_phi_idx[i][pos_mask]
        pos_val = sorted_phi_values[i][pos_mask]
        selected_idx = pos_idx[rank_i: rank_i + top_k]
        selected_val = pos_val[rank_i: rank_i + top_k]
        if len(selected_idx) == 0:
            logging.warning(
                f"Term {i} has no positive-φ features at rank_i={rank_i} in {phi_path.name}; "
                f"ablation will be a no-op for this term."
            )
        elif len(selected_idx) < top_k:
            logging.warning(
                f"Term {i} has only {len(selected_idx)} positive-φ features "
                f"(requested top_k={top_k}, rank_i={rank_i}) in {phi_path.name}."
            )
        feature_ids_list.append(selected_idx)
        feature_values_list.append(selected_val)
    logging.info(
        f"Loaded up to top-{top_k} positive-φ features per linguistic term "
        f"(rank_i={rank_i}) from {phi_path}"
    )
    return feature_ids_list, feature_values_list

def log_feature_sentence_distribution(
    phi_path: Path,
    feature_ids: np.ndarray,
    n_sentences: int | None = None,
    output_dir: Path | None = None,
    linguistic_terms_map: dict[str, int] | None = None,
) -> None:
    h5_path = phi_path.with_suffix(".h5")
    if not h5_path.exists():
        logging.warning(f"H5 file not found at {h5_path}, skipping feature distribution logging.")
        return

    feature_ids = np.asarray(feature_ids)
    # Build per-term rows: list of (term_label, 1D feature array)
    if feature_ids.ndim == 2:
        if linguistic_terms_map is not None:
            idx_to_term = {v: k for k, v in linguistic_terms_map.items()}
            term_rows = [(idx_to_term.get(i, str(i)), feature_ids[i]) for i in range(feature_ids.shape[0])]
        else:
            term_rows = [(str(i), feature_ids[i]) for i in range(feature_ids.shape[0])]
    else:
        term_rows = [("all", feature_ids.ravel())]

    with h5py.File(h5_path, "r") as h5f:
        offsets = np.asarray(h5f["offsets"])
        feature_idx_ds = h5f["feature_idx"]
        token_idx_ds = h5f["token_idx"]
        total = int(h5f.attrs.get("n_sentences", len(offsets) - 1))
        if n_sentences is not None:
            total = min(total, n_sentences)
        n_pairs = total // 2  # ensure even

        pair_good_feat_sets: list[set] = []
        pair_bad_feat_sets: list[set] = []
        for pair_i in range(n_pairs):
            good_i, bad_i = pair_i * 2, pair_i * 2 + 1

            gs, ge = int(offsets[good_i]), int(offsets[good_i + 1])
            if ge > gs:
                good_tok = np.asarray(token_idx_ds[gs:ge])
                good_feat = np.asarray(feature_idx_ds[gs:ge])
                pair_good_feat_sets.append(set(int(f) for f in good_feat[good_tok != 0]))
            else:
                pair_good_feat_sets.append(set())

            bs, be = int(offsets[bad_i]), int(offsets[bad_i + 1])
            if be > bs:
                bad_tok = np.asarray(token_idx_ds[bs:be])
                bad_feat = np.asarray(feature_idx_ds[bs:be])
                pair_bad_feat_sets.append(set(int(f) for f in bad_feat[bad_tok != 0]))
            else:
                pair_bad_feat_sets.append(set())

    all_lines = []
    for term_label, term_fids in term_rows:
        feature_id_set = set(int(f) for f in term_fids)
        sent_counts: dict[int, dict[str, int]] = {fid: {"bad_only": 0, "good_only": 0, "both": 0} for fid in feature_id_set}
        for good_feat_set, bad_feat_set in zip(pair_good_feat_sets, pair_bad_feat_sets):
            for fid in feature_id_set:
                in_good = fid in good_feat_set
                in_bad  = fid in bad_feat_set
                if in_bad and not in_good:
                    sent_counts[fid]["bad_only"] += 1
                elif in_good and not in_bad:
                    sent_counts[fid]["good_only"] += 1
                elif in_good and in_bad:
                    sent_counts[fid]["both"] += 1

        lines = [f"Feature distribution for phenomenon '{term_label}' ({phi_path.stem}, n_pairs={n_pairs}):"]
        lines.append(f"  {'Feature':>10}  {'bad_only':>10}  {'good_only':>10}  {'both':>6}  {'bad_only_%':>11}  {'good_only_%':>12}")
        for fid in term_fids:
            fid = int(fid)
            bo = sent_counts[fid]["bad_only"]
            go = sent_counts[fid]["good_only"]
            bt = sent_counts[fid]["both"]
            lines.append(f"  {fid:>10}  {bo:>10}  {go:>10}  {bt:>6}  {100.0*bo/n_pairs:>10.1f}%  {100.0*go/n_pairs:>11.1f}%")
        all_lines.extend(lines)
        all_lines.append("")

    text = "\n".join(all_lines)
    # logging.info(text)
    if output_dir is not None:
        out_path = output_dir / f"{phi_path.stem}_feature_distribution.txt"
        out_path.write_text(text + "\n")

def load_phi_scores(phi_path: Path) -> np.ndarray:
    """Reconstruct dense φ score array of shape (n_terms, n_features).

    The .npz stores features sorted by descending φ; this inverts the sort
    to produce a dense array indexed by feature id.
    """
    data = np.load(phi_path)
    sorted_phi_idx = data["sorted_phi_idx"]        # (n_terms, n_features)
    sorted_phi_values = data["sorted_phi_values"]  # (n_terms, n_features)
    n_terms, n_features = sorted_phi_idx.shape
    phi_dense = np.zeros((n_terms, n_features), dtype=np.float32)
    for i in range(n_terms):
        phi_dense[i, sorted_phi_idx[i]] = sorted_phi_values[i]
    return phi_dense


def save_dataset_evaluation_to_jsonl(output_dir, model_name, dataset_name, save_outputs, phi_files,
                                     all_layers_loglikelihoods_base, all_layers_loglikelihoods_abl, all_layers_loglikelihoods_random):
    # all_layers_loglikelihoods_*[layer_idx] is a dict: term -> list[float] (interleaved good/bad)
    n_layers = len(phi_files)
    bonferroni_alpha = 0.001 / n_layers
    terms = list(all_layers_loglikelihoods_base[0].keys())

    all_terms_max_delta_acc = []
    output_lines = []

    for term in terms:
        term_base   = [all_layers_loglikelihoods_base[l][term]   for l in range(n_layers)]
        term_abl    = [all_layers_loglikelihoods_abl[l][term]    for l in range(n_layers)]
        term_random = [all_layers_loglikelihoods_random[l][term] for l in range(n_layers)]

        all_pref_margins_cv  = {}
        all_pref_margins_wf  = {}
        all_ctrl_margins_cv  = {}
        all_ctrl_margins_wf  = {}
        all_sel_margins_cv   = {}
        all_sel_margins_wf   = {}
        avg_pref_cv_layers   = []
        avg_pref_wf_layers   = []
        avg_ctrl_cv_layers   = []
        avg_ctrl_wf_layers   = []
        avg_sel_cv_layers    = []
        avg_sel_wf_layers    = []
        delta_acc_abl_layers    = []
        delta_acc_random_layers = []

        for layer_idx in range(n_layers):
            base   = term_base[layer_idx]
            abl    = term_abl[layer_idx]
            random_ = term_random[layer_idx]

            good_base   = base[0::2];   bad_base   = base[1::2]
            good_abl    = abl[0::2];    bad_abl    = abl[1::2]
            good_random = random_[0::2]; bad_random = random_[1::2]
            pref_cv  = calculate_preference_margin(bad_abl,    bad_base)
            ctrl_cv  = calculate_preference_margin(bad_random, bad_base)
            pref_wf  = calculate_preference_margin(good_abl,    good_base)
            ctrl_wf  = calculate_preference_margin(good_random, good_base)
            pref_sel = calculate_preference_margin(bad_abl, good_abl)
            ctrl_sel = calculate_preference_margin(bad_random, good_random)
            acc_abl    = calculate_accuracy(calculate_preference_margin(good_abl,    bad_abl))
            acc_base   = calculate_accuracy(calculate_preference_margin(good_base,   bad_base))
            acc_random = calculate_accuracy(calculate_preference_margin(good_random, bad_random))

            all_pref_margins_cv[layer_idx]  = pref_cv
            all_pref_margins_wf[layer_idx]  = pref_wf
            all_ctrl_margins_cv[layer_idx]  = ctrl_cv
            all_ctrl_margins_wf[layer_idx]  = ctrl_wf
            all_sel_margins_cv[layer_idx]   = pref_sel
            all_sel_margins_wf[layer_idx]   = ctrl_sel
            avg_pref_cv_layers.append(np.mean(pref_cv))
            avg_pref_wf_layers.append(np.mean(pref_wf))
            avg_ctrl_cv_layers.append(np.mean(ctrl_cv))
            avg_ctrl_wf_layers.append(np.mean(ctrl_wf))
            avg_sel_cv_layers.append(np.mean(pref_sel))
            avg_sel_wf_layers.append(np.mean(ctrl_sel))
            delta_acc_abl_layers.append(acc_abl - acc_base)
            delta_acc_random_layers.append(acc_random - acc_base)

        min_delta_acc = min(delta_acc_abl_layers)
        l_star = delta_acc_abl_layers.index(min_delta_acc)
        all_terms_max_delta_acc.append(min_delta_acc)
        
        try:
            wstat_c, wpval_c = wilcoxon(all_pref_margins_cv[l_star], all_ctrl_margins_cv[l_star], alternative="greater")
            wstat_w, wpval_w = wilcoxon(all_pref_margins_wf[l_star], all_ctrl_margins_wf[l_star], alternative="two-sided")
            wstat_s, wpval_s = wilcoxon(all_sel_margins_cv[l_star], all_sel_margins_wf[l_star], alternative="greater")
        except:
            logging.warning(f"Error occurred while performing Wilcoxon test for term '{term}', setting p-values to NaN.")
            wstat_c, wpval_c = np.nan, np.nan
            wstat_w, wpval_w = np.nan, np.nan
            wstat_s, wpval_s = np.nan, np.nan

        if np.all(np.array(delta_acc_abl_layers) == np.array(delta_acc_random_layers)):
            logging.warning(f"Delta accuracy values are identical for abl and random layers for term '{term}', cannot perform Wilcoxon test; setting p-value to NaN.")
            wstat_d, wpval_d = np.nan, np.nan
        else:
            wstat_d, wpval_d = wilcoxon(delta_acc_abl_layers, delta_acc_random_layers, alternative="greater")

        output_lines.append(f"\n=== Linguistic term: {term} ===")
        output_lines.append(f"Best layer: {phi_files[l_star].stem}")
        output_lines.append(f"Max delta accuracy (abl - base): {min_delta_acc:.6f}")
        output_lines.append(f"Wilcoxon (constraint,   alpha={bonferroni_alpha:.6f}): stat={wstat_c:.6f}, p={wpval_c:.6e}, sig={wpval_c < bonferroni_alpha}")
        output_lines.append(f"Wilcoxon (well-form.,   alpha={bonferroni_alpha:.6f}): stat={wstat_w:.6f}, p={wpval_w:.6e}, sig={wpval_w < bonferroni_alpha}")
        output_lines.append(f"Wilcoxon (selectivity,  alpha={bonferroni_alpha:.6f}): stat={wstat_s:.6f}, p={wpval_s:.6e}, sig={wpval_s < bonferroni_alpha}")
        output_lines.append(f"Wilcoxon (delta_acc,    alpha={bonferroni_alpha:.6f}): stat={wstat_d:.6f}, p={wpval_d:.6e}, sig={wpval_d < bonferroni_alpha}")
        output_lines.append("  Layer                           avg_pref_cv  avg_pref_wf  avg_ctrl_cv  avg_ctrl_wf  avg_sel_cv   avg_sel_rand   Wcv_p       Www_p       Wsel_p      delta_acc_abl  delta_acc_rnd  cv>0  wf~0  sel>0" )
        for layer_idx in range(n_layers):
            try:
                wcl, wpl = wilcoxon(all_pref_margins_cv[layer_idx], all_ctrl_margins_cv[layer_idx], alternative="greater")
                wwl, wpl_w = wilcoxon(all_pref_margins_wf[layer_idx], all_ctrl_margins_wf[layer_idx], alternative="two-sided")
                wsl, wpl_s = wilcoxon(all_sel_margins_cv[layer_idx], all_sel_margins_wf[layer_idx], alternative="greater")
            except:
                logging.warning(f"Error occurred while performing Wilcoxon test for layer {layer_idx}, setting p-values to NaN.")
                wcl, wpl = np.nan, np.nan
                wwl, wpl_w = np.nan, np.nan
                wsl, wpl_s = np.nan, np.nan
            cv_pass = "YES" if avg_pref_cv_layers[layer_idx] > 0 else "NO"
            wf_pass = "YES" if abs(avg_pref_wf_layers[layer_idx]) <= 0.05 else "NO"
            sel_pass = "YES" if avg_sel_cv_layers[layer_idx] > 0 else "NO"
            output_lines.append(
                f"  {phi_files[layer_idx].stem:<30}  "
                f"{avg_pref_cv_layers[layer_idx]:>11.4f}  {avg_pref_wf_layers[layer_idx]:>11.4f}  "
                f"{avg_ctrl_cv_layers[layer_idx]:>11.4f}  {avg_ctrl_wf_layers[layer_idx]:>11.4f}  "
                f"{avg_sel_cv_layers[layer_idx]:>11.4f}  {avg_sel_wf_layers[layer_idx]:>11.4f}  "
                f"{wpl:.3e}  {wpl_w:.3e}  {wpl_s:.3e}  "
                f"{delta_acc_abl_layers[layer_idx]:>13.6f}  {delta_acc_random_layers[layer_idx]:>13.6f}  "
                f"{cv_pass:>4}  {wf_pass:>4}  {sel_pass:>4}"
            )

        logging.info(
            f"[{term}] Wilcoxon constraint: stat={wstat_c:.4f} p={wpval_c:.3e} | "
            f"well-form.: stat={wstat_w:.4f} p={wpval_w:.3e} | "
            f"selectivity: stat={wstat_s:.4f} p={wpval_s:.3e} | "
            f"delta_acc: stat={wstat_d:.4f} p={wpval_d:.3e}"
        )

    if save_outputs:
        output_path = output_dir / f"{model_name}_{dataset_name}_summary.txt"
        output_path.write_text("\n".join(output_lines) + "\n")

    return float(np.mean(all_terms_max_delta_acc))


def main(args):
    """
    Scan all *_phi.npz files in --phi_dir, run ablation per layer, and report
    max_l mean_margin_bad to test: max_l (1/|D|) sum_i M_{i,l,bad} > 0.
    """
    model = load_model(args.model_name)
    release, _ = get_sae_release_id(args.model_name)
    phi_dir = args.phi_dir / args.split_name
    phi_files = sorted(phi_dir.glob("*.npz"))
    if not phi_files:
        raise FileNotFoundError(f"No *.npz files found in {phi_dir}")

    # Process the dataset
    with open(args.dataset_path) as f:
        data = [json.loads(line) for line in f]
    if args.n_pairs is not None:
        data = data[: args.n_pairs]

    sampled_sentences = {}
    linguistic_terms_set = set()
    for dataline in data:
        linguistic_term = dataline[0]["linguistics_term"]
        if linguistic_term in sampled_sentences:
            sampled_sentences[linguistic_term].append(dataline[0]["prompt"])
            sampled_sentences[linguistic_term].append(dataline[1]["prompt"])
        else:
            sampled_sentences[linguistic_term] = [dataline[0]["prompt"], dataline[1]["prompt"]]
        linguistic_terms_set.add(linguistic_term)
    
    # Create a mapping from linguistic terms to indices
    linguistic_terms_map = {term: i for i, term in enumerate(sorted(linguistic_terms_set))}

    if isinstance(args.output_dir, str):
        output_dir = Path(args.output_dir)
    else:
        output_dir = args.output_dir
    model_name = args.model_name.replace("/", "_")
    output_dir = output_dir / model_name / "ablation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "ablation_feature_stats.csv"
    stats_path.write_text("sae_id,linguistic_term,batch_start,batch_size,eligible_total,ablated_total\n")
    
    batch_size = args.batch_size

    all_layers_avg_sent_loglik_abl = []
    all_layers_avg_sent_loglik_base = []
    all_layers_avg_sent_loglik_random = []
    for phi_path in phi_files:
        # Infer sae_id from filename
        sae_id = phi_path.stem
        phi_scores = load_phi_scores(phi_path)  # (n_terms, n_features)
        feature_ids = load_positive_feature_ids(phi_path, top_k=args.top_k, rank_i=args.rank_i)  # for logging only
        log_feature_sentence_distribution(args.phi_dir / phi_path.name, feature_ids[0], n_sentences=args.n_pairs * 2 if args.n_pairs else None, output_dir=output_dir, linguistic_terms_map=linguistic_terms_map)
        sae = load_sae_model(release, sae_id)
        # Collect log-likelihoods per linguistic term for this layer
        layer_loglik_abl    = {term: [] for term in sampled_sentences}
        layer_loglik_base   = {term: [] for term in sampled_sentences}
        layer_loglik_random = {term: [] for term in sampled_sentences}
        for linguistic_term, sentences in tqdm.tqdm(sampled_sentences.items(), desc=f"{sae_id} terms", leave=False):
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i+batch_size]
                term_phi_scores = phi_scores[linguistic_terms_map[linguistic_term]]
                batch_avg_sent_loglik_abl, batch_avg_sent_loglik_base, batch_avg_sent_loglik_random = run_model_with_ablation(
                    model, args.model_name, sae, sae_id, term_phi_scores, args.top_k, batch_sentences,
                    stats_path=stats_path, linguistic_term=linguistic_term, batch_start=i,
                )
                layer_loglik_abl[linguistic_term].extend(batch_avg_sent_loglik_abl.tolist())
                layer_loglik_base[linguistic_term].extend(batch_avg_sent_loglik_base.tolist())
                layer_loglik_random[linguistic_term].extend(batch_avg_sent_loglik_random.tolist())
        all_layers_avg_sent_loglik_abl.append(layer_loglik_abl)
        all_layers_avg_sent_loglik_base.append(layer_loglik_base)
        all_layers_avg_sent_loglik_random.append(layer_loglik_random)
    return save_dataset_evaluation_to_jsonl(
        output_dir, model_name, args.dataset_name, args.save_outputs, phi_files,
        all_layers_avg_sent_loglik_base, all_layers_avg_sent_loglik_abl, all_layers_avg_sent_loglik_random
    )

def run_model_with_ablation(
    model,
    model_name,
    sae: SAE,
    sae_id: str,
    phi_scores: np.ndarray,
    top_k: int,
    sampled_sentences: list[str],
    prepend_bos: bool = True,
    stats_path: Path | None = None,
    linguistic_term: str | None = None,
    batch_start: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer_number = extract_layer_number(sae_id)
    if 'gpt2' in model_name:
        # GPT-2: no pad token, uses eos for right-padding
        pad_id = model.tokenizer.eos_token_id
        hook_block_name = f'blocks.{layer_number}.hook_in.hook_sae_acts_post'
    else:
        # Gemma and others: left-padded; fall back to eos_token_id if pad_token_id is None
        pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        hook_block_name = f'blocks.{layer_number}.hook_out.hook_sae_acts_post'
    input_ids = model.to_tokens(sampled_sentences, prepend_bos=prepend_bos)
    attention_mask, full_batch_div = compute_attention_mask_and_divergence(input_ids, pad_id)
    # Convert φ scores and masks to tensors on device
    phi_t  = torch.from_numpy(phi_scores).to(input_ids.device)
    attn_t = attention_mask.to(input_ids.device)
    hook_stats: list[dict] = []
    # ---- ablated pass: top-k φ-ranked features active at/after divergence token ----
    logits_abl = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=True,
        use_error_term=True,
        fwd_hooks=[(hook_block_name, partial(
            ablate_top_k_active_phi,
            phi_scores=phi_t, top_k=top_k,
            attention_mask=attn_t, divergence_indices=full_batch_div,
            stats_sink=hook_stats,
        ))],
    )
    if stats_path is not None and hook_stats:
        s = hook_stats[-1]
        with stats_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{sae_id},{linguistic_term or ''},{batch_start if batch_start is not None else -1},"
                f"{s['batch_size']},{s['eligible_total']},{s['ablated_total']}\n"
            )
    model.reset_saes()
    model.reset_hooks()
    # ---- baseline pass ------------------------------------------------
    logits_base = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=prepend_bos,
        use_error_term=True,
        fwd_hooks=[],
    )
    model.reset_saes()
    model.reset_hooks()
    # ---- random pass: top-k random features active at/after divergence token --------
    logits_random = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=True,
        use_error_term=True,
        fwd_hooks=[(hook_block_name, partial(
            ablate_random_active,
            top_k=top_k, attention_mask=attn_t, divergence_indices=full_batch_div,
        ))],
    )
    model.reset_saes()
    model.reset_hooks()
    batch_avg_sent_loglik_abl = compute_log_likelihood(logits_abl, input_ids, attention_mask)
    batch_avg_sent_loglik_base = compute_log_likelihood(logits_base, input_ids, attention_mask)
    batch_avg_sent_loglik_random = compute_log_likelihood(logits_random, input_ids, attention_mask)
    return batch_avg_sent_loglik_abl, batch_avg_sent_loglik_base, batch_avg_sent_loglik_random


def compute_attention_mask_and_divergence(
    input_ids: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (attention_mask, divergence_indices) for interleaved good/bad batches.

    input_ids is expected in interleaved order:
    [good_0, bad_0, good_1, bad_1, ...].
    divergence_indices has shape [batch] and assigns the same first-difference
    index to each good/bad sentence pair while ignoring padding-only positions.
    """
    attention_mask = (input_ids != pad_id).long()
    good_ids = input_ids[0::2]      # [n_pairs, seq]
    bad_ids = input_ids[1::2]       # [n_pairs, seq]
    good_attn = attention_mask[0::2]
    bad_attn = attention_mask[1::2]

    # Compare only at positions that are real tokens in both sentences.
    both_real = (good_attn & bad_attn).bool()  # [n_pairs, seq]
    diff = (good_ids != bad_ids) & both_real   # [n_pairs, seq]

    # If a pair has no valid difference (rare), fall back to first both-real token.
    has_diff = diff.any(dim=1)
    first_diff = diff.float().argmax(dim=1)
    first_both_real = both_real.float().argmax(dim=1)
    pair_div = torch.where(has_diff, first_diff, first_both_real)

    full_batch_div = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    full_batch_div[0::2] = pair_div
    full_batch_div[1::2] = pair_div
    return attention_mask, full_batch_div


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_grad_enabled(False)
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
    parser.add_argument('--top_k', type=int, default=5, help='Number of top positive-phi features to ablate per layer.')
    parser.add_argument('--rank_i', type=int, default=0, help='Rank index for random feature selection (if using --random_features).')
    parser.add_argument(
        "--output_dir", type=Path,
        default=PROJECT_ROOT / "output" / "features",
    )
    parser.add_argument("--split_name", type=str, default="test", help="Name of the dataset split (e.g., 'train', 'test') to look for in --phi_dir.")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--n_pairs", type=int, default=None,
        help="Limit to first N pairs. Defaults to all.",
    )
    parser.add_argument("--save_outputs", action="store_true", help="Whether to save detailed outputs to files.")
    args = parser.parse_args()
    main(args)