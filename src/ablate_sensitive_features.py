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
from util.blimp_util import make_blimp_prefix_mask


def ablate_features(sae_acts, hook, feature_ids, feature_values, attention_mask, is_control=False):
    feature_ids = torch.as_tensor(feature_ids, device=sae_acts.device)
    feature_values = torch.as_tensor(feature_values, device=sae_acts.device)
    good = sae_acts[0::2]   # shape: (n_pairs, seq, hidden)
    bad  = sae_acts[1::2]
    attention_mask = attention_mask[0::2]  # shape: (n_pairs, seq)
    good_feats = good[:, :, feature_ids] # reranking features by phi rank, shape: (n_pairs, seq, n_features)
    bad_feats  = bad[:, :, feature_ids]

    active_mask = (bad_feats != 0) & (good_feats == 0)
    
    active_mask = active_mask & attention_mask.unsqueeze(-1)  # only consider active features at attended positions

    if not is_control:
        selected_idx = active_mask.float().topk(1, dim=-1).indices
    else:
        n_pairs, seq, n_feats = active_mask.shape
        flat = active_mask.float().view(-1, n_feats)          # (n_pairs*seq, n_features)

        n_active = flat.sum(dim=-1).min().item()
        k = max(1, min(1, int(n_active)))

        selected_idx = ((flat + 1e-10)                            # epsilon avoids all-zero rows
                    .multinomial(k, replacement=False)       # (n_pairs*seq, k)
                    .view(n_pairs, seq, k))                  # (n_pairs, seq, k)
    actual_idx = feature_ids[selected_idx]

    bad_vals = bad.gather(dim=2, index=actual_idx)     # Shape: (8, 7, 1)
    good_vals = good.gather(dim=2, index=actual_idx)   # Shape: (8, 7, 1)

    mask_at_idx = (bad_vals != 0) & (good_vals == 0)

    mask_at_idx = mask_at_idx & attention_mask.unsqueeze(-1).bool()

    b_coords, seq_coords, _ = torch.where(mask_at_idx)


    valid_targets = actual_idx[b_coords, seq_coords, 0]

    # mode_seq = seq_coords.mode().values
    # mask = seq_coords == mode_seq
    # b_coords = b_coords[mask]
    # seq_coords = seq_coords[mask]
    # valid_targets = valid_targets[mask]
    # import pdb; pdb.set_trace()
    bad[b_coords, seq_coords, valid_targets] = 0

    return sae_acts

def load_pos_neg_feature_ids(phi_path: Path, top_k: int = 10, rank_i: int = 0, positive_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k feature indices by ascending/descending phi score."""
    data = np.load(phi_path)
    if positive_only:
        sorted_phi_idx = data["sorted_phi_idx"].copy()
        sorted_phi_values = data["sorted_phi_values"].copy()
    else:
        sorted_phi_idx = data["sorted_phi_idx"][:, ::-1].copy()
        sorted_phi_values = data["sorted_phi_values"][:, ::-1].copy()
    # feature_ids = sorted_phi_idx[:, rank_i: rank_i + top_k]
    # logging.info(
    #     f"Loaded {len(feature_ids)} {'positive' if positive_only else 'negative'}-phi features "
    #     f"(out of {len(sorted_phi_idx)} total) from {phi_path}, using top {top_k}"
    # )
    return (sorted_phi_idx, sorted_phi_values)

def log_feature_sentence_distribution(
    phi_path: Path,
    feature_ids: np.ndarray,
    n_sentences: int | None = None,
    output_dir: Path | None = None,
    linguistic_terms_map: dict[str, int] | None = None,
) -> None:
    """For each top feature, log pair-level counts per linguistic phenomenon."""
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


def load_random_feature_ids(phi_path: Path, top_k: int = 10, exclude_feature_ids: np.ndarray | None = None) -> np.ndarray:
    """Return top_k random feature indices per linguistic phenomenon, using a different RNG seed per term."""
    data = np.load(phi_path)
    sorted_phi_idx = data["sorted_phi_idx"]
    sorted_phi_values = data["sorted_phi_values"]

    exclude_2d = np.asarray(exclude_feature_ids) if exclude_feature_ids is not None else None
    result_rows = []
    for term_idx in range(sorted_phi_idx.shape[0]):
        term_active = np.unique(sorted_phi_idx[term_idx][sorted_phi_values[term_idx] != 0])
        if exclude_2d is not None:
            term_excl = exclude_2d[term_idx] if exclude_2d.ndim == 2 else exclude_2d.ravel()
            term_active = term_active[~np.isin(term_active, term_excl)]
        rng = np.random.default_rng(seed=term_idx)
        if len(term_active) == 0:
            logging.warning(f"Term {term_idx}: no activated features available for random sampling; using zeros.")
            result_rows.append(np.zeros(top_k, dtype=sorted_phi_idx.dtype))
            continue
        sampled = rng.choice(term_active, size=top_k, replace=len(term_active) < top_k)
        logging.info(
            f"Term {term_idx}: sampled {len(sampled)} random control features "
            f"(from {len(term_active)} activated eligible) from {phi_path}"
        )
        result_rows.append(sampled)
    return np.stack(result_rows)
    

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
        output_lines.append("  Layer                           avg_pref_cv  avg_pref_wf  avg_ctrl_cv  avg_ctrl_wf  avg_sel_cv   avg_sel_wf   Wcv_p       Www_p       Wsel_p      delta_acc_abl  delta_acc_rnd  cv>0  wf~0  sel>0" )
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
    
    batch_size = args.batch_size

    all_layers_avg_sent_loglik_abl = []
    all_layers_avg_sent_loglik_base = []
    all_layers_avg_sent_loglik_random = []
    for phi_path in phi_files:
        # Infer sae_id from filename
        sae_id = phi_path.stem
        feature_ids = load_pos_neg_feature_ids(phi_path, top_k=args.top_k, rank_i=args.rank_i, positive_only=args.positive_only)
        log_feature_sentence_distribution(args.phi_dir / phi_path.name, feature_ids[0], n_sentences=args.n_pairs * 2 if args.n_pairs else None, output_dir=output_dir, linguistic_terms_map=linguistic_terms_map)
        random_feature_ids = load_random_feature_ids(phi_path, top_k=args.top_k, exclude_feature_ids=feature_ids[0])
        sae = load_sae_model(release, sae_id)
        # Collect log-likelihoods per linguistic term for this layer
        layer_loglik_abl    = {term: [] for term in sampled_sentences}
        layer_loglik_base   = {term: [] for term in sampled_sentences}
        layer_loglik_random = {term: [] for term in sampled_sentences}
        for linguistic_term, sentences in tqdm.tqdm(sampled_sentences.items(), desc=f"{sae_id} terms", leave=False):
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i+batch_size]
                feature_ids_batch = feature_ids[0][linguistic_terms_map[linguistic_term]]
                feature_values_batch = feature_ids[1][linguistic_terms_map[linguistic_term]]
                random_feature_ids_batch = random_feature_ids[linguistic_terms_map[linguistic_term]]
                batch_avg_sent_loglik_abl, batch_avg_sent_loglik_base, batch_avg_sent_loglik_random = run_model_with_ablation(
                    model, args.model_name, sae, sae_id, feature_ids_batch, feature_values_batch, random_feature_ids_batch, batch_sentences
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
    feature_ids: np.ndarray,
    feature_values: np.ndarray,
    random_feature_ids: np.ndarray,
    sampled_sentences: list[str],
    prepend_bos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer_number = extract_layer_number(sae_id)
    if 'gpt2' in model_name:
        # GPT 2 specifics
        pad_id = model.tokenizer.eos_token_id
        hook_block_name = f'blocks.{layer_number}.hook_in.hook_sae_acts_post'
    else:
        pad_id = model.tokenizer.pad_token_id
        hook_block_name = f'blocks.{layer_number}.hook_out.hook_sae_acts_post'
    input_ids = model.to_tokens(sampled_sentences, prepend_bos=prepend_bos)
    masks, _ = make_blimp_prefix_mask(input_ids)
    attention_mask = (input_ids != pad_id).long() * masks.long()  # shape: (B, L)
    # ---- ablated pass ------------------------------------------------
    logits_abl = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=prepend_bos,
        use_error_term=True,
        fwd_hooks=[(hook_block_name, partial(ablate_features, feature_ids=feature_ids, feature_values=feature_values, attention_mask=attention_mask))],
    )
    model.reset_saes()
    model.reset_hooks()
    # ---- baseline pass (SAE active, no features ablated) -------------
    logits_base = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=prepend_bos,
        use_error_term=True,
        fwd_hooks=[],
    )
    model.reset_saes()
    model.reset_hooks()
    # ---- random pass ------------------------------------------------
    logits_random = model.run_with_hooks_with_saes(
        sampled_sentences,
        saes=[sae],
        prepend_bos=prepend_bos,
        use_error_term=True,
        fwd_hooks=[(hook_block_name, partial(ablate_features, feature_ids=feature_ids, feature_values=feature_values, attention_mask=attention_mask, is_control=True))],
    )

    batch_avg_sent_loglik_abl = compute_log_likelihood(logits_abl, input_ids, attention_mask)
    batch_avg_sent_loglik_base = compute_log_likelihood(logits_base, input_ids, attention_mask)
    batch_avg_sent_loglik_random = compute_log_likelihood(logits_random, input_ids, attention_mask)
    return batch_avg_sent_loglik_abl, batch_avg_sent_loglik_base, batch_avg_sent_loglik_random


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    parser.add_argument("--positive_only", action="store_true", help="Only ablate features with positive phi.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--n_pairs", type=int, default=None,
        help="Limit to first N pairs. Defaults to all.",
    )
    parser.add_argument("--save_outputs", action="store_true", help="Whether to save detailed outputs to files.")
    args = parser.parse_args()
    main(args)