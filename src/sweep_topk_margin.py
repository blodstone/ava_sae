"""
Sweep top_k from 1..max_k, run zero-ablation for each k, record mean_margin_bad,
then plot the curve.

Usage:
    python sweep_topk_margin.py \
        --phi_path output/features_test/google_gemma-3-270m/layer_5_width_65k_l0_medium_phi.npz \
        --sae_id layer_5_width_65k_l0_medium \
        --max_k 10 \
        --n_pairs 100
"""

import os
import sys

def _early_gpu_ids() -> str:
    if '--gpu_ids' in sys.argv:
        return sys.argv[sys.argv.index('--gpu_ids') + 1]
    return '3,4'

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = _early_gpu_ids()

import argparse
import json
import logging
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sae_lens import SAE, HookedSAETransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
torch.set_grad_enabled(False)


def _parse_layer_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("_")[1])


def _compute_avg_loglik(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * shift_mask
    return token_logp.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)


def _zero_ablate(
    feature_acts: torch.Tensor,
    resid_post: torch.Tensor,
    sae: SAE,
    feature_ids: np.ndarray,
) -> torch.Tensor:
    feature_acts_ablated = feature_acts.clone()
    fid_tensor = torch.from_numpy(feature_ids).long().to(feature_acts.device)
    feature_acts_ablated[:, :, fid_tensor] = 0.0
    sae_out_orig = sae.decode(feature_acts)
    sae_out_ablated = sae.decode(feature_acts_ablated)
    return sae_out_ablated + (resid_post - sae_out_orig)


def run_one_k(
    model: HookedSAETransformer,
    sae: SAE,
    hook_name: str,
    pad_id: int,
    feature_ids_k: np.ndarray,
    sampled_sentences: list[str],
    batch_size: int,
) -> tuple[float, float, float]:
    """Returns (mean_margin_bad, mean_margin_good, ablated_accuracy)."""
    margins_bad: list[float] = []
    margins_good: list[float] = []
    correct_abl: list[bool] = []

    for batch_start in range(0, len(sampled_sentences), batch_size):
        batch = sampled_sentences[batch_start : batch_start + batch_size]

        logits_base_raw, cache = model.run_with_cache(
            batch,
            prepend_bos=True,
            names_filter=lambda n: n == hook_name,
        )
        logits_base = cast(torch.Tensor, logits_base_raw)
        input_ids = model.to_tokens(batch, prepend_bos=True)
        attn_mask = cast(torch.Tensor, (input_ids != pad_id)).long()
        avg_loglik_base = _compute_avg_loglik(input_ids, attn_mask, logits_base)
        del logits_base

        resid_post = cache[hook_name]
        feature_acts = sae.encode(resid_post)
        modified_acts = _zero_ablate(feature_acts, resid_post, sae, feature_ids_k)
        del cache, feature_acts, resid_post

        frozen = modified_acts.detach()
        model.reset_hooks()
        model.add_hook(hook_name, lambda tensor, hook: frozen)
        logits_abl = cast(torch.Tensor, model(batch, prepend_bos=True))
        model.reset_hooks()
        del modified_acts, frozen

        avg_loglik_abl = _compute_avg_loglik(input_ids, attn_mask, logits_abl)
        del logits_abl

        for j in range(0, len(batch), 2):

            bad_base = avg_loglik_base[j + 1].item()
            bad_abl = avg_loglik_abl[j + 1].item()
            good_base = avg_loglik_base[j].item()
            good_abl = avg_loglik_abl[j].item()
            margins_bad.append(bad_abl - bad_base)
            margins_good.append(good_abl - good_base)
            correct_abl.append(good_base > bad_abl)

    mean_mb = float(np.mean(margins_bad))
    mean_mg = float(np.mean(margins_good))
    acc_abl = float(np.mean(correct_abl))
    return mean_mb, mean_mg, acc_abl


def _load_candidates(
    phi_path: Path, phi_sign: str, max_k: int
) -> tuple[np.ndarray, int]:
    """Return (candidate_idx, effective_max_k) for a phi .npz file."""
    data = np.load(phi_path)
    sorted_phi_idx: np.ndarray = data["sorted_phi_idx"]
    sorted_phi_values: np.ndarray = data["sorted_phi_values"]
    if phi_sign == "positive":
        mask = sorted_phi_values > 0
    else:
        mask = sorted_phi_values < 0
    candidate_idx = sorted_phi_idx[mask]
    candidate_vals = sorted_phi_values[mask]
    if phi_sign == "negative":
        order = np.argsort(candidate_vals)
        candidate_idx = candidate_idx[order]
    return candidate_idx, min(max_k, len(candidate_idx))


def sweep_one(
    phi_path: Path,
    sae_id: str,
    model: HookedSAETransformer,
    sae: SAE,
    sampled_sentences: list[str],
    pad_id: int,
    phi_sign: str,
    max_k: int,
    batch_size: int,
    out_path: Path,
) -> None:
    candidate_idx, effective_max_k = _load_candidates(phi_path, phi_sign, max_k)
    if effective_max_k == 0:
        logging.warning(f"No {phi_sign}-phi features in {phi_path.name}, skipping.")
        return
    logging.info(f"[{sae_id}] Sweeping top_k=1..{effective_max_k} over {phi_sign}-phi features")

    layer = _parse_layer_from_sae_id(sae_id)
    hook_name = f"blocks.{layer}.hook_resid_post"

    ks: list[int] = []
    mean_margin_bads: list[float] = []
    mean_margin_goods: list[float] = []
    ablated_accs: list[float] = []

    for k in range(1, effective_max_k + 1):
        feature_ids_k = candidate_idx[:k].astype(np.int32)
        mean_mb, mean_mg, acc_abl = run_one_k(
            model, sae, hook_name, pad_id, feature_ids_k, sampled_sentences, batch_size,
        )
        ks.append(k)
        mean_margin_bads.append(mean_mb)
        mean_margin_goods.append(mean_mg)
        ablated_accs.append(acc_abl)
        logging.info(
            f"[{sae_id}] top_k={k:3d}  mean_margin_bad={mean_mb:+.4f}  mean_margin_good={mean_mg:+.4f}  ablated_acc={acc_abl:.4f}"
        )

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax1.plot(ks, mean_margin_bads, marker="o", linewidth=1.5)
    ax1.axhline(0, color="red", linestyle="--", linewidth=0.8, label="M=0")
    ax1.set_ylabel(r"$(1/|D|)\sum_i M_{i,l,\mathrm{bad}}$")
    ax1.set_title(f"Mean margin_bad vs top_k ({phi_sign}-phi, {sae_id})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(ks, mean_margin_goods, marker="^", color="blue", linewidth=1.5)
    ax2.axhline(0, color="red", linestyle="--", linewidth=0.8, label="M=0")
    ax2.set_ylabel("Mean margin good")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax3.plot(ks, ablated_accs, marker="s", color="orange", linewidth=1.5)
    ax3.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax3.set_xlabel("top_k features ablated")
    ax3.set_ylabel("Ablated accuracy")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"[{sae_id}] Saved chart to {out_path}")


def main(args):
    # Load dataset once
    with open(args.dataset_path) as f:
        raw = [json.loads(line) for line in f]
    if args.n_pairs is not None:
        raw = raw[: args.n_pairs]
    sampled_sentences: list[str] = []
    for d in raw:
        sampled_sentences.append(d[0]["prompt"])
        sampled_sentences.append(d[1]["prompt"])

    model = HookedSAETransformer.from_pretrained(args.model_name, device="cuda")
    model.eval()
    assert model.tokenizer is not None
    _raw_pad = model.tokenizer.pad_token_id
    pad_id = int(_raw_pad) if _raw_pad is not None else 0  # type: ignore[arg-type]

    if args.phi_dir is not None:
        npz_files = sorted(args.phi_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {args.phi_dir}")
        for phi_path in npz_files:
            sae_id = phi_path.stem   # filename without extension = sae_id
            out_path = args.out_path.parent / f"sweep_topk_{sae_id}_{args.phi_sign}.png"
            sae = SAE.from_pretrained(args.sae_release, sae_id, device="cuda")
            sae.eval()
            sweep_one(
                phi_path, sae_id, model, sae, sampled_sentences, pad_id,
                args.phi_sign, args.max_k, args.batch_size, out_path,
            )
            del sae
            torch.cuda.empty_cache()
    else:
        if args.phi_path is None:
            raise ValueError("Provide either --phi_path or --phi_dir.")
        sae_id = args.sae_id
        sae = SAE.from_pretrained(args.sae_release, sae_id, device="cuda")
        sae.eval()
        sweep_one(
            args.phi_path, sae_id, model, sae, sampled_sentences, pad_id,
            args.phi_sign, args.max_k, args.batch_size, args.out_path,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sweep top_k and plot mean_margin_bad.")
    parser.add_argument("--phi_path", type=Path, default=None, help="Single .npz phi file.")
    parser.add_argument("--phi_dir", type=Path, default=None, help="Directory of .npz files; runs sweep for each.")
    parser.add_argument("--sae_id", type=str, default="layer_5_width_65k_l0_medium", help="SAE id (single-file mode only).")
    parser.add_argument("--phi_sign", type=str, default="positive", choices=["positive", "negative"])
    parser.add_argument("--max_k", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="google/gemma-3-270m")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-2-270m-pt-res")
    parser.add_argument(
        "--dataset_path", type=Path,
        default=PROJECT_ROOT / "data" / "input_data" / "blimp_data.jsonl",
    )
    parser.add_argument(
        "--out_path", type=Path,
        default=PROJECT_ROOT / "output" / "sweep_topk_margin.png",
        help="Output chart path (single-file mode). In --phi_dir mode, charts are saved alongside as sweep_topk_<sae_id>_<sign>.png.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_pairs", type=int, default=None)
    parser.add_argument("--gpu_ids", type=str, default="3,4")
    args = parser.parse_args()
    main(args)
