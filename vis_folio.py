import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors


def find_weight_file(sae_dir: Path) -> Path:
    candidates = [
        sae_dir / "sae_weights.safetensors",
        sae_dir / "model.safetensors",
        sae_dir / "weights.safetensors",
        sae_dir / "sae.pt",
        sae_dir / "model.pt",
        sae_dir / "state_dict.pt",
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback: pick first safetensors/pt file in folder
    all_files = list(sae_dir.glob("*.safetensors")) + list(sae_dir.glob("*.pt"))
    if not all_files:
        raise FileNotFoundError(f"No SAE weight file found in {sae_dir}")
    return all_files[0]


def load_state_dict(weight_file: Path) -> Dict[str, torch.Tensor]:
    if weight_file.suffix == ".safetensors":
        return load_safetensors(str(weight_file))
    obj = torch.load(weight_file, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj
    raise ValueError(f"Unsupported checkpoint format in {weight_file}")


def pick_tensor(state_dict: Dict[str, torch.Tensor], keys: Tuple[str, ...]) -> Optional[torch.Tensor]:
    for k in keys:
        if k in state_dict:
            return state_dict[k]
    # fuzzy match fallback
    for k in state_dict:
        lk = k.lower()
        if any(kk.lower() in lk for kk in keys):
            t = state_dict[k]
            if isinstance(t, torch.Tensor):
                return t
    return None


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, dim=-1)
    return x @ x.T


def pca_2d(x: torch.Tensor) -> np.ndarray:
    # x: [n_features, d_in]
    x_centered = x - x.mean(dim=0, keepdim=True)
    # torch.pca_lowrank is stable and available on CPU
    q = min(16, min(x_centered.shape) - 1)
    U, S, _ = torch.pca_lowrank(x_centered, q=q)
    x2 = U[:, :2] * S[:2]
    return x2.cpu().numpy()


def save_plots(
    w_dec: torch.Tensor,
    out_dir: Path,
    top_k_heatmap: int = 128,
    top_k_annotate: int = 20,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure shape [n_features, d_in]
    # SAE decoder is often [d_sae, d_in], but some checkpoints store [d_in, d_sae]
    if w_dec.shape[0] < w_dec.shape[1]:
        # likely [d_sae, d_in] already; keep
        feats = w_dec.float().cpu()
    else:
        # if uncertain, assume larger dim is d_sae and transpose
        feats = w_dec.T.float().cpu()

    n_features = feats.shape[0]
    norms = feats.norm(dim=1).numpy()

    # 1) Norm histogram
    plt.figure(figsize=(8, 5))
    plt.hist(norms, bins=60)
    plt.title("SAE Feature (Decoder Row) Norms")
    plt.xlabel("||W_dec[i]||")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_norm_hist.png", dpi=180)
    plt.close()

    # 2) PCA scatter
    coords = pca_2d(feats)
    plt.figure(figsize=(8, 7))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=norms, s=10, alpha=0.8)
    plt.colorbar(sc, label="Feature norm")
    plt.title("SAE Decoder Features (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # annotate highest-norm features
    top_idx = np.argsort(norms)[-top_k_annotate:]
    for i in top_idx:
        plt.annotate(str(i), (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.9)

    plt.tight_layout()
    plt.savefig(out_dir / "feature_pca.png", dpi=180)
    plt.close()

    # 3) Cosine similarity heatmap on top-k norm features
    k = min(top_k_heatmap, n_features)
    topk_idx = torch.tensor(np.argsort(norms)[-k:], dtype=torch.long)
    top_feats = feats[topk_idx]
    sim = cosine_similarity_matrix(top_feats).numpy()

    plt.figure(figsize=(8, 7))
    plt.imshow(sim, interpolation="nearest")
    plt.colorbar(label="cosine similarity")
    plt.title(f"Cosine Similarity (Top {k} Norm Features)")
    plt.xlabel("Feature index (top-k subset)")
    plt.ylabel("Feature index (top-k subset)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_cosine_heatmap_topk.png", dpi=180)
    plt.close()

    # 4) nearest neighbors table
    full_sim = cosine_similarity_matrix(feats)  # [n, n]
    full_sim.fill_diagonal_(-1.0)
    nn_vals, nn_idx = torch.topk(full_sim, k=5, dim=1)

    rows = []
    for i in range(n_features):
        rows.append(
            {
                "feature_id": i,
                "norm": float(norms[i]),
                "nn1_id": int(nn_idx[i, 0]),
                "nn1_cos": float(nn_vals[i, 0]),
                "nn2_id": int(nn_idx[i, 1]),
                "nn2_cos": float(nn_vals[i, 1]),
                "nn3_id": int(nn_idx[i, 2]),
                "nn3_cos": float(nn_vals[i, 2]),
                "nn4_id": int(nn_idx[i, 3]),
                "nn4_cos": float(nn_vals[i, 3]),
                "nn5_id": int(nn_idx[i, 4]),
                "nn5_cos": float(nn_vals[i, 4]),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "feature_nearest_neighbors.csv", index=False)

    # summary json
    summary = {
        "n_features": int(n_features),
        "d_in": int(feats.shape[1]),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Visualize trained SAE features")
    parser.add_argument(
        "--sae-dir",
        type=str,
        required=True,
        help="Directory produced by sparse_autoencoder.save_inference_model()",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="sae_feature_viz",
        help="Where to write plots and CSV files",
    )
    parser.add_argument("--top-k-heatmap", type=int, default=128)
    parser.add_argument("--top-k-annotate", type=int, default=20)
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    out_dir = Path(args.out_dir)

    weight_file = find_weight_file(sae_dir)
    state_dict = load_state_dict(weight_file)

    w_dec = pick_tensor(
        state_dict,
        keys=("W_dec", "decoder.weight", "sae.W_dec", "autoencoder.W_dec"),
    )
    if w_dec is None:
        available = "\n".join(list(state_dict.keys())[:80])
        raise KeyError(
            "Could not find decoder matrix in checkpoint. "
            "Tried keys: W_dec, decoder.weight, sae.W_dec, autoencoder.W_dec.\n"
            f"Available keys (first 80):\n{available}"
        )

    save_plots(
        w_dec=w_dec,
        out_dir=out_dir,
        top_k_heatmap=args.top_k_heatmap,
        top_k_annotate=args.top_k_annotate,
    )
    print(f"Done. Visualizations saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()