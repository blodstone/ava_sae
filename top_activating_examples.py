"""
Module for finding top activating examples per SAE (Sparse Autoencoder) feature.

This script loads a pre-trained SAE and language model, then processes a dataset to identify
the token-level examples that maximally activate each feature in the SAE's latent space.
Results are saved as CSV and JSON outputs with token context information.
"""
import argparse
import csv
import heapq
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    files = list(sae_dir.glob("*.safetensors")) + list(sae_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No SAE weight file found in: {sae_dir}")
    return files[0]


def load_state_dict(weight_file: Path) -> Dict[str, torch.Tensor]:
    if weight_file.suffix == ".safetensors":
        return load_safetensors(str(weight_file))
    obj = torch.load(weight_file, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unsupported checkpoint format: {weight_file}")


def pick_tensor(state_dict: Dict[str, torch.Tensor], keys: Tuple[str, ...]) -> Optional[torch.Tensor]:
    for k in keys:
        if k in state_dict and isinstance(state_dict[k], torch.Tensor):
            return state_dict[k]
    for k, v in state_dict.items():
        lk = k.lower()
        if any(kk.lower() in lk for kk in keys) and isinstance(v, torch.Tensor):
            return v
    return None


def load_sae_params(sae_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sd = load_state_dict(find_weight_file(sae_dir))

    w_enc = pick_tensor(sd, ("W_enc", "encoder.weight", "sae.W_enc", "autoencoder.W_enc"))
    b_enc = pick_tensor(sd, ("b_enc", "encoder.bias", "sae.b_enc", "autoencoder.b_enc"))
    w_dec = pick_tensor(sd, ("W_dec", "decoder.weight", "sae.W_dec", "autoencoder.W_dec"))
    b_dec = pick_tensor(sd, ("b_dec", "decoder.bias", "sae.b_dec", "autoencoder.b_dec"))

    if w_enc is None or b_enc is None or w_dec is None:
        keys = "\n".join(list(sd.keys())[:120])
        raise KeyError(
            "Could not find SAE tensors. Need W_enc, b_enc, W_dec.\n"
            f"Available keys (first 120):\n{keys}"
        )
    if b_dec is None:
        # safe fallback if b_dec is absent
        d_in = w_dec.shape[1] if w_dec.shape[0] < w_dec.shape[1] else w_dec.shape[0]
        b_dec = torch.zeros(d_in, dtype=w_enc.dtype)

    # enforce shapes: W_enc [d_in, d_sae], W_dec [d_sae, d_in]
    if w_dec.shape[0] > w_dec.shape[1]:
        # likely [d_in, d_sae]
        w_dec = w_dec.T
    if w_enc.shape[0] > w_enc.shape[1]:
        # likely [d_sae, d_in]
        w_enc = w_enc.T

    return w_enc.float(), b_enc.float(), w_dec.float(), b_dec.float()


def load_local_or_hf_dataset(dataset_path: str, split: str) -> Dataset:
    p = Path(dataset_path)

    def _select_split(ds_obj, wanted_split: str) -> Dataset:
        if isinstance(ds_obj, DatasetDict):
            if wanted_split in ds_obj:
                return ds_obj[wanted_split]
            # fallbacks
            for alt in ("train", "validation", "test"):
                if alt in ds_obj:
                    return ds_obj[alt]
            return ds_obj[next(iter(ds_obj.keys()))]
        return ds_obj

    def _loader_from_suffix(suffix: str) -> str:
        s = suffix.lower()
        if s in (".jsonl", ".json"):
            return "json"
        if s == ".csv":
            return "csv"
        if s == ".parquet":
            return "parquet"
        if s == ".txt":
            return "text"
        raise ValueError(f"Unsupported file extension: {suffix}")

    # Local path handling
    if p.exists():
        # 1) Single file (jsonl/json/csv/parquet/txt)
        if p.is_file():
            loader = _loader_from_suffix(p.suffix)
            ds_obj = load_dataset(loader, data_files=str(p))
            return _select_split(ds_obj, split)

        # 2) Directory: try HF save_to_disk first, then raw files
        if p.is_dir():
            try:
                ds_obj = load_from_disk(str(p))
                return _select_split(ds_obj, split)
            except FileNotFoundError:
                pass  # not a HF dataset directory; try raw files below

            # discover raw dataset files in folder
            files = []
            for pat in ("*.jsonl", "*.json", "*.csv", "*.parquet", "*.txt"):
                files.extend(sorted(p.glob(pat)))

            if not files:
                raise FileNotFoundError(
                    f"No supported dataset files found in directory: {p}. "
                    "Expected one of: .jsonl, .json, .csv, .parquet, .txt"
                )

            # use one loader type at a time (prefer json/jsonl if mixed)
            priority = [".jsonl", ".json", ".parquet", ".csv", ".txt"]
            chosen_suffix = None
            for sfx in priority:
                if any(f.suffix.lower() == sfx for f in files):
                    chosen_suffix = sfx
                    break
            chosen = [str(f) for f in files if f.suffix.lower() == chosen_suffix]
            loader = _loader_from_suffix(chosen_suffix)

            # split inference from filename
            data_files = {"train": []}
            for f in chosen:
                name = Path(f).name.lower()
                if "validation" in name or "valid" in name or "val" in name:
                    data_files.setdefault("validation", []).append(f)
                elif "test" in name:
                    data_files.setdefault("test", []).append(f)
                elif "train" in name:
                    data_files.setdefault("train", []).append(f)
                else:
                    data_files.setdefault("train", []).append(f)

            ds_obj = load_dataset(loader, data_files=data_files)
            return _select_split(ds_obj, split)

    # 3) HF hub dataset id
    ds_obj = load_dataset(dataset_path, split=split)
    return _select_split(ds_obj, split)


def pick_text_field(ds: Dataset) -> Optional[str]:
    for c in ("text", "content", "prompt", "completion", "document"):
        if c in ds.column_names:
            return c
    return None


def iter_batches(
    ds: Dataset,
    tokenizer,
    batch_size: int,
    seq_len: int,
    text_field: Optional[str],
    max_examples: Optional[int],
):
    n = len(ds) if max_examples is None else min(len(ds), max_examples)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id

    i = 0
    while i < n:
        rows = ds[i : min(i + batch_size, n)]
        seq_ids = list(range(i, min(i + batch_size, n)))

        if "input_ids" in rows:
            ids_list = rows["input_ids"]
        else:
            texts = rows[text_field] if text_field else [""] * len(seq_ids)
            tok = tokenizer(
                texts,
                truncation=True,
                max_length=seq_len,
                padding=False,
                return_attention_mask=False,
            )
            ids_list = tok["input_ids"]

        trimmed = [x[:seq_len] for x in ids_list]
        max_l = max(len(x) for x in trimmed) if trimmed else 1
        max_l = min(max_l, seq_len)

        input_ids = []
        attn = []
        for x in trimmed:
            x = x[:max_l]
            pad_n = max_l - len(x)
            input_ids.append(x + [pad_id] * pad_n)
            attn.append([1] * len(x) + [0] * pad_n)

        yield (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            seq_ids,
        )
        i += batch_size


def parse_feature_ids(feature_ids: str) -> List[int]:
    return [int(x.strip()) for x in feature_ids.split(",") if x.strip()]


def get_top_norm_features(w_dec: torch.Tensor, n_features: int) -> List[int]:
    norms = torch.norm(w_dec, dim=1)
    n = min(n_features, w_dec.shape[0])
    return torch.topk(norms, k=n).indices.cpu().tolist()


def decode_context(tokenizer, ids: List[int], pos: int, window: int) -> Tuple[str, str]:
    l = max(0, pos - window)
    r = min(len(ids), pos + window + 1)
    ctx_ids = ids[l:r]
    tok_id = ids[pos]
    ctx = tokenizer.decode(ctx_ids, skip_special_tokens=False)
    tok = tokenizer.decode([tok_id], skip_special_tokens=False)
    return tok, ctx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Top activating token-level examples per SAE feature")
    ap.add_argument("--sae-dir", type=str, required=True)
    ap.add_argument("input_file", type=str, help="Path to input dataset (local path or HF id)")
    ap.add_argument("--out-dir", type=str, default="sae_top_examples")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--layer-idx", type=int, default=20)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--top-k-per-feature", type=int, default=20)
    ap.add_argument("--feature-ids", type=str, default="")
    ap.add_argument("--n-features", type=int, default=128, help="Used if --feature-ids is empty")
    ap.add_argument("--context-window", type=int, default=12)
    ap.add_argument("--max-examples", type=int, default=2000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def run_top_activating_examples(args: argparse.Namespace) -> None:

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w_enc, b_enc, w_dec, b_dec = load_sae_params(Path(args.sae_dir))
    d_sae = w_dec.shape[0]

    if args.feature_ids.strip():
        feature_ids = [f for f in parse_feature_ids(args.feature_ids) if 0 <= f < d_sae]
    else:
        feature_ids = get_top_norm_features(w_dec, args.n_features)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=None,
    ).to(args.device)
    model.eval()

    # target module: model.model.layers[layer].mlp (Qwen-style)
    target_module = model.model.layers[args.layer_idx].mlp
    hook_cache = {}

    def hook_fn(_module, _inp, out):
        hook_cache["mlp_out"] = out[0] if isinstance(out, tuple) else out

    handle = target_module.register_forward_hook(hook_fn)

    ds = load_local_or_hf_dataset(args.input_file, args.split)
    text_field = None if "input_ids" in ds.column_names else pick_text_field(ds)
    if "input_ids" not in ds.column_names and text_field is None:
        raise ValueError(f"No input_ids and no text-like field found. Columns: {ds.column_names}")

    # heaps: feature_id -> min-heap[(score, tie_breaker, record_dict)]
    heaps: Dict[int, List[Tuple[float, int, dict]]] = {f: [] for f in feature_ids}
    tie_breaker = itertools.count()

    w_enc = w_enc.to(args.device)
    b_enc = b_enc.to(args.device)
    b_dec = b_dec.to(args.device)

    with torch.no_grad():
        for input_ids, attn_mask, seq_ids in iter_batches(
            ds,
            tokenizer,
            args.batch_size,
            args.seq_len,
            text_field,
            args.max_examples,
        ):
            input_ids = input_ids.to(args.device)
            attn_mask = attn_mask.to(args.device)

            hook_cache.clear()
            _ = model(input_ids=input_ids, attention_mask=attn_mask)

            x = hook_cache["mlp_out"].float()  # [B,S,d_in]
            if x.dim() != 3:
                raise RuntimeError(f"Unexpected hooked activation shape: {tuple(x.shape)}")

            # SAE encode: relu((x - b_dec) @ W_enc + b_enc)
            pre = torch.einsum("bsd,df->bsf", x - b_dec.view(1, 1, -1), w_enc) + b_enc.view(1, 1, -1)
            acts = torch.relu(pre)  # [B,S,d_sae]

            B, S, _ = acts.shape
            ids_cpu = input_ids.detach().cpu().tolist()
            mask_cpu = attn_mask.detach().cpu()

            for f in feature_ids:
                a = acts[:, :, f]  # [B,S]
                a = a * mask_cpu.to(a.device)  # ignore padding
                flat = a.reshape(-1)
                if flat.numel() == 0:
                    continue

                local_k = min(args.top_k_per_feature, flat.numel())
                vals, idxs = torch.topk(flat, k=local_k)
                for v, idx in zip(vals.tolist(), idxs.tolist()):
                    if v <= 0:
                        continue
                    b = idx // S
                    t = idx % S
                    tok, ctx = decode_context(tokenizer, ids_cpu[b], t, args.context_window)
                    rec = {
                        "feature_id": f,
                        "score": float(v),
                        "token_id": int(ids_cpu[b][t]),
                        "token_str": tok.replace("\n", "\\n"),
                        "context": ctx.replace("\n", "\\n"),
                        "seq_id": int(seq_ids[b]),
                        "token_pos": int(t),
                    }

                    heap = heaps[f]
                    item = (rec["score"], next(tie_breaker), rec)
                    if len(heap) < args.top_k_per_feature:
                        heapq.heappush(heap, item)
                    elif rec["score"] > heap[0][0]:
                        heapq.heapreplace(heap, item)

    handle.remove()

    # write CSV
    csv_path = out_dir / "top_activating_examples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_id",
                "rank",
                "score",
                "token_id",
                "token_str",
                "context",
                "seq_id",
                "token_pos",
            ],
        )
        writer.writeheader()
        for feat in sorted(heaps.keys()):
            rows = sorted((x[2] for x in heaps[feat]), key=lambda r: r["score"], reverse=True)
            for rank, r in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "feature_id": r["feature_id"],
                        "rank": rank,
                        "score": f"{r['score']:.6f}",
                        "token_id": r["token_id"],
                        "token_str": r["token_str"],
                        "context": r["context"],
                        "seq_id": r["seq_id"],
                        "token_pos": r["token_pos"],
                    }
                )

    # write compact JSON per feature
    per_feature = {}
    for feat in sorted(heaps.keys()):
        rows = sorted((x[2] for x in heaps[feat]), key=lambda r: r["score"], reverse=True)
        per_feature[str(feat)] = rows

    with open(out_dir / "top_activating_examples.json", "w", encoding="utf-8") as f:
        json.dump(per_feature, f, indent=2, ensure_ascii=False)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {out_dir / 'top_activating_examples.json'}")


def main() -> None:
    args = parse_args()
    run_top_activating_examples(args)


if __name__ == "__main__":
    main()