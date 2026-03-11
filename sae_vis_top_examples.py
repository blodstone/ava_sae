"""
Generate SAE visualizations for top features using model activations.

This script loads a Sparse Autoencoder (SAE) and a language model, processes
text data, and creates interactive visualizations showing which features activate
for different inputs. Handles version compatibility across different sae-vis APIs.
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from sae_lens import SAE
from sae_vis import SaeVisConfig, SaeVisData
import inspect


def build_vis_cfg(n_features: int):
    sig = inspect.signature(SaeVisConfig.__init__)
    allowed = set(sig.parameters.keys())

    kwargs = {}
    # only pass keys that exist in this installed version
    if "n_features" in allowed:
        kwargs["n_features"] = n_features
    if "features" in allowed:
        kwargs["features"] = n_features
    if "feature_centric_layout" in allowed:
        kwargs["feature_centric_layout"] = True

    return SaeVisConfig(**kwargs)


def create_vis_data_compat(model, sae, tokens, attention_mask, cfg):
    sig = inspect.signature(SaeVisData.create)
    p = set(sig.parameters.keys())

    # Try most common signatures first
    if {"model", "sae", "tokens", "cfg"}.issubset(p):
        kwargs = {"model": model, "sae": sae, "tokens": tokens, "cfg": cfg}
        if "attention_mask" in p:
            kwargs["attention_mask"] = attention_mask
        return SaeVisData.create(**kwargs)

    if {"model", "sae", "token_ids", "cfg"}.issubset(p):
        kwargs = {"model": model, "sae": sae, "token_ids": tokens, "cfg": cfg}
        if "attention_mask" in p:
            kwargs["attention_mask"] = attention_mask
        return SaeVisData.create(**kwargs)

    # last-resort attempt
    try:
        return SaeVisData.create(model=model, sae=sae, tokens=tokens, attention_mask=attention_mask, cfg=cfg)
    except TypeError:
        return SaeVisData.create(model, sae, tokens, cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae-dir", type=str, required=True)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--hook-name", type=str, default="blocks.20.hook_mlp_out")
    p.add_argument("--dataset-jsonl", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="folio_sae_vis")
    p.add_argument("--n-features", type=int, default=128)
    p.add_argument("--max-prompts", type=int, default=256)
    p.add_argument("--ctx-len", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SAE
    sae = SAE.load_from_pretrained(args.sae_dir, device=device)

    # Model + tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    # Data
    ds = load_dataset("json", data_files=args.dataset_jsonl, split="train")
    ds = ds.select(range(min(len(ds), args.max_prompts)))

    # Robust text extraction
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]
    raw_texts = ds[text_col]

    texts = []
    for x in raw_texts:
        if x is None:
            continue
        if isinstance(x, str):
            s = x.strip()
        else:
            s = str(x).strip()
        if s:
            texts.append(s)

    if not texts:
        raise ValueError(
            f"No valid text found in column '{text_col}'. Columns: {ds.column_names}"
        )

    toks = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.ctx_len,
    )
    tokens = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)

    # Remove hook_name from config construction
    cfg = build_vis_cfg(args.n_features)

    vis_data = create_vis_data_compat(
        model=model,
        sae=sae,
        tokens=tokens,
        attention_mask=attention_mask,
        cfg=cfg,
    )

    # Save with compatibility fallback
    if hasattr(vis_data, "save_feature_centric_vis"):
        vis_data.save_feature_centric_vis(out_dir)
    elif hasattr(vis_data, "save"):
        vis_data.save(out_dir)
    else:
        raise AttributeError("No known save method on SaeVisData in this version.")


if __name__ == "__main__":
    main()