"""Evaluate zero-shot FOLIO prompting with GPT-2 family models."""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = ("True", "False", "Uncertain")


def parse_torch_dtype(dtype_name: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported --dtype value: {dtype_name}")
    return mapping[dtype_name]


def infer_input_device(model, fallback_device: torch.device) -> torch.device:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        # Prefer a CUDA entry point when the model is sharded.
        for target in model.hf_device_map.values():
            if isinstance(target, str) and target.startswith("cuda"):
                return torch.device(target)
        for target in model.hf_device_map.values():
            if isinstance(target, str) and target == "cpu":
                return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def build_max_memory(max_memory_per_gpu: Optional[str], cpu_offload_gb: Optional[int]):
    if max_memory_per_gpu is None and cpu_offload_gb is None:
        return None

    max_memory = {}
    if max_memory_per_gpu is not None:
        for gpu_idx in range(torch.cuda.device_count()):
            max_memory[gpu_idx] = max_memory_per_gpu
    if cpu_offload_gb is not None:
        max_memory["cpu"] = f"{cpu_offload_gb}GiB"
    return max_memory


def load_model(args):
    torch_dtype = parse_torch_dtype(args.dtype)
    max_memory = build_max_memory(args.max_memory_per_gpu, args.cpu_offload_gb)

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
    }

    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

    return AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)


def normalize_label(label: str) -> str:
    label = str(label).strip()
    mapping = {
        "true": "True",
        "false": "False",
        "uncertain": "Uncertain",
        "unknown": "Uncertain",
    }
    return mapping.get(label.lower(), label)


def build_from_folio_row(row: dict) -> dict:
    premises_text = row["premises"]
    conclusion_text = row["conclusion"]
    prefix = (
        "Given the following premises, determine whether the conclusion is True, False, or Uncertain.\n\n"
        "Premises:\n"
        + premises_text
        + "\n\nConclusion:\n"
    )
    prompt = prefix + conclusion_text + "\nAnswer with one of: True, False, Uncertain.\nAnswer: "
    return {
        "dataset": "folio",
        "prompt": prompt,
        "id": row["story_id"],
        "label": normalize_label(row["label"]),
    }


def sequence_log_likelihood(texts, tokenizer, model, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = out.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * shift_mask
    sent_loglik = token_logp.sum(dim=1)
    return sent_loglik.tolist()


def evaluate_batch(batch_data, tokenizer, model, device):
    texts = []
    for sample in batch_data:
        for label in LABELS:
            texts.append(sample["prompt"] + label)
    # import pdb; pdb.set_trace()
    scores = sequence_log_likelihood(texts, tokenizer, model, device)

    for i, sample in enumerate(batch_data):
        candidate_scores = scores[i * len(LABELS):(i + 1) * len(LABELS)]
        pred_idx = max(range(len(LABELS)), key=lambda j: candidate_scores[j])
        sample["prediction"] = LABELS[pred_idx]


def main(args):
    folio_path = Path(args.folio_path)
    with folio_path.open() as f:
        folio_rows = [json.loads(line) for line in f]
    folio_data = [build_from_folio_row(row) for row in tqdm.tqdm(folio_rows)]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_model(args)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device_map == "none":
        model = model.to(default_device)
    input_device = infer_input_device(model, default_device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    batch_size = args.batch_size
    for i in tqdm.tqdm(range(0, len(folio_data), batch_size)):
        batch_data = folio_data[i:i + batch_size]
        evaluate_batch(batch_data, tokenizer, model, input_device)
        for sample in batch_data:
            if sample["prediction"] == sample["label"]:
                correct += 1

    accuracy = correct / len(folio_data) if folio_data else 0.0
    print(f"Model: {args.model_name}")
    print(f"Examples: {len(folio_data)}")
    print(f"Accuracy: {accuracy:.4f}")

    output_path = Path(args.output_path)
    with output_path.open("w") as f:
        for sample in folio_data:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate zero-shot prompting on FOLIO with GPT-2")
    parser.add_argument(
        "--folio_path",
        type=str,
        default="/home/users1/hardy/hardy/datasets/FOLIO/folio_v2_validation.jsonl",
        help="Path to FOLIO JSONL file",
    )
    parser.add_argument("--output_path", type=str, default="folio_results.jsonl", help="Path to save predictions")
    parser.add_argument("--model_name", type=str, default="gpt2", help="HF causal model name (e.g., gpt2, gpt2-large)")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        choices=["none", "auto", "balanced", "balanced_low_0", "sequential"],
        help="How to place model across devices; use auto/balanced for multi-GPU",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help='Per-GPU memory cap for sharding, e.g. "20GiB"',
    )
    parser.add_argument(
        "--cpu_offload_gb",
        type=int,
        default=None,
        help="Optional CPU RAM budget (GiB) for offload when using device_map",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model loading dtype",
    )
    main(parser.parse_args())
