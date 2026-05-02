"""
Extract features from specified layers of a language model using SAE, and compute log-likelihoods for a dataset of minimal pairs. 
"""
import os
import sys
import logging
import re
import tqdm
import h5py
import torch
import json
from pathlib import Path
import argparse
import random
import numpy as np

from sae_lens import SAE, SAETransformerBridge

from util.buffered_writer import BufferedFeatureWriter


# os.environ['HF_HUB_OFFLINE'] = '1'  # SAEs are already cached; skip remote freshness checks every batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

torch.set_grad_enabled(False)

def load_sae_model(sae_release: str, sae_id: str):
    sae = SAE.from_pretrained(sae_release, sae_id, device='cuda')
    sae.eval()
    return sae

def load_model(model_name: str):
    model = SAETransformerBridge.boot_transformers(model_name, device="cuda")
    model.eval()
    return model

def compute_log_likelihood(logits, input_ids, attention_mask):
    shift_logits = logits[:, :-1, :]      # [B, T, D]: position i predicts token i+1 (BOS at 0 predicts t1)
    shift_labels = input_ids[:, 1:]       # [B, T]: labels are shifted left (BOS is not a label, last token is not predicted)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    token_logp = token_logp * attention_mask[:, 1:]  # [B, T]: mask padding
    
    # Aggregate log-likelihoods
    sent_loglik = token_logp.sum(dim=1)   # [B]
    batch_avg_sent_loglik = sent_loglik / attention_mask[:, 1:].sum(dim=1)  # [B]: average loglik per token (excluding padding and BOS)
    return batch_avg_sent_loglik

def run_model_with_targeted_cache(model, model_name, sae, sae_id, sampled_sentences, prepend_bos=True):
    with torch.no_grad():
        layer_number = extract_layer_number(sae_id)
        if 'gpt2' in model_name:
            # GPT 2 specifics
            hook_block_name = f'blocks.{layer_number}.hook_in.hook_sae_acts_post'
        else:
            hook_block_name = f'blocks.{layer_number}.hook_out.hook_sae_acts_post'

        logits, cache = model.run_with_cache_with_saes(
            sampled_sentences,
            saes=[sae],
            prepend_bos=prepend_bos
        )
        input_ids = model.to_tokens(sampled_sentences, prepend_bos=prepend_bos)
        pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        attention_mask = (input_ids != pad_id).long()
        batch_avg_sent_loglik = compute_log_likelihood(logits, input_ids, attention_mask)
        feature_acts = cache[hook_block_name]

        return batch_avg_sent_loglik.tolist(), input_ids, attention_mask, feature_acts


def calculate_preference_margin(loglik1, loglik2):
    # loglik1 and loglik2 are lists of log-likelihoods for two sets of sentences (e.g., correct vs incorrect)
    margins = [l1 - l2 for l1, l2 in zip(loglik1, loglik2)]
    return margins

def calculate_accuracy(margins):
    # Calculate the percentage of cases where margin > 0 (model prefers the first set of sentences)
    correct = sum(1 for m in margins if m > 0)
    total = len(margins)
    accuracy = correct / total if total > 0 else 0
    return accuracy

def extract_layer_number(layer_str: str) -> int:
    """
    Extracts the layer number from various layer string formats:
    - "blocks.{i}.hook_resid_pre"     -> i
    - "layer_{i}_width_1m_l0_big"      -> i
    - "layer{i}"                -> i
    """
    patterns = [
        r"blocks\.(\d+)\.",        # blocks.{i}.hook_resid_post
        r"layer_(\d+)_",           # layer_{i}_width_1m_l0_big
        r"layer(\d+)",           # layer{i}.sae.pt
    ]

    for pattern in patterns:
        match = re.search(pattern, layer_str)
        if match:
            return int(match.group(1))

    raise ValueError(f"Could not extract layer number from: {layer_str!r}")

def get_sae_release_id(model_name) -> tuple[str, list]:
    # Map model names to SAE release IDs
    mapping_release = {
        "gpt2": "gpt2-small-res-jb",
        "google/gemma-3-270m": "gemma-scope-2-270m-pt-res",
        "google/gemma-3-1b-pt": "gemma-scope-2-1b-pt-res",
        "google/gemma-3-4b-pt": "gemma-scope-2-4b-pt-res",
        "google/gemma-3-12b-pt": "gemma-scope-2-12b-pt-res",
        "google/gemma-3-27b-pt": "gemma-scope-2-27b-pt-res",
        "Qwen/Qwen3.5-2B": "qwen-scope-3.5-2b-base-w32k-l100",
        "Qwen/Qwen3.5-9B": "qwen-scope-3.5-9b-base-w64k-l100",
    }   
    
    mapping_sae_id = {
        "gpt2": [f"blocks.{i}.hook_resid_pre" for i in range(12)],
        "google/gemma-3-270m": [f"layer_{str(i)}_width_262k_l0_big" for i in [5,9,12,15]],
        "google/gemma-3-1b-pt": [f"layer_{str(i)}_width_262k_l0_big" for i in [7,13,17,22]],
        "google/gemma-3-4b-pt": [f"layer_{str(i)}_width_262k_l0_big" for i in [9,17,22,29]],
        "google/gemma-3-12b-pt": [f"layer_{str(i)}_width_262k_l0_big" for i in [12,24,31,41]],
        "google/gemma-3-27b-pt": [f"layer_{str(i)}_width_262k_l0_big" for i in [16,31,40,53]],
        "Qwen/Qwen3.5-2B": [f"layer{i}" for i in [6, 12, 16, 20]],
        "Qwen/Qwen3.5-9B": [f"layer{i}" for i in [8, 16, 21, 27]],
    }
    release = mapping_release.get(model_name, None)
    sae_ids = mapping_sae_id.get(model_name, None)
    if release is None or sae_ids is None:
        raise ValueError(f"Model name {model_name} not found in mapping. Please provide a valid model name.")
    return release, sae_ids

def save_features_to_hdf5(writers, h5_files, sae_id, feature_acts, input_ids, attention_mask, batch_start_idx, output_dir, n_sentences):
    if sae_id not in writers:
        h5f = h5py.File(output_dir / f"{sae_id}.h5", "w")
        h5_files[sae_id] = h5f
        writers[sae_id] = BufferedFeatureWriter(
            h5f,
            n_sentences=n_sentences,
            n_features=feature_acts.shape[-1],  # infer F from activation last dim
            capacity_multiplier=0.1,           # tune: lower = less RAM, more resizes
        )
    writers[sae_id].add_batch(feature_acts, input_ids, attention_mask, batch_start_idx=batch_start_idx)

def save_dataset_evaluation_to_jsonl(output_dir, dataset_name, sentences, loglikelihoods):
    output_path = output_dir / f"{dataset_name}_evaluation.jsonl"
    with open(output_path, "w") as f:
        for sent, loglik in zip(sentences, loglikelihoods):
            json_line = json.dumps({"sentence": sent, "log_likelihood": loglik})
            f.write(json_line + "\n")
    logging.info(f"Saved dataset evaluation results to {output_path}")
    good_avg_loglik = loglikelihoods[0::2]  # Even indices are "good" sentences
    bad_avg_loglik = loglikelihoods[1::2]   # Odd indices are "bad" sentences  
    preference_margins = calculate_preference_margin(good_avg_loglik, bad_avg_loglik)
    accuracy = calculate_accuracy(preference_margins)
    output_path = output_dir / f"{dataset_name}_preference_margins.jsonl"
    with open(output_path, "w") as f:
        for sent_good, sent_bad, loglik_good, loglik_bad, margin in zip(sentences[0::2], sentences[1::2], good_avg_loglik, bad_avg_loglik, preference_margins):
            json_line = json.dumps({
                "sentence_good": sent_good,
                "sentence_bad": sent_bad,
                "log_likelihood_good": loglik_good,
                "log_likelihood_bad": loglik_bad,
                "preference_margin": margin
            })
            f.write(json_line + "\n")
    output_path = output_dir / f"{dataset_name}_accuracy.txt"
    with open(output_path, "w") as f:
        f.write(f"Preference margin accuracy: {accuracy:.4f}\n")
    logging.info(f"Saved preference margins and accuracy to {output_path}")
    

def main(args):
    # Load model 
    model = load_model(args.model_name)
    release, sae_ids = get_sae_release_id(args.model_name)
    # Load SAE models all at once (only if lazy_load_saes is False; otherwise they will be loaded per batch to save RAM)
    if not args.lazy_load_saes:
        saes = {sae_id: load_sae_model(release, sae_id) for sae_id in sae_ids}  # Load all SAEs into a dict
    else:
        saes = {}  # Will load lazily in the loop
    # Process the dataset
    with open(args.dataset_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    if args.sample_size is not None:
        data = data[:args.sample_size]
    sampled_sentences = []
    for dataline in data:
        sampled_sentences.append(dataline[0]['prompt'])
        sampled_sentences.append(dataline[1]['prompt'])
    logging.info(f"Sampled {len(sampled_sentences)} sentences for interpretation.")
    batch_size = args.batch_size

    output_dir = args.output_dir / args.model_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_files = {}
    writers = {}
    n_sentences = len(sampled_sentences)
    cache = None
    is_finished_calc_log = False # Flag for calculating log likelihood once only as they are the same across SAEs
    try:
        
        if args.lazy_load_saes:
            logging.info("SAE models will be loaded lazily per batch to save RAM.")
            for sae_id in sae_ids:
                logging.debug(f"Processing with SAE ID: {sae_id}")
                sae = load_sae_model(release, sae_id)  # Load SAE for this layer only when we reach it in the loop
                all_avg_sent_loglik = []
                for i in tqdm.tqdm(range(0, len(sampled_sentences), batch_size)):
                    batch_sentences = sampled_sentences[i:i+batch_size]
                    batch_avg_sent_loglik, input_ids, attention_mask, feature_acts = run_model_with_targeted_cache(
                        model,
                        args.model_name,
                        sae, # type: ignore
                        sae_id,
                        batch_sentences,
                        prepend_bos=False if args.model_name.startswith("gpt2") else True, 
                    )
                    all_avg_sent_loglik.extend(batch_avg_sent_loglik)

                    # Save features to HDF5 using buffered writer
                    save_features_to_hdf5(writers, h5_files, sae_id, feature_acts, input_ids, attention_mask, batch_start_idx=i, output_dir=output_dir, n_sentences=n_sentences)
                if not is_finished_calc_log:
                    save_dataset_evaluation_to_jsonl(output_dir, args.dataset_name, sampled_sentences, all_avg_sent_loglik)
                    is_finished_calc_log = True
            for sae_id, writer in writers.items():
                writer.finalize()
        else:
            logging.info("All SAE models loaded at once. This may use more RAM but will be faster.")
            all_avg_sent_loglik = []
            for i in tqdm.tqdm(range(0, len(sampled_sentences), batch_size)):
                batch_sentences = sampled_sentences[i:i+batch_size]
                is_first_sae = True
                for sae_id in sae_ids:
                    logging.debug(f"Processing with SAE ID: {sae_id}")
                    sae = saes[sae_id]  # type: ignore
                    batch_avg_sent_loglik, input_ids, attention_mask, feature_acts = run_model_with_targeted_cache(
                        model,
                        args.model_name,
                        sae, # type: ignore
                        sae_id,
                        batch_sentences,
                        prepend_bos=False if args.model_name.startswith("gpt2") else True, 
                    )
                    if is_first_sae:
                        all_avg_sent_loglik.extend(batch_avg_sent_loglik)
                        is_first_sae = False

                    # Save features to HDF5 using buffered writer
                    save_features_to_hdf5(writers, h5_files, sae_id, feature_acts, input_ids, attention_mask, batch_start_idx=i, output_dir=output_dir, n_sentences=n_sentences)
            if not is_finished_calc_log:
                save_dataset_evaluation_to_jsonl(output_dir, args.dataset_name, sampled_sentences, all_avg_sent_loglik)
                is_finished_calc_log = True
            for sae_id, writer in writers.items():
                writer.finalize()
    finally:
        if cache is not None:
            del cache
        for h5py_f in h5_files.values():
            h5py_f.close()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--dataset_name', type=str, default='blimp')
    parser.add_argument('--dataset_path', type=str, default=PROJECT_ROOT / "data" / "input_data" / "blimp_data.jsonl")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=Path, default=PROJECT_ROOT / "output" / "features") # tune: change to "features" for full run
    parser.add_argument('--sample_size', type=int, default=None, help="Number of sentences to sample from the dataset for testing. Set to None for no limit.")
    parser.add_argument('--lazy_load_saes', action='store_true', help="Whether to load SAEs lazily per batch (saves RAM but slower).")
    args = parser.parse_args()
    main(args)