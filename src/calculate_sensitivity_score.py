from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import argparse
import tqdm
from pathlib import Path
from typing import Any, cast
import json
import h5py
import numpy as np

def _process_one_wrapper(args_tuple):
    h5_path, sampled_sentences, linguistic_terms_map, out_path, sentence_idx, start_idx, end_idx = args_tuple
    return _process_one(h5_path, sampled_sentences, linguistic_terms_map, out_path, sentence_idx, start_idx, end_idx)

def _resolve_sentence_indices(n_sentences: int, sentence_idx: int | None, start_idx: int, end_idx: int | None):
    if sentence_idx is not None:
        if sentence_idx < 0 or sentence_idx >= n_sentences:
            raise ValueError(f"sentence_idx={sentence_idx} is out of range [0, {n_sentences - 1}].")
        return [sentence_idx]

    if start_idx < 0 or start_idx >= n_sentences:
        raise ValueError(f"start_idx={start_idx} is out of range [0, {n_sentences - 1}].")

    stop = n_sentences if end_idx is None else end_idx
    if stop <= start_idx or stop > n_sentences:
        raise ValueError(f"end_idx={stop} must satisfy start_idx < end_idx <= {n_sentences}.")
    return list(range(start_idx, stop))


def _read_sentence_sparse(h5f: h5py.File, sent_idx: int):
    offsets = cast(h5py.Dataset, h5f["offsets"])
    ds_token_idx = cast(h5py.Dataset, h5f["token_idx"])
    ds_feature_idx = cast(h5py.Dataset, h5f["feature_idx"])
    ds_feature_values = cast(h5py.Dataset, h5f["feature_values"])
    ds_tokens = cast(h5py.Dataset, h5f["tokens"])

    start = int(offsets[sent_idx])
    end = int(offsets[sent_idx + 1])

    token_idx = ds_token_idx[start:end]
    feature_idx = ds_feature_idx[start:end]
    feature_values = ds_feature_values[start:end]
    tokens = ds_tokens[sent_idx]

    return {
        "sentence_idx": sent_idx,
        "tokens": np.asarray(tokens, dtype=np.int32),
        "token_idx": np.asarray(token_idx, dtype=np.int32),
        "feature_idx": np.asarray(feature_idx, dtype=np.int32),
        "feature_values": np.asarray(feature_values, dtype=np.float32),
    }


def compute_phi_l(
    h5f: h5py.File,
    sampled_sentences: list[str],
    linguistic_terms_map: dict[str, int],
    sentence_indices: list[int],
    n_features: int,
):
    if len(sentence_indices) % 2 != 0:
        raise ValueError("sentence_indices must contain an even number of entries (good/bad pairs).")
    n_linguistic_terms = len(linguistic_terms_map) 
    n_pairs = len(sentence_indices) // 2
    bad_phi_sum = np.zeros((n_linguistic_terms, n_features), dtype=np.float64)
    good_phi_sum = np.zeros((n_linguistic_terms, n_features), dtype=np.float64)
    bad_phi_count = np.zeros((n_linguistic_terms, n_features), dtype=np.float64)
    good_phi_count = np.zeros((n_linguistic_terms, n_features), dtype=np.float64)
    phi_sum = np.zeros((n_linguistic_terms, n_features), dtype=np.float64)
    instance_count = np.zeros(n_linguistic_terms, dtype=np.int32)
    total_tokens = 0

    # Track pair-level counts for correct distribution reporting
    bad_only_pair_count = np.zeros((n_linguistic_terms, n_features), dtype=np.int32)   # bad but not good (φ-eligible)
    good_only_pair_count = np.zeros((n_linguistic_terms, n_features), dtype=np.int32)  # good but not bad
    both_pair_count = np.zeros((n_linguistic_terms, n_features), dtype=np.int32)       # both good and bad

    for idx in tqdm.tqdm(range(0, len(sampled_sentences), 2), desc="Processing sentence pairs"):
        linguistic_term, global_idx = sampled_sentences[idx]  
        linguistic_term_index = linguistic_terms_map[linguistic_term]
        good_idx = sentence_indices[global_idx]
        bad_idx = sentence_indices[global_idx + 1]
        good_sent_data = _read_sentence_sparse(h5f, good_idx)
        bad_sent_data = _read_sentence_sparse(h5f, bad_idx)
        
        t = min(len(good_sent_data["tokens"]), len(bad_sent_data["tokens"]))
        # if t == 0:
        #     continue
        # Exclude BOS token (index 0) from both sentences before computing phi
        good_bos_mask = good_sent_data["token_idx"] != 0
        bad_bos_mask  = bad_sent_data["token_idx"]  != 0
        good_feat_set = set(good_sent_data["feature_idx"][good_bos_mask])
        bad_feat_set  = set(bad_sent_data["feature_idx"][bad_bos_mask])

        # Pair-level set operations
        bad_only  = bad_feat_set - good_feat_set   # φ-eligible
        good_only = good_feat_set - bad_feat_set
        both      = good_feat_set & bad_feat_set
        bad_only_mask = ~np.isin(bad_sent_data['feature_idx'], good_sent_data['feature_idx'])

        np.add.at(bad_only_pair_count,  (linguistic_term_index, list(bad_only)),  1)
        np.add.at(good_only_pair_count, (linguistic_term_index, list(good_only)), 1)
        np.add.at(both_pair_count,      (linguistic_term_index, list(both)),      1)

        # accumulation (unchanged)

        np.add.at(bad_phi_sum, (linguistic_term_index, bad_sent_data['feature_idx']), bad_sent_data['feature_values'].astype(np.float64))
        np.add.at(good_phi_sum, (linguistic_term_index, good_sent_data['feature_idx']), good_sent_data['feature_values'].astype(np.float64))
        np.add.at(bad_phi_count, (linguistic_term_index, bad_sent_data['feature_idx']), 1)
        np.add.at(good_phi_count, (linguistic_term_index, good_sent_data['feature_idx']), 1)

        total_tokens += t
    bad_phi_avg  = np.where(bad_phi_count  > 0, bad_phi_sum  / np.where(bad_phi_count  > 0, bad_phi_count,  1), 0.0)
    good_phi_avg = np.where(good_phi_count > 0, good_phi_sum / np.where(good_phi_count > 0, good_phi_count, 1), 0.0)
    # phi_sum = (bad_phi_avg - good_phi_avg).astype(np.float32)
    # import pdb; pdb.set_trace()
    phi_sum = (bad_only_pair_count).astype(np.float32) - (good_only_pair_count).astype(np.float32)  # φ-eligible pairs only
    if total_tokens == 0 or n_pairs == 0:
        raise ValueError("No sentence pairs provided.")

    # distribution = {
    #     "bad_only_pairs":  bad_only_pair_count,   # should match φ; good_% = 0 by construction
    #     "good_only_pairs": good_only_pair_count,
    #     "both_pairs":      both_pair_count,
    #     "n_pairs":         n_pairs,
    # }

    # distribution_lines = ["Distribution of feature presence across good/bad pairs:"]
    # for category, counts in distribution.items():
    #     if category != "n_pairs":
    #         distribution_lines.append(
    #             f"  {category}: {np.sum(counts)} features (mean {np.mean(counts):.2f} pairs/feature)"
    #         )
    #     else:
    #         distribution_lines.append(f"  {category}: {counts}")
    # distribution["summary_text"] = "\n".join(distribution_lines)
    
    # phi_l = (phi_sum / float(total_tokens)).astype(np.float32)
    return phi_sum

def _process_one(h5_path: Path, sampled_sentences: list[str], linguistic_terms_map: dict[str, int], out_path: Path, sentence_idx, start_idx: int, end_idx) -> None:

    with h5py.File(h5_path, "r") as h5f:
        ds_offsets = cast(h5py.Dataset, h5f["offsets"])
        n_features = cast(int, h5f.attrs.get("n_features", 0))
        n_sentences = int(h5f.attrs.get("n_sentences", len(ds_offsets) - 1))
        sentence_indices = _resolve_sentence_indices(
            n_sentences=n_sentences,
            sentence_idx=sentence_idx,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        phi_l = compute_phi_l(h5f, sampled_sentences=sampled_sentences, linguistic_terms_map=linguistic_terms_map, sentence_indices=sentence_indices, n_features=n_features)
    sorted_phi_idx = np.argsort(phi_l, axis=1)[:, ::-1]
    sorted_phi_values = np.take_along_axis(phi_l, sorted_phi_idx, axis=1)   
    np.savez_compressed(
        out_path,
        sorted_phi_idx=sorted_phi_idx,
        sorted_phi_values=sorted_phi_values,
    )

    # distribution_txt_path = out_path.with_name(f"{out_path.stem}_distribution.txt")
    # distribution_txt_path.write_text(cast(str, distribution["summary_text"]) + "\n", encoding="utf-8")


def main(args):
    
    with open(args.dataset_path) as f:
        data = [json.loads(line) for line in f]

    sampled_sentences = []
    linguistic_terms_set = set()
    for dataline in data:
        sampled_sentences.append((dataline[0]["linguistics_term"], dataline[0]['global_idx']))  # Extract linguistic phenomenon
        sampled_sentences.append((dataline[1]["linguistics_term"], dataline[1]['global_idx']))  # Extract linguistic phenomenon
        linguistic_terms_set.add(dataline[0]["linguistics_term"])
        linguistic_terms_set.add(dataline[1]["linguistics_term"])
    # Create a mapping from linguistic terms to indices
    
    linguistic_terms_map = {term: i for i, term in enumerate(sorted(linguistic_terms_set))}

    if args.h5_dir is not None:
        h5_files = sorted(args.h5_dir.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {args.h5_dir}")
        tasks = []
        for h5_path in h5_files:
            out_dir = h5_path.parent / args.split_name
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / h5_path.name.replace(".h5", ".npz")
            tasks.append((h5_path, sampled_sentences, linguistic_terms_map, out_path,
                  args.sentence_idx, args.start_idx, args.end_idx))
            # logging.info(f"Split: {args.split_name}, Processing {h5_path.name} -> {out_path.name}")
            # _process_one(h5_path, sampled_sentences, linguistic_terms_map, out_path, args.sentence_idx, args.start_idx, args.end_idx, args.lambda_)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(_process_one_wrapper, t): t[0] for t in tasks}
            for future in as_completed(futures):
                h5_path = futures[future]
                try:
                    future.result()
                    logging.info(f"Done: {h5_path.name}")
                except Exception as e:
                    logging.error(f"Failed {h5_path.name}: {e}")
    else:
        if args.h5_path is None:
            raise ValueError("Provide either --h5_path or --h5_dir.")
        out_path = args.out_path if args.out_path is not None else args.h5_path.with_suffix(".npz")
        _process_one(args.h5_path, sampled_sentences, linguistic_terms_map, out_path, args.sentence_idx, args.start_idx, args.end_idx)


if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description="Compute SAE sensitivity score Phi from sparse HDF5 features.")
    parser.add_argument(
        "--dataset_path", type=Path,
        default=PROJECT_ROOT / "data" / "input_data" / "blimp_data.jsonl",
    )
    parser.add_argument("--split_name", type=str, default="valid", help="Which BLiMP split to use (e.g., 'valid', 'test').")
    parser.add_argument("--h5_path", type=Path, default=None, help="Path to one SAE layer .h5 file.")
    parser.add_argument("--h5_dir", type=Path, default=None, help="Directory of .h5 files; processes all and saves .npz alongside each.")
    parser.add_argument("--out_path", type=Path, default=None, help="Output .npz path (single-file mode only). Defaults to h5_path with .npz extension.")
    parser.add_argument("--sentence_idx", type=int, default=None, help="Process a single sentence index.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start of sentence range (inclusive).")
    parser.add_argument("--end_idx", type=int, default=None, help="End of sentence range (exclusive). Defaults to all sentences.")
    parser.add_argument("--lambda_", type=float, default=1.0, help="Regularization parameter.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for processing multiple .h5 files.")
    args = parser.parse_args()
    main(args)
