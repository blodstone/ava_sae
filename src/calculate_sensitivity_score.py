import logging
import argparse
import tqdm
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

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
    sentence_indices: list[int],
    n_features: int,
    lambda_: float = 1.0
) -> np.ndarray:
    if len(sentence_indices) % 2 != 0:
        raise ValueError("sentence_indices must contain an even number of entries (good/bad pairs).")

    n_pairs = len(sentence_indices) // 2
    phi_sum = np.zeros(n_features, dtype=np.float64)
    total_tokens = 0

    # Track pair-level counts for correct distribution reporting
    bad_only_pair_count = np.zeros(n_features, dtype=np.int32)   # bad but not good (φ-eligible)
    good_only_pair_count = np.zeros(n_features, dtype=np.int32)  # good but not bad
    both_pair_count = np.zeros(n_features, dtype=np.int32)       # both good and bad

    for pos in tqdm.tqdm(range(0, len(sentence_indices), 2)):
        good_idx = sentence_indices[pos]
        bad_idx = sentence_indices[pos + 1]

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

        np.add.at(bad_only_pair_count,  list(bad_only),  1)
        np.add.at(good_only_pair_count, list(good_only), 1)
        np.add.at(both_pair_count,      list(both),      1)

        # φ accumulation (unchanged)
        np.add.at(phi_sum, bad_sent_data['feature_idx'], bad_sent_data['feature_values'].astype(np.float64))
        np.add.at(phi_sum, good_sent_data['feature_idx'], -good_sent_data['feature_values'].astype(np.float64))


        total_tokens += t

    if total_tokens == 0 or n_pairs == 0:
        raise ValueError("No sentence pairs provided.")

    distribution = {
        "bad_only_pairs":  bad_only_pair_count,   # should match φ; good_% = 0 by construction
        "good_only_pairs": good_only_pair_count,
        "both_pairs":      both_pair_count,
        "n_pairs":         n_pairs,
    }

    print("Distribution of feature presence across good/bad pairs:")
    for category, counts in distribution.items():
        if category != "n_pairs":
            print(f"  {category}: {np.sum(counts)} features (mean {np.mean(counts):.2f} pairs/feature)")
        else:
            print(f"  {category}: {counts}")
    # both_threshold = max(1, int(0.02 * n_pairs))  # e.g. 20 pairs out of 1000
    # both_mask = both_pair_count > both_threshold
    # phi_sum[both_mask] = 0.0
    
    # logging.info(
    #     f"Zeroed out {both_mask.sum()} features that appeared in both "
    #     f"good and bad sentences (both_pair_count > {both_threshold})."
    # )
    return (phi_sum / float(total_tokens)).astype(np.float32)

def _process_one(h5_path: Path, out_path: Path, sentence_idx, start_idx: int, end_idx, lambda_: float = 1.0) -> None:
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
        phi_l = compute_phi_l(h5f, sentence_indices=sentence_indices, n_features=n_features, lambda_=lambda_)

    sorted_phi_idx = np.argsort(phi_l)[::-1]
    sorted_phi_values = phi_l[sorted_phi_idx]
    np.savez_compressed(
        out_path,
        sorted_phi_idx=sorted_phi_idx,
        sorted_phi_values=sorted_phi_values,
    )


def main(args):
    if args.h5_dir is not None:
        h5_files = sorted(args.h5_dir.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in {args.h5_dir}")
        for h5_path in h5_files:
            out_path = h5_path.with_suffix(".npz")
            import logging
            logging.info(f"Processing {h5_path.name} -> {out_path.name}")
            _process_one(h5_path, out_path, args.sentence_idx, args.start_idx, args.end_idx, args.lambda_)
    else:
        if args.h5_path is None:
            raise ValueError("Provide either --h5_path or --h5_dir.")
        out_path = args.out_path if args.out_path is not None else args.h5_path.with_suffix(".npz")
        _process_one(args.h5_path, out_path, args.sentence_idx, args.start_idx, args.end_idx, args.lambda_)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description="Compute SAE sensitivity score Phi from sparse HDF5 features.")
    parser.add_argument("--h5_path", type=Path, default=None, help="Path to one SAE layer .h5 file.")
    parser.add_argument("--h5_dir", type=Path, default=None, help="Directory of .h5 files; processes all and saves .npz alongside each.")
    parser.add_argument("--out_path", type=Path, default=None, help="Output .npz path (single-file mode only). Defaults to h5_path with .npz extension.")
    parser.add_argument("--sentence_idx", type=int, default=None, help="Process a single sentence index.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start of sentence range (inclusive).")
    parser.add_argument("--end_idx", type=int, default=None, help="End of sentence range (exclusive). Defaults to all sentences.")
    parser.add_argument("--lambda_", type=float, default=1.0, help="Regularization parameter.")

    args = parser.parse_args()
    main(args)
