import json
from typing import List
import torch

def build_from_blimp_row(row: dict) -> List:
    sentences = [row['sentence_good'], row['sentence_bad']]
    results = []
    for idx, sentence in enumerate(sentences):
        prompt = sentence
        results.append({
            "dataset": "blimp",
            "prompt": prompt,
            "id": row['UID'] + '#' + row['pairID'] + '#' + str(idx),
            "linguistics_term": row['linguistics_term'],
            "label": 'bad' if idx == 1 else 'good'
        })
    return results

def make_blimp_prefix_mask(input_ids: torch.Tensor):
    """
    For each consecutive pair of rows (i, i+1) in input_ids, build a boolean
    attention mask where the shared prefix (from start up to, but NOT including,
    the first differing token) is masked as False, and the differing token
    onward is True (unmasked).

    Args:
        input_ids:    (B, L) tensor, B must be even (pairs)

    Returns:
        mask:          (B, L) bool tensor — False = masked prefix, True = unmasked
        diff_positions: (B//2,) tensor — position of the differing token per pair
    """
    assert input_ids.size(0) % 2 == 0, "Batch size must be even (pairs)"
    B, L = input_ids.shape

    # Start with everything unmasked (True)
    mask = torch.ones(B, L, dtype=torch.bool, device=input_ids.device)
    diff_positions = torch.zeros(B // 2, dtype=torch.long, device=input_ids.device)

    for p in range(B // 2):
        i, j = 2 * p, 2 * p + 1

        # First position where the two sentences differ
        differs = (input_ids[i] != input_ids[j]).nonzero(as_tuple=True)[0]
        assert len(differs) > 0, f"Pair ({i},{j}) has no differing token!"
        diff_pos = differs[0].item()

        diff_positions[p] = diff_pos

        # Mask the shared prefix (before the differing token) as False
        mask[i, :diff_pos] = False
        mask[i, diff_pos+1:] = False
        mask[j, :diff_pos] = False
        mask[j, diff_pos+1:] = False

    return mask, diff_positions