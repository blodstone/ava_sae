"""Tests for ablate_top_k_active_phi and ablate_random_active hooks."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import pytest
from simple_ablate_sensitive_features import (
    ablate_top_k_active_phi,
    ablate_random_active,
    compute_attention_mask_and_divergence,
)
from extract_features import load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_acts(batch, seq, n_features, fill=0.0):
    return torch.full((batch, seq, n_features), fill, dtype=torch.float32)


def _first_divergence_index_no_padding(good_ids: torch.Tensor, bad_ids: torch.Tensor) -> int:
    """Compute first token mismatch index on unpadded token id sequences."""
    min_len = min(good_ids.numel(), bad_ids.numel())
    for i in range(min_len):
        if int(good_ids[i]) != int(bad_ids[i]):
            return i
    return max(min_len - 1, 0)


def _assert_padding_side(attn: torch.Tensor, side: str) -> None:
    """Assert binary attention mask follows left- or right-padding convention."""
    # Remove rows with no padding to avoid vacuous checks.
    rows = [r for r in range(attn.shape[0]) if (attn[r] == 0).any()]
    assert rows, "Expected at least one padded row to validate padding side"
    for r in rows:
        row = attn[r]
        zeros = (row == 0).nonzero(as_tuple=True)[0]
        if side == "left":
            # Left padding means zeros form a prefix.
            assert torch.all(row[: zeros.numel()] == 0)
            assert torch.all(row[zeros.numel() :] == 1)
        elif side == "right":
            # Right padding means zeros form a suffix.
            first_zero = int(zeros[0])
            assert torch.all(row[:first_zero] == 1)
            assert torch.all(row[first_zero:] == 0)
        else:
            raise ValueError(f"Unknown side: {side}")


# ---------------------------------------------------------------------------
# ablate_top_k_active_phi
# ---------------------------------------------------------------------------

class TestAblateTopKActivePhi:

    def test_pre_divergence_positions_unchanged(self):
        """Tokens BEFORE the divergence index must never be modified."""
        B, S, F = 2, 6, 10
        sae_acts = torch.ones(B, S, F)
        # Feature 0 has the highest phi; it is active everywhere
        phi_scores = torch.zeros(F)
        phi_scores[0] = 1.0
        attention_mask = torch.ones(B, S, dtype=torch.long)
        # Divergence at position 3 for both samples
        divergence_indices = torch.tensor([3, 3])

        result = ablate_top_k_active_phi(
            sae_acts, hook=None,
            phi_scores=phi_scores, top_k=1,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        # Positions 0..2 must be untouched
        assert torch.allclose(result[:, :3, :], sae_acts[:, :3, :]), \
            "Pre-divergence positions were incorrectly modified"

    def test_post_divergence_top_phi_feature_zeroed(self):
        """The highest-φ active feature must be zeroed at t >= divergence_index."""
        B, S, F = 1, 5, 4
        sae_acts = torch.ones(B, S, F)
        phi_scores = torch.tensor([0.1, 0.9, 0.5, 0.3])  # feature 1 is top
        attention_mask = torch.ones(B, S, dtype=torch.long)
        divergence_indices = torch.tensor([2])

        result = ablate_top_k_active_phi(
            sae_acts, hook=None,
            phi_scores=phi_scores, top_k=1,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        # Feature 1 must be 0 at positions 2, 3, 4
        assert torch.all(result[0, 2:, 1] == 0.0), \
            "Top-φ feature was not zeroed at post-divergence positions"
        # Feature 1 must be 1 at positions 0, 1
        assert torch.all(result[0, :2, 1] == 1.0), \
            "Top-φ feature was incorrectly zeroed at pre-divergence positions"
        # Other features must be untouched everywhere
        for f in [0, 2, 3]:
            assert torch.all(result[0, :, f] == 1.0), \
                f"Feature {f} was incorrectly modified"

    def test_padding_positions_not_considered_for_activity(self):
        """A feature active ONLY in padding positions must not be selected for ablation."""
        B, S, F = 1, 6, 4
        sae_acts = torch.zeros(B, S, F)
        # Feature 2 fires only at position 0 (a left-pad position)
        sae_acts[0, 0, 2] = 5.0
        # Feature 0 fires at position 3 (real token, post-divergence)
        sae_acts[0, 3, 0] = 1.0

        phi_scores = torch.tensor([0.5, 0.0, 0.9, 0.0])  # feature 2 has highest phi
        # Left-padding: positions 0-1 are padding, 2-5 are real
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
        divergence_indices = torch.tensor([3])  # divergence at position 3

        result = ablate_top_k_active_phi(
            sae_acts, hook=None,
            phi_scores=phi_scores, top_k=1,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        # Feature 2 should NOT be zeroed (it was only active in padding)
        assert result[0, 0, 2] == 5.0, \
            "Feature active only in padding was incorrectly zeroed"
        # Feature 0 should be zeroed at position 3 (it is the top active real-token feature)
        assert result[0, 3, 0] == 0.0, \
            "Top-φ feature active at real post-divergence position was not zeroed"

    def test_right_padding_not_considered_for_activity(self):
        """A feature active ONLY in right-padding positions must not be selected."""
        B, S, F = 1, 6, 4
        sae_acts = torch.zeros(B, S, F)
        # Feature 3 fires only at position 5 (right-pad)
        sae_acts[0, 5, 3] = 5.0
        # Feature 0 fires at position 2 (real, post-divergence)
        sae_acts[0, 2, 0] = 1.0

        phi_scores = torch.tensor([0.4, 0.0, 0.0, 0.9])  # feature 3 highest
        # Right-padding: positions 0-4 real, position 5 pad
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long)
        divergence_indices = torch.tensor([2])

        result = ablate_top_k_active_phi(
            sae_acts, hook=None,
            phi_scores=phi_scores, top_k=1,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        assert result[0, 5, 3] == 5.0, \
            "Feature active only in right-padding was incorrectly zeroed"
        assert result[0, 2, 0] == 0.0, \
            "Top-φ real post-divergence feature was not zeroed"

    def test_only_positive_phi_features_selected(self):
        """Features with φ <= 0 must never be ablated even if active."""
        B, S, F = 1, 4, 3
        sae_acts = torch.ones(B, S, F)
        phi_scores = torch.tensor([-1.0, 0.0, 0.8])  # only feature 2 is positive
        attention_mask = torch.ones(B, S, dtype=torch.long)
        divergence_indices = torch.tensor([1])

        result = ablate_top_k_active_phi(
            sae_acts, hook=None,
            phi_scores=phi_scores, top_k=3,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        # Features 0 and 1 must be untouched
        assert torch.all(result[0, :, 0] == 1.0), "Negative-φ feature was ablated"
        assert torch.all(result[0, :, 1] == 1.0), "Zero-φ feature was ablated"
        # Feature 2 must be zeroed post-divergence
        assert torch.all(result[0, 1:, 2] == 0.0), "Positive-φ feature was not ablated"
        assert result[0, 0, 2] == 1.0, "Positive-φ feature ablated before divergence"

    def test_no_active_features_returns_unchanged(self):
        """If no features are active post-divergence, sae_acts must be returned unchanged."""
        B, S, F = 2, 5, 8
        sae_acts = torch.zeros(B, S, F)
        phi_scores = torch.ones(F)
        attention_mask = torch.ones(B, S, dtype=torch.long)
        divergence_indices = torch.tensor([2, 2])

        result = ablate_top_k_active_phi(
            sae_acts, hook=None,
            phi_scores=phi_scores, top_k=3,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        assert torch.allclose(result, sae_acts), \
            "Hook modified acts when no features were active"

    def test_top_k_respected(self):
        """Exactly top_k features (by φ) must be zeroed, not more."""
        B, S, F = 1, 4, 6
        sae_acts = torch.ones(B, S, F)
        # phi: feature 5 > 4 > 3 > 2 > 1 > 0
        phi_scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        attention_mask = torch.ones(B, S, dtype=torch.long)
        divergence_indices = torch.tensor([0])

        result = ablate_top_k_active_phi(
            sae_acts, hook=None,
            phi_scores=phi_scores, top_k=2,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        zeroed = (result[0].sum(dim=0) == 0)  # features zeroed across all positions
        assert zeroed[5] and zeroed[4], "Top-2 features were not zeroed"
        assert not zeroed[3] and not zeroed[2], "More than top_k features were zeroed"

    def test_per_sample_divergence_indices(self):
        """Each sample in the batch uses its own divergence index independently."""
        B, S, F = 2, 5, 4
        sae_acts = torch.ones(B, S, F)
        phi_scores = torch.tensor([0.0, 0.0, 0.0, 1.0])  # feature 3 is top
        attention_mask = torch.ones(B, S, dtype=torch.long)
        # Sample 0: divergence at 1; sample 1: divergence at 3
        divergence_indices = torch.tensor([1, 3])

        result = ablate_top_k_active_phi(
            sae_acts, hook=None,
            phi_scores=phi_scores, top_k=1,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        # Sample 0: feature 3 zeroed from position 1 onward
        assert result[0, 0, 3] == 1.0, "Sample 0 pre-divergence was modified"
        assert torch.all(result[0, 1:, 3] == 0.0), "Sample 0 post-divergence not zeroed"

        # Sample 1: feature 3 zeroed from position 3 onward
        assert torch.all(result[1, :3, 3] == 1.0), "Sample 1 pre-divergence was modified"
        assert torch.all(result[1, 3:, 3] == 0.0), "Sample 1 post-divergence not zeroed"


# ---------------------------------------------------------------------------
# ablate_random_active
# ---------------------------------------------------------------------------

class TestAblateRandomActive:

    def test_pre_divergence_unchanged(self):
        B, S, F = 2, 6, 10
        sae_acts = torch.ones(B, S, F)
        attention_mask = torch.ones(B, S, dtype=torch.long)
        divergence_indices = torch.tensor([3, 3])

        result = ablate_random_active(
            sae_acts, hook=None, top_k=2,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        assert torch.allclose(result[:, :3, :], sae_acts[:, :3, :]), \
            "Pre-divergence positions were modified"

    def test_padding_positions_not_selected(self):
        """Features active only in padding must not be chosen for ablation."""
        B, S, F = 1, 6, 4
        sae_acts = torch.zeros(B, S, F)
        # Only padding position 0 activates feature 2
        sae_acts[0, 0, 2] = 5.0
        # Real post-divergence positions activate feature 1
        sae_acts[0, 3, 1] = 1.0
        sae_acts[0, 4, 1] = 1.0

        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)
        divergence_indices = torch.tensor([3])

        # Run many times; feature 2 must NEVER be zeroed
        for _ in range(20):
            result = ablate_random_active(
                sae_acts.clone(), hook=None, top_k=1,
                attention_mask=attention_mask,
                divergence_indices=divergence_indices,
            )
            assert result[0, 0, 2] == 5.0, \
                "Feature active only in padding was selected for random ablation"
            # Must actually ablate real post-divergence activity as well.
            assert (result[0, 3:, 1] == 0.0).any(), \
                "No real post-divergence feature was ablated"

    def test_exactly_top_k_features_zeroed(self):
        """Random ablation must zero at most top_k distinct features post-divergence."""
        torch.manual_seed(0)
        B, S, F = 1, 8, 20
        sae_acts = torch.rand(B, S, F) + 0.1  # all features active everywhere
        attention_mask = torch.ones(B, S, dtype=torch.long)
        divergence_indices = torch.tensor([2])
        top_k = 5

        result = ablate_random_active(
            sae_acts, hook=None, top_k=top_k,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        # Count features zeroed at ANY post-divergence position
        post_div = result[0, 2:, :]              # [S-2, F]
        zeroed_features = (post_div == 0).all(dim=0).sum().item()
        assert zeroed_features == top_k, \
            f"Expected {top_k} zeroed features, got {zeroed_features}"

    def test_no_active_features_returns_unchanged(self):
        B, S, F = 1, 5, 6
        sae_acts = torch.zeros(B, S, F)
        attention_mask = torch.ones(B, S, dtype=torch.long)
        divergence_indices = torch.tensor([1])

        result = ablate_random_active(
            sae_acts, hook=None, top_k=3,
            attention_mask=attention_mask,
            divergence_indices=divergence_indices,
        )

        assert torch.allclose(result, sae_acts)


@pytest.mark.integration
class TestRealModelPaddingAndDivergence:
    """Optional integration tests using real model tokenizers through load_model."""

    @staticmethod
    def _require_real_tests_enabled():
        if os.getenv("RUN_REAL_MODEL_TESTS") != "1":
            pytest.skip("Set RUN_REAL_MODEL_TESTS=1 to run real model integration tests")

    def test_gpt2_padding_and_divergence(self):
        self._require_real_tests_enabled()
        model_name = os.getenv("REAL_TEST_GPT2_MODEL", "gpt2")
        try:
            model = load_model(model_name)
        except Exception as exc:
            pytest.skip(f"Could not load GPT-2 model '{model_name}': {exc}")

        # Interleaved good/bad pairs with varying sentence lengths to force padding.
        sentences = [
            "The old man walks to the market every morning.",
            "The old man walk to the market every morning.",
            "A cat sleeps.",
            "A cat sleep.",
        ]
        input_ids = model.to_tokens(sentences, prepend_bos=True)
        pad_id = model.tokenizer.eos_token_id
        attn, full_div = compute_attention_mask_and_divergence(input_ids, pad_id)

        _assert_padding_side(attn, "right")

        # Validate divergence against per-pair unpadded manual computation.
        for i in range(0, input_ids.shape[0], 2):
            g = input_ids[i][attn[i].bool()]
            b = input_ids[i + 1][attn[i + 1].bool()]
            expected = _first_divergence_index_no_padding(g, b)
            assert int(full_div[i]) == expected
            assert int(full_div[i + 1]) == expected

    def test_gemma_padding_and_divergence(self):
        self._require_real_tests_enabled()
        model_name = os.getenv("REAL_TEST_GEMMA_MODEL", "google/gemma-2-2b")
        try:
            model = load_model(model_name)
        except Exception as exc:
            pytest.skip(f"Could not load Gemma model '{model_name}': {exc}")

        sentences = [
            "The old man walks to the market every morning.",
            "The old man walk to the market every morning.",
            "A cat sleeps.",
            "A cat sleep.",
        ]
        input_ids = model.to_tokens(sentences, prepend_bos=True)
        pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        attn, full_div = compute_attention_mask_and_divergence(input_ids, pad_id)

        _assert_padding_side(attn, "left")

        for i in range(0, input_ids.shape[0], 2):
            g = input_ids[i][attn[i].bool()]
            b = input_ids[i + 1][attn[i + 1].bool()]
            expected = _first_divergence_index_no_padding(g, b)
            assert int(full_div[i]) == expected
            assert int(full_div[i + 1]) == expected
