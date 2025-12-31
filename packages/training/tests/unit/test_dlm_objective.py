"""Tests for DLMObjective (Diffusion Language Model / Fast-dLLM)."""

import pytest
import torch

from wrinklefree.objectives.dlm import DLMObjective


class TestDLMPreprocessing:
    """Test DLM preprocessing (masking)."""

    def test_masks_tokens(self):
        """Test that DLM preprocessing masks tokens correctly."""
        mask_token_id = 999
        obj = DLMObjective(mask_token_id=mask_token_id, mask_prob=0.5)

        # Create input that doesn't contain mask token
        input_ids = torch.randint(0, 100, (2, 10))
        input_ids[input_ids == mask_token_id] = 0

        batch = {"input_ids": input_ids.clone()}

        # Run preprocessing
        processed = obj.preprocess_batch(batch)

        # Check that some tokens are masked (with high prob, should have some)
        masked_ids = processed["input_ids"]
        labels = processed["dlm_labels"]

        # Check masking was applied
        mask = masked_ids == mask_token_id
        assert mask.any(), "No tokens were masked (prob=0.5 should mask some)"

        # Check that labels match original inputs where masked
        assert torch.all(labels[mask] == input_ids[mask])

        # Check that unmasked positions are ignored in labels
        assert torch.all(labels[~mask] == -100)

    def test_preserves_first_last_tokens(self):
        """Test that first and last tokens are never masked."""
        mask_token_id = 999
        # Use high mask prob to ensure masking would happen
        obj = DLMObjective(mask_token_id=mask_token_id, mask_prob=0.9)

        input_ids = torch.randint(0, 100, (4, 20))
        batch = {"input_ids": input_ids.clone()}

        processed = obj.preprocess_batch(batch)
        masked_ids = processed["input_ids"]

        # First and last tokens should never be masked
        assert (masked_ids[:, 0] != mask_token_id).all()
        assert (masked_ids[:, -1] != mask_token_id).all()

    def test_respects_attention_mask(self):
        """Test that padding tokens are not masked."""
        mask_token_id = 999
        obj = DLMObjective(mask_token_id=mask_token_id, mask_prob=0.9)

        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        # Mark last 3 tokens as padding
        attention_mask[:, -3:] = 0

        batch = {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask,
        }

        processed = obj.preprocess_batch(batch)
        masked_ids = processed["input_ids"]
        labels = processed["dlm_labels"]

        # Padding positions should not be masked
        assert (masked_ids[:, -3:] != mask_token_id).all()
        # Padding positions should have ignore_index in labels
        assert (labels[:, -3:] == -100).all()

    def test_stores_original_input(self):
        """Test that original input_ids are stored for debugging."""
        obj = DLMObjective(mask_token_id=999, mask_prob=0.5)
        input_ids = torch.randint(0, 100, (2, 10))
        batch = {"input_ids": input_ids.clone()}

        processed = obj.preprocess_batch(batch)

        assert "_original_input_ids" in processed
        assert torch.equal(processed["_original_input_ids"], input_ids)


class TestDLMForward:
    """Test DLM forward pass (loss computation)."""

    def test_computes_loss(self):
        """Test DLM forward pass loss calculation."""
        obj = DLMObjective(mask_token_id=999)

        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Create dummy labels with one masked position per sequence
        labels = torch.full((batch_size, seq_len), -100)
        labels[0, 2] = 5  # Target for token 2 is ID 5
        labels[1, 4] = 10  # Target for token 4 is ID 10

        batch = {"dlm_labels": labels}
        model_outputs = {"logits": logits}

        output = obj.forward(model_outputs, batch)

        assert output.loss > 0
        assert output.loss.item() == output.metrics["loss"].item()
        assert output.metrics["num_masked"] == 2
        assert output.metrics["mask_ratio"] == 2 / (batch_size * seq_len)

    def test_raises_without_labels(self):
        """Test that forward raises if dlm_labels missing."""
        obj = DLMObjective(mask_token_id=999)

        logits = torch.randn(2, 8, 100)
        batch = {}  # Missing dlm_labels
        model_outputs = {"logits": logits}

        with pytest.raises(ValueError, match="dlm_labels not found"):
            obj.forward(model_outputs, batch)

    def test_no_shift_like_clm(self):
        """Verify DLM doesn't shift labels like CLM does.

        In CLM: predict next token (labels shifted by 1)
        In DLM: predict masked token at same position (no shift)
        """
        obj = DLMObjective(mask_token_id=999)

        # Create logits where position 2 has high prob for token 5
        batch_size, seq_len, vocab_size = 1, 8, 10
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[0, 2, 5] = 10.0  # High logit for token 5 at position 2

        # Label says position 2 should be token 5
        labels = torch.full((batch_size, seq_len), -100)
        labels[0, 2] = 5

        batch = {"dlm_labels": labels}
        model_outputs = {"logits": logits}

        output = obj.forward(model_outputs, batch)

        # With correct prediction, loss should be low
        assert output.loss < 1.0


class TestDLMIntegration:
    """Integration tests for DLM with ObjectiveManager."""

    def test_factory_creates_dlm(self):
        """Test that factory can create DLM objective."""
        from wrinklefree.objectives.factory import create_objective

        obj = create_objective("dlm", {"mask_token_id": 999, "mask_prob": 0.2})

        assert isinstance(obj, DLMObjective)
        assert obj.mask_token_id == 999
        assert obj.mask_prob == 0.2

    def test_dlm_in_objective_manager(self):
        """Test DLM works with ObjectiveManager."""
        from wrinklefree.objectives.manager import ObjectiveManager

        dlm = DLMObjective(mask_token_id=999, mask_prob=0.15)

        manager = ObjectiveManager(
            objectives={"dlm": dlm},
            weights={"dlm": 1.0},
        )

        assert "dlm" in manager.objectives
        assert manager.any_modifies_input  # DLM modifies input
