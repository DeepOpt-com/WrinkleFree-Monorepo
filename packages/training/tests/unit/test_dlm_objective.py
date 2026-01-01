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


class TestDLMConfigSaving:
    """Test dlm_config.json format for checkpoint saving."""

    def test_dlm_config_json_format(self):
        """Verify dlm_config.json has expected fields matching deprecated train_dlm.py."""
        obj = DLMObjective(mask_token_id=999, mask_prob=0.15)

        # This is the config format saved by ContinuedPretrainingTrainer.save_checkpoint()
        config = {
            "mask_token_id": obj.mask_token_id,
            "mask_prob": obj.mask_prob,
            "ignore_index": obj.ignore_index,
            "training_method": "unified-dlm",
        }

        # Verify expected keys
        expected_keys = {"mask_token_id", "mask_prob", "ignore_index", "training_method"}
        assert set(config.keys()) == expected_keys

        # Verify values
        assert config["mask_token_id"] == 999
        assert config["mask_prob"] == 0.15
        assert config["ignore_index"] == -100  # Default
        assert config["training_method"] == "unified-dlm"

    def test_dlm_objective_has_required_attributes(self):
        """Verify DLMObjective exposes all attributes needed for config saving."""
        obj = DLMObjective(mask_token_id=42, mask_prob=0.2, ignore_index=-200)

        # These attributes are accessed by save_checkpoint() for dlm_config.json
        assert hasattr(obj, "mask_token_id")
        assert hasattr(obj, "mask_prob")
        assert hasattr(obj, "ignore_index")

        assert obj.mask_token_id == 42
        assert obj.mask_prob == 0.2
        assert obj.ignore_index == -200

    def test_dlm_config_json_file_creation(self, tmp_path):
        """Test that dlm_config.json is created correctly during checkpoint save."""
        import json

        # Simulate what save_checkpoint does
        checkpoint_dir = tmp_path / "checkpoints" / "test_checkpoint"
        checkpoint_dir.mkdir(parents=True)

        dlm_obj = DLMObjective(mask_token_id=999, mask_prob=0.15)

        # This mirrors the logic in ContinuedPretrainingTrainer.save_checkpoint()
        dlm_config = {
            "mask_token_id": dlm_obj.mask_token_id,
            "mask_prob": dlm_obj.mask_prob,
            "ignore_index": dlm_obj.ignore_index,
            "training_method": "unified-dlm",
        }
        with open(checkpoint_dir / "dlm_config.json", "w") as f:
            json.dump(dlm_config, f, indent=2)

        # Verify file was created
        assert (checkpoint_dir / "dlm_config.json").exists()

        # Verify content
        with open(checkpoint_dir / "dlm_config.json") as f:
            saved_config = json.load(f)

        assert saved_config["mask_token_id"] == 999
        assert saved_config["mask_prob"] == 0.15
        assert saved_config["ignore_index"] == -100
        assert saved_config["training_method"] == "unified-dlm"

    def test_objective_manager_dlm_access(self):
        """Test that ObjectiveManager provides access to DLM objective for config extraction."""
        from wrinklefree.objectives.manager import ObjectiveManager

        dlm = DLMObjective(mask_token_id=42, mask_prob=0.3)
        manager = ObjectiveManager(
            objectives={"dlm": dlm},
            weights={"dlm": 1.0},
        )

        # This is how save_checkpoint accesses the DLM objective
        # objectives is a ModuleDict, so use dictionary-style access
        assert "dlm" in manager.objectives
        retrieved_dlm = manager.objectives["dlm"]
        assert retrieved_dlm is not None
        assert retrieved_dlm.mask_token_id == 42
        assert retrieved_dlm.mask_prob == 0.3
