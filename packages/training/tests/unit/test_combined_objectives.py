"""Tests for combined multi-task objectives (LM + DLM).

Tests the interaction between ContinuePretrainObjective and DLMObjective
when running multi-task training on the same data.
"""

import pytest
import torch

from wrinklefree.objectives.continue_pretrain import ContinuePretrainObjective
from wrinklefree.objectives.dlm import DLMObjective
from wrinklefree.objectives.manager import ObjectiveManager


class TestDLMStoresOriginals:
    """Test that DLM stores original labels for multi-task training."""

    def test_stores_original_labels(self):
        """Test that DLM stores _original_labels in batch."""
        obj = DLMObjective(mask_token_id=999, mask_prob=0.5)

        input_ids = torch.randint(0, 100, (2, 10))
        labels = input_ids.clone()  # Labels match input for LM

        batch = {
            "input_ids": input_ids.clone(),
            "labels": labels.clone(),
        }

        processed = obj.preprocess_batch(batch)

        assert "_original_labels" in processed
        # With complementary masks (default), batch is doubled so _original_labels is also doubled
        # First half should match original labels
        assert torch.equal(processed["_original_labels"][:2], labels)

    def test_original_labels_not_masked(self):
        """Test that _original_labels are not masked."""
        obj = DLMObjective(mask_token_id=999, mask_prob=0.9)  # High mask prob

        input_ids = torch.randint(0, 100, (4, 20))
        labels = input_ids.clone()

        batch = {
            "input_ids": input_ids.clone(),
            "labels": labels.clone(),
        }

        processed = obj.preprocess_batch(batch)

        # Original labels should not contain mask token
        assert not (processed["_original_labels"] == 999).any()
        # But input_ids should be masked
        assert (processed["input_ids"] == 999).any()


class TestContinuePretrainUsesOriginalLabels:
    """Test that ContinuePretrainObjective uses _original_labels if present."""

    def test_uses_original_labels_when_present(self):
        """Test CE objective uses _original_labels from DLM preprocessing."""
        obj = ContinuePretrainObjective()

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Create batch with both labels and _original_labels
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        original_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        batch = {
            "labels": labels,
            "_original_labels": original_labels,  # Should be used instead
        }
        model_outputs = {"logits": logits}

        # Run forward with mocked labels to verify which is used
        # We'll create a scenario where original_labels gives lower loss
        perfect_logits = torch.zeros(batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for s in range(seq_len - 1):
                perfect_logits[b, s, original_labels[b, s + 1]] = 10.0

        model_outputs_perfect = {"logits": perfect_logits}

        output = obj(model_outputs_perfect, batch)

        # Loss should be low because we're predicting original_labels correctly
        assert output.loss.item() < 1.0

    def test_falls_back_to_labels(self):
        """Test CE falls back to labels if _original_labels not present."""
        obj = ContinuePretrainObjective()

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        batch = {"labels": labels}  # No _original_labels
        model_outputs = {"logits": logits}

        output = obj(model_outputs, batch)

        assert output.loss.item() > 0  # Should work normally


class TestCombinedObjectiveManager:
    """Test ObjectiveManager with both CE and DLM objectives."""

    def test_combined_ce_dlm_forward(self):
        """Test running CE + DLM together on same data."""
        cp_obj = ContinuePretrainObjective()
        dlm_obj = DLMObjective(mask_token_id=999, mask_prob=0.15)

        manager = ObjectiveManager(
            objectives={"continue_pretrain": cp_obj, "dlm": dlm_obj},
            weights={"continue_pretrain": 1.0, "dlm": 0.5},
        )

        # Create batch
        batch_size, seq_len, vocab_size = 2, 10, 100
        input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len))  # Avoid mask token
        labels = input_ids.clone()

        batch = {
            "input_ids": input_ids,
            "labels": labels,
        }

        # Preprocess (DLM applies masking)
        processed_batch = manager.preprocess_batch(batch)

        # Verify originals are stored
        assert "_original_input_ids" in processed_batch
        assert "_original_labels" in processed_batch

        # With complementary masks, batch is doubled (2 -> 4)
        doubled_batch_size = processed_batch["input_ids"].shape[0]
        logits = torch.randn(doubled_batch_size, seq_len, vocab_size)
        model_outputs = {"logits": logits}

        # Forward pass
        output = manager(model_outputs, processed_batch)

        # Verify both objectives computed
        assert "continue_pretrain" in output.objective_outputs
        assert "dlm" in output.objective_outputs

        # Verify weighted combination
        cp_loss = output.objective_outputs["continue_pretrain"].loss
        dlm_loss = output.objective_outputs["dlm"].loss
        expected_total = 1.0 * cp_loss + 0.5 * dlm_loss

        assert torch.allclose(output.loss, expected_total, rtol=1e-5)

    def test_wandb_metrics_includes_both(self):
        """Test WandB metrics include both objective losses."""
        cp_obj = ContinuePretrainObjective()
        dlm_obj = DLMObjective(mask_token_id=999, mask_prob=0.15)

        manager = ObjectiveManager(
            objectives={"continue_pretrain": cp_obj, "dlm": dlm_obj},
            weights={"continue_pretrain": 1.0, "dlm": 0.5},
        )

        batch_size, seq_len, vocab_size = 2, 10, 100
        input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
        labels = input_ids.clone()

        batch = {"input_ids": input_ids, "labels": labels}
        processed_batch = manager.preprocess_batch(batch)

        # With complementary masks, batch is doubled
        doubled_batch_size = processed_batch["input_ids"].shape[0]
        logits = torch.randn(doubled_batch_size, seq_len, vocab_size)
        model_outputs = {"logits": logits}

        output = manager(model_outputs, processed_batch)
        metrics = manager.get_wandb_metrics(output)

        # Verify all expected metrics
        assert "train/loss" in metrics
        assert "train/continue_pretrain_loss" in metrics
        assert "train/dlm_loss" in metrics
        assert "train/dlm_num_masked" in metrics
        assert "schedule/continue_pretrain_weight" in metrics
        assert "schedule/dlm_weight" in metrics

    def test_modifies_input_flag(self):
        """Test that any_modifies_input is correctly set."""
        cp_obj = ContinuePretrainObjective()  # doesn't modify
        dlm_obj = DLMObjective(mask_token_id=999)  # modifies

        # Only CE
        manager1 = ObjectiveManager(objectives={"cp": cp_obj})
        assert not manager1.any_modifies_input

        # Only DLM
        manager2 = ObjectiveManager(objectives={"dlm": dlm_obj})
        assert manager2.any_modifies_input

        # Both
        manager3 = ObjectiveManager(objectives={"cp": cp_obj, "dlm": dlm_obj})
        assert manager3.any_modifies_input


class TestConfigurableResume:
    """Test configurable checkpoint resume behavior."""

    def test_resume_config_defaults(self):
        """Test default resume configuration values."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "training": {
                "resume": {
                    "load_optimizer_state": True,
                    "load_scheduler_state": True,
                    "load_training_state": True,
                    "strict_model_load": True,
                }
            }
        })

        resume_config = config.training.resume

        assert resume_config.load_optimizer_state is True
        assert resume_config.load_scheduler_state is True
        assert resume_config.load_training_state is True
        assert resume_config.strict_model_load is True

    def test_resume_config_overrides(self):
        """Test resume config can be overridden."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "training": {
                "resume": {
                    "load_optimizer_state": False,  # Fresh optimizer
                    "load_scheduler_state": False,  # Fresh scheduler
                    "load_training_state": True,    # Keep step count
                }
            }
        })

        resume_config = config.training.resume

        assert resume_config.load_optimizer_state is False
        assert resume_config.load_scheduler_state is False
        assert resume_config.load_training_state is True


class TestFactoryWithDLM:
    """Test factory creates DLM correctly with combined objectives."""

    def test_create_manager_with_dlm_enabled(self):
        """Test creating manager with both CE and DLM enabled."""
        from wrinklefree.objectives.factory import create_objective_manager

        config = {
            "objectives": {
                "continue_pretrain": {"enabled": True, "weight": 1.0},
                "dlm": {"enabled": True, "weight": 0.5, "mask_token_id": 999, "mask_prob": 0.15},
            },
        }

        manager = create_objective_manager(config, total_steps=1000)

        assert "continue_pretrain" in manager.objectives
        assert "dlm" in manager.objectives
        assert manager.base_weights["continue_pretrain"] == 1.0
        assert manager.base_weights["dlm"] == 0.5
        assert manager.any_modifies_input

    def test_dlm_requires_mask_token_id(self):
        """Test that DLM requires mask_token_id when enabled."""
        from wrinklefree.objectives.factory import create_objective_manager

        config = {
            "objectives": {
                "continue_pretrain": {"enabled": True},
                "dlm": {"enabled": True, "weight": 0.5},  # No mask_token_id
            },
        }

        # Should raise because mask_token_id is required
        with pytest.raises(KeyError):
            create_objective_manager(config, total_steps=1000)
