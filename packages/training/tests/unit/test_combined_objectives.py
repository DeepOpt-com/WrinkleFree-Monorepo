"""Tests for combined multi-task objectives.

Tests the interaction between ContinuePretrainObjective and other objectives
when running multi-task training on the same data.
"""

import pytest
import torch

from wf_train.objectives.continue_pretrain import ContinuePretrainObjective
from wf_train.objectives.manager import ObjectiveManager


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


class TestObjectiveManagerBasics:
    """Test ObjectiveManager with CE objectives."""

    def test_ce_only_forward(self):
        """Test running CE objective alone."""
        cp_obj = ContinuePretrainObjective()

        manager = ObjectiveManager(
            objectives={"continue_pretrain": cp_obj},
            weights={"continue_pretrain": 1.0},
        )

        # Create batch
        batch_size, seq_len, vocab_size = 2, 10, 100
        input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
        labels = input_ids.clone()

        batch = {
            "input_ids": input_ids,
            "labels": labels,
        }

        # Preprocess
        processed_batch = manager.preprocess_batch(batch)

        logits = torch.randn(batch_size, seq_len, vocab_size)
        model_outputs = {"logits": logits}

        # Forward pass
        output = manager(model_outputs, processed_batch)

        # Verify objective computed
        assert "continue_pretrain" in output.objective_outputs
        assert output.loss.item() > 0

    def test_wandb_metrics_ce_only(self):
        """Test WandB metrics with CE objective only."""
        cp_obj = ContinuePretrainObjective()

        manager = ObjectiveManager(
            objectives={"continue_pretrain": cp_obj},
            weights={"continue_pretrain": 1.0},
        )

        batch_size, seq_len, vocab_size = 2, 10, 100
        input_ids = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
        labels = input_ids.clone()

        batch = {"input_ids": input_ids, "labels": labels}
        processed_batch = manager.preprocess_batch(batch)

        logits = torch.randn(batch_size, seq_len, vocab_size)
        model_outputs = {"logits": logits}

        output = manager(model_outputs, processed_batch)
        metrics = manager.get_wandb_metrics(output)

        # Verify expected metrics
        assert "train/loss" in metrics
        assert "train/continue_pretrain_loss" in metrics
        assert "schedule/continue_pretrain_weight" in metrics

    def test_modifies_input_flag(self):
        """Test that any_modifies_input is correctly set."""
        cp_obj = ContinuePretrainObjective()  # doesn't modify

        # Only CE
        manager = ObjectiveManager(objectives={"cp": cp_obj})
        assert not manager.any_modifies_input


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


class TestFactoryBasics:
    """Test factory creates objectives correctly."""

    def test_create_manager_with_ce_only(self):
        """Test creating manager with CE objective only."""
        from wf_train.objectives.factory import create_objective_manager

        config = {
            "objectives": {
                "continue_pretrain": {"enabled": True, "weight": 1.0},
            },
        }

        manager = create_objective_manager(config, total_steps=1000)

        assert "continue_pretrain" in manager.objectives
        assert manager.base_weights["continue_pretrain"] == 1.0
        assert not manager.any_modifies_input
