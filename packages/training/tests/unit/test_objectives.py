"""Tests for the objectives system."""

import pytest
import torch
import torch.nn as nn

from wf_train.objectives.base import Objective, ObjectiveOutput
from wf_train.objectives.continue_pretrain import ContinuePretrainObjective
from wf_train.objectives.layerwise import LayerwiseDistillationObjective, LayerwiseLossType
from wf_train.objectives.manager import (
    CurriculumPhase,
    CurriculumScheduler,
    ObjectiveManager,
)
from wf_train.objectives.factory import create_objective_manager


class TestObjectiveOutput:
    """Test ObjectiveOutput dataclass."""

    def test_basic_output(self):
        """Test creating basic output."""
        loss = torch.tensor(1.5)
        output = ObjectiveOutput(loss=loss)
        assert output.loss == loss
        assert output.metrics == {}
        assert output.ce_loss is None

    def test_output_with_metrics(self):
        """Test output with metrics."""
        loss = torch.tensor(1.5)
        metrics = {"perplexity": torch.tensor(4.5)}
        output = ObjectiveOutput(loss=loss, metrics=metrics)
        assert output.metrics["perplexity"].item() == 4.5

    def test_output_with_ce_loss(self):
        """Test output with CE loss for logging."""
        loss = torch.tensor(1.5)
        ce_loss = torch.tensor(1.2)
        output = ObjectiveOutput(loss=loss, ce_loss=ce_loss)
        assert abs(output.ce_loss.item() - 1.2) < 1e-5


class TestContinuePretrainObjective:
    """Test ContinuePretrainObjective."""

    def test_init(self):
        """Test initialization."""
        obj = ContinuePretrainObjective()
        assert obj.name == "continue_pretrain"
        assert not obj.requires_teacher
        assert not obj.requires_hidden_states
        assert not obj.modifies_input

    def test_cross_entropy_computation(self):
        """Test cross-entropy loss computation."""
        obj = ContinuePretrainObjective()

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        model_outputs = {"logits": logits}
        batch = {"labels": labels}

        output = obj(model_outputs, batch)

        assert output.loss.shape == ()
        assert output.loss.item() > 0
        assert "perplexity" in output.metrics
        assert output.ce_loss is not None

    def test_label_shifting(self):
        """Test that labels are shifted for next token prediction."""
        obj = ContinuePretrainObjective()

        # Create simple case where we can verify shifting
        batch_size, seq_len, vocab_size = 1, 5, 10
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 2, 3, 4, 5]])

        # Set logits to make prediction easy
        for i in range(seq_len - 1):
            logits[0, i, labels[0, i + 1]] = 10.0

        model_outputs = {"logits": logits}
        batch = {"labels": labels}

        output = obj(model_outputs, batch)
        # Loss should be low since predictions align with shifted labels
        assert output.loss.item() < 1.0

    def test_perplexity_metric(self):
        """Test perplexity computation."""
        obj = ContinuePretrainObjective()

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        model_outputs = {"logits": logits}
        batch = {"labels": labels}

        output = obj(model_outputs, batch)

        # Perplexity should be exp(loss)
        expected_ppl = torch.exp(output.loss.detach())
        assert torch.allclose(output.metrics["perplexity"], expected_ppl)


class TestLayerwiseDistillationObjective:
    """Test LayerwiseDistillationObjective."""

    def test_init(self):
        """Test initialization."""
        obj = LayerwiseDistillationObjective()
        assert obj.name == "layerwise_distill"
        assert obj.requires_teacher
        assert obj.requires_hidden_states
        assert not obj.modifies_input

    def test_requires_teacher_outputs(self):
        """Test that it raises without teacher outputs."""
        obj = LayerwiseDistillationObjective()

        model_outputs = {"logits": torch.randn(2, 10, 100), "hidden_states": (torch.randn(2, 10, 64),)}
        batch = {}

        with pytest.raises(ValueError, match="requires teacher_outputs"):
            obj(model_outputs, batch)

    def test_hidden_state_alignment(self):
        """Test hidden state alignment loss."""
        obj = LayerwiseDistillationObjective(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        batch_size, seq_len, hidden_size = 2, 10, 64
        num_layers = 4

        # Create hidden states (including embedding layer at index 0)
        student_hidden = tuple(torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1))
        teacher_hidden = tuple(torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1))

        model_outputs = {"logits": torch.randn(2, 10, 100), "hidden_states": student_hidden}
        teacher_outputs = {"hidden_states": teacher_hidden}
        batch = {}

        output = obj(model_outputs, batch, teacher_outputs)

        assert output.loss.shape == ()
        assert output.loss.item() > 0
        assert "mean_layer_loss" in output.metrics

    def test_layer_weights_progressive(self):
        """Test progressive layer weights."""
        obj = LayerwiseDistillationObjective(layer_weights="progressive")

        weights = obj._get_layer_weights(4)

        # Progressive weights should increase
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]
        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_layer_weights_exponential(self):
        """Test exponential layer weights."""
        obj = LayerwiseDistillationObjective(layer_weights="exponential")

        weights = obj._get_layer_weights(4)

        # Exponential weights should increase faster
        assert weights[-1] > weights[0] * 4
        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_mse_normalized_loss(self):
        """Test MSE normalized loss computation."""
        obj = LayerwiseDistillationObjective(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        batch_size, seq_len, hidden_size = 2, 10, 64

        # Identical hidden states should have near-zero loss
        student_h = torch.randn(batch_size, seq_len, hidden_size)
        teacher_h = student_h.clone()

        loss = obj._mse_loss(student_h, teacher_h, None, normalize=True)
        assert loss.item() < 1e-6


class TestCurriculumScheduler:
    """Test CurriculumScheduler."""

    def test_init(self):
        """Test initialization."""
        phases = [
            CurriculumPhase("warmup", 0.2, {"continue_pretrain": 1.0}),
            CurriculumPhase("main", 1.0, {"continue_pretrain": 1.0, "dlm": 0.1}),
        ]
        scheduler = CurriculumScheduler(phases, total_steps=1000)

        assert scheduler.total_steps == 1000
        assert len(scheduler.phases) == 2

    def test_phase_transitions(self):
        """Test phase transitions."""
        phases = [
            CurriculumPhase("warmup", 0.2, {"continue_pretrain": 1.0}),
            CurriculumPhase("main", 1.0, {"continue_pretrain": 0.8, "dlm": 0.2}),
        ]
        scheduler = CurriculumScheduler(phases, total_steps=100)

        # At step 0, should be in warmup
        assert scheduler.get_current_phase().name == "warmup"

        # At step 30, should be in main
        for _ in range(30):
            scheduler.step()
        assert scheduler.get_current_phase().name == "main"

    def test_linear_interpolation(self):
        """Test linear weight interpolation."""
        phases = [
            CurriculumPhase("start", 0.5, {"obj1": 1.0, "obj2": 0.0}),
            CurriculumPhase("end", 1.0, {"obj1": 0.0, "obj2": 1.0}),
        ]
        scheduler = CurriculumScheduler(phases, total_steps=100, interpolation="linear")

        # Move to middle of second phase (step 75 = 75% progress)
        for _ in range(75):
            scheduler.step()

        weights = scheduler.get_weights()
        # At 75% progress (middle of second phase), should be interpolated
        assert 0.4 < weights["obj1"] < 0.6
        assert 0.4 < weights["obj2"] < 0.6

    def test_weight_at_boundary(self):
        """Test weights at phase boundaries."""
        phases = [
            CurriculumPhase("phase1", 0.5, {"obj1": 1.0}),
            CurriculumPhase("phase2", 1.0, {"obj1": 0.5}),
        ]
        scheduler = CurriculumScheduler(phases, total_steps=100)

        # At step 50 (exactly at boundary)
        for _ in range(50):
            scheduler.step()

        weights = scheduler.get_weights()
        # Should be at or near the start of phase2
        assert weights["obj1"] <= 1.0

    def test_data_config_switching(self):
        """Test data config switching between phases."""
        phases = [
            CurriculumPhase("warmup", 0.2, {"cp": 1.0}, data_config="fineweb"),
            CurriculumPhase("main", 1.0, {"cp": 1.0}, data_config="mixed_pretrain"),
        ]
        scheduler = CurriculumScheduler(phases, total_steps=100)

        assert scheduler.get_data_config() == "fineweb"

        for _ in range(25):
            scheduler.step()

        assert scheduler.get_data_config() == "mixed_pretrain"

    def test_state_dict(self):
        """Test state serialization."""
        phases = [CurriculumPhase("main", 1.0, {"cp": 1.0})]
        scheduler = CurriculumScheduler(phases, total_steps=100)

        for _ in range(50):
            scheduler.step()

        state = scheduler.state_dict()
        assert state["current_step"] == 50

        scheduler2 = CurriculumScheduler(phases, total_steps=100)
        scheduler2.load_state_dict(state)
        assert scheduler2.current_step == 50


class TestObjectiveManager:
    """Test ObjectiveManager."""

    def test_init(self):
        """Test initialization."""
        objectives = {
            "continue_pretrain": ContinuePretrainObjective(),
        }
        manager = ObjectiveManager(objectives)

        assert len(manager.objectives) == 1
        assert not manager.requires_teacher
        assert not manager.requires_hidden_states

    def test_weighted_combination(self):
        """Test weighted loss combination."""
        objectives = {
            "obj1": ContinuePretrainObjective(),
            "obj2": ContinuePretrainObjective(),
        }
        weights = {"obj1": 1.0, "obj2": 0.5}
        manager = ObjectiveManager(objectives, weights=weights)

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        model_outputs = {"logits": logits}
        batch = {"labels": labels}

        output = manager(model_outputs, batch)

        # Combined loss should be weighted sum
        obj1_loss = output.objective_outputs["obj1"].loss
        obj2_loss = output.objective_outputs["obj2"].loss
        expected = 1.0 * obj1_loss + 0.5 * obj2_loss

        assert torch.allclose(output.loss, expected)

    def test_requires_teacher_aggregation(self):
        """Test requires_teacher is correctly aggregated."""
        objectives = {
            "cp": ContinuePretrainObjective(),
            "lw": LayerwiseDistillationObjective(),
        }
        manager = ObjectiveManager(objectives)

        assert manager.requires_teacher  # Because layerwise needs it
        assert manager.requires_hidden_states

    def test_wandb_metrics_format(self):
        """Test wandb metrics generation."""
        objectives = {"continue_pretrain": ContinuePretrainObjective()}
        manager = ObjectiveManager(objectives)

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        model_outputs = {"logits": logits}
        batch = {"labels": labels}

        output = manager(model_outputs, batch)
        metrics = manager.get_wandb_metrics(output)

        assert "train/loss" in metrics
        assert "train/loss_unweighted_ce" in metrics
        assert "train/perplexity" in metrics
        assert "train/continue_pretrain_loss" in metrics
        assert "schedule/continue_pretrain_weight" in metrics


class TestFactory:
    """Test objective factory functions."""

    def test_create_from_config(self):
        """Test creating ObjectiveManager from config."""
        config = {
            "objectives": {
                "continue_pretrain": {
                    "enabled": True,
                    "weight": 1.0,
                },
            },
            "curriculum": {
                "enabled": False,
            },
        }

        manager = create_objective_manager(config, total_steps=1000)

        assert len(manager.objectives) == 1
        assert "continue_pretrain" in manager.objectives

    def test_disabled_objectives(self):
        """Test that disabled objectives are not created."""
        config = {
            "objectives": {
                "continue_pretrain": {"enabled": True, "weight": 1.0},
                "layerwise_distill": {"enabled": False, "weight": 0.5},
            },
        }

        manager = create_objective_manager(config, total_steps=1000)

        assert len(manager.objectives) == 1
        assert "continue_pretrain" in manager.objectives
        assert "layerwise_distill" not in manager.objectives

    def test_with_curriculum(self):
        """Test creating manager with curriculum."""
        config = {
            "objectives": {
                "continue_pretrain": {"enabled": True, "weight": 1.0},
            },
            "curriculum": {
                "enabled": True,
                "interpolation": "linear",
                "phases": [
                    {"name": "warmup", "end_ratio": 0.2, "objectives": {"continue_pretrain": 1.0}},
                    {"name": "main", "end_ratio": 1.0, "objectives": {"continue_pretrain": 0.8}},
                ],
            },
        }

        manager = create_objective_manager(config, total_steps=1000)

        assert manager.curriculum is not None
        assert len(manager.curriculum.phases) == 2

    def test_default_objective(self):
        """Test that default objective is created when none specified."""
        config = {"objectives": {}}

        manager = create_objective_manager(config, total_steps=1000)

        assert len(manager.objectives) == 1
        assert "continue_pretrain" in manager.objectives
