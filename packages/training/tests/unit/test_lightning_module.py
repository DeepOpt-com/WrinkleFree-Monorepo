"""Unit tests for the Lightning module."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from wrinklefree.lightning.module import WrinkleFreeLightningModule
from wrinklefree.objectives.base import Objective, ObjectiveOutput
from wrinklefree.objectives.manager import ObjectiveManager


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
    ):
        x = self.embed(input_ids)
        hidden_states = [x] if output_hidden_states else None

        for layer in self.layers:
            x = torch.relu(layer(x))
            if output_hidden_states:
                hidden_states.append(x)

        logits = self.lm_head(x)

        if return_dict:
            result = MagicMock()
            result.logits = logits
            result.hidden_states = tuple(hidden_states) if hidden_states else None
            result.attentions = None
            return result
        return logits


class DummyObjective(Objective):
    """Simple objective for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def forward(self, model_outputs, batch, teacher_outputs=None):
        logits = model_outputs["logits"]
        labels = batch.get("labels", batch["input_ids"])

        # Simple cross-entropy
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return ObjectiveOutput(
            loss=loss,
            metrics={"ce_loss": loss.detach()},
            ce_loss=loss.detach(),
        )


class TestWrinkleFreeLightningModule:
    """Tests for WrinkleFreeLightningModule."""

    @pytest.fixture
    def model(self):
        return DummyModel()

    @pytest.fixture
    def objective_manager(self):
        objectives = {"dummy": DummyObjective()}
        return ObjectiveManager(objectives=objectives, weights={"dummy": 1.0})

    @pytest.fixture
    def module(self, model, objective_manager):
        return WrinkleFreeLightningModule(
            model=model,
            objective_manager=objective_manager,
            optimizer_cfg={"type": "adamw", "learning_rate": 1e-4},
        )

    def test_init(self, module):
        """Test module initialization."""
        assert module.model is not None
        assert module.objective_manager is not None
        assert module.tokens_processed == 0

    def test_forward(self, module):
        """Test forward pass."""
        batch_size, seq_len = 2, 16
        batch = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
        }

        outputs = module.forward(**batch)

        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 1000)

    def test_training_step(self, module):
        """Test training step returns loss."""
        batch_size, seq_len = 2, 16
        batch = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, 100, (batch_size, seq_len)),
        }

        # Mock the log methods
        module.log = MagicMock()
        module.log_dict = MagicMock()

        loss = module.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)

    def test_configure_optimizers_adamw(self, model, objective_manager):
        """Test AdamW optimizer configuration."""
        module = WrinkleFreeLightningModule(
            model=model,
            objective_manager=objective_manager,
            optimizer_cfg={"type": "adamw", "learning_rate": 1e-4, "weight_decay": 0.1},
        )

        result = module.configure_optimizers()

        # Should return just the optimizer (no scheduler by default)
        assert isinstance(result, torch.optim.AdamW)

    def test_configure_optimizers_with_scheduler(self, model, objective_manager):
        """Test optimizer with scheduler configuration."""
        module = WrinkleFreeLightningModule(
            model=model,
            objective_manager=objective_manager,
            optimizer_cfg={"type": "adamw", "learning_rate": 1e-4},
            scheduler_cfg={
                "type": "cosine_warmup",
                "warmup_steps": 100,
                "max_steps": 1000,
            },
        )

        result = module.configure_optimizers()

        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_checkpoint_state(self, module):
        """Test checkpoint save/load includes custom state."""
        module.tokens_processed = 12345

        checkpoint = {}
        module.on_save_checkpoint(checkpoint)

        assert checkpoint["tokens_processed"] == 12345

        # Reset and load
        module.tokens_processed = 0
        module.on_load_checkpoint(checkpoint)

        assert module.tokens_processed == 12345

    def test_teacher_model_frozen(self, model, objective_manager):
        """Test teacher model is frozen when provided."""
        teacher = DummyModel()

        module = WrinkleFreeLightningModule(
            model=model,
            objective_manager=objective_manager,
            teacher_model=teacher,
        )

        # Teacher should be frozen
        for param in module.teacher_model.parameters():
            assert not param.requires_grad


class TestObjectiveManagerIntegration:
    """Test integration with ObjectiveManager."""

    def test_preprocess_batch_called(self):
        """Test that preprocess_batch is called on the batch."""
        model = DummyModel()
        objective = DummyObjective()
        objective.modifies_input = True
        objective.preprocess_batch = MagicMock(side_effect=lambda x: x)

        manager = ObjectiveManager(
            objectives={"dummy": objective},
            weights={"dummy": 1.0},
        )

        module = WrinkleFreeLightningModule(
            model=model,
            objective_manager=manager,
        )
        module.log = MagicMock()
        module.log_dict = MagicMock()

        batch = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "labels": torch.randint(0, 100, (2, 16)),
        }

        module.training_step(batch, batch_idx=0)

        objective.preprocess_batch.assert_called_once()

    def test_curriculum_stepped(self):
        """Test that curriculum is stepped after each training step."""
        from wrinklefree.objectives.manager import CurriculumPhase, CurriculumScheduler

        model = DummyModel()
        objective = DummyObjective()

        phases = [
            CurriculumPhase(name="main", end_ratio=1.0, objective_weights={"dummy": 1.0}),
        ]
        curriculum = CurriculumScheduler(phases=phases, total_steps=100)

        manager = ObjectiveManager(
            objectives={"dummy": objective},
            weights={"dummy": 1.0},
            curriculum=curriculum,
        )

        module = WrinkleFreeLightningModule(
            model=model,
            objective_manager=manager,
        )
        module.log = MagicMock()
        module.log_dict = MagicMock()

        batch = {
            "input_ids": torch.randint(0, 100, (2, 16)),
            "labels": torch.randint(0, 100, (2, 16)),
        }

        initial_step = curriculum.current_step
        module.training_step(batch, batch_idx=0)

        assert curriculum.current_step == initial_step + 1
