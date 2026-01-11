"""Integration tests for unified training with ObjectiveManager.

Tests the end-to-end flow of:
1. Creating ObjectiveManager from config
2. Passing to ContinuedPretrainingTrainer
3. Forward step using _forward_step_objectives
4. Curriculum weight updates
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from wf_train.objectives import (
    ObjectiveManager,
    ContinuePretrainObjective,
    DistillObjective,
    LayerWiseConfig,
    create_objective_manager,
)
from wf_train.training._legacy.continued_pretraining import ContinuedPretrainingTrainer


class SimpleLMHead(nn.Module):
    """Minimal LM model for testing."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        output_hidden_states: bool = False,
    ) -> dict:
        x = self.embedding(input_ids)
        hidden_states = [x] if output_hidden_states else None

        for layer in self.layers:
            x = torch.relu(layer(x))
            if output_hidden_states:
                hidden_states.append(x)

        logits = self.lm_head(x)

        result = {"logits": logits}
        if output_hidden_states:
            result["hidden_states"] = tuple(hidden_states)
        return result


def create_dummy_dataloader(batch_size: int = 4, seq_len: int = 16, vocab_size: int = 100, num_batches: int = 10):
    """Create a dummy dataloader for testing."""
    input_ids = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Convert to dict format
    class DictDataLoader:
        def __init__(self, dl):
            self.dl = dl

        def __iter__(self):
            for batch in self.dl:
                yield {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2],
                }

        def __len__(self):
            return len(self.dl)

    return DictDataLoader(dataloader)


class TestUnifiedTrainerForwardStep:
    """Test _forward_step_objectives in ContinuedPretrainingTrainer."""

    @pytest.fixture
    def model(self):
        return SimpleLMHead(vocab_size=100, hidden_size=64, num_layers=2)

    @pytest.fixture
    def dataloader(self):
        return create_dummy_dataloader(batch_size=4, seq_len=16, vocab_size=100)

    @pytest.fixture
    def config(self):
        return OmegaConf.create({
            "training": {
                "batch_size": 4,
                "max_seq_length": 16,
                "gradient_accumulation_steps": 1,
                "total_tokens": 10000,
                "gradient_clipping": 1.0,
                "max_steps": 10,
                "logging": {"log_interval": 5, "wandb": {"enabled": False}},
                "checkpoint": {"save_interval": 100},
                "objectives": {
                    "continue_pretrain": {"enabled": True, "weight": 1.0},
                },
                "curriculum": {"enabled": False},
            },
            "output_dir": "/tmp/test_unified",
        })

    def test_forward_step_with_objective_manager(self, model, dataloader, config):
        """Test that _forward_step_objectives works correctly."""
        # Create ObjectiveManager
        objective_manager = ObjectiveManager(
            objectives={"continue_pretrain": ContinuePretrainObjective()},
            weights={"continue_pretrain": 1.0},
        )

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create trainer
        trainer = ContinuedPretrainingTrainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=dataloader,
            config=config,
            objective_manager=objective_manager,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
        )

        # Get a batch
        batch = next(iter(dataloader))

        # Test forward step
        loss_dict = trainer._forward_step(batch)

        # Verify outputs
        assert "loss" in loss_dict
        assert loss_dict["loss"].item() > 0
        assert "tokens_processed" in loss_dict
        assert "continue_pretrain_loss" in loss_dict
        assert "perplexity" in loss_dict
        assert "weight_continue_pretrain" in loss_dict

    def test_forward_step_routes_to_objectives(self, model, dataloader, config):
        """Test that _forward_step routes to _forward_step_objectives when manager present."""
        objective_manager = ObjectiveManager(
            objectives={"continue_pretrain": ContinuePretrainObjective()},
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        trainer = ContinuedPretrainingTrainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=dataloader,
            config=config,
            objective_manager=objective_manager,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
        )

        assert trainer.use_objective_manager is True
        assert trainer.loss_fn is None

    def test_forward_step_without_objective_manager(self, model, dataloader, config):
        """Test fallback to _forward_step_lm when no ObjectiveManager."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        trainer = ContinuedPretrainingTrainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=dataloader,
            config=config,
            objective_manager=None,  # No manager
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
        )

        assert trainer.use_objective_manager is False
        assert trainer.loss_fn is not None

        # Test forward step
        batch = next(iter(dataloader))
        loss_dict = trainer._forward_step(batch)

        assert "loss" in loss_dict
        assert loss_dict["loss"].item() > 0


class TestUnifiedTrainerCurriculum:
    """Test curriculum scheduling in unified trainer."""

    def test_curriculum_weights_update(self):
        """Test that curriculum weights are updated during training."""
        from wf_train.objectives.manager import CurriculumPhase, CurriculumScheduler

        # Create curriculum
        phases = [
            CurriculumPhase("phase1", 0.5, {"continue_pretrain": 1.0}),
            CurriculumPhase("phase2", 1.0, {"continue_pretrain": 0.5}),
        ]
        curriculum = CurriculumScheduler(phases, total_steps=100)

        objectives = {"continue_pretrain": ContinuePretrainObjective()}
        manager = ObjectiveManager(objectives, curriculum=curriculum)

        # Initial weights
        weights = manager.get_current_weights()
        assert weights["continue_pretrain"] == 1.0

        # Step curriculum to phase 2
        for _ in range(60):
            manager.step_curriculum()

        weights = manager.get_current_weights()
        # Should be interpolating toward 0.5
        assert weights["continue_pretrain"] < 1.0


class TestUnifiedTrainerFromConfig:
    """Test creating unified trainer from Hydra config."""

    def test_create_from_unified_config(self):
        """Test creating ObjectiveManager from unified training config."""
        config = {
            "objectives": {
                "continue_pretrain": {
                    "enabled": True,
                    "weight": 1.0,
                    "ignore_index": -100,
                },
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

        assert len(manager.objectives) == 1
        assert "continue_pretrain" in manager.objectives
        assert manager.curriculum is not None
        assert len(manager.curriculum.phases) == 2

    def test_multiple_objectives(self):
        """Test creating manager with multiple objectives."""
        config = {
            "objectives": {
                "continue_pretrain": {"enabled": True, "weight": 1.0},
                "layerwise_distill": {"enabled": True, "weight": 0.5, "loss_type": "mse_normalized"},
            },
        }

        manager = create_objective_manager(config, total_steps=1000)

        assert len(manager.objectives) == 2
        assert manager.requires_teacher  # layerwise requires teacher
        assert manager.requires_hidden_states


class TestUnifiedTrainerMetrics:
    """Test metrics generation in unified trainer."""

    def test_wandb_metrics_format(self):
        """Test that wandb metrics are correctly formatted."""
        objectives = {"continue_pretrain": ContinuePretrainObjective()}
        manager = ObjectiveManager(objectives, weights={"continue_pretrain": 1.0})

        # Create dummy inputs
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))

        output = manager(
            model_outputs={"logits": logits},
            batch={"labels": labels},
        )

        metrics = manager.get_wandb_metrics(output)

        # Check required metrics
        assert "train/loss" in metrics
        assert "train/loss_unweighted_ce" in metrics
        assert "train/perplexity" in metrics
        assert "train/continue_pretrain_loss" in metrics
        assert "schedule/continue_pretrain_weight" in metrics

        # Check values are reasonable
        assert metrics["train/loss"] > 0
        assert metrics["train/perplexity"] > 1.0  # perplexity >= 1
        assert metrics["schedule/continue_pretrain_weight"] == 1.0
