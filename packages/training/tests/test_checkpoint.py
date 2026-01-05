"""Smoke tests for checkpoint save/load functionality."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from wf_train.training._legacy.trainer import Trainer, create_optimizer, create_scheduler


class SimpleModel(nn.Module):
    """A tiny model for testing checkpointing (no GPU required)."""

    def __init__(self, input_size: int = 32, hidden_size: int = 64, output_size: int = 32):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.norm(x)
        x = self.linear2(x)
        return x


class TestCheckpointSaveLoad:
    """Tests for checkpoint saving and loading."""

    def test_save_checkpoint_creates_file(self):
        """Test that save_checkpoint creates the expected file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            config = OmegaConf.create({
                "output_dir": str(output_dir),
                "max_steps": 10,
            })

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                config=config,
                device=torch.device("cpu"),
            )

            # Simulate some training state
            trainer.global_step = 100
            trainer.epoch = 2
            trainer.best_eval_loss = 0.5
            trainer.train_losses = [1.0, 0.8, 0.6]
            trainer.eval_losses = [0.9, 0.7]

            # Save checkpoint
            trainer.save_checkpoint("test_ckpt")

            # Verify file exists
            checkpoint_path = output_dir / "checkpoints" / "test_ckpt" / "checkpoint.pt"
            assert checkpoint_path.exists(), f"Checkpoint file not found at {checkpoint_path}"

    def test_load_checkpoint_restores_state(self):
        """Test that load_checkpoint correctly restores training state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create and save a checkpoint
            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            config = OmegaConf.create({
                "output_dir": str(output_dir),
                "max_steps": 10,
            })

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                config=config,
                device=torch.device("cpu"),
            )

            # Set training state
            trainer.global_step = 42
            trainer.epoch = 3
            trainer.best_eval_loss = 0.25
            trainer.train_losses = [1.0, 0.5, 0.3]
            trainer.eval_losses = [0.8, 0.4]

            # Do a fake forward/backward to populate optimizer state
            x = torch.randn(4, 32)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()

            # Save checkpoint
            trainer.save_checkpoint("restore_test")

            # Create a fresh trainer with new model/optimizer
            new_model = SimpleModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

            new_trainer = Trainer(
                model=new_model,
                optimizer=new_optimizer,
                config=config,
                device=torch.device("cpu"),
            )

            # Verify initial state is different
            assert new_trainer.global_step == 0
            assert new_trainer.epoch == 0

            # Load checkpoint
            checkpoint_dir = output_dir / "checkpoints" / "restore_test"
            new_trainer.load_checkpoint(checkpoint_dir)

            # Verify state is restored
            assert new_trainer.global_step == 42, f"global_step not restored: {new_trainer.global_step}"
            assert new_trainer.epoch == 3, f"epoch not restored: {new_trainer.epoch}"
            assert new_trainer.best_eval_loss == 0.25, f"best_eval_loss not restored: {new_trainer.best_eval_loss}"
            assert new_trainer.train_losses == [1.0, 0.5, 0.3], f"train_losses not restored"
            assert new_trainer.eval_losses == [0.8, 0.4], f"eval_losses not restored"

    def test_model_weights_restored_correctly(self):
        """Test that model weights are correctly saved and restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            config = OmegaConf.create({
                "output_dir": str(output_dir),
                "max_steps": 10,
            })

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                config=config,
                device=torch.device("cpu"),
            )

            # Modify weights to non-default values
            with torch.no_grad():
                for param in model.parameters():
                    param.fill_(0.123)

            # Save checkpoint
            trainer.save_checkpoint("weights_test")

            # Create fresh model with different weights
            new_model = SimpleModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

            new_trainer = Trainer(
                model=new_model,
                optimizer=new_optimizer,
                config=config,
                device=torch.device("cpu"),
            )

            # Verify weights are different before loading
            for param in new_model.parameters():
                assert not torch.allclose(param, torch.full_like(param, 0.123)), \
                    "New model should have different weights before load"

            # Load checkpoint
            checkpoint_dir = output_dir / "checkpoints" / "weights_test"
            new_trainer.load_checkpoint(checkpoint_dir)

            # Verify weights are restored
            for param in new_model.parameters():
                assert torch.allclose(param, torch.full_like(param, 0.123)), \
                    "Model weights not correctly restored"

    def test_scheduler_state_saved_and_restored(self):
        """Test that scheduler state is correctly saved and restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = create_scheduler(
                optimizer,
                scheduler_type="cosine",
                num_training_steps=1000,
                num_warmup_steps=100,
            )

            config = OmegaConf.create({
                "output_dir": str(output_dir),
                "max_steps": 10,
            })

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                device=torch.device("cpu"),
            )

            # Step scheduler a few times
            for _ in range(50):
                scheduler.step()

            original_lr = optimizer.param_groups[0]["lr"]
            trainer.global_step = 50

            # Save checkpoint
            trainer.save_checkpoint("scheduler_test")

            # Create fresh setup
            new_model = SimpleModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
            new_scheduler = create_scheduler(
                new_optimizer,
                scheduler_type="cosine",
                num_training_steps=1000,
                num_warmup_steps=100,
            )

            # Verify LR is different before loading
            assert new_optimizer.param_groups[0]["lr"] != original_lr

            new_trainer = Trainer(
                model=new_model,
                optimizer=new_optimizer,
                scheduler=new_scheduler,
                config=config,
                device=torch.device("cpu"),
            )

            # Load checkpoint
            checkpoint_dir = output_dir / "checkpoints" / "scheduler_test"
            new_trainer.load_checkpoint(checkpoint_dir)

            # Verify scheduler state is restored (LR should match)
            restored_lr = new_optimizer.param_groups[0]["lr"]
            assert abs(restored_lr - original_lr) < 1e-8, \
                f"Scheduler LR not restored: {restored_lr} vs {original_lr}"

    def test_checkpoint_with_file_path(self):
        """Test loading checkpoint by direct file path instead of directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            config = OmegaConf.create({
                "output_dir": str(output_dir),
                "max_steps": 10,
            })

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                config=config,
                device=torch.device("cpu"),
            )

            trainer.global_step = 99
            trainer.save_checkpoint("path_test")

            # Load by file path directly
            new_model = SimpleModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

            new_trainer = Trainer(
                model=new_model,
                optimizer=new_optimizer,
                config=config,
                device=torch.device("cpu"),
            )

            # Load using direct file path
            checkpoint_file = output_dir / "checkpoints" / "path_test" / "checkpoint.pt"
            new_trainer.load_checkpoint(checkpoint_file)

            assert new_trainer.global_step == 99


class TestCheckpointContents:
    """Tests for checkpoint file contents."""

    def test_checkpoint_contains_expected_keys(self):
        """Test that checkpoint contains all expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            model = SimpleModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

            config = OmegaConf.create({
                "output_dir": str(output_dir),
                "max_steps": 10,
            })

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                device=torch.device("cpu"),
            )

            trainer.save_checkpoint("keys_test")

            # Load and inspect checkpoint
            checkpoint_path = output_dir / "checkpoints" / "keys_test" / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            expected_keys = {
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "global_step",
                "epoch",
                "best_eval_loss",
                "train_losses",
                "eval_losses",
            }

            assert set(checkpoint.keys()) == expected_keys, \
                f"Missing keys: {expected_keys - set(checkpoint.keys())}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
