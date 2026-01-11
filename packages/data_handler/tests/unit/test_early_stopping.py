"""Tests for PlateauEarlyStopping."""

import json
import tempfile
from pathlib import Path

import pytest

from wf_data.training import PlateauEarlyStopping


class TestPlateauEarlyStopping:
    """Unit tests for PlateauEarlyStopping class."""

    def test_disabled_never_stops(self):
        """When disabled, check() always returns False."""
        es = PlateauEarlyStopping(patience=1, enabled=False)

        # Even with many plateau checks, should never stop
        for _ in range(100):
            assert es.check(1.0, step=0) is False

        assert es.stopped_early is False

    def test_basic_plateau_detection(self):
        """Detects plateau after patience consecutive non-improvements."""
        es = PlateauEarlyStopping(
            patience=3,
            min_delta=0.01,
            min_evals=2,
            enabled=True,
            rank=0,
        )

        # Improving losses - should not trigger
        assert es.check(2.0, step=0) is False  # First eval
        assert es.check(1.9, step=1) is False  # Improved
        assert es.check(1.8, step=2) is False  # Improved

        # Plateau (within min_delta of best)
        assert es.check(1.79, step=3) is False  # wait=1
        assert es.check(1.79, step=4) is False  # wait=2
        assert es.check(1.79, step=5) is True   # wait=3, triggers!

        assert es.stopped_early is True
        assert es.best == pytest.approx(1.8, abs=0.01)

    def test_min_evals_respected(self):
        """Should not stop before min_evals checks."""
        es = PlateauEarlyStopping(
            patience=1,
            min_delta=0.01,
            min_evals=5,
            enabled=True,
            rank=0,
        )

        # All same loss - would trigger after patience=1, but min_evals=5
        for i in range(4):
            assert es.check(1.0, step=i) is False

        # 5th eval - now can trigger
        assert es.check(1.0, step=4) is True

    def test_improvement_resets_wait(self):
        """Improvement resets the wait counter."""
        es = PlateauEarlyStopping(
            patience=3,
            min_delta=0.01,
            min_evals=1,
            enabled=True,
            rank=0,
        )

        es.check(2.0, step=0)
        es.check(1.99, step=1)  # wait=1 (not improving by min_delta)
        es.check(1.99, step=2)  # wait=2

        # Big improvement - resets wait
        es.check(1.5, step=3)
        assert es.wait == 0
        assert es.best == pytest.approx(1.5)

    def test_max_mode(self):
        """Mode='max' works for accuracy-like metrics."""
        es = PlateauEarlyStopping(
            patience=2,
            min_delta=0.01,
            min_evals=1,
            mode="max",
            enabled=True,
            rank=0,
        )

        es.check(0.5, step=0)   # First
        es.check(0.6, step=1)   # Improved
        es.check(0.59, step=2)  # wait=1

        assert es.check(0.59, step=3) is True  # wait=2, triggers
        assert es.best == pytest.approx(0.6)

    def test_state_dict_round_trip(self):
        """State can be saved and restored correctly."""
        es1 = PlateauEarlyStopping(patience=5, enabled=True, rank=0)

        # Run some checks
        es1.check(2.0, step=0)
        es1.check(1.9, step=1)
        es1.check(1.89, step=2)  # wait=1

        # Save state
        state = es1.state_dict()

        # Create new instance and restore
        es2 = PlateauEarlyStopping(patience=5, enabled=True, rank=0)
        es2.load_state_dict(state)

        assert es2.best == es1.best
        assert es2.wait == es1.wait
        assert es2.eval_count == es1.eval_count
        assert es2.best_step == es1.best_step

    def test_save_json(self):
        """JSON file is saved correctly."""
        es = PlateauEarlyStopping(
            patience=3,
            min_delta=0.05,
            min_evals=2,
            enabled=True,
            rank=0,
        )

        es.check(2.0, step=0)
        es.check(1.5, step=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            es.save_json(Path(tmpdir))

            json_path = Path(tmpdir) / "early_stopping.json"
            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)

            assert data["best_metric"] == pytest.approx(1.5)
            assert data["best_step"] == 1
            assert data["total_evals"] == 2
            assert data["config"]["patience"] == 3
            assert data["config"]["min_delta"] == 0.05


class TestPlateauIntegration:
    """Integration test with simulated training loop."""

    def test_simulated_training_plateau(self):
        """Simulate a training run that plateaus."""
        es = PlateauEarlyStopping(
            patience=3,
            min_delta=0.01,  # Larger delta so plateau is clearer
            min_evals=5,
            enabled=True,
            rank=0,
        )

        # Simulate loss curve: drops then plateaus
        losses = (
            [3.0, 2.5, 2.2, 2.0, 1.9]  # Improving (5 evals = min_evals)
            + [1.895, 1.894, 1.893]  # Plateau: < min_delta=0.01 improvement
        )

        stopped_at = None
        for step, loss in enumerate(losses):
            if es.check(loss, step):
                stopped_at = step
                break

        assert stopped_at is not None, f"Should have triggered. wait={es.wait}, best={es.best}"
        assert stopped_at == 7  # After 3 patience steps (indices 5,6,7)
        assert es.stopped_early is True

    def test_continuous_improvement_no_stop(self):
        """Continuous improvement should never trigger early stopping."""
        es = PlateauEarlyStopping(
            patience=3,
            min_delta=0.001,
            min_evals=5,
            enabled=True,
            rank=0,
        )

        # Steadily decreasing loss
        for step in range(100):
            loss = 5.0 - step * 0.02  # Decreases by 0.02 each step
            if es.check(loss, step):
                pytest.fail(f"Should not stop with continuous improvement at step {step}")

        assert es.stopped_early is False
        assert es.wait == 0  # Never accumulated wait


class TestTinyModelPlateau:
    """Test with actual tiny model to force plateau."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch not available"
    )
    def test_tiny_model_plateau(self):
        """Train tiny model on trivial task until it plateaus."""
        import torch
        import torch.nn as nn

        # Tiny model: 2-layer MLP (~1K params)
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        # Trivial task: identity (easy to overfit, will plateau)
        torch.manual_seed(42)
        X = torch.randn(32, 8)
        y = X.clone()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        es = PlateauEarlyStopping(
            patience=10,
            min_delta=1e-6,  # Very small - will plateau once loss stabilizes
            min_evals=20,
            enabled=True,
            rank=0,
        )

        stopped_at = None
        for step in range(500):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            if es.check(loss.item(), step):
                stopped_at = step
                break

        # Should have plateaued (this task is easy to overfit)
        assert stopped_at is not None, f"Should have stopped. Final loss: {loss.item():.6f}, wait: {es.wait}"
        assert stopped_at < 200, f"Should plateau quickly on trivial task, stopped at {stopped_at}"
        assert es.stopped_early is True

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch not available"
    )
    def test_tiny_model_plateau_with_wandb(self):
        """Train tiny model with WandB - triggers real alert email."""
        import os
        import torch
        import torch.nn as nn

        # Skip if no WandB API key
        if not os.environ.get("WANDB_API_KEY"):
            pytest.skip("WANDB_API_KEY not set - skipping WandB integration test")

        import wandb

        # Initialize WandB run
        wandb.init(
            project="wrinklefree-tests",
            name="early-stopping-test",
            tags=["test", "early-stopping"],
            config={"test": "plateau_detection"},
        )

        try:
            # Tiny model: 2-layer MLP (~1K params)
            model = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
            )

            # Trivial task: identity
            torch.manual_seed(42)
            X = torch.randn(32, 8)
            y = X.clone()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            criterion = nn.MSELoss()

            es = PlateauEarlyStopping(
                patience=10,
                min_delta=1e-6,
                min_evals=20,
                enabled=True,
                rank=0,
            )

            stopped_at = None
            for step in range(500):
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                # Log to WandB
                wandb.log({"train/loss": loss.item()}, step=step)

                if es.check(loss.item(), step):
                    stopped_at = step
                    break

            assert stopped_at is not None, "Should have triggered early stopping"
            assert es.stopped_early is True

            # Verify WandB summary was set
            assert wandb.run.summary.get("early_stopped") is True
            assert wandb.run.summary.get("plateau_loss") is not None

        finally:
            wandb.finish()
