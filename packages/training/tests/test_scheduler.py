"""Tests for learning rate schedulers."""

import pytest
import torch
from wf_train.training._legacy.trainer import create_scheduler


class TestWSDScheduler:
    """Tests for Warmup-Stable-Decay (WSD) scheduler."""

    def test_wsd_three_phases(self):
        """WSD should have warmup, stable, and decay phases."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="wsd",
            num_training_steps=100,
            num_warmup_steps=10,
            num_decay_steps=20,  # 80 stable steps
            min_lr_ratio=0.0,
            decay_type="linear",
        )

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # Phase 1: Warmup (steps 0-9) - LR increases
        assert lrs[0] < lrs[9], "LR should increase during warmup"

        # Phase 2: Stable (steps 10-79) - LR constant at peak
        assert abs(lrs[10] - 1.0) < 0.01, f"LR should be ~1.0 at start of stable, got {lrs[10]}"
        assert abs(lrs[79] - 1.0) < 0.01, f"LR should be ~1.0 at end of stable, got {lrs[79]}"

        # Phase 3: Decay (steps 80-99) - LR decreases
        assert lrs[80] > lrs[99], "LR should decrease during decay"
        assert lrs[99] < 0.1, f"LR should be near 0 at end, got {lrs[99]}"

    def test_wsd_default_decay_ratio(self):
        """When decay_steps not specified, should use 20% of total."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="wsd",
            num_training_steps=1000,
            num_warmup_steps=100,
            # num_decay_steps not specified - should be 200 (20%)
            min_lr_ratio=0.0,
        )

        lrs = []
        for _ in range(1000):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # Stable phase should end around step 800 (1000 - 200 decay)
        # Check LR is still at peak around step 700
        assert abs(lrs[700] - 1.0) < 0.01, f"LR should be ~1.0 during stable phase, got {lrs[700]}"
        # Check LR is decaying at step 900
        assert lrs[900] < 0.6, f"LR should be decaying at step 900, got {lrs[900]}"

    def test_wsd_cosine_decay(self):
        """WSD with cosine decay type."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="wsd",
            num_training_steps=100,
            num_warmup_steps=10,
            num_decay_steps=20,
            min_lr_ratio=0.0,
            decay_type="cosine",
        )

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # Cosine decay should be smooth
        assert lrs[85] > lrs[95], "Cosine decay should decrease"
        # Cosine starts slower than linear
        assert lrs[85] > 0.7, f"Cosine decay starts slow, got {lrs[85]}"

    def test_wsd_min_lr_ratio(self):
        """WSD should decay to min_lr_ratio * peak_lr."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="wsd",
            num_training_steps=100,
            num_warmup_steps=10,
            num_decay_steps=20,
            min_lr_ratio=0.1,  # Decay to 10% of peak
            decay_type="linear",
        )

        # Run to end
        for _ in range(100):
            optimizer.step()
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert abs(final_lr - 0.1) < 0.02, f"Final LR should be ~0.1, got {final_lr}"


class TestCosineScheduler:
    """Tests for cosine scheduler (existing behavior)."""

    def test_cosine_warmup_and_decay(self):
        """Cosine scheduler should warmup then decay."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_training_steps=100,
            num_warmup_steps=10,
            min_lr_ratio=0.1,
        )

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # Warmup
        assert lrs[0] < lrs[9], "LR should increase during warmup"
        # Decay starts immediately after warmup
        assert lrs[10] > lrs[50], "LR should start decaying after warmup"
        # Final LR near min
        assert lrs[99] < 0.2, f"Final LR should be near min, got {lrs[99]}"


class TestConstantScheduler:
    """Tests for constant scheduler."""

    def test_constant_lr(self):
        """Constant scheduler should not change LR."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        scheduler = create_scheduler(
            optimizer,
            scheduler_type="constant",
            num_training_steps=100,
        )

        for _ in range(100):
            optimizer.step()
            scheduler.step()

        assert optimizer.param_groups[0]["lr"] == 0.5, "LR should remain constant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
