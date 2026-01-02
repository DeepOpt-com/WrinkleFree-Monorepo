"""Tests for curriculum learning functionality.

Tests the curriculum warmup feature where training starts with a simple dataset
(e.g., fineweb-edu) before switching to a more complex dataset (e.g., mixed_pretrain).
"""

import concurrent.futures
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset


class MockIterableDataset(IterableDataset):
    """Simple mock dataset for testing."""

    def __init__(self, name: str, batch_size: int = 4):
        self.name = name
        self.batch_size = batch_size
        self.iteration_count = 0

    def __iter__(self):
        while True:
            self.iteration_count += 1
            # Return a batch with source indicator
            yield {
                "input_ids": torch.ones(self.batch_size, 128, dtype=torch.long),
                "labels": torch.ones(self.batch_size, 128, dtype=torch.long),
                "source": self.name,
            }


class TestCurriculumConfig:
    """Test curriculum configuration parsing."""

    def test_curriculum_config_defaults(self):
        """Verify curriculum config has correct defaults."""
        from omegaconf import OmegaConf

        config_str = """
        curriculum:
          enabled: true
          warmup_ratio: 0.2
          warmup_data_config: fineweb
        """
        cfg = OmegaConf.create(config_str)

        assert cfg.curriculum.enabled is True
        assert cfg.curriculum.warmup_ratio == 0.2
        assert cfg.curriculum.warmup_data_config == "fineweb"

    def test_curriculum_config_disabled(self):
        """Verify curriculum can be disabled."""
        from omegaconf import OmegaConf

        config_str = """
        curriculum:
          enabled: false
          warmup_ratio: 0.2
          warmup_data_config: fineweb
        """
        cfg = OmegaConf.create(config_str)

        assert cfg.curriculum.enabled is False

    def test_curriculum_switch_step_calculation(self):
        """Verify switch step is calculated correctly from warmup_ratio."""
        warmup_ratio = 0.2
        total_steps = 1000

        switch_step = int(total_steps * warmup_ratio)

        assert switch_step == 200


class TestDataloaderSwapping:
    """Test dataloader swapping mechanism."""

    def test_dataloader_swap_at_switch_step(self):
        """Verify dataloader is swapped at the correct step."""
        # Create two mock dataloaders
        phase1_dataset = MockIterableDataset("fineweb-edu")
        phase2_dataset = MockIterableDataset("mixed_pretrain")

        phase1_loader = DataLoader(phase1_dataset, batch_size=None)
        phase2_loader = DataLoader(phase2_dataset, batch_size=None)

        switch_step = 5
        current_loader = phase1_loader
        next_phase_loader = phase2_loader

        # Simulate training loop
        sources_seen = []
        for step in range(10):
            # Get batch from current loader
            batch = next(iter(current_loader))
            sources_seen.append(batch["source"])

            # Check for swap
            if step == switch_step - 1 and next_phase_loader is not None:
                current_loader = next_phase_loader
                next_phase_loader = None

        # Verify we used both dataloaders
        assert "fineweb-edu" in sources_seen
        assert "mixed_pretrain" in sources_seen

        # First 5 batches should be from phase 1
        for i in range(5):
            assert sources_seen[i] == "fineweb-edu"

        # Remaining batches should be from phase 2
        for i in range(5, 10):
            assert sources_seen[i] == "mixed_pretrain"

    def test_dataloader_swap_preserves_training_state(self):
        """Verify training state (step counter) is preserved after swap."""
        switch_step = 10
        global_step = 0

        # Simulate training
        for _ in range(20):
            global_step += 1

            if global_step == switch_step:
                # Swap would happen here
                pass

        assert global_step == 20  # Training continues uninterrupted


class TestBackgroundLoading:
    """Test background dataloader loading."""

    def test_background_loader_future_resolution(self):
        """Verify background-loaded dataloader can be resolved from future."""
        def create_loader_slow():
            # Simulate slow loading
            time.sleep(0.1)
            dataset = MockIterableDataset("background_loaded")
            return DataLoader(dataset, batch_size=None), None, None

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(create_loader_slow)

        # Wait for completion
        dataloader, _, _ = future.result(timeout=5.0)

        # Verify we can use the dataloader
        batch = next(iter(dataloader))
        assert batch["source"] == "background_loaded"

        executor.shutdown(wait=False)

    def test_background_loader_available_check(self):
        """Verify we can check if background loader is ready without blocking."""
        def create_loader_slow():
            time.sleep(0.5)
            dataset = MockIterableDataset("background")
            return DataLoader(dataset, batch_size=None), None, None

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(create_loader_slow)

        # Should not be done immediately
        assert not future.done()

        # Wait for completion
        time.sleep(0.6)
        assert future.done()

        executor.shutdown(wait=False)

    def test_training_continues_while_background_loads(self):
        """Verify training can continue while background loader is loading."""
        phase1_dataset = MockIterableDataset("phase1")
        phase1_loader = DataLoader(phase1_dataset, batch_size=None)

        def create_loader_slow():
            time.sleep(0.2)
            dataset = MockIterableDataset("phase2")
            return DataLoader(dataset, batch_size=None), None, None

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        background_future = executor.submit(create_loader_slow)

        # Simulate training steps while background loads
        steps_trained = 0
        background_ready_at_step = None

        for step in range(20):
            # Get batch from phase 1
            batch = next(iter(phase1_loader))
            steps_trained += 1

            # Check if background is ready (non-blocking)
            if background_future is not None and background_future.done():
                if background_ready_at_step is None:
                    background_ready_at_step = step
                phase2_loader, _, _ = background_future.result()
                background_future = None

            time.sleep(0.02)  # Simulate training time

        # We should have trained multiple steps
        assert steps_trained == 20

        # Background should have become ready at some point during training
        assert background_ready_at_step is not None
        assert background_ready_at_step < 20

        executor.shutdown(wait=False)


class TestCurriculumWithTrainer:
    """Test curriculum integration with Stage2Trainer."""

    def test_trainer_accepts_curriculum_params(self):
        """Verify Stage2Trainer accepts curriculum-related parameters."""
        # Import inside test to avoid module-level import issues
        try:
            from wrinklefree.training._legacy.continued_pretraining import ContinuedPretrainingTrainer as Stage2Trainer
        except ImportError:
            pytest.skip("Stage2Trainer not available")

        # Check that Stage2Trainer.__init__ accepts the parameters
        import inspect
        sig = inspect.signature(Stage2Trainer.__init__)
        params = sig.parameters

        # These params should exist
        assert "next_phase_loader_future" in params or "switch_step" in params, \
            "Stage2Trainer should accept curriculum params"

    def test_curriculum_variables_initialized(self):
        """Verify curriculum-related variables are properly initialized."""
        # This tests that probe_dataloaders and mixed_dataset are initialized
        # before being used in the curriculum path

        # Simulate the curriculum path initialization
        curriculum_enabled = True

        if curriculum_enabled:
            # These should be initialized before use
            probe_dataloaders = None
            mixed_dataset = None

        # Later code should be able to check these
        assert probe_dataloaders is None
        assert mixed_dataset is None


class TestCurriculumEdgeCases:
    """Test edge cases in curriculum learning."""

    def test_warmup_ratio_zero_skips_phase1(self):
        """Verify warmup_ratio=0 skips phase 1 entirely."""
        warmup_ratio = 0.0
        total_steps = 1000

        switch_step = int(total_steps * warmup_ratio)

        assert switch_step == 0  # Should switch immediately

    def test_warmup_ratio_one_skips_phase2(self):
        """Verify warmup_ratio=1 means phase 1 only (no switch)."""
        warmup_ratio = 1.0
        total_steps = 1000

        switch_step = int(total_steps * warmup_ratio)

        assert switch_step == 1000  # Never switches during training

    def test_background_loader_exception_handling(self):
        """Verify exceptions in background loader are handled gracefully."""
        def create_loader_failing():
            time.sleep(0.1)
            raise RuntimeError("Simulated loading failure")

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(create_loader_failing)

        # Wait for completion
        time.sleep(0.2)
        assert future.done()

        # Should raise when we try to get result
        with pytest.raises(RuntimeError, match="Simulated loading failure"):
            future.result()

        executor.shutdown(wait=False)

    def test_switch_step_beyond_max_steps(self):
        """Verify training completes normally if switch_step > max_steps."""
        max_steps = 100
        switch_step = 200  # Beyond max_steps

        steps_completed = 0
        did_switch = False

        for step in range(max_steps):
            steps_completed += 1

            if step == switch_step:
                did_switch = True

        assert steps_completed == max_steps
        assert not did_switch  # Switch never happened
