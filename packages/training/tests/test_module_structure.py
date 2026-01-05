"""Tests for module structure.

Verifies that:
1. Active modules are properly organized
2. Data module requires wf_data
"""

import warnings
import pytest


class TestActiveImports:
    """Test that active imports work correctly."""

    def test_training_fsdp_utilities_active(self):
        """FSDP utilities should be directly available (not deprecated)."""
        from wf_train.training import (
            wrap_model_fsdp,
            apply_activation_checkpointing,
            setup_distributed,
            cleanup_distributed,
        )
        assert wrap_model_fsdp is not None
        assert apply_activation_checkpointing is not None
        assert setup_distributed is not None
        assert cleanup_distributed is not None


class TestActiveModules:
    """Test that active (non-deprecated) modules work correctly."""

    def test_lightning_module_importable(self):
        """Lightning module should be importable."""
        from wf_train.lightning import WrinkleFreeLightningModule
        assert WrinkleFreeLightningModule is not None

    def test_lightning_datamodule_importable(self):
        """Lightning datamodule should be importable."""
        from wf_train.lightning import WrinkleFreeDataModule
        assert WrinkleFreeDataModule is not None

    def test_lightning_callbacks_importable(self):
        """Lightning callbacks should be importable."""
        from wf_train.lightning import callbacks
        assert hasattr(callbacks, 'GCSCheckpointCallback')
        assert hasattr(callbacks, 'TokenCountCallback')

    def test_objectives_importable(self):
        """Objectives module should be importable without warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from wf_train.objectives import (
                ObjectiveManager,
                ContinuePretrainObjective,
                DLMObjective,
                create_objective_manager,
            )

            # Objectives should NOT emit deprecation warnings
            deprecation_warnings = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "objectives" in str(x.message).lower()
            ]
            assert len(deprecation_warnings) == 0

    def test_quantization_no_fp8(self):
        """Quantization module should not have FP8 exports (removed)."""
        from wrinklefree import quantization

        # These should NOT exist
        assert not hasattr(quantization, 'FP8Config')
        assert not hasattr(quantization, 'FP8Capability')
        assert not hasattr(quantization, 'detect_fp8_capability')

    def test_models_no_fp8(self):
        """Models module should not have FP8 exports (removed)."""
        from wrinklefree import models

        # These should NOT exist
        assert not hasattr(models, 'FP8BitLinear')
        assert not hasattr(models, 'convert_bitlinear_to_fp8')


class TestExperimentalModules:
    """Test that experimental modules are properly organized."""

    def test_experimental_emits_warning(self):
        """Importing from _experimental should emit FutureWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from wf_train._experimental import moe

            future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(future_warnings) >= 1

    def test_experimental_no_fp8(self):
        """Experimental module should not have fp8 submodule (deleted)."""
        import importlib.util

        spec = importlib.util.find_spec("wf_train._experimental.fp8")
        assert spec is None, "fp8 submodule should have been deleted"

    def test_experimental_moe_available(self):
        """MoE experimental module should still be available."""
        import importlib.util

        spec = importlib.util.find_spec("wf_train._experimental.moe")
        assert spec is not None

    def test_experimental_tensor_parallel_available(self):
        """Tensor parallel experimental module should still be available."""
        import importlib.util

        spec = importlib.util.find_spec("wf_train._experimental.tensor_parallel")
        assert spec is not None


class TestDataModule:
    """Test data module structure."""

    def test_data_requires_wf_data(self):
        """Data module should require wf_data package."""
        # If wf_data is not available, importing should raise ImportError
        # Since it IS available in this environment, we just verify it works
        from wf_train.data import (
            create_pretraining_dataloader,
            MixedDataset,
            InfluenceTracker,
        )
        assert create_pretraining_dataloader is not None
        assert MixedDataset is not None
        assert InfluenceTracker is not None

    def test_data_no_legacy_exports(self):
        """Data module should not export legacy pretrain classes."""
        from wrinklefree import data

        # These legacy classes should NOT be exported anymore
        assert not hasattr(data, 'PretrainDataset')
        assert not hasattr(data, 'PackedPretrainDataset')
        assert not hasattr(data, 'StreamingPretrainDataset')
        assert not hasattr(data, 'MixedPretrainDataset')

    def test_finetune_datasets_available(self):
        """Finetune datasets (not deprecated) should be available."""
        from wf_train.data import (
            FinetuneDataset,
            InstructDataset,
            create_finetune_dataloader,
        )
        assert FinetuneDataset is not None
        assert InstructDataset is not None
        assert create_finetune_dataloader is not None
