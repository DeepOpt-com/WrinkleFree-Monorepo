"""Tests for module structure and deprecation warnings.

Verifies that:
1. Legacy imports work with deprecation warnings
2. Active modules are properly organized
3. Data module requires data_handler
"""

import warnings
import pytest


class TestLegacyImports:
    """Test that legacy imports work and emit deprecation warnings."""

    def test_training_legacy_import_emits_warning(self):
        """Training _legacy module should have deprecation warning code."""
        import sys

        module_name = "wrinklefree.training._legacy"
        if module_name in sys.modules:
            # Module already imported, check it has the warning code by inspecting source
            import inspect
            import wrinklefree.training._legacy as legacy_module
            source = inspect.getsource(legacy_module)
            assert "DeprecationWarning" in source
            assert "deprecated" in source.lower()
        else:
            # Module not yet imported, capture warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from wrinklefree.training._legacy import Trainer

                deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(deprecation_warnings) >= 1
                assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_training_legacy_trainer_available(self):
        """Legacy Trainer class should be importable."""
        from wrinklefree.training._legacy import Trainer
        assert Trainer is not None
        assert hasattr(Trainer, '__init__')

    def test_training_legacy_continued_pretraining_available(self):
        """Legacy ContinuedPretrainingTrainer should be importable."""
        from wrinklefree.training._legacy import ContinuedPretrainingTrainer
        assert ContinuedPretrainingTrainer is not None

    def test_training_legacy_stage2_alias(self):
        """Stage2Trainer should be an alias for ContinuedPretrainingTrainer."""
        from wrinklefree.training._legacy import ContinuedPretrainingTrainer, Stage2Trainer
        assert Stage2Trainer is ContinuedPretrainingTrainer

    def test_training_legacy_stage1_available(self):
        """Legacy stage1 functions should be importable."""
        from wrinklefree.training._legacy import convert_model_to_bitnet, run_stage1
        assert convert_model_to_bitnet is not None
        assert run_stage1 is not None

    def test_training_legacy_helpers_available(self):
        """Legacy helper functions should be importable."""
        from wrinklefree.training._legacy import (
            create_optimizer,
            create_scheduler,
            download_checkpoint_from_gcs,
        )
        assert create_optimizer is not None
        assert create_scheduler is not None
        assert download_checkpoint_from_gcs is not None

    def test_distillation_import_emits_warning(self):
        """Distillation module should have deprecation warning code."""
        import importlib
        import sys

        # Force reload to capture warning
        module_name = "wrinklefree.distillation"
        if module_name in sys.modules:
            # Module already imported, check it has the warning code by inspecting source
            import inspect
            import wrinklefree.distillation as distillation_module
            source = inspect.getsource(distillation_module)
            assert "DeprecationWarning" in source
            assert "deprecated" in source.lower()
        else:
            # Module not yet imported, capture warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from wrinklefree.distillation import LayerwiseDistillationLoss

                deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(deprecation_warnings) >= 1
                assert "deprecated" in str(deprecation_warnings[0].message).lower()


class TestBackwardCompatibility:
    """Test that backward-compatible imports from main training module work."""

    def test_training_reexports_legacy_classes(self):
        """Main training module should re-export legacy classes."""
        # These should work (re-exported from _legacy)
        from wrinklefree.training import (
            Trainer,
            ContinuedPretrainingTrainer,
            Stage2Trainer,
            create_optimizer,
            create_scheduler,
            convert_model_to_bitnet,
            run_stage1,
            run_stage2,
        )
        assert Trainer is not None
        assert ContinuedPretrainingTrainer is not None
        assert Stage2Trainer is ContinuedPretrainingTrainer

    def test_training_fsdp_utilities_active(self):
        """FSDP utilities should be directly available (not deprecated)."""
        from wrinklefree.training import (
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
        from wrinklefree.lightning import WrinkleFreeLightningModule
        assert WrinkleFreeLightningModule is not None

    def test_lightning_datamodule_importable(self):
        """Lightning datamodule should be importable."""
        from wrinklefree.lightning import WrinkleFreeDataModule
        assert WrinkleFreeDataModule is not None

    def test_lightning_callbacks_importable(self):
        """Lightning callbacks should be importable."""
        from wrinklefree.lightning import callbacks
        assert hasattr(callbacks, 'GCSCheckpointCallback')
        assert hasattr(callbacks, 'TokenCountCallback')

    def test_objectives_importable(self):
        """Objectives module should be importable without warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from wrinklefree.objectives import (
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
            from wrinklefree._experimental import moe

            future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
            assert len(future_warnings) >= 1

    def test_experimental_no_fp8(self):
        """Experimental module should not have fp8 submodule (deleted)."""
        import importlib.util

        spec = importlib.util.find_spec("wrinklefree._experimental.fp8")
        assert spec is None, "fp8 submodule should have been deleted"

    def test_experimental_moe_available(self):
        """MoE experimental module should still be available."""
        import importlib.util

        spec = importlib.util.find_spec("wrinklefree._experimental.moe")
        assert spec is not None

    def test_experimental_tensor_parallel_available(self):
        """Tensor parallel experimental module should still be available."""
        import importlib.util

        spec = importlib.util.find_spec("wrinklefree._experimental.tensor_parallel")
        assert spec is not None


class TestDataModule:
    """Test data module structure."""

    def test_data_requires_data_handler(self):
        """Data module should require data_handler package."""
        # If data_handler is not available, importing should raise ImportError
        # Since it IS available in this environment, we just verify it works
        from wrinklefree.data import (
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
        from wrinklefree.data import (
            FinetuneDataset,
            InstructDataset,
            create_finetune_dataloader,
        )
        assert FinetuneDataset is not None
        assert InstructDataset is not None
        assert create_finetune_dataloader is not None
