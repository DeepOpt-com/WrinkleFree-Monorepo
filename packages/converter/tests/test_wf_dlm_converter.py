"""Tests for wf_dlm_converter source modules.

These tests import and validate the wf_dlm_converter package to ensure coverage.
"""

import pytest
from unittest.mock import MagicMock, patch
import inspect


class TestConstants:
    """Tests for wf_dlm_converter.constants module."""

    def test_import_constants(self):
        """Test that constants module can be imported."""
        from wf_dlm_converter.constants import (
            MODAL_APP_NAME,
            MODAL_VOLUME_DLM_OUTPUTS,
            MODAL_VOLUME_CHECKPOINTS,
            MODAL_VOLUME_HF_CACHE,
            GCS_BUCKET,
            GCS_DLM_PREFIX,
            DEFAULT_MODEL,
            DEFAULT_BLOCK_SIZE,
            DEFAULT_DIFFUSION_STEPS,
            DEFAULT_TOTAL_TOKENS,
            DEFAULT_WANDB_PROJECT,
            DEFAULT_LEARNING_RATE,
            DEFAULT_BATCH_SIZE,
            DEFAULT_MAX_SEQ_LENGTH,
            CONVERSION_TIMEOUT,
            VALIDATION_TIMEOUT,
            RunIdPrefix,
            SUPPORTED_GPU_TYPES,
            DEFAULT_GPU_TYPE,
            MaskToken,
        )

        # Verify types
        assert isinstance(MODAL_APP_NAME, str)
        assert isinstance(DEFAULT_BLOCK_SIZE, int)
        assert isinstance(DEFAULT_LEARNING_RATE, float)
        assert isinstance(SUPPORTED_GPU_TYPES, frozenset)

    def test_run_id_prefix_enum(self):
        """Test RunIdPrefix enum values."""
        from wf_dlm_converter.constants import RunIdPrefix

        assert RunIdPrefix.CONVERT.value == "dlm-convert-"
        assert RunIdPrefix.VALIDATE.value == "dlm-validate-"

        # Test enum is usable as string
        run_id = f"{RunIdPrefix.CONVERT.value}test-run"
        assert run_id == "dlm-convert-test-run"

    def test_mask_token_class(self):
        """Test MaskToken class constants."""
        from wf_dlm_converter.constants import MaskToken

        assert MaskToken.ID == 32000
        assert MaskToken.TEXT == "[MASK]"

    def test_default_values_reasonable(self):
        """Test default values are reasonable."""
        from wf_dlm_converter.constants import (
            DEFAULT_BLOCK_SIZE,
            DEFAULT_DIFFUSION_STEPS,
            DEFAULT_TOTAL_TOKENS,
            DEFAULT_LEARNING_RATE,
            DEFAULT_BATCH_SIZE,
            DEFAULT_MAX_SEQ_LENGTH,
        )

        assert DEFAULT_BLOCK_SIZE > 0
        assert DEFAULT_DIFFUSION_STEPS > 0
        assert DEFAULT_TOTAL_TOKENS > 0
        assert 0 < DEFAULT_LEARNING_RATE < 1
        assert DEFAULT_BATCH_SIZE > 0
        assert DEFAULT_MAX_SEQ_LENGTH > 0

    def test_timeouts_reasonable(self):
        """Test timeout values are reasonable."""
        from wf_dlm_converter.constants import (
            CONVERSION_TIMEOUT,
            VALIDATION_TIMEOUT,
        )

        assert CONVERSION_TIMEOUT == 24 * 60 * 60  # 24 hours
        assert VALIDATION_TIMEOUT == 30 * 60  # 30 minutes
        assert VALIDATION_TIMEOUT < CONVERSION_TIMEOUT

    def test_supported_gpu_types(self):
        """Test SUPPORTED_GPU_TYPES contains expected GPUs."""
        from wf_dlm_converter.constants import SUPPORTED_GPU_TYPES, DEFAULT_GPU_TYPE

        assert "H100" in SUPPORTED_GPU_TYPES
        assert "A100" in SUPPORTED_GPU_TYPES
        assert "A10G" in SUPPORTED_GPU_TYPES
        assert DEFAULT_GPU_TYPE in SUPPORTED_GPU_TYPES

    def test_gcs_config(self):
        """Test GCS configuration is set."""
        from wf_dlm_converter.constants import GCS_BUCKET, GCS_DLM_PREFIX

        assert isinstance(GCS_BUCKET, str)
        assert len(GCS_BUCKET) > 0
        assert isinstance(GCS_DLM_PREFIX, str)


class TestCore:
    """Tests for wf_dlm_converter.core module."""

    def test_import_core(self):
        """Test that core module can be imported."""
        from wf_dlm_converter.core import convert, validate, logs, cancel

        assert callable(convert)
        assert callable(validate)
        assert callable(logs)
        assert callable(cancel)

    def test_convert_signature(self):
        """Test convert function has expected signature."""
        from wf_dlm_converter.core import convert

        sig = inspect.signature(convert)
        params = list(sig.parameters.keys())

        # Required params
        assert "model" in params
        assert "checkpoint_path" in params

        # Optional params with defaults
        assert "output_path" in params
        assert "total_tokens" in params
        assert "block_size" in params
        assert "backend" in params
        assert "gpu" in params

    def test_validate_signature(self):
        """Test validate function has expected signature."""
        from wf_dlm_converter.core import validate

        sig = inspect.signature(validate)
        params = list(sig.parameters.keys())

        assert "model_path" in params
        assert "test_prompt" in params
        assert "block_size" in params
        assert "diffusion_steps" in params

    def test_logs_signature(self):
        """Test logs function has expected signature."""
        from wf_dlm_converter.core import logs

        sig = inspect.signature(logs)
        params = list(sig.parameters.keys())

        assert "run_id" in params
        assert "follow" in params

    def test_cancel_signature(self):
        """Test cancel function has expected signature."""
        from wf_dlm_converter.core import cancel

        sig = inspect.signature(cancel)
        params = list(sig.parameters.keys())

        assert "run_id" in params

    def test_cancel_unknown_run_id(self):
        """Test cancel returns error for unknown run ID format."""
        from wf_dlm_converter.core import cancel

        result = cancel("unknown-format-xyz")
        assert result["success"] is False
        assert "Unknown run ID" in result["error"]


class TestCLI:
    """Tests for wf_dlm_converter.cli module."""

    def test_import_cli(self):
        """Test that cli module can be imported."""
        from wf_dlm_converter import cli

        assert cli is not None


class TestInit:
    """Tests for wf_dlm_converter package init."""

    def test_package_import(self):
        """Test that wf_dlm_converter package can be imported."""
        import wf_dlm_converter

        assert wf_dlm_converter is not None

    def test_package_exports(self):
        """Test that main exports are available."""
        from wf_dlm_converter import convert, validate

        assert callable(convert)
        assert callable(validate)


class TestModels:
    """Tests for wf_dlm_converter.models module."""

    def test_import_models(self):
        """Test that models module can be imported."""
        from wf_dlm_converter import models

        assert models is not None


class TestConversion:
    """Tests for wf_dlm_converter.conversion module."""

    def test_import_conversion(self):
        """Test that conversion module can be imported."""
        from wf_dlm_converter import conversion

        assert conversion is not None


class TestModal:
    """Tests for wf_dlm_converter.modal module."""

    def test_import_modal_module(self):
        """Test that modal module can be imported."""
        from wf_dlm_converter import modal

        assert modal is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
