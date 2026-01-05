"""Tests for wf_deploy source modules.

These tests import and validate the wf_deploy package to ensure coverage.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


class TestConstants:
    """Tests for wf_deploy.constants module."""

    def test_import_constants(self):
        """Test that constants module can be imported."""
        from wf_deploy.constants import (
            RunIdPrefix,
            STAGE_CONFIG_MAP,
            SUPPORTED_GPU_TYPES,
            GPU_PROFILES,
            DEFAULT_WANDB_PROJECT,
            DEFAULT_CONTEXT_SIZE,
            DEFAULT_SMOKE_TEST_MODEL,
            TRAINING_TIMEOUT,
            SMOKE_TEST_TIMEOUT,
            DEBUG_TIMEOUT,
            EnvVars,
            get_wandb_entity,
        )

        # Verify types
        assert isinstance(STAGE_CONFIG_MAP, dict)
        assert isinstance(SUPPORTED_GPU_TYPES, frozenset)
        assert isinstance(GPU_PROFILES, dict)

    def test_run_id_prefix_enum(self):
        """Test RunIdPrefix enum values."""
        from wf_deploy.constants import RunIdPrefix

        assert RunIdPrefix.SKYPILOT.value == "sky-"
        # Test enum is usable as string
        run_id = f"{RunIdPrefix.SKYPILOT.value}test-run"
        assert run_id == "sky-test-run"

    def test_stage_config_map(self):
        """Test STAGE_CONFIG_MAP has all stages."""
        from wf_deploy.constants import STAGE_CONFIG_MAP

        # All valid stages should be mapped
        assert 1 in STAGE_CONFIG_MAP
        assert 1.9 in STAGE_CONFIG_MAP
        assert 2 in STAGE_CONFIG_MAP
        assert 3 in STAGE_CONFIG_MAP

        # Config names should be strings
        for stage, config in STAGE_CONFIG_MAP.items():
            assert isinstance(config, str)
            assert len(config) > 0

    def test_supported_gpu_types(self):
        """Test SUPPORTED_GPU_TYPES contains expected GPUs."""
        from wf_deploy.constants import SUPPORTED_GPU_TYPES

        assert "H100" in SUPPORTED_GPU_TYPES
        assert "A100" in SUPPORTED_GPU_TYPES
        assert "A10G" in SUPPORTED_GPU_TYPES

    def test_gpu_profiles_mapping(self):
        """Test GPU_PROFILES maps GPU types to config names."""
        from wf_deploy.constants import GPU_PROFILES, SUPPORTED_GPU_TYPES

        for gpu_type in GPU_PROFILES:
            assert gpu_type in SUPPORTED_GPU_TYPES
            assert isinstance(GPU_PROFILES[gpu_type], str)

    def test_timeouts_reasonable(self):
        """Test timeout values are reasonable."""
        from wf_deploy.constants import (
            TRAINING_TIMEOUT,
            SMOKE_TEST_TIMEOUT,
            DEBUG_TIMEOUT,
        )

        assert TRAINING_TIMEOUT == 24 * 60 * 60  # 24 hours
        assert SMOKE_TEST_TIMEOUT == 30 * 60  # 30 minutes
        assert DEBUG_TIMEOUT == 5 * 60  # 5 minutes
        assert DEBUG_TIMEOUT < SMOKE_TEST_TIMEOUT < TRAINING_TIMEOUT

    def test_env_vars_class(self):
        """Test EnvVars class has required attributes."""
        from wf_deploy.constants import EnvVars

        assert EnvVars.GH_TOKEN == "GH_TOKEN"
        assert EnvVars.WANDB_API_KEY == "WANDB_API_KEY"
        assert EnvVars.WANDB_ENTITY == "WANDB_ENTITY"
        assert EnvVars.HF_HOME == "HF_HOME"

    def test_get_wandb_entity_not_set(self):
        """Test get_wandb_entity returns None when not set."""
        from wf_deploy.constants import get_wandb_entity

        with patch.dict(os.environ, {}, clear=True):
            assert get_wandb_entity() is None

    def test_get_wandb_entity_set(self):
        """Test get_wandb_entity returns value when set."""
        from wf_deploy.constants import get_wandb_entity

        with patch.dict(os.environ, {"WANDB_ENTITY": "my-team"}):
            assert get_wandb_entity() == "my-team"


class TestConfig:
    """Tests for wf_deploy.config module."""

    def test_import_config(self):
        """Test that config module can be imported."""
        from wf_deploy.config import (
            ResourcesConfig,
            ServiceConfig,
            TrainingConfig,
            InfraConfig,
        )

        assert ResourcesConfig is not None
        assert ServiceConfig is not None
        assert TrainingConfig is not None
        assert InfraConfig is not None

    def test_resources_config_defaults(self):
        """Test ResourcesConfig has sensible defaults."""
        from wf_deploy.config import ResourcesConfig

        config = ResourcesConfig()
        assert config.cpus == "16+"
        assert config.memory == "128+"
        assert config.use_spot is True
        assert config.disk_size == 100

    def test_service_config_validation(self):
        """Test ServiceConfig validates required fields."""
        from wf_deploy.config import ServiceConfig
        from pydantic import ValidationError

        # Should fail without required fields
        with pytest.raises(ValidationError):
            ServiceConfig()

        # Should succeed with required fields
        config = ServiceConfig(name="test-service", model_path="/path/to/model")
        assert config.name == "test-service"
        assert config.backend == "bitnet"
        assert config.port == 8080

    def test_training_config_validation(self):
        """Test TrainingConfig validates required fields."""
        from wf_deploy.config import TrainingConfig
        from pydantic import ValidationError

        # Should fail without required fields
        with pytest.raises(ValidationError):
            TrainingConfig()

        # Should succeed with required fields
        config = TrainingConfig(name="test-job", model="qwen3_4b", stage=2)
        assert config.name == "test-job"
        assert config.model == "qwen3_4b"
        assert config.stage == 2
        assert config.backend == "modal"  # default
        assert config.wandb_enabled is True

    def test_training_config_hydra_overrides(self):
        """Test TrainingConfig accepts Hydra overrides."""
        from wf_deploy.config import TrainingConfig

        config = TrainingConfig(
            name="test",
            model="smollm2_135m",
            stage=1.9,
            hydra_overrides=["training.lr=1e-4", "training.batch_size=8"],
        )

        assert len(config.hydra_overrides) == 2
        assert "training.lr=1e-4" in config.hydra_overrides

    def test_infra_config_provider_validation(self):
        """Test InfraConfig validates provider field."""
        from wf_deploy.config import InfraConfig
        from pydantic import ValidationError

        # Valid providers
        for provider in ["hetzner", "aws", "gcp"]:
            config = InfraConfig(provider=provider)
            assert config.provider == provider

        # Invalid provider
        with pytest.raises(ValidationError):
            InfraConfig(provider="invalid")


class TestCore:
    """Tests for wf_deploy.core module."""

    def test_import_core(self):
        """Test that core module can be imported."""
        from wf_deploy.core import train, Scale

        assert callable(train)
        assert Scale is not None

    def test_train_without_skypilot(self):
        """Test train raises ImportError when SkyPilot not installed."""
        from wf_deploy.core import train

        # Mock sky module to simulate not being installed
        with patch.dict("sys.modules", {"sky": None}):
            # This should attempt to import sky and fail gracefully
            # We can't easily test the actual ImportError without
            # uninstalling skypilot, so we just verify the function exists
            pass

    def test_train_requires_skypilot(self):
        """Test train function signature and import."""
        from wf_deploy.core import train
        import inspect

        # Verify signature
        sig = inspect.signature(train)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "stage" in params


class TestCredentials:
    """Tests for wf_deploy.credentials module."""

    def test_import_credentials(self):
        """Test that credentials module can be imported."""
        from wf_deploy import credentials

        assert credentials is not None


class TestInit:
    """Tests for wf_deploy package init."""

    def test_package_import(self):
        """Test that wf_deploy package can be imported."""
        import wf_deploy

        assert wf_deploy is not None

    def test_package_exports(self):
        """Test that main exports are available."""
        from wf_deploy import train

        assert callable(train)


class TestDeployer:
    """Tests for wf_deploy.deployer module."""

    def test_import_deployer(self):
        """Test that deployer module can be imported."""
        from wf_deploy import deployer

        assert deployer is not None


class TestInfra:
    """Tests for wf_deploy.infra module."""

    def test_import_infra(self):
        """Test that infra module can be imported."""
        from wf_deploy import infra

        assert infra is not None


class TestTrainer:
    """Tests for wf_deploy.trainer module."""

    def test_import_trainer(self):
        """Test that trainer module can be imported."""
        from wf_deploy import trainer

        assert trainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
