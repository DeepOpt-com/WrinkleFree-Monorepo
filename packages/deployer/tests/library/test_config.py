"""Tests for wf_deploy.config module."""

import pytest
from pydantic import ValidationError

from wf_deploy.config import (
    ResourcesConfig,
    ServiceConfig,
    TrainingConfig,
    InfraConfig,
)


class TestResourcesConfig:
    """Tests for ResourcesConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ResourcesConfig()
        assert config.cpus == "16+"
        assert config.memory == "128+"
        assert config.use_spot is True
        assert config.accelerators is None

    def test_custom_values(self):
        """Test custom values."""
        config = ResourcesConfig(
            cpus="32+",
            memory="256+",
            accelerators="A100:8",
            cloud="aws",
            use_spot=False,
        )
        assert config.cpus == "32+"
        assert config.accelerators == "A100:8"
        assert config.cloud == "aws"
        assert config.use_spot is False


class TestServiceConfig:
    """Tests for ServiceConfig."""

    def test_minimal_config(self):
        """Test minimal required config."""
        config = ServiceConfig(
            name="test-service",
            model_path="gs://bucket/model.gguf",
        )
        assert config.name == "test-service"
        assert config.backend == "bitnet"
        assert config.port == 8080

    def test_full_config(self):
        """Test full config with all options."""
        config = ServiceConfig(
            name="test-service",
            backend="vllm",
            model_path="s3://bucket/model.gguf",
            port=9000,
            context_size=8192,
            min_replicas=3,
            max_replicas=20,
            target_qps=10.0,
            resources=ResourcesConfig(
                accelerators="L4:1",
                cloud="gcp",
            ),
        )
        assert config.backend == "vllm"
        assert config.port == 9000
        assert config.context_size == 8192
        assert config.resources.accelerators == "L4:1"

    def test_missing_required_field(self):
        """Test error on missing required field."""
        with pytest.raises(ValidationError):
            ServiceConfig(name="test")  # missing model_path

    def test_invalid_backend(self):
        """Test error on invalid backend."""
        with pytest.raises(ValidationError):
            ServiceConfig(
                name="test",
                model_path="/path/to/model",
                backend="invalid",  # type: ignore
            )


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_minimal_config(self):
        """Test minimal required config."""
        config = TrainingConfig(
            name="train-job",
            model="qwen3_4b",
            stage=2,
            checkpoint_bucket="my-bucket",
        )
        assert config.model == "qwen3_4b"
        assert config.stage == 2
        assert config.checkpoint_store == "modal"  # Default changed to Modal
        assert config.accelerators == "H100:4"

    def test_full_config(self):
        """Test full config with all options."""
        config = TrainingConfig(
            name="train-job",
            model="llama3_8b",
            stage=3,
            checkpoint_bucket="my-bucket",
            checkpoint_store="gcs",
            accelerators="A100:8",
            cloud="gcp",
            use_spot=False,
            wandb_project="my-project",
        )
        assert config.checkpoint_store == "gcs"
        assert config.accelerators == "A100:8"
        assert config.use_spot is False


class TestInfraConfig:
    """Tests for InfraConfig."""

    def test_hetzner_config(self):
        """Test Hetzner config."""
        config = InfraConfig(
            provider="hetzner",
            server_count=5,
            server_type="ax102",
        )
        assert config.provider == "hetzner"
        assert config.server_count == 5

    def test_gcp_config(self):
        """Test GCP config."""
        config = InfraConfig(
            provider="gcp",
            project_id="my-project",
            region="us-west1",
        )
        assert config.provider == "gcp"
        assert config.project_id == "my-project"

    def test_aws_config(self):
        """Test AWS config."""
        config = InfraConfig(
            provider="aws",
            bucket_name="my-bucket",
        )
        assert config.provider == "aws"
        assert config.bucket_name == "my-bucket"

    def test_invalid_provider(self):
        """Test error on invalid provider."""
        with pytest.raises(ValidationError):
            InfraConfig(provider="invalid")  # type: ignore
