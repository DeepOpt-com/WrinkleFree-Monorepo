"""Tests for wf_deployer.deployer module."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from wf_deployer.config import ServiceConfig, ResourcesConfig
from wf_deployer.credentials import Credentials


@pytest.fixture
def mock_sky():
    """Create and inject mock sky module."""
    mock = MagicMock()
    mock.serve.up.return_value = "request-123"
    mock.serve.down.return_value = "request-456"
    mock.serve.status.return_value = "request-789"
    mock.serve.logs.return_value = "request-logs"
    mock.serve.update.return_value = "request-update"
    mock.get.return_value = ("test-service", "https://endpoint.example.com")
    mock.stream_and_get.return_value = None
    mock.Resources = MagicMock()
    mock.Task = MagicMock()
    mock.clouds.CLOUD_REGISTRY.from_str = MagicMock(return_value="aws")
    return mock


@pytest.fixture
def mock_credentials():
    """Create mock credentials."""
    return Credentials(
        aws_access_key_id="AKIATEST",
        aws_secret_access_key="secret",
        gcp_project_id="test-project",
    )


@pytest.fixture
def service_config():
    """Create test service config."""
    return ServiceConfig(
        name="test-service",
        backend="bitnet",
        model_path="gs://bucket/model.gguf",
        port=8080,
        context_size=4096,
        min_replicas=1,
        max_replicas=5,
        resources=ResourcesConfig(
            cpus="16+",
            memory="128+",
            use_spot=True,
        ),
    )


class TestDeployer:
    """Tests for Deployer class."""

    def test_init(self, service_config, mock_credentials, monkeypatch):
        """Test Deployer initialization."""
        from wf_deployer.deployer import Deployer

        # Clear env vars first
        for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
            monkeypatch.delenv(key, raising=False)

        deployer = Deployer(service_config, mock_credentials)

        assert deployer.config == service_config
        assert deployer.credentials == mock_credentials
        # Credentials should be applied to environment
        assert os.environ.get("AWS_ACCESS_KEY_ID") == "AKIATEST"

    def test_init_without_credentials(self, service_config, monkeypatch):
        """Test Deployer initialization without explicit credentials."""
        from wf_deployer.deployer import Deployer

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "FROM_ENV")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret_env")

        deployer = Deployer(service_config)

        assert deployer.credentials.aws_access_key_id == "FROM_ENV"

    def test_get_envs(self, service_config, mock_credentials):
        """Test environment variables generation."""
        from wf_deployer.deployer import Deployer

        deployer = Deployer(service_config, mock_credentials)
        envs = deployer._get_envs()

        assert envs["BACKEND"] == "bitnet"
        assert envs["MODEL_PATH"] == "gs://bucket/model.gguf"
        assert envs["CONTEXT_SIZE"] == "4096"
        assert envs["PORT"] == "8080"
        assert envs["HOST"] == "0.0.0.0"

    def test_up_detached(self, service_config, mock_credentials, mock_sky):
        """Test deploying service in detached mode."""
        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.deployer import Deployer

            deployer = Deployer(service_config, mock_credentials)
            result = deployer.up(detach=True)

            assert "test-service" in result
            assert "request-123" in result

    def test_up_blocking(self, service_config, mock_credentials, mock_sky):
        """Test deploying service in blocking mode."""
        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.deployer import Deployer

            deployer = Deployer(service_config, mock_credentials)
            result = deployer.up(detach=False)

            assert result == "https://endpoint.example.com"

    def test_down(self, service_config, mock_credentials, mock_sky):
        """Test tearing down service."""
        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.deployer import Deployer

            deployer = Deployer(service_config, mock_credentials)
            deployer.down()

            mock_sky.serve.down.assert_called_once_with("test-service")

    def test_status(self, service_config, mock_credentials, mock_sky):
        """Test getting service status."""
        mock_sky.get.return_value = {"status": "RUNNING", "replicas": 3}

        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.deployer import Deployer

            deployer = Deployer(service_config, mock_credentials)
            status = deployer.status()

            assert status["status"] == "RUNNING"

    def test_logs(self, service_config, mock_credentials, mock_sky):
        """Test getting service logs."""
        mock_sky.get.return_value = "Log output here..."

        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.deployer import Deployer

            deployer = Deployer(service_config, mock_credentials)
            logs = deployer.logs(follow=False)

            assert logs == "Log output here..."

    def test_logs_follow(self, service_config, mock_credentials, mock_sky):
        """Test streaming service logs."""
        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.deployer import Deployer

            deployer = Deployer(service_config, mock_credentials)
            result = deployer.logs(follow=True)

            assert result == ""
            mock_sky.stream_and_get.assert_called_once()

    def test_update(self, service_config, mock_credentials, mock_sky):
        """Test updating service configuration."""
        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.deployer import Deployer

            deployer = Deployer(service_config, mock_credentials)
            deployer.update(min_replicas=3, max_replicas=10)

            mock_sky.serve.update.assert_called_once_with(
                "test-service", min_replicas=3, max_replicas=10
            )
