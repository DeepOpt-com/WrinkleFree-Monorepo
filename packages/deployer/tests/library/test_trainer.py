"""Tests for wf_deployer.trainer module (SkyPilot backend)."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from wf_deployer.config import TrainingConfig
from wf_deployer.credentials import Credentials


@pytest.fixture
def mock_sky():
    """Create and inject mock sky module."""
    mock = MagicMock()
    mock.jobs.launch.return_value = "request-123"
    mock.jobs.queue_v2.return_value = "request-queue"
    mock.jobs.logs.return_value = "request-logs"
    mock.jobs.cancel.return_value = "request-cancel"
    mock.get.return_value = (42, MagicMock())
    mock.stream_and_get.return_value = None
    mock.Resources = MagicMock()
    mock.Task = MagicMock()
    mock.Storage = MagicMock()
    mock.clouds.CLOUD_REGISTRY.from_str = MagicMock(return_value="runpod")
    return mock


@pytest.fixture
def mock_credentials():
    """Create mock credentials."""
    return Credentials(
        aws_access_key_id="AKIATEST",
        aws_secret_access_key="secret",
        runpod_api_key="runpod-key",
        checkpoint_bucket="test-checkpoints",
    )


@pytest.fixture
def skypilot_training_config():
    """Create test training config for SkyPilot backend."""
    return TrainingConfig(
        name="test-training",
        model="qwen3_4b",
        stage=2,  # Integer stage
        backend="skypilot",  # Explicitly use SkyPilot
        checkpoint_bucket="test-checkpoints",
        checkpoint_store="s3",
        accelerators="H100:4",
        cloud="runpod",
        use_spot=True,
        wandb_project="test-project",
    )


class TestTrainerSkyPilot:
    """Tests for Trainer class with SkyPilot backend."""

    def test_init(self, skypilot_training_config, mock_credentials, monkeypatch):
        """Test Trainer initialization with SkyPilot."""
        from wf_deployer.trainer import Trainer

        for key in ["AWS_ACCESS_KEY_ID", "RUNPOD_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        trainer = Trainer(skypilot_training_config, mock_credentials)

        assert trainer.config == skypilot_training_config
        assert trainer.credentials == mock_credentials
        assert trainer.config.backend == "skypilot"
        assert os.environ.get("RUNPOD_API_KEY") == "runpod-key"

    def test_init_without_credentials(self, skypilot_training_config, monkeypatch):
        """Test Trainer initialization without explicit credentials."""
        from wf_deployer.trainer import Trainer

        monkeypatch.setenv("RUNPOD_API_KEY", "from-env")

        trainer = Trainer(skypilot_training_config)

        assert trainer.credentials.runpod_api_key == "from-env"

    def test_get_envs(self, skypilot_training_config, mock_credentials):
        """Test environment variables generation."""
        from wf_deployer.trainer import Trainer

        trainer = Trainer(skypilot_training_config, mock_credentials)
        envs = trainer._get_envs()

        assert envs["MODEL"] == "qwen3_4b"
        assert envs["STAGE"] == "2.0"  # Stage is float now
        assert envs["WANDB_PROJECT"] == "test-project"
        assert envs["CHECKPOINT_BUCKET"] == "test-checkpoints"
        assert envs["CHECKPOINT_STORE"] == "s3"

    def test_get_job_name(self, skypilot_training_config, mock_credentials):
        """Test job name generation."""
        from wf_deployer.trainer import Trainer

        trainer = Trainer(skypilot_training_config, mock_credentials)
        name = trainer._get_job_name()

        assert name == "wrinklefree-train-qwen3_4b-stage2.0"

    def test_launch_detached(self, skypilot_training_config, mock_credentials, mock_sky):
        """Test launching training job in detached mode."""
        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.trainer import Trainer

            trainer = Trainer(skypilot_training_config, mock_credentials)
            result = trainer.launch(detach=True)

            assert result == "request-123"

    def test_launch_blocking(self, skypilot_training_config, mock_credentials, mock_sky):
        """Test launching training job in blocking mode."""
        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.trainer import Trainer

            trainer = Trainer(skypilot_training_config, mock_credentials)
            result = trainer.launch(detach=False)

            assert result == "42"  # String now

    def test_status(self, skypilot_training_config, mock_credentials, mock_sky):
        """Test getting job status."""
        mock_sky.get.return_value = [
            {"name": "wrinklefree-train-qwen3_4b-stage2.0", "status": "RUNNING"},
            {"name": "other-job", "status": "PENDING"},
        ]

        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.trainer import Trainer

            trainer = Trainer(skypilot_training_config, mock_credentials)
            trainer._current_run_id = "some-id"  # Set a run ID
            status = trainer._status_skypilot()

            assert status["name"] == "wrinklefree-train-qwen3_4b-stage2.0"
            assert status["status"] == "RUNNING"

    def test_status_not_found(self, skypilot_training_config, mock_credentials, mock_sky):
        """Test getting status when job not found."""
        mock_sky.get.return_value = [{"name": "other-job", "status": "RUNNING"}]

        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.trainer import Trainer

            trainer = Trainer(skypilot_training_config, mock_credentials)
            trainer._current_run_id = "some-id"
            status = trainer._status_skypilot()

            assert status["status"] == "not_found"

    def test_logs(self, skypilot_training_config, mock_credentials, mock_sky):
        """Test getting job logs."""
        mock_sky.get.return_value = "Training epoch 1/10..."

        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.trainer import Trainer

            trainer = Trainer(skypilot_training_config, mock_credentials)
            trainer._current_run_id = "some-id"
            logs = trainer._logs_skypilot(follow=False)

            assert logs == "Training epoch 1/10..."

    def test_logs_follow(self, skypilot_training_config, mock_credentials, mock_sky):
        """Test streaming job logs."""
        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.trainer import Trainer

            trainer = Trainer(skypilot_training_config, mock_credentials)
            trainer._current_run_id = "some-id"
            result = trainer._logs_skypilot(follow=True)

            assert result == ""
            mock_sky.stream_and_get.assert_called_once()

    def test_cancel(self, skypilot_training_config, mock_credentials, mock_sky):
        """Test canceling training job."""
        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.trainer import Trainer

            trainer = Trainer(skypilot_training_config, mock_credentials)
            trainer._current_run_id = "some-id"
            trainer._cancel_skypilot()

            mock_sky.jobs.cancel.assert_called_once_with(
                name="wrinklefree-train-qwen3_4b-stage2.0"
            )

    def test_list_jobs(self, skypilot_training_config, mock_credentials, mock_sky):
        """Test listing all jobs."""
        mock_sky.get.return_value = [
            {"name": "job1", "status": "RUNNING"},
            {"name": "job2", "status": "COMPLETED"},
        ]

        with patch.dict(sys.modules, {"sky": mock_sky}):
            from wf_deployer.trainer import Trainer

            trainer = Trainer(skypilot_training_config, mock_credentials)
            jobs = trainer._list_skypilot()

            assert len(jobs) == 2
            assert jobs[0]["name"] == "job1"
