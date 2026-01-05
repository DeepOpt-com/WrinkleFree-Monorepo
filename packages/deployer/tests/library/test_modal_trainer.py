"""Tests for Modal training integration."""

import pytest
from unittest.mock import MagicMock, patch

from wf_deploy.config import TrainingConfig


@pytest.fixture
def modal_training_config():
    """Create test training config for Modal backend."""
    return TrainingConfig(
        name="test-modal-training",
        model="qwen3_4b",
        stage=2,
        backend="modal",  # Modal is default
        data="fineweb",
        max_steps=100,
        wandb_enabled=False,
    )


@pytest.fixture
def skypilot_training_config():
    """Create test training config for SkyPilot backend."""
    return TrainingConfig(
        name="test-skypilot-training",
        model="qwen3_4b",
        stage=2,
        backend="skypilot",
        checkpoint_bucket="test-checkpoints",
        checkpoint_store="s3",
    )


class TestTrainingConfig:
    """Tests for TrainingConfig with Modal fields."""

    def test_default_backend_is_modal(self):
        """Modal should be the default backend."""
        config = TrainingConfig(
            name="test",
            model="qwen3_4b",
            stage=2,
        )
        assert config.backend == "modal"

    def test_modal_config_fields(self, modal_training_config):
        """Test Modal-specific config fields."""
        assert modal_training_config.backend == "modal"
        assert modal_training_config.gpu == "H100"  # Default
        assert modal_training_config.data == "fineweb"
        assert modal_training_config.max_steps == 100
        assert modal_training_config.wandb_enabled is False

    def test_skypilot_config_fields(self, skypilot_training_config):
        """Test SkyPilot-specific config fields."""
        assert skypilot_training_config.backend == "skypilot"
        assert skypilot_training_config.checkpoint_bucket == "test-checkpoints"
        assert skypilot_training_config.checkpoint_store == "s3"

    def test_hydra_overrides(self):
        """Test hydra_overrides field."""
        config = TrainingConfig(
            name="test",
            model="qwen3_4b",
            stage=2,
            hydra_overrides=["training.lr=1e-4", "training.batch_size=8"],
        )
        assert len(config.hydra_overrides) == 2
        assert "training.lr=1e-4" in config.hydra_overrides


class TestTrainerBackendSelection:
    """Tests for Trainer backend selection."""

    def test_modal_backend_initialization(self, modal_training_config):
        """Test that Modal backend is initialized correctly."""
        with patch("wf_deploy.modal_deployer.ModalTrainer") as mock_modal:
            from wf_deploy.trainer import Trainer

            trainer = Trainer(modal_training_config)
            # Modal backend should be initialized
            assert trainer.config.backend == "modal"

    def test_skypilot_backend_initialization(self, skypilot_training_config, monkeypatch):
        """Test that SkyPilot backend is initialized correctly."""
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

        from wf_deploy.trainer import Trainer

        trainer = Trainer(skypilot_training_config)
        assert trainer.config.backend == "skypilot"


class TestTrainerFromJson:
    """Tests for Trainer.from_json method."""

    def test_from_json_basic(self):
        """Test creating Trainer from JSON config."""
        with patch("wf_deploy.modal_deployer.ModalTrainer"):
            from wf_deploy.trainer import Trainer

            trainer = Trainer.from_json({
                "model": "qwen3_4b",
                "stage": 2,
            })

            assert trainer.config.model == "qwen3_4b"
            assert trainer.config.stage == 2
            assert trainer.config.backend == "modal"  # Default

    def test_from_json_with_all_fields(self):
        """Test creating Trainer from JSON with all fields."""
        with patch("wf_deploy.modal_deployer.ModalTrainer"):
            from wf_deploy.trainer import Trainer

            trainer = Trainer.from_json({
                "model": "smollm2_135m",
                "stage": 1.9,
                "data": "fineweb",
                "max_steps": 1000,
                "max_tokens": 1000000,
                "wandb_enabled": False,
            })

            assert trainer.config.model == "smollm2_135m"
            assert trainer.config.stage == 1.9
            assert trainer.config.max_steps == 1000
            assert trainer.config.wandb_enabled is False

    def test_from_json_auto_generates_name(self):
        """Test that name is auto-generated if not provided."""
        with patch("wf_deploy.modal_deployer.ModalTrainer"):
            from wf_deploy.trainer import Trainer

            trainer = Trainer.from_json({
                "model": "qwen3_4b",
                "stage": 2,
            })

            assert trainer.config.name == "qwen3_4b-s2"


class TestQuickLaunch:
    """Tests for quick_launch convenience function."""

    def test_quick_launch_basic(self):
        """Test basic quick_launch call."""
        with patch("wf_deploy.trainer.Trainer") as mock_trainer_class:
            mock_instance = MagicMock()
            mock_instance.launch.return_value = "run-123"
            mock_trainer_class.return_value = mock_instance

            from wf_deploy.trainer import quick_launch

            run_id = quick_launch("qwen3_4b", stage=2)

            assert run_id == "run-123"
            mock_instance.launch.assert_called_once()

    def test_quick_launch_with_max_steps(self):
        """Test quick_launch with max_steps."""
        with patch("wf_deploy.trainer.Trainer") as mock_trainer_class:
            mock_instance = MagicMock()
            mock_instance.launch.return_value = "run-456"
            mock_trainer_class.return_value = mock_instance

            from wf_deploy.trainer import quick_launch

            run_id = quick_launch("smollm2_135m", stage=1.9, max_steps=100)

            assert run_id == "run-456"
            # Check that TrainingConfig was created with correct params
            call_args = mock_trainer_class.call_args
            config = call_args[0][0]
            assert config.model == "smollm2_135m"
            assert config.stage == 1.9
            assert config.max_steps == 100


class TestModalTrainingConfig:
    """Tests for ModalTrainingConfig dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        from wf_deploy.modal_deployer import ModalTrainingConfig

        config = ModalTrainingConfig(
            model="qwen3_4b",
            stage=2,
            max_steps=1000,
        )
        data = config.to_dict()

        assert data["model"] == "qwen3_4b"
        assert data["stage"] == 2
        assert data["max_steps"] == 1000

    def test_from_dict(self):
        """Test deserialization from dict."""
        from wf_deploy.modal_deployer import ModalTrainingConfig

        data = {
            "model": "smollm2_135m",
            "stage": 1.9,
            "wandb_enabled": False,
        }
        config = ModalTrainingConfig.from_dict(data)

        assert config.model == "smollm2_135m"
        assert config.stage == 1.9
        assert config.wandb_enabled is False


class TestCLI:
    """Tests for the wf CLI."""

    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from wf_deploy.cli import cli, train, runs, logs, smoke

        assert cli is not None
        assert train is not None
        assert runs is not None

    def test_cli_train_help(self):
        """Test that train command has proper help text."""
        from click.testing import CliRunner
        from wf_deploy.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])

        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--stage" in result.output
        assert "--cloud" in result.output  # Cloud provider option
        # max_steps is now passed via Hydra overrides, not a CLI flag
