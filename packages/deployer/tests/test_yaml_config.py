"""Tests for SkyPilot YAML configuration files."""

from pathlib import Path

import pytest
import yaml


DEPLOYER_DIR = Path(__file__).parent.parent
SKYPILOT_DIR = DEPLOYER_DIR / "skypilot"


class TestTrainYaml:
    """Tests for train.yaml configuration."""

    @pytest.fixture
    def train_config(self):
        """Load train.yaml configuration."""
        config_path = SKYPILOT_DIR / "train.yaml"
        assert config_path.exists(), f"train.yaml not found at {config_path}"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_has_nebius_cloud(self, train_config):
        """train.yaml should use Nebius cloud."""
        resources = train_config.get("resources", {})
        cloud = resources.get("cloud")
        assert cloud == "nebius", f"Expected nebius cloud, got {cloud}"

    def test_has_required_envs(self, train_config):
        """train.yaml should have required environment variables."""
        envs = train_config.get("envs", {})
        required = ["MODEL", "TRAINING_CONFIG", "GCS_BUCKET", "WANDB_PROJECT"]
        for key in required:
            assert key in envs, f"Missing required env: {key}"

    def test_setup_installs_uv(self, train_config):
        """Setup should install uv if not present."""
        setup = train_config.get("setup", "")
        assert "uv" in setup, "Setup should reference uv"

    def test_run_uses_dispatch_train(self, train_config):
        """Run should use dispatch_train.py."""
        run = train_config.get("run", "")
        assert "dispatch_train.py" in run, "Run should use dispatch_train.py"


class TestSmokeTestYaml:
    """Tests for smoke_test.yaml configuration."""

    @pytest.fixture
    def smoke_config(self):
        """Load smoke_test.yaml configuration."""
        config_path = SKYPILOT_DIR / "smoke_test.yaml"
        assert config_path.exists(), f"smoke_test.yaml not found at {config_path}"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_uses_runpod(self, smoke_config):
        """smoke_test.yaml should use RunPod cloud."""
        resources = smoke_config.get("resources", {})
        cloud = resources.get("cloud")
        assert cloud == "runpod", f"Expected runpod cloud, got {cloud}"

    def test_has_objective_env(self, smoke_config):
        """smoke_test.yaml should have OBJECTIVE env var."""
        envs = smoke_config.get("envs", {})
        assert "OBJECTIVE" in envs, "Missing OBJECTIVE env var"

    def test_run_uses_dispatch_smoke(self, smoke_config):
        """Run should use dispatch_smoke.py."""
        run = smoke_config.get("run", "")
        assert "dispatch_smoke.py" in run, "Run should use dispatch_smoke.py"


class TestAllYamlsHaveBasicStructure:
    """Ensure all SkyPilot YAMLs have required structure."""

    @pytest.mark.parametrize("yaml_file", [
        "train.yaml",
        "smoke_test.yaml",
        "eval.yaml",
    ])
    def test_yaml_has_required_sections(self, yaml_file):
        """Each training YAML should have name, resources, and run sections."""
        config_path = SKYPILOT_DIR / yaml_file
        if not config_path.exists():
            pytest.skip(f"{yaml_file} not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "name" in config, f"{yaml_file}: missing 'name'"
        assert "resources" in config, f"{yaml_file}: missing 'resources'"
        assert "run" in config, f"{yaml_file}: missing 'run'"

    def test_service_yaml_has_service_block(self):
        """service.yaml should have a service block (SkyServe format)."""
        config_path = SKYPILOT_DIR / "service.yaml"
        if not config_path.exists():
            pytest.skip("service.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "service" in config, "service.yaml: missing 'service'"
        assert "resources" in config, "service.yaml: missing 'resources'"
