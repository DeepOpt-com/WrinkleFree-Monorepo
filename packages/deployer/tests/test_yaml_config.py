"""Tests for SkyPilot YAML configuration files."""

from pathlib import Path

import pytest
import yaml


DEPLOYER_DIR = Path(__file__).parent.parent
SKYPILOT_DIR = DEPLOYER_DIR / "skypilot"

# Expected Docker image URL
EXPECTED_IMAGE = "docker:gcr.io/wrinklefree-481904/wf-train:latest"


class TestTrainYaml:
    """Tests for train.yaml configuration."""

    @pytest.fixture
    def train_config(self):
        """Load train.yaml configuration."""
        config_path = SKYPILOT_DIR / "train.yaml"
        assert config_path.exists(), f"train.yaml not found at {config_path}"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_has_docker_image(self, train_config):
        """train.yaml should specify Docker image."""
        resources = train_config.get("resources", {})
        image_id = resources.get("image_id")
        assert image_id is not None, "resources.image_id should be set"
        assert image_id == EXPECTED_IMAGE, f"Expected {EXPECTED_IMAGE}, got {image_id}"

    def test_has_nebius_cloud(self, train_config):
        """train.yaml should use Nebius cloud."""
        resources = train_config.get("resources", {})
        cloud = resources.get("cloud")
        assert cloud == "nebius", f"Expected nebius cloud, got {cloud}"

    def test_setup_activates_venv(self, train_config):
        """Setup should activate the Docker image's venv."""
        setup = train_config.get("setup", "")
        assert "/app/.venv/bin/activate" in setup, "Setup should activate /app/.venv"

    def test_setup_installs_editable_packages(self, train_config):
        """Setup should install editable packages."""
        setup = train_config.get("setup", "")
        assert "pip install -e" in setup, "Setup should install editable packages"
        assert "--no-deps" in setup, "Should use --no-deps to skip deps (already in image)"

    def test_run_uses_python_not_uv(self, train_config):
        """Run should use python directly, not uv run."""
        run = train_config.get("run", "")
        assert "python scripts/train.py" in run, "Run should use python directly"
        assert "uv run python" not in run, "Run should not use uv run (deps in image)"

    def test_run_activates_venv(self, train_config):
        """Run should activate the Docker image's venv."""
        run = train_config.get("run", "")
        assert "/app/.venv/bin/activate" in run, "Run should activate /app/.venv"


class TestDlmTrainYaml:
    """Tests for dlm_train.yaml configuration."""

    @pytest.fixture
    def dlm_config(self):
        """Load dlm_train.yaml configuration."""
        config_path = SKYPILOT_DIR / "dlm_train.yaml"
        assert config_path.exists(), f"dlm_train.yaml not found at {config_path}"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_has_docker_image(self, dlm_config):
        """dlm_train.yaml should specify Docker image."""
        resources = dlm_config.get("resources", {})
        image_id = resources.get("image_id")
        assert image_id is not None, "resources.image_id should be set"
        assert image_id == EXPECTED_IMAGE, f"Expected {EXPECTED_IMAGE}, got {image_id}"

    def test_setup_activates_venv(self, dlm_config):
        """Setup should activate the Docker image's venv."""
        setup = dlm_config.get("setup", "")
        assert "/app/.venv/bin/activate" in setup, "Setup should activate /app/.venv"


class TestSmokeTestYaml:
    """Tests for smoke_test.yaml configuration."""

    @pytest.fixture
    def smoke_config(self):
        """Load smoke_test.yaml configuration."""
        config_path = SKYPILOT_DIR / "smoke_test.yaml"
        assert config_path.exists(), f"smoke_test.yaml not found at {config_path}"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_has_docker_image(self, smoke_config):
        """smoke_test.yaml should specify Docker image."""
        resources = smoke_config.get("resources", {})
        image_id = resources.get("image_id")
        assert image_id is not None, "resources.image_id should be set"
        assert image_id == EXPECTED_IMAGE, f"Expected {EXPECTED_IMAGE}, got {image_id}"

    def test_uses_runpod(self, smoke_config):
        """smoke_test.yaml should use RunPod cloud."""
        resources = smoke_config.get("resources", {})
        cloud = resources.get("cloud")
        assert cloud == "runpod", f"Expected runpod cloud, got {cloud}"


class TestSmokeTestNebiusYaml:
    """Tests for smoke_test_nebius.yaml configuration."""

    @pytest.fixture
    def smoke_nebius_config(self):
        """Load smoke_test_nebius.yaml configuration."""
        config_path = SKYPILOT_DIR / "smoke_test_nebius.yaml"
        assert config_path.exists(), f"smoke_test_nebius.yaml not found at {config_path}"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_has_docker_image(self, smoke_nebius_config):
        """smoke_test_nebius.yaml should specify Docker image."""
        resources = smoke_nebius_config.get("resources", {})
        image_id = resources.get("image_id")
        assert image_id is not None, "resources.image_id should be set"
        assert image_id == EXPECTED_IMAGE, f"Expected {EXPECTED_IMAGE}, got {image_id}"

    def test_uses_nebius(self, smoke_nebius_config):
        """smoke_test_nebius.yaml should use Nebius cloud."""
        resources = smoke_nebius_config.get("resources", {})
        cloud = resources.get("cloud")
        assert cloud == "nebius", f"Expected nebius cloud, got {cloud}"

    def test_uses_8xh100(self, smoke_nebius_config):
        """smoke_test_nebius.yaml should use 8x H100."""
        resources = smoke_nebius_config.get("resources", {})
        accelerators = resources.get("accelerators")
        assert accelerators == "H100:8", f"Expected H100:8, got {accelerators}"


class TestAllYamlsHaveDockerImage:
    """Ensure all SkyPilot YAMLs that should have Docker images do."""

    @pytest.mark.parametrize("yaml_file", [
        "train.yaml",
        "dlm_train.yaml",
        "smoke_test.yaml",
        "smoke_test_nebius.yaml",
    ])
    def test_yaml_has_docker_image(self, yaml_file):
        """Each training YAML should have the Docker image configured."""
        config_path = SKYPILOT_DIR / yaml_file
        assert config_path.exists(), f"{yaml_file} not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        resources = config.get("resources", {})
        image_id = resources.get("image_id")
        assert image_id is not None, f"{yaml_file}: resources.image_id should be set"
        assert image_id.startswith("docker:"), f"{yaml_file}: image_id should start with 'docker:'"
        assert "wf-train" in image_id, f"{yaml_file}: image should be wf-train"
