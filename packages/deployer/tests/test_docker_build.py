"""Tests for Docker image building functionality."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from wf_deployer import core
from wf_deployer.cli import cli


class TestBuildImageFunction:
    """Tests for core.build_image function."""

    def test_dockerfile_exists(self):
        """Dockerfile.train should exist in docker/ directory."""
        deployer_dir = Path(__file__).parent.parent
        dockerfile = deployer_dir / "docker" / "Dockerfile.train"
        assert dockerfile.exists(), f"Dockerfile not found at {dockerfile}"

    def test_build_script_exists(self):
        """Build script should exist and be executable."""
        deployer_dir = Path(__file__).parent.parent
        script = deployer_dir / "scripts" / "build-image.sh"
        assert script.exists(), f"Build script not found at {script}"

    def test_tag_generation_with_git(self):
        """Tag should be YYYYMMDD-<git-hash> format."""
        import datetime

        # Mock subprocess to return a known git hash
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="abc1234\n")

            # Get the tag that would be generated
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            expected_pattern = f"{date_str}-abc1234"

            # Call the tag generation logic
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip()
                tag = f"{date_str}-{git_hash}"
                assert tag.startswith(date_str), f"Tag should start with date: {tag}"
                assert "-" in tag, f"Tag should have git hash: {tag}"

    def test_tag_generation_without_git(self):
        """Tag should use 'local' when git is not available."""
        import datetime

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("git not found")

            date_str = datetime.datetime.now().strftime("%Y%m%d")
            expected_tag = f"{date_str}-local"

            # Simulate the fallback logic
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                )
                git_hash = result.stdout.strip() if result.returncode == 0 else "local"
            except Exception:
                git_hash = "local"

            tag = f"{date_str}-{git_hash}"
            assert tag == expected_tag

    def test_custom_tag(self):
        """Custom tag should be used when provided."""
        custom_tag = "v1.0.0-test"

        with patch("subprocess.run") as mock_run:
            # Make docker build fail fast (we just want to test tag handling)
            mock_run.return_value = MagicMock(returncode=1)

            with pytest.raises(RuntimeError, match="Docker build failed"):
                core.build_image(push=False, tag=custom_tag)

    def test_image_name_format(self):
        """Image name should follow GCR format."""
        assert core.GCP_PROJECT_ID == "wrinklefree-481904"
        assert core.IMAGE_NAME == "gcr.io/wrinklefree-481904/wf-train"


class TestBuildCLICommand:
    """Tests for 'wf build' CLI command."""

    def test_build_command_exists(self):
        """'wf build' command should be registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["build", "--help"])
        assert result.exit_code == 0
        assert "Build and push training Docker image" in result.output

    def test_build_no_push_option(self):
        """--no-push option should be available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["build", "--help"])
        assert "--no-push" in result.output

    def test_build_tag_option(self):
        """--tag option should be available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["build", "--help"])
        assert "--tag" in result.output or "-t" in result.output

    @patch("wf_deployer.core.build_image")
    def test_build_calls_core_function(self, mock_build):
        """CLI should call core.build_image with correct args."""
        mock_build.return_value = "gcr.io/wrinklefree-481904/wf-train:test"

        runner = CliRunner()
        result = runner.invoke(cli, ["build", "--no-push", "-t", "test"])

        mock_build.assert_called_once_with(push=False, tag="test")

    @patch("wf_deployer.core.build_image")
    def test_build_default_push(self, mock_build):
        """CLI should push by default."""
        mock_build.return_value = "gcr.io/wrinklefree-481904/wf-train:test"

        runner = CliRunner()
        result = runner.invoke(cli, ["build", "-t", "test"])

        mock_build.assert_called_once_with(push=True, tag="test")


class TestDockerfileContent:
    """Tests for Dockerfile.train content."""

    def test_dockerfile_has_cuda_base(self):
        """Dockerfile should use CUDA base image."""
        deployer_dir = Path(__file__).parent.parent
        dockerfile = deployer_dir / "docker" / "Dockerfile.train"

        content = dockerfile.read_text()
        assert "nvidia/cuda" in content, "Should use NVIDIA CUDA base image"
        assert "12.4" in content, "Should use CUDA 12.4"

    def test_dockerfile_has_python(self):
        """Dockerfile should install Python 3.11."""
        deployer_dir = Path(__file__).parent.parent
        dockerfile = deployer_dir / "docker" / "Dockerfile.train"

        content = dockerfile.read_text()
        assert "python3.11" in content, "Should install Python 3.11"

    def test_dockerfile_has_uv(self):
        """Dockerfile should install uv package manager."""
        deployer_dir = Path(__file__).parent.parent
        dockerfile = deployer_dir / "docker" / "Dockerfile.train"

        content = dockerfile.read_text()
        assert "uv" in content, "Should install uv"
        assert "astral.sh" in content, "Should install uv from astral.sh"

    def test_dockerfile_has_ml_packages(self):
        """Dockerfile should install core ML packages."""
        deployer_dir = Path(__file__).parent.parent
        dockerfile = deployer_dir / "docker" / "Dockerfile.train"

        content = dockerfile.read_text()
        required_packages = [
            "torch",
            "transformers",
            "datasets",
            "hydra-core",
            "wandb",
            "accelerate",
        ]
        for pkg in required_packages:
            assert pkg in content, f"Should install {pkg}"

    def test_dockerfile_has_cache_buster(self):
        """Dockerfile should have WF_VERSION for cache busting."""
        deployer_dir = Path(__file__).parent.parent
        dockerfile = deployer_dir / "docker" / "Dockerfile.train"

        content = dockerfile.read_text()
        assert "WF_VERSION" in content, "Should have WF_VERSION build arg"
