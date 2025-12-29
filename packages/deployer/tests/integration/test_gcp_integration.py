"""
Integration tests for GCP Terraform configurations.

These tests validate GCP configuration using fake-gcs-server for storage
emulation and basic Terraform validation.

Run with:
    docker compose -f docker-compose.localstack.yml up -d
    uv run pytest tests/integration/test_gcp_integration.py -v
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Check if terraform is available
TERRAFORM_AVAILABLE = shutil.which("terraform") is not None

# Check if fake-gcs-server is available
FAKE_GCS_AVAILABLE = False
try:
    import requests

    try:
        resp = requests.get("http://localhost:4443/storage/v1/b", timeout=2)
        FAKE_GCS_AVAILABLE = resp.status_code in [200, 404]  # 404 is OK (no buckets)
    except requests.exceptions.RequestException:
        pass
except ImportError:
    pass


pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture(scope="module")
def terraform_gcp_dir():
    """Path to Terraform GCP configuration."""
    return Path(__file__).parent.parent.parent / "terraform" / "gcp"


class TestFakeGCSHealth:
    """Test fake-gcs-server is properly configured."""

    @pytest.mark.skipif(
        not FAKE_GCS_AVAILABLE, reason="fake-gcs-server not available"
    )
    def test_fake_gcs_is_healthy(self):
        """fake-gcs-server should be responding."""
        import requests

        response = requests.get("http://localhost:4443/storage/v1/b", timeout=5)
        # 200 = has buckets, 404 = no buckets yet (both are valid)
        assert response.status_code in [200, 404]


class TestGCPTerraformValidation:
    """Test GCP Terraform configuration validity."""

    def test_terraform_init_succeeds(self, terraform_gcp_dir):
        """Terraform init should succeed for GCP module."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_gcp_dir.exists():
            pytest.skip("GCP terraform directory not found")

        result = subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=terraform_gcp_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform init failed: {result.stderr}"

    def test_terraform_validate_succeeds(self, terraform_gcp_dir):
        """Terraform validate should succeed for GCP module."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_gcp_dir.exists():
            pytest.skip("GCP terraform directory not found")

        # Ensure initialized
        subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=terraform_gcp_dir,
            capture_output=True,
        )

        result = subprocess.run(
            ["terraform", "validate"],
            cwd=terraform_gcp_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform validate failed: {result.stderr}"

    def test_terraform_fmt_check(self, terraform_gcp_dir):
        """Terraform files should be properly formatted."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_gcp_dir.exists():
            pytest.skip("GCP terraform directory not found")

        result = subprocess.run(
            ["terraform", "fmt", "-check", "-recursive"],
            cwd=terraform_gcp_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Terraform files not properly formatted. Run: terraform fmt -recursive\n"
            f"Files needing format: {result.stdout}"
        )


class TestGCPConfigurationChecks:
    """Test GCP configuration file contents."""

    def test_networking_file_exists(self, terraform_gcp_dir):
        """networking.tf should exist."""
        networking_file = terraform_gcp_dir / "networking.tf"
        assert networking_file.exists(), "networking.tf should exist"

    def test_variables_have_descriptions(self, terraform_gcp_dir):
        """All variables should have descriptions."""
        variables_file = terraform_gcp_dir / "variables.tf"
        if not variables_file.exists():
            pytest.skip("variables.tf not found")

        content = variables_file.read_text()
        # Count variable blocks and description attributes
        import re

        variable_blocks = len(re.findall(r'variable\s+"[^"]+"', content))
        descriptions = len(re.findall(r"description\s*=", content))

        assert descriptions >= variable_blocks, (
            f"Not all variables have descriptions. "
            f"Found {variable_blocks} variables but only {descriptions} descriptions."
        )

    def test_outputs_have_descriptions(self, terraform_gcp_dir):
        """All outputs should have descriptions."""
        outputs_file = terraform_gcp_dir / "outputs.tf"
        if not outputs_file.exists():
            pytest.skip("outputs.tf not found")

        content = outputs_file.read_text()
        import re

        output_blocks = len(re.findall(r'output\s+"[^"]+"', content))
        descriptions = len(re.findall(r"description\s*=", content))

        assert descriptions >= output_blocks, (
            f"Not all outputs have descriptions. "
            f"Found {output_blocks} outputs but only {descriptions} descriptions."
        )


class TestGCPSecurityConfiguration:
    """Test GCP security-related configuration."""

    def test_firewall_rules_exist(self, terraform_gcp_dir):
        """Firewall rules should be defined."""
        networking_file = terraform_gcp_dir / "networking.tf"
        if not networking_file.exists():
            pytest.skip("networking.tf not found")

        content = networking_file.read_text()
        assert "google_compute_firewall" in content, (
            "Firewall rules should be defined"
        )
        # Check for essential rules
        assert "allow_ssh" in content, "SSH firewall rule should exist"
        assert "allow_inference" in content, "Inference firewall rule should exist"
        assert "allow_health_check" in content, "Health check firewall rule should exist"
