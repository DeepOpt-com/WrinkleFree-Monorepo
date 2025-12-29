"""
Integration tests for Hetzner Terraform configurations.

These tests validate Hetzner Cloud configuration for base layer infrastructure
including SSH keys, networks, and firewall rules.

Run with:
    uv run pytest tests/integration/test_hetzner_integration.py -v
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

# Check if terraform is available
TERRAFORM_AVAILABLE = shutil.which("terraform") is not None


pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture(scope="module")
def terraform_hetzner_dir():
    """Path to Terraform Hetzner configuration."""
    return Path(__file__).parent.parent.parent / "terraform" / "hetzner"


class TestHetznerConfigurationChecks:
    """Test Hetzner configuration file contents."""

    def test_main_tf_exists(self, terraform_hetzner_dir):
        """main.tf should exist."""
        main_file = terraform_hetzner_dir / "main.tf"
        assert main_file.exists(), "main.tf should exist"

    def test_variables_file_exists(self, terraform_hetzner_dir):
        """variables.tf should exist."""
        variables_file = terraform_hetzner_dir / "variables.tf"
        assert variables_file.exists(), "variables.tf should exist"

    def test_outputs_file_exists(self, terraform_hetzner_dir):
        """outputs.tf should exist."""
        outputs_file = terraform_hetzner_dir / "outputs.tf"
        assert outputs_file.exists(), "outputs.tf should exist"

    def test_providers_file_exists(self, terraform_hetzner_dir):
        """providers.tf should exist."""
        providers_file = terraform_hetzner_dir / "providers.tf"
        assert providers_file.exists(), "providers.tf should exist"

    def test_variables_have_descriptions(self, terraform_hetzner_dir):
        """All variables should have descriptions."""
        variables_file = terraform_hetzner_dir / "variables.tf"
        if not variables_file.exists():
            pytest.skip("variables.tf not found")

        content = variables_file.read_text()
        variable_blocks = len(re.findall(r'variable\s+"[^"]+"', content))
        descriptions = len(re.findall(r"description\s*=", content))

        assert descriptions >= variable_blocks, (
            f"Not all variables have descriptions. "
            f"Found {variable_blocks} variables but only {descriptions} descriptions."
        )

    def test_outputs_have_descriptions(self, terraform_hetzner_dir):
        """All outputs should have descriptions."""
        outputs_file = terraform_hetzner_dir / "outputs.tf"
        if not outputs_file.exists():
            pytest.skip("outputs.tf not found")

        content = outputs_file.read_text()
        output_blocks = len(re.findall(r'output\s+"[^"]+"', content))
        descriptions = len(re.findall(r"description\s*=", content))

        assert descriptions >= output_blocks, (
            f"Not all outputs have descriptions. "
            f"Found {output_blocks} outputs but only {descriptions} descriptions."
        )


class TestHetznerNetworkConfiguration:
    """Test Hetzner network configuration."""

    def test_network_resource_defined(self, terraform_hetzner_dir):
        """Network resource should be defined."""
        main_file = terraform_hetzner_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        assert "hcloud_network" in content, (
            "Hetzner Cloud network should be defined"
        )

    def test_subnet_defined(self, terraform_hetzner_dir):
        """Subnet should be defined for the network."""
        main_file = terraform_hetzner_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        assert "hcloud_network_subnet" in content, (
            "Network subnet should be defined"
        )

    def test_firewall_defined(self, terraform_hetzner_dir):
        """Firewall should be defined."""
        main_file = terraform_hetzner_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        assert "hcloud_firewall" in content, (
            "Firewall should be defined"
        )


class TestHetznerSecurityConfiguration:
    """Test Hetzner security-related configuration."""

    def test_ssh_key_resource_defined(self, terraform_hetzner_dir):
        """SSH key resource should be defined."""
        main_file = terraform_hetzner_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        assert "hcloud_ssh_key" in content, (
            "SSH key resource should be defined"
        )

    def test_firewall_has_ssh_rule(self, terraform_hetzner_dir):
        """Firewall should have SSH rule."""
        main_file = terraform_hetzner_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        # Check for port 22 in firewall rules
        assert "22" in content, (
            "Firewall should have SSH port 22 rule"
        )

    def test_firewall_has_inference_port(self, terraform_hetzner_dir):
        """Firewall should have inference port rule."""
        main_file = terraform_hetzner_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        # Check for inference port (can be hardcoded or via variable)
        has_inference_port = "inference_port" in content or "8080" in content
        assert has_inference_port, (
            "Firewall should have inference port rule (var.inference_port or 8080)"
        )


class TestHetznerSkyPilotIntegration:
    """Test SkyPilot integration outputs."""

    def test_ssh_node_pools_output_exists(self, terraform_hetzner_dir):
        """SSH node pools config output should exist."""
        outputs_file = terraform_hetzner_dir / "outputs.tf"
        if not outputs_file.exists():
            pytest.skip("outputs.tf not found")

        content = outputs_file.read_text()
        assert "ssh_node_pools" in content.lower() or "node_pool" in content.lower(), (
            "Output for SkyPilot SSH node pools should exist"
        )

    def test_server_ips_output_exists(self, terraform_hetzner_dir):
        """Server IPs output should exist for configuration."""
        outputs_file = terraform_hetzner_dir / "outputs.tf"
        if not outputs_file.exists():
            pytest.skip("outputs.tf not found")

        content = outputs_file.read_text()
        # Should output either server IPs or the full config
        assert "ip" in content.lower() or "server" in content.lower(), (
            "Output for server IPs should exist"
        )


class TestHetznerTerraformValidation:
    """Test Hetzner Terraform configuration validity."""

    def test_terraform_init_succeeds(self, terraform_hetzner_dir):
        """Terraform init should succeed for Hetzner module."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_hetzner_dir.exists():
            pytest.skip("Hetzner terraform directory not found")

        result = subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=terraform_hetzner_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform init failed: {result.stderr}"

    def test_terraform_validate_succeeds(self, terraform_hetzner_dir):
        """Terraform validate should succeed for Hetzner module."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_hetzner_dir.exists():
            pytest.skip("Hetzner terraform directory not found")

        # Ensure initialized
        subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=terraform_hetzner_dir,
            capture_output=True,
        )

        result = subprocess.run(
            ["terraform", "validate"],
            cwd=terraform_hetzner_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform validate failed: {result.stderr}"

    def test_terraform_fmt_check(self, terraform_hetzner_dir):
        """Terraform files should be properly formatted."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_hetzner_dir.exists():
            pytest.skip("Hetzner terraform directory not found")

        result = subprocess.run(
            ["terraform", "fmt", "-check", "-recursive"],
            cwd=terraform_hetzner_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Terraform files not properly formatted. Run: terraform fmt -recursive\n"
            f"Files needing format: {result.stdout}"
        )
