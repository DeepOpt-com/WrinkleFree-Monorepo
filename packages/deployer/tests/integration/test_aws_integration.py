"""
Integration tests for AWS Terraform configurations.

These tests validate AWS configuration for S3 checkpoint storage
and IAM roles for SkyPilot access.

Run with:
    uv run pytest tests/integration/test_aws_integration.py -v
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
def terraform_aws_dir():
    """Path to Terraform AWS configuration."""
    return Path(__file__).parent.parent.parent / "terraform" / "aws"


class TestAWSConfigurationChecks:
    """Test AWS configuration file contents."""

    def test_main_tf_exists(self, terraform_aws_dir):
        """main.tf should exist."""
        main_file = terraform_aws_dir / "main.tf"
        assert main_file.exists(), "main.tf should exist"

    def test_variables_file_exists(self, terraform_aws_dir):
        """variables.tf should exist."""
        variables_file = terraform_aws_dir / "variables.tf"
        assert variables_file.exists(), "variables.tf should exist"

    def test_outputs_file_exists(self, terraform_aws_dir):
        """outputs.tf should exist."""
        outputs_file = terraform_aws_dir / "outputs.tf"
        assert outputs_file.exists(), "outputs.tf should exist"

    def test_variables_have_descriptions(self, terraform_aws_dir):
        """All variables should have descriptions."""
        variables_file = terraform_aws_dir / "variables.tf"
        if not variables_file.exists():
            pytest.skip("variables.tf not found")

        content = variables_file.read_text()
        variable_blocks = len(re.findall(r'variable\s+"[^"]+"', content))
        descriptions = len(re.findall(r"description\s*=", content))

        assert descriptions >= variable_blocks, (
            f"Not all variables have descriptions. "
            f"Found {variable_blocks} variables but only {descriptions} descriptions."
        )

    def test_outputs_have_descriptions(self, terraform_aws_dir):
        """All outputs should have descriptions."""
        outputs_file = terraform_aws_dir / "outputs.tf"
        if not outputs_file.exists():
            pytest.skip("outputs.tf not found")

        content = outputs_file.read_text()
        output_blocks = len(re.findall(r'output\s+"[^"]+"', content))
        descriptions = len(re.findall(r"description\s*=", content))

        assert descriptions >= output_blocks, (
            f"Not all outputs have descriptions. "
            f"Found {output_blocks} outputs but only {descriptions} descriptions."
        )


class TestAWSSecurityConfiguration:
    """Test AWS security-related configuration."""

    def test_s3_bucket_encryption_configured(self, terraform_aws_dir):
        """S3 buckets should have encryption configured."""
        main_file = terraform_aws_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        assert "aws_s3_bucket_server_side_encryption_configuration" in content, (
            "S3 bucket encryption should be configured"
        )

    def test_s3_public_access_blocked(self, terraform_aws_dir):
        """S3 buckets should block public access."""
        main_file = terraform_aws_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        assert "aws_s3_bucket_public_access_block" in content, (
            "S3 public access block should be configured"
        )
        assert "block_public_acls" in content, (
            "Should block public ACLs"
        )
        assert "block_public_policy" in content, (
            "Should block public policy"
        )

    def test_s3_versioning_enabled(self, terraform_aws_dir):
        """S3 buckets should have versioning enabled for checkpoints."""
        main_file = terraform_aws_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        assert "aws_s3_bucket_versioning" in content, (
            "S3 bucket versioning should be configured"
        )

    def test_iam_role_defined(self, terraform_aws_dir):
        """IAM role for SkyPilot should be defined."""
        main_file = terraform_aws_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        assert "aws_iam_role" in content, (
            "IAM role should be defined"
        )

    def test_iam_policy_uses_least_privilege(self, terraform_aws_dir):
        """IAM policy should follow least privilege principle."""
        main_file = terraform_aws_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        # Check for specific S3 actions rather than s3:*
        assert "s3:GetObject" in content or "s3:PutObject" in content, (
            "IAM policy should use specific S3 actions, not s3:*"
        )
        # Should NOT have overly permissive policies
        assert '"*"' not in content or "Resource" in content, (
            "IAM policy should not use wildcard resources carelessly"
        )


class TestAWSCheckpointStorage:
    """Test checkpoint storage configuration."""

    def test_checkpoint_bucket_defined(self, terraform_aws_dir):
        """Checkpoint bucket should be defined."""
        main_file = terraform_aws_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        assert "checkpoint" in content.lower(), (
            "Checkpoint bucket should be defined"
        )

    def test_lifecycle_rules_for_archival(self, terraform_aws_dir):
        """Lifecycle rules should be configured for cost optimization."""
        main_file = terraform_aws_dir / "main.tf"
        if not main_file.exists():
            pytest.skip("main.tf not found")

        content = main_file.read_text()
        # Check for lifecycle configuration
        assert "aws_s3_bucket_lifecycle_configuration" in content, (
            "S3 lifecycle configuration should exist for checkpoint archival"
        )


class TestAWSTerraformValidation:
    """Test AWS Terraform configuration validity."""

    def test_terraform_init_succeeds(self, terraform_aws_dir):
        """Terraform init should succeed for AWS module."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_aws_dir.exists():
            pytest.skip("AWS terraform directory not found")

        result = subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=terraform_aws_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform init failed: {result.stderr}"

    def test_terraform_validate_succeeds(self, terraform_aws_dir):
        """Terraform validate should succeed for AWS module."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_aws_dir.exists():
            pytest.skip("AWS terraform directory not found")

        # Ensure initialized
        subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=terraform_aws_dir,
            capture_output=True,
        )

        result = subprocess.run(
            ["terraform", "validate"],
            cwd=terraform_aws_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform validate failed: {result.stderr}"

    def test_terraform_fmt_check(self, terraform_aws_dir):
        """Terraform files should be properly formatted."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_aws_dir.exists():
            pytest.skip("AWS terraform directory not found")

        result = subprocess.run(
            ["terraform", "fmt", "-check", "-recursive"],
            cwd=terraform_aws_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Terraform files not properly formatted. Run: terraform fmt -recursive\n"
            f"Files needing format: {result.stdout}"
        )
