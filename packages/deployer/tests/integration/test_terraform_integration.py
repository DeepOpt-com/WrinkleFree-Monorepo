"""
Integration tests for Terraform configurations against LocalStack.

These tests validate that Terraform configurations work correctly
with AWS-compatible APIs using LocalStack for emulation.

Run with:
    docker compose -f docker-compose.localstack.yml up -d
    uv run pytest tests/integration/test_terraform_integration.py -v -m localstack
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

# Check if terraform is available
TERRAFORM_AVAILABLE = shutil.which("terraform") is not None

# Check if LocalStack is available
LOCALSTACK_AVAILABLE = False
try:
    import boto3
    import requests

    try:
        resp = requests.get("http://localhost:4566/_localstack/health", timeout=2)
        LOCALSTACK_AVAILABLE = resp.status_code == 200
    except requests.exceptions.RequestException:
        pass
except ImportError:
    pass


pytestmark = [
    pytest.mark.localstack,
    pytest.mark.integration,
    pytest.mark.skipif(
        not LOCALSTACK_AVAILABLE,
        reason="LocalStack not available. Run: docker compose -f docker-compose.localstack.yml up -d",
    ),
]


@pytest.fixture(scope="module")
def localstack_ec2_client():
    """Create boto3 EC2 client configured for LocalStack."""
    import boto3

    return boto3.client(
        "ec2",
        endpoint_url="http://localhost:4566",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )


@pytest.fixture(scope="module")
def localstack_s3_client():
    """Create boto3 S3 client configured for LocalStack."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url="http://localhost:4566",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )


@pytest.fixture(scope="module")
def terraform_hetzner_dir():
    """Path to Terraform Hetzner configuration."""
    return Path(__file__).parent.parent.parent / "terraform" / "hetzner"


class TestLocalStackHealth:
    """Test LocalStack is properly configured."""

    def test_localstack_is_healthy(self):
        """LocalStack should be healthy and responding."""
        import requests

        response = requests.get("http://localhost:4566/_localstack/health", timeout=5)
        assert response.status_code == 200

        health = response.json()
        assert health.get("services", {}).get("ec2") in ["running", "available"]

    def test_ec2_service_available(self, localstack_ec2_client):
        """EC2 service should be available."""
        response = localstack_ec2_client.describe_vpcs()
        assert "Vpcs" in response

    def test_s3_service_available(self, localstack_s3_client):
        """S3 service should be available."""
        response = localstack_s3_client.list_buckets()
        assert "Buckets" in response


class TestVPCConfiguration:
    """Test VPC resources created by init script."""

    def test_vpc_exists(self, localstack_ec2_client):
        """VPC should exist with correct configuration."""
        response = localstack_ec2_client.describe_vpcs(
            Filters=[{"Name": "tag:Name", "Values": ["wrinklefree-test"]}]
        )
        vpcs = response["Vpcs"]
        assert len(vpcs) >= 1, "VPC 'wrinklefree-test' should exist"

        vpc = vpcs[0]
        assert vpc["CidrBlock"] == "10.0.0.0/16"

    def test_subnet_exists(self, localstack_ec2_client):
        """Subnet should exist in the VPC."""
        response = localstack_ec2_client.describe_subnets(
            Filters=[{"Name": "tag:Name", "Values": ["wrinklefree-test-subnet"]}]
        )
        subnets = response["Subnets"]
        assert len(subnets) >= 1, "Subnet should exist"

        subnet = subnets[0]
        assert subnet["CidrBlock"] == "10.0.1.0/24"

    def test_security_group_exists(self, localstack_ec2_client):
        """Security group should exist with correct rules."""
        response = localstack_ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": ["wrinklefree-test-sg"]}]
        )
        sgs = response["SecurityGroups"]
        assert len(sgs) >= 1, "Security group should exist"

        sg = sgs[0]
        # Check for SSH rule
        ssh_rules = [
            r
            for r in sg.get("IpPermissions", [])
            if r.get("FromPort") == 22 and r.get("ToPort") == 22
        ]
        assert len(ssh_rules) >= 1, "SSH rule should exist"

        # Check for inference port rule
        inference_rules = [
            r
            for r in sg.get("IpPermissions", [])
            if r.get("FromPort") == 8080 and r.get("ToPort") == 8080
        ]
        assert len(inference_rules) >= 1, "Inference port rule should exist"


class TestS3Integration:
    """Test S3 bucket created by init script."""

    def test_test_bucket_exists(self, localstack_s3_client):
        """Test artifacts bucket should exist."""
        response = localstack_s3_client.list_buckets()
        bucket_names = [b["Name"] for b in response["Buckets"]]
        assert "wrinklefree-test-artifacts" in bucket_names

    def test_can_upload_to_bucket(self, localstack_s3_client):
        """Should be able to upload objects to test bucket."""
        localstack_s3_client.put_object(
            Bucket="wrinklefree-test-artifacts",
            Key="test/integration-test.txt",
            Body=b"Integration test content",
        )

        response = localstack_s3_client.get_object(
            Bucket="wrinklefree-test-artifacts", Key="test/integration-test.txt"
        )
        content = response["Body"].read()
        assert content == b"Integration test content"


class TestTerraformValidation:
    """Test Terraform configuration validity."""

    def test_terraform_validate_hetzner(self, terraform_hetzner_dir):
        """Hetzner Terraform configuration should be valid."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        if not terraform_hetzner_dir.exists():
            pytest.skip("Hetzner terraform directory not found")

        # Initialize terraform (skip backend)
        result = subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=terraform_hetzner_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform init failed: {result.stderr}"

        # Validate configuration
        result = subprocess.run(
            ["terraform", "validate"],
            cwd=terraform_hetzner_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform validate failed: {result.stderr}"

    def test_terraform_validate_gcp(self):
        """GCP Terraform configuration should be valid."""
        if not TERRAFORM_AVAILABLE:
            pytest.fail(
                "terraform binary not found in PATH. "
                "Install Terraform 1.6+: https://developer.hashicorp.com/terraform/downloads"
            )
        gcp_dir = Path(__file__).parent.parent.parent / "terraform" / "gcp"
        if not gcp_dir.exists():
            pytest.skip("GCP terraform directory not found")

        # Initialize terraform (skip backend)
        result = subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=gcp_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform init failed: {result.stderr}"

        # Validate configuration
        result = subprocess.run(
            ["terraform", "validate"],
            cwd=gcp_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Terraform validate failed: {result.stderr}"


class TestSecurityGroupRules:
    """Test security group rule configuration."""

    def test_ssh_not_open_to_world_in_production_sg(self, localstack_ec2_client):
        """SSH should not be open to 0.0.0.0/0 in properly configured security groups."""
        response = localstack_ec2_client.describe_security_groups(
            Filters=[{"Name": "group-name", "Values": ["wrinklefree-test-sg"]}]
        )

        for sg in response["SecurityGroups"]:
            for rule in sg.get("IpPermissions", []):
                if rule.get("FromPort") == 22:
                    # Check that SSH is not open to the world
                    for ip_range in rule.get("IpRanges", []):
                        cidr = ip_range.get("CidrIp", "")
                        # Our test SG uses 10.0.0.0/8, not 0.0.0.0/0
                        assert cidr != "0.0.0.0/0", (
                            f"Security group {sg['GroupName']} has SSH open to 0.0.0.0/0"
                        )
