"""Tests for wf_deployer.credentials module."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from wf_deployer.credentials import Credentials


class TestCredentials:
    """Tests for Credentials class."""

    def test_defaults(self):
        """Test default values."""
        creds = Credentials()
        assert creds.aws_access_key_id is None
        assert creds.aws_region == "us-east-1"
        assert creds.checkpoint_store == "s3"
        assert creds.hetzner_server_ips == []

    def test_custom_values(self):
        """Test custom values."""
        creds = Credentials(
            aws_access_key_id="AKIATEST",
            aws_secret_access_key="secret",
            gcp_project_id="my-project",
            hetzner_server_ips=["10.0.0.1", "10.0.0.2"],
        )
        assert creds.aws_access_key_id == "AKIATEST"
        assert creds.gcp_project_id == "my-project"
        assert len(creds.hetzner_server_ips) == 2

    def test_from_env_file(self, tmp_path):
        """Test loading from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            """
AWS_ACCESS_KEY_ID=AKIATEST123
AWS_SECRET_ACCESS_KEY=secret123
AWS_DEFAULT_REGION=eu-west-1
GCP_PROJECT_ID=test-project
HETZNER_SERVER_IPS=10.0.0.1,10.0.0.2,10.0.0.3
CHECKPOINT_BUCKET=my-checkpoints
"""
        )

        creds = Credentials.from_env_file(env_file)

        assert creds.aws_access_key_id == "AKIATEST123"
        assert creds.aws_secret_access_key == "secret123"
        assert creds.aws_region == "eu-west-1"
        assert creds.gcp_project_id == "test-project"
        assert creds.hetzner_server_ips == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
        assert creds.checkpoint_bucket == "my-checkpoints"

    def test_from_env_file_fallback_to_env_vars(self, tmp_path, monkeypatch):
        """Test fallback to environment variables."""
        # Set env var
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "FROM_ENV")
        monkeypatch.setenv("GCP_PROJECT_ID", "env-project")

        # Create partial .env file (missing AWS_ACCESS_KEY_ID)
        env_file = tmp_path / ".env"
        env_file.write_text("CHECKPOINT_BUCKET=from-file\n")

        creds = Credentials.from_env_file(env_file)

        # AWS_ACCESS_KEY_ID from env (not in file)
        assert creds.aws_access_key_id == "FROM_ENV"
        # CHECKPOINT_BUCKET from file
        assert creds.checkpoint_bucket == "from-file"
        # GCP_PROJECT_ID from env (not in file)
        assert creds.gcp_project_id == "env-project"

    def test_from_env_file_nonexistent(self, tmp_path, monkeypatch):
        """Test loading from nonexistent file falls back to env."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "FROM_ENV")

        creds = Credentials.from_env_file(tmp_path / "nonexistent.env")

        assert creds.aws_access_key_id == "FROM_ENV"

    def test_from_env(self, monkeypatch):
        """Test loading from environment only."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAENV")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secretenv")
        monkeypatch.setenv("RUNPOD_API_KEY", "runpod123")

        creds = Credentials.from_env()

        assert creds.aws_access_key_id == "AKIAENV"
        assert creds.aws_secret_access_key == "secretenv"
        assert creds.runpod_api_key == "runpod123"

    def test_apply_to_env(self, monkeypatch):
        """Test exporting credentials to environment."""
        # Clear any existing env vars
        for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "GCP_PROJECT_ID"]:
            monkeypatch.delenv(key, raising=False)

        creds = Credentials(
            aws_access_key_id="AKIATEST",
            aws_secret_access_key="secret",
            gcp_project_id="test-project",
        )

        creds.apply_to_env()

        assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIATEST"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret"
        assert os.environ["GCP_PROJECT_ID"] == "test-project"

    def test_has_aws(self):
        """Test has_aws method."""
        assert not Credentials().has_aws()
        assert not Credentials(aws_access_key_id="key").has_aws()  # missing secret
        assert Credentials(
            aws_access_key_id="key",
            aws_secret_access_key="secret",
        ).has_aws()

    def test_has_gcp(self):
        """Test has_gcp method."""
        assert not Credentials().has_gcp()
        assert Credentials(gcp_project_id="project").has_gcp()
        assert Credentials(
            google_application_credentials=Path("/path/to/key.json")
        ).has_gcp()

    def test_has_hetzner(self):
        """Test has_hetzner method."""
        assert not Credentials().has_hetzner()
        assert Credentials(hetzner_api_token="token").has_hetzner()

    def test_has_runpod(self):
        """Test has_runpod method."""
        assert not Credentials().has_runpod()
        assert Credentials(runpod_api_key="key").has_runpod()
