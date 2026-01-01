"""Credentials management for WrinkleFree Deployer.

Supports loading credentials from:
- .env files (explicit path)
- Environment variables (fallback)
- Direct construction

Example usage:
    # Load from .env file
    creds = Credentials.from_env_file(".env")

    # Load from existing environment
    creds = Credentials.from_env()

    # Export to environment for SkyPilot/Terraform
    creds.apply_to_env()
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Credentials(BaseModel):
    """Cloud credentials with multiple loading strategies."""

    # AWS
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret key"
    )
    aws_region: str = Field(default="us-east-1", description="AWS region")

    # GCP
    google_application_credentials: Optional[Path] = Field(
        default=None, description="Path to GCP service account JSON"
    )
    gcp_project_id: Optional[str] = Field(default=None, description="GCP project ID")

    # Hetzner
    hetzner_api_token: Optional[str] = Field(
        default=None, description="Hetzner Cloud API token"
    )
    hetzner_ssh_key_path: Optional[Path] = Field(
        default=None, description="Path to Hetzner SSH private key"
    )
    hetzner_server_ips: list[str] = Field(
        default_factory=list, description="Hetzner dedicated server IPs"
    )

    # RunPod
    runpod_api_key: Optional[str] = Field(default=None, description="RunPod API key")

    # Vast.ai
    vastai_api_key: Optional[str] = Field(default=None, description="Vast.ai API key")

    # Model/Training
    checkpoint_bucket: Optional[str] = Field(
        default=None, description="Checkpoint bucket name"
    )
    checkpoint_store: str = Field(default="s3", description="Checkpoint storage type")

    # W&B
    wandb_api_key: Optional[str] = Field(default=None, description="W&B API key")

    @classmethod
    def from_env_file(cls, path: str | Path) -> "Credentials":
        """Load from .env file, with fallback to existing env vars.

        Args:
            path: Path to .env file. If file doesn't exist or path is empty,
                  only environment variables are used.

        Returns:
            Credentials instance with merged values.
        """
        # Import here to avoid requiring python-dotenv if not used
        try:
            from dotenv import dotenv_values
        except ImportError:
            raise ImportError(
                "python-dotenv is required for .env file loading. "
                "Install with: pip install python-dotenv"
            )

        # Load from file if it exists
        file_values: dict[str, str | None] = {}
        if path and Path(path).exists():
            file_values = dotenv_values(path)

        # Credential precedence: .env file > environment variable > None
        # This allows .env to override existing env vars for local development
        # while still working in environments where credentials are set globally
        def get(key: str) -> str | None:
            return file_values.get(key) or os.environ.get(key)

        return cls(
            aws_access_key_id=get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get("AWS_SECRET_ACCESS_KEY"),
            aws_region=get("AWS_DEFAULT_REGION") or "us-east-1",
            google_application_credentials=(
                Path(p) if (p := get("GOOGLE_APPLICATION_CREDENTIALS")) else None
            ),
            gcp_project_id=get("GCP_PROJECT_ID"),
            hetzner_api_token=get("HETZNER_API_TOKEN"),
            hetzner_ssh_key_path=(
                Path(p) if (p := get("HETZNER_SSH_KEY_PATH")) else None
            ),
            hetzner_server_ips=(
                [ip.strip() for ip in get("HETZNER_SERVER_IPS").split(",") if ip.strip()]
                if get("HETZNER_SERVER_IPS")
                else []
            ),
            runpod_api_key=get("RUNPOD_API_KEY"),
            vastai_api_key=get("VASTAI_API_KEY"),
            checkpoint_bucket=get("CHECKPOINT_BUCKET"),
            checkpoint_store=get("CHECKPOINT_STORE") or "s3",
            wandb_api_key=get("WANDB_API_KEY"),
        )

    @classmethod
    def from_env(cls) -> "Credentials":
        """Load only from existing environment variables.

        Returns:
            Credentials instance from environment.
        """
        return cls.from_env_file("")

    def apply_to_env(self) -> None:
        """Export credentials to environment variables.

        This is useful for tools like SkyPilot and Terraform that read
        credentials from environment variables.
        """
        if self.aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        if self.aws_region:
            os.environ["AWS_DEFAULT_REGION"] = self.aws_region
        if self.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
                self.google_application_credentials
            )
        if self.gcp_project_id:
            os.environ["GCP_PROJECT_ID"] = self.gcp_project_id
        if self.hetzner_api_token:
            os.environ["HETZNER_API_TOKEN"] = self.hetzner_api_token
        if self.runpod_api_key:
            os.environ["RUNPOD_API_KEY"] = self.runpod_api_key
        if self.vastai_api_key:
            os.environ["VASTAI_API_KEY"] = self.vastai_api_key
        if self.wandb_api_key:
            os.environ["WANDB_API_KEY"] = self.wandb_api_key

    def has_aws(self) -> bool:
        """Check if AWS credentials are configured."""
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    def has_gcp(self) -> bool:
        """Check if GCP credentials are configured."""
        return bool(self.google_application_credentials or self.gcp_project_id)

    def has_hetzner(self) -> bool:
        """Check if Hetzner credentials are configured."""
        return bool(self.hetzner_api_token)

    def has_runpod(self) -> bool:
        """Check if RunPod credentials are configured."""
        return bool(self.runpod_api_key)

    def has_vastai(self) -> bool:
        """Check if Vast.ai credentials are configured."""
        return bool(self.vastai_api_key)
