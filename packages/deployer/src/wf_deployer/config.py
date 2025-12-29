"""Pydantic configuration models for WrinkleFree Deployer.

These models provide type-safe, validated configuration for deployment,
training, and infrastructure operations.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from wf_deployer.constants import (
    DEFAULT_DATA,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_CONTEXT_SIZE,
)


class ResourcesConfig(BaseModel):
    """Resource requirements for SkyPilot tasks."""

    cpus: str = Field(default="16+", description="CPU requirement (e.g., '16+', '32')")
    memory: str = Field(default="128+", description="Memory requirement (e.g., '128+', '256')")
    accelerators: Optional[str] = Field(
        default=None, description="GPU accelerators (e.g., 'L4:1', 'A100:8')"
    )
    cloud: Optional[str] = Field(
        default=None, description="Cloud provider (e.g., 'aws', 'gcp', 'runpod')"
    )
    region: Optional[str] = Field(default=None, description="Cloud region")
    use_spot: bool = Field(default=True, description="Use spot/preemptible instances")
    disk_size: int = Field(default=100, description="Disk size in GB")


class ServiceConfig(BaseModel):
    """Configuration for deploying an inference service."""

    name: str = Field(..., description="Service name (used as SkyServe service name)")
    backend: Literal["bitnet", "vllm"] = Field(
        default="bitnet", description="Inference backend"
    )
    model_path: str = Field(
        ..., description="Path to model (local, gs://, s3://, or hf://)"
    )
    resources: ResourcesConfig = Field(
        default_factory=ResourcesConfig, description="Resource requirements"
    )

    # Server config
    port: int = Field(default=8080, description="Server port")
    host: str = Field(default="0.0.0.0", description="Server host")
    context_size: int = Field(default=DEFAULT_CONTEXT_SIZE, description="Context window size")
    num_threads: int = Field(default=0, description="CPU threads (0=auto)")

    # Scaling config
    min_replicas: int = Field(default=1, description="Minimum replicas")
    max_replicas: int = Field(default=10, description="Maximum replicas")
    target_qps: float = Field(default=5.0, description="Target QPS per replica")

    # Health check config
    readiness_path: str = Field(default="/health", description="Health check path")
    initial_delay_seconds: int = Field(
        default=120, description="Initial delay for health check"
    )


class TrainingConfig(BaseModel):
    """Configuration for launching a training job."""

    name: str = Field(..., description="Job name")
    model: str = Field(..., description="Model config name (e.g., 'qwen3_4b')")
    stage: float = Field(..., description="Training stage (1, 1.9, 2, or 3)")

    # Backend selection (Modal is default)
    backend: Literal["modal", "skypilot"] = Field(
        default="modal", description="Deployment backend (modal or skypilot)"
    )

    # Checkpoint storage (used by SkyPilot backend)
    checkpoint_bucket: Optional[str] = Field(
        default=None, description="Bucket name for checkpoints (SkyPilot only)"
    )
    checkpoint_store: Literal["s3", "gcs", "r2", "azure", "modal"] = Field(
        default="modal", description="Storage backend"
    )

    # Resources
    accelerators: str = Field(default="H100:4", description="GPU accelerators")
    gpu: Literal["H100", "A100-80GB", "A100", "A10G", "L4", "T4"] = Field(
        default="H100", description="GPU type for Modal backend"
    )
    cloud: Optional[str] = Field(default="runpod", description="Cloud provider (SkyPilot)")
    use_spot: bool = Field(default=True, description="Use spot instances (SkyPilot)")

    # Training config
    workdir: str = Field(
        default="../WrinkleFree-1.58Quant", description="Training code directory"
    )
    data: str = Field(default=DEFAULT_DATA, description="Data config name")
    max_steps: Optional[int] = Field(default=None, description="Max training steps")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens to train on")

    # W&B config
    wandb_project: str = Field(default=DEFAULT_WANDB_PROJECT, description="W&B project name")
    wandb_enabled: bool = Field(default=True, description="Enable W&B logging")

    # Hydra overrides for advanced config
    hydra_overrides: list[str] = Field(
        default_factory=list, description="Additional Hydra CLI overrides"
    )


class InfraConfig(BaseModel):
    """Configuration for Terraform infrastructure provisioning."""

    provider: Literal["hetzner", "aws", "gcp"] = Field(
        ..., description="Cloud provider"
    )

    # Hetzner-specific
    server_count: int = Field(default=3, description="Number of Hetzner servers")
    server_type: str = Field(default="ax102", description="Hetzner server type")

    # GCP-specific
    project_id: Optional[str] = Field(default=None, description="GCP project ID")
    region: str = Field(default="us-central1", description="Cloud region")

    # AWS-specific
    bucket_name: Optional[str] = Field(default=None, description="S3 bucket name")

    # Common
    auto_approve: bool = Field(
        default=False, description="Auto-approve terraform apply"
    )
