"""WrinkleFree Deployer - Simple training launcher.

This is a LAUNCHER, not a training framework. The actual training
config lives in WrinkleFree-1.58Quant/configs/ (Hydra).

We just need to know:
- WHAT to train: model, stage
- HOW to customize: hydra overrides

Training runs on SkyPilot for managed GPU jobs with spot recovery.

Example:
    from wf_deploy import train

    # Basic - launches on SkyPilot
    run_id = train("qwen3_4b", stage=2)

    # With overrides
    run_id = train("qwen3_4b", stage=2, overrides=["training.lr=1e-4"])

    # With specific scale
    run_id = train("qwen3_4b", stage=2, scale="large")

CLI:
    wf train -m qwen3_4b -s 2
    wf train -m qwen3_4b -s 2 training.lr=1e-4
    wf logs <run_id>
"""

# =============================================================================
# Simple interface (recommended)
# =============================================================================
from wf_deploy.core import (
    # Training
    train,
    train_fairy2,
    logs,
    cancel,
    list_runs,
    smoke_test,
    # Serving
    serve,
    serve_down,
    serve_status,
)

# =============================================================================
# Legacy interface (for backward compatibility)
# =============================================================================
from wf_deploy.config import (
    ResourcesConfig,
    ServiceConfig,
    TrainingConfig,
    InfraConfig,
)
from wf_deploy.credentials import Credentials
from wf_deploy.deployer import Deployer
from wf_deploy.trainer import Trainer, quick_launch
from wf_deploy.infra import Infra

__all__ = [
    # Simple interface (use these!)
    "train",
    "train_fairy2",
    "logs",
    "cancel",
    "list_runs",
    "smoke_test",
    # Legacy (still works, but simpler interface above is preferred)
    "TrainingConfig",
    "ServiceConfig",
    "ResourcesConfig",
    "InfraConfig",
    "Credentials",
    "Deployer",
    "Trainer",
    "Infra",
    "quick_launch",
]

__version__ = "0.2.0"
