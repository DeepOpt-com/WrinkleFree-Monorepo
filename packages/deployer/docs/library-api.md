# WrinkleFree Deployer Python API

A Pydantic-based Python library for training and deploying LLM inference services.

## Installation

```bash
# From the repository
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Training with Modal (Default)

```python
from wf_deployer import quick_launch, Trainer

# One-liner for AI tools
run_id = quick_launch("qwen3_4b", stage=2, max_steps=10000)

# Full control
trainer = Trainer.from_json({
    "model": "qwen3_4b",
    "stage": 2,
    "max_steps": 10000,
})
run_id = trainer.launch()
print(trainer.status(run_id))
trainer.logs(run_id, follow=True)
```

### Training with SkyPilot

```python
from wf_deployer import TrainingConfig, Trainer, Credentials

creds = Credentials.from_env_file(".env")

config = TrainingConfig(
    name="qwen3-stage2",
    model="qwen3_4b",
    stage=2,
    backend="skypilot",  # Use SkyPilot instead of Modal
    checkpoint_bucket="my-checkpoints",
)

trainer = Trainer(config, creds)
job_id = trainer.launch()
trainer.logs(follow=True)
```

### Deploy a Service

```python
from wf_deployer import ServiceConfig, Deployer, Credentials

creds = Credentials.from_env_file(".env")

config = ServiceConfig(
    name="my-llm-service",
    backend="bitnet",
    model_path="gs://my-bucket/model.gguf",
)

deployer = Deployer(config, creds)
endpoint = deployer.up()
print(f"Service ready at: {endpoint}")
```

## CLI Reference

The `wf` command provides a unified CLI for training.

### Training Commands

```bash
# Launch training
wf train --model qwen3_4b --stage 2
wf train --model smollm2_135m --stage 1.9 --max-steps 100

# With options
wf train --model qwen3_4b --stage 2 --no-wandb --skip-recovery

# Smoke test
wf smoke
wf smoke --model qwen3_4b
```

### Monitoring Commands

```bash
# List runs
wf runs
wf runs <run_id>  # Details of specific run

# View logs
wf logs <run_id>
wf logs <run_id> -f  # Follow mode

# Cancel run
wf cancel <run_id>
```

### Setup Commands

```bash
# Interactive setup guide
wf setup
```

## Configuration Reference

### TrainingConfig

Configuration for launching training jobs.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | *required* | Job name |
| `model` | str | *required* | Model config name (e.g., "qwen3_4b") |
| `stage` | float | *required* | Training stage (1, 1.9, 2, or 3) |
| `backend` | "modal" \| "skypilot" | "modal" | Training backend |
| `data` | str | "fineweb" | Data config name |
| `max_steps` | int \| None | None | Maximum training steps |
| `max_tokens` | int \| None | None | Maximum tokens to train on |
| `gpu` | str | "H100" | GPU type (Modal) |
| `wandb_enabled` | bool | True | Enable W&B logging |
| `wandb_project` | str | "wrinklefree" | W&B project name |
| `hydra_overrides` | list[str] | [] | Additional Hydra CLI overrides |
| `checkpoint_bucket` | str \| None | None | Bucket for checkpoints (SkyPilot) |
| `checkpoint_store` | str | "modal" | Storage backend |
| `accelerators` | str | "H100:4" | GPU accelerators (SkyPilot) |
| `cloud` | str \| None | "runpod" | Cloud provider (SkyPilot) |
| `use_spot` | bool | True | Use spot instances (SkyPilot) |

### ServiceConfig

Configuration for deploying an inference service.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | *required* | Service name |
| `backend` | "bitnet" \| "vllm" | "bitnet" | Inference backend |
| `model_path` | str | *required* | Path to model (local, gs://, s3://, hf://) |
| `port` | int | 8080 | Server port |
| `context_size` | int | 4096 | Context window size |
| `min_replicas` | int | 1 | Minimum replicas |
| `max_replicas` | int | 10 | Maximum replicas |
| `target_qps` | float | 5.0 | Target QPS per replica |
| `resources` | ResourcesConfig | *defaults* | Resource requirements |

### Credentials

Cloud credentials with multiple loading strategies.

```python
# From .env file (with fallback to environment)
creds = Credentials.from_env_file(".env")

# From environment only
creds = Credentials.from_env()

# Export to environment (for SkyPilot/Terraform)
creds.apply_to_env()
```

## API Reference

### Trainer

```python
class Trainer:
    def __init__(self, config: TrainingConfig, credentials: Credentials = None): ...

    @classmethod
    def from_json(cls, config_dict: dict) -> "Trainer":
        """Create trainer from JSON config (for AI tools)."""

    def launch(self, detach: bool = True) -> str:
        """Launch training job. Returns run_id."""

    def status(self, run_id: str = None) -> dict:
        """Get job status."""

    def logs(self, run_id: str = None, follow: bool = False) -> str:
        """Get job logs."""

    def cancel(self, run_id: str = None) -> bool:
        """Cancel the training job."""

    def list_runs(self, limit: int = 20) -> list[dict]:
        """List recent training runs."""

    def smoke_test(self, model: str = "smollm2_135m") -> dict:
        """Run pipeline validation."""
```

### quick_launch

```python
def quick_launch(
    model: str = "qwen3_4b",
    stage: float = 2,
    max_steps: int | None = None,
    **kwargs,
) -> str:
    """Quick launch a training run with minimal config.

    Args:
        model: Model config name
        stage: Training stage (1, 1.9, 2, or 3)
        max_steps: Maximum training steps
        **kwargs: Additional TrainingConfig fields

    Returns:
        Run ID
    """
```

### ModalTrainer

Direct access to Modal training functions.

```python
from wf_deployer import ModalTrainer

trainer = ModalTrainer()

# Launch training
run_id = trainer.launch(
    model="qwen3_4b",
    stage=2,
    max_steps=10000,
    detach=True,
)

# From JSON
run_id = trainer.launch_json({
    "model": "qwen3_4b",
    "stage": 2,
})

# Monitor
trainer.status(run_id)
trainer.logs(run_id, follow=True)
trainer.cancel(run_id)

# List runs
trainer.list_runs(limit=20)

# Smoke test
trainer.smoke_test("smollm2_135m")
```

### Deployer

```python
class Deployer:
    def __init__(self, config: ServiceConfig, credentials: Credentials = None): ...

    def up(self, detach: bool = True) -> str:
        """Deploy the service. Returns endpoint URL."""

    def down(self) -> None:
        """Tear down the service."""

    def status(self) -> dict:
        """Get service status."""

    def logs(self, follow: bool = False) -> str:
        """Get service logs."""
```

## Modal Integration

### Key Concepts

**Automatic Shutdown:**
Modal functions shut down automatically when complete. You only pay for compute time used.

**Fingerprint-Based Resume:**
Training runs are identified by SHA256(config + git commit). Same config = auto-resume:
```python
# First run - trains from scratch
quick_launch("qwen3_4b", stage=2)

# Later - resumes from checkpoint
quick_launch("qwen3_4b", stage=2)
```

**Persistent Volumes:**
- `wrinklefree-checkpoints`: Training checkpoints
- `wrinklefree-hf-cache`: HuggingFace models/datasets

### Modal Secrets

Required secrets for Modal training:
```bash
modal secret create wandb-api-key WANDB_API_KEY=<key>
modal secret create huggingface-token HF_TOKEN=<token>
```

## Backward Compatibility

The library works alongside existing CLI workflows:

```bash
# These still work
sky serve up skypilot/service.yaml --name wrinklefree
sky jobs launch skypilot/train.yaml -e MODEL=qwen3_4b
modal run src/wf_deployer/modal_deployer.py --model qwen3_4b --stage 2
```

## Running Tests

```bash
# Run all library tests
uv run pytest tests/library/ -v

# Run with coverage
uv run pytest tests/library/ --cov=src/wf_deployer --cov-report=html
```
