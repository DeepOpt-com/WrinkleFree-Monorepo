"""Configuration dataclasses for distillation training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TeacherConfig:
    """Configuration for teacher model."""

    # Model specification
    model_name: Optional[str] = None  # None = infer from student checkpoint

    # vLLM settings
    use_vllm: bool = False
    vllm_url: str = "http://localhost:8000"
    vllm_top_k_logprobs: int = 100

    # Local teacher settings
    load_in_4bit: bool = False
    offload_to_cpu: bool = False


@dataclass
class LossConfig:
    """Configuration for distillation loss."""

    # Loss weights (BitDistill defaults)
    lambda_logits: float = 10.0  # Weight for logits distillation
    gamma_attention: float = 1e-5  # Weight for attention distillation (0 = disabled)

    # Temperature for KL divergence
    temperature: float = 5.0

    # Attention distillation settings
    use_relation_distill: bool = True  # Use BitDistill A·Aᵀ relations
    distill_layer: int = -1  # Layer for attention distillation (-1 = last)


@dataclass
class DistillationConfig:
    """Full configuration for distillation training."""

    # Student checkpoint
    student_checkpoint_path: str = ""

    # Teacher configuration
    teacher: TeacherConfig = field(default_factory=TeacherConfig)

    # Loss configuration
    loss: LossConfig = field(default_factory=LossConfig)

    # Training hyperparameters
    max_steps: int = 5000
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    gradient_clipping: float = 1.0

    # Optimizer
    optimizer_type: str = "adamw"  # adamw, adamw_8bit, muon
    learning_rate: float = 2.4e-3
    weight_decay: float = 0.024

    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1

    # Checkpointing
    save_interval: int = 500
    keep_last_n: int = 3
    output_dir: str = "outputs/distillation"

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    wandb_enabled: bool = True
    wandb_project: str = "wrinklefree-distillation"

    # Influence-based rebalancing
    influence_enabled: bool = True
    influence_update_interval: int = 1000
    influence_learning_rate: float = 0.2

    # Data (cheapertraining config name)
    data_config_name: str = "mixed_pretrain"
    max_seq_length: int = 1024

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.student_checkpoint_path)
