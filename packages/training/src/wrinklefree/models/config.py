"""Model configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BitNetConfig:
    """
    Configuration for BitNet models.

    This configuration supports LLaMA-style architectures with
    BitNet quantization (1.58-bit weights, 8-bit activations).

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of KV heads for GQA (None = MHA)
        head_dim: Dimension per attention head
        max_position_embeddings: Maximum sequence length
        rope_theta: Base for RoPE frequency calculation
        attention_dropout: Dropout probability for attention
        hidden_act: Activation function in FFN ("relu2" or "silu")
        rms_norm_eps: Epsilon for RMSNorm
        tie_word_embeddings: Whether to tie input/output embeddings
        use_flash_attention: Whether to use Flash Attention
        use_cache: Whether to enable KV cache for inference
        pad_token_id: ID for padding token
        bos_token_id: ID for beginning-of-sequence token
        eos_token_id: ID for end-of-sequence token
    """

    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: Optional[int] = 8
    head_dim: Optional[int] = None
    max_position_embeddings: int = 4096
    rope_theta: float = 500000.0
    attention_dropout: float = 0.0
    hidden_act: str = "relu2"
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    use_flash_attention: bool = True
    use_cache: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        """Compute derived values."""
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads


@dataclass
class BitNetTrainingConfig:
    """
    Training configuration for BitNet models.

    Includes settings for the 3-stage BitDistill training pipeline.

    Args:
        stage: Training stage (1=SubLN insertion, 2=pretrain, 3=distill)
        max_steps: Maximum training steps
        max_seq_length: Maximum sequence length
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        warmup_steps: Number of warmup steps
        lr_scheduler_type: Type of learning rate scheduler
        gradient_clipping: Maximum gradient norm
        mixed_precision: Mixed precision type ("bf16", "fp16", "no")
        use_8bit_optimizer: Whether to use 8-bit AdamW
        quantization_warmup_steps: Steps for quantization warmup (stage 2/3)
        lambda_logits: Logits distillation loss weight (stage 3)
        gamma_attention: Attention distillation loss weight (stage 3)
        temperature: Distillation temperature (stage 3)
    """

    stage: int = 3
    max_steps: int = 5000
    max_seq_length: int = 512
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    gradient_clipping: float = 1.0
    mixed_precision: str = "bf16"
    use_8bit_optimizer: bool = True
    quantization_warmup_steps: int = 1000
    lambda_logits: float = 10.0
    gamma_attention: float = 1e-5
    temperature: float = 5.0


@dataclass
class BitNetDistributedConfig:
    """
    Distributed training configuration.

    Args:
        strategy: Distributed strategy ("fsdp", "ddp", "single")
        sharding_strategy: FSDP sharding strategy
        mixed_precision_enabled: Enable mixed precision in FSDP
        activation_checkpointing: Enable activation checkpointing
        num_gpus: Number of GPUs per node
        num_nodes: Number of nodes
    """

    strategy: str = "fsdp"
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision_enabled: bool = True
    activation_checkpointing: bool = True
    num_gpus: int = 1
    num_nodes: int = 1


# Preset configurations for common model sizes
BITNET_CONFIGS = {
    "bitnet-1b": BitNetConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=16,
        num_kv_heads=4,
        max_position_embeddings=4096,
    ),
    "bitnet-3b": BitNetConfig(
        vocab_size=128256,
        hidden_size=3200,
        intermediate_size=8640,
        num_hidden_layers=26,
        num_attention_heads=32,
        num_kv_heads=8,
        max_position_embeddings=4096,
    ),
    "bitnet-7b": BitNetConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        max_position_embeddings=4096,
    ),
    "bitnet-13b": BitNetConfig(
        vocab_size=128256,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_kv_heads=8,
        max_position_embeddings=4096,
    ),
}


def get_config(name: str) -> BitNetConfig:
    """Get a preset configuration by name."""
    if name not in BITNET_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(BITNET_CONFIGS.keys())}")
    return BITNET_CONFIGS[name]
