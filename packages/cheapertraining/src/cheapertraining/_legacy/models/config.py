"""Model configuration dataclasses.

Based on MobileLLM-R1 paper (arXiv:2509.24945) architecture specifications.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MobileLLMConfig:
    """Configuration for MobileLLM model architecture.

    Architecture features from the paper:
    - QK-norm for training stability
    - Grouped Query Attention (GQA) for efficiency
    - Weight sharing between input/output embeddings
    - RoPE positional embeddings
    - RMSNorm for layer normalization

    Reference: https://arxiv.org/abs/2509.24945
    """

    # Core architecture
    num_layers: int = 22
    num_heads: int = 24
    num_kv_heads: int = 6  # For GQA; if equal to num_heads, uses MHA
    embed_dim: int = 1536
    hidden_dim: int = 6144  # FFN intermediate dimension
    vocab_size: int = 151_936  # Qwen2.5 tokenizer vocabulary size
    max_seq_len: int = 32_768

    # Architecture features
    use_qk_norm: bool = True  # QK-norm for training stability
    use_weight_sharing: bool = True  # Share input/output embeddings

    # RoPE configuration
    rope_base: float = 500_000.0
    rope_scaling: Optional[dict] = None  # For extended context

    # Normalization
    norm_eps: float = 1e-5

    # Dropout (typically 0 for pretraining)
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # MoE configuration (for very large models like DeepSeek-V3)
    use_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    moe_intermediate_size: Optional[int] = None

    # Initialization
    initializer_range: float = 0.02

    # Gradient checkpointing for memory efficiency
    use_gradient_checkpointing: bool = False
    gradient_checkpointing_mode: str = "quantized"  # "standard" or "quantized" (INT8)

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.embed_dim // self.num_heads

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        return self.num_heads // self.num_kv_heads

    def __post_init__(self):
        """Validate configuration."""
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )


@dataclass
class MobileLLM140MConfig(MobileLLMConfig):
    """140M parameter configuration from the paper.

    Table 2 in arXiv:2509.24945:
    - 15 layers, 9 heads, 3 KV-heads
    - dim=576, hidden_dim=2048
    """

    num_layers: int = 15
    num_heads: int = 9
    num_kv_heads: int = 3
    embed_dim: int = 576
    hidden_dim: int = 2048


@dataclass
class MobileLLM360MConfig(MobileLLMConfig):
    """360M parameter configuration from the paper.

    Table 2 in arXiv:2509.24945:
    - 15 layers, 16 heads, 4 KV-heads
    - dim=1024, hidden_dim=4096
    """

    num_layers: int = 15
    num_heads: int = 16
    num_kv_heads: int = 4
    embed_dim: int = 1024
    hidden_dim: int = 4096


@dataclass
class MobileLLM950MConfig(MobileLLMConfig):
    """950M parameter configuration from the paper.

    Table 2 in arXiv:2509.24945:
    - 22 layers, 24 heads, 6 KV-heads
    - dim=1536, hidden_dim=6144
    """

    num_layers: int = 22
    num_heads: int = 24
    num_kv_heads: int = 6
    embed_dim: int = 1536
    hidden_dim: int = 6144


@dataclass
class MobileLLM7BConfig(MobileLLMConfig):
    """7B parameter configuration for scaling studies.

    Based on LLaMA-2 7B architecture scaled with MobileLLM features.
    """

    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    embed_dim: int = 4096
    hidden_dim: int = 11008


@dataclass
class MobileLLM70BConfig(MobileLLMConfig):
    """70B parameter configuration for scaling studies.

    Based on LLaMA-2 70B architecture scaled with MobileLLM features.
    """

    num_layers: int = 80
    num_heads: int = 64
    num_kv_heads: int = 8
    embed_dim: int = 8192
    hidden_dim: int = 28672


@dataclass
class MobileLLM671BConfig(MobileLLMConfig):
    """671B parameter configuration for DeepSeek-V3 scale.

    Based on DeepSeek-V3 architecture with MoE.
    Reference: https://arxiv.org/html/2412.19437v1
    """

    num_layers: int = 61
    num_heads: int = 128
    num_kv_heads: int = 8
    embed_dim: int = 7168
    hidden_dim: int = 18432

    # MoE configuration
    use_moe: bool = True
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048


def get_config(name: str) -> MobileLLMConfig:
    """Get a predefined configuration by name.

    Args:
        name: Configuration name (e.g., "140m", "950m", "7b", "70b", "671b")

    Returns:
        MobileLLMConfig instance
    """
    configs = {
        "140m": MobileLLM140MConfig,
        "360m": MobileLLM360MConfig,
        "950m": MobileLLM950MConfig,
        "7b": MobileLLM7BConfig,
        "70b": MobileLLM70BConfig,
        "671b": MobileLLM671BConfig,
    }

    name = name.lower().replace("-", "").replace("_", "")
    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")

    return configs[name]()
