"""Configuration dataclasses for influence function computation.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) and AutoMixer paper.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class InfluenceTarget(Enum):
    """Which layers to use for influence calculation.

    Following AutoMixer's discriminative layer selection strategy,
    embedding and output layers contain the most discriminative signal.
    """
    EMBEDDING_ONLY = "embedding"
    OUTPUT_ONLY = "output"
    EMBEDDING_AND_OUTPUT = "both"


@dataclass
class InfluenceConfig:
    """Configuration for influence function computation.

    The DataInf algorithm approximates influence without Hessian inversion:
        I(z_train, z_probe) = <grad_train, grad_probe> / (lambda + ||grad_train||^2)
    """

    # Layer selection (AutoMixer discriminative layer selection)
    target_layers: InfluenceTarget = InfluenceTarget.EMBEDDING_AND_OUTPUT

    # DataInf regularization parameter
    # Prevents division by zero and stabilizes the estimate
    lambda_reg: float = 1e-4

    # Computation settings
    batch_size: int = 32
    use_fp16: bool = True
    num_workers: int = 4

    # Caching settings
    cache_gradients: bool = True
    cache_dir: Optional[str] = None

    # Gradient computation
    max_grad_norm: float = 1.0  # Clip gradients during extraction


@dataclass
class ProbeSetConfig:
    """Configuration for creating representative probe sets.

    Reference: MobileLLM-R1 Phase I - Data Curation
    Steps:
    1. Quality filtering (FineWeb-Edu classifier, min_score >= 4)
    2. Ask-LLM scoring (keep top 10%)
    3. Semantic deduplication
    """

    # Total probe set size
    probe_set_size: int = 10000
    seed: int = 42

    # Quality filtering thresholds
    fineweb_edu_min_score: float = 4.0
    ask_llm_top_fraction: float = 0.10  # Keep top 10%

    # Deduplication settings
    dedup_similarity_threshold: float = 0.85
    dedup_method: str = "minhash"  # "minhash" or "semantic"
    minhash_num_perm: int = 128

    # Domain categories for probe set
    domains: List[str] = field(default_factory=lambda: ["code", "math", "knowledge"])
    samples_per_domain: int = 3333  # ~10k total for 3 domains

    # Tokenization settings
    max_length: int = 2048


@dataclass
class MixtureOptimizationConfig:
    """Configuration for Phase II mixture weight calculation.

    Reference: MobileLLM-R1 Phase II - Pre-Training
    Determines optimal dataset weights using influence on probe set.
    """

    # Number of samples to evaluate from each dataset
    samples_per_dataset: int = 1000

    # Weight constraints
    normalize_weights: bool = True
    min_weight: float = 0.01  # Minimum dataset weight (1%)
    max_weight: float = 0.90  # Maximum dataset weight (90%)

    # Update settings
    weight_update_interval: int = 10000  # Steps between weight updates

    # Smoothing
    influence_smoothing: float = 0.1  # EMA smoothing for influence scores


@dataclass
class SelfBoostingConfig:
    """Configuration for Phase III mid-training self-boosting.

    Reference: MobileLLM-R1 Phase III - Mid-Training
    Uses current model to filter training data via rejection sampling.
    """

    # Filtering threshold
    # Samples with influence <= threshold are rejected
    influence_threshold: float = 0.0

    # Iterative compression stages
    num_stages: int = 2
    compression_ratio_per_stage: float = 0.5  # Keep 50% per stage

    # Recomputation interval
    # How often to refresh probe gradients with updated model
    recompute_interval: int = 1000

    # Buffer size for rejection sampling
    buffer_size: int = 1000

    # Minimum batch size after filtering
    # If filtering removes too many samples, keep at least this many
    min_batch_size: int = 1


# ============================================================================
# InfluenceDistillation Configurations (arXiv:2505.19051)
# ============================================================================


@dataclass
class JVPEmbeddingConfig:
    """Configuration for JVP-based embeddings.

    JVP (Jacobian-Vector Product) embeddings capture the sensitivity of
    transformer layer outputs to parameter perturbations. This provides
    a cheap approximation to gradient information.

    Reference: "Efficient Data Selection at Scale via Influence Distillation"
    (arXiv:2505.19051)
    """

    # Number of transformer blocks to use for JVP computation
    # First K layers typically capture most discriminative information
    num_jvp_layers: int = 4

    # Number of random tangent vectors for JVP
    # More vectors = higher-dimensional embedding = better approximation
    num_tangent_vectors: int = 2

    # Random seed for reproducible tangent vectors
    seed: int = 42

    # Whether to project embeddings via Randomized Hadamard Transform
    use_hadamard_projection: bool = True

    # Target dimension after projection (must be power of 2)
    projection_dim: int = 131072  # 2^17

    # Batch size for JVP computation
    batch_size: int = 32


@dataclass
class LandmarkConfig:
    """Configuration for landmark selection.

    Landmarks are a small subset of source samples for which we compute
    accurate gradients. Other samples' influences are approximated via
    KRR (Kernel Ridge Regression) from these landmarks.

    Reference: Section 4.2 of arXiv:2505.19051
    """

    # Number of landmark samples to select
    num_landmarks: int = 4096

    # Selection strategy:
    # - "random": Uniform random selection (fastest)
    # - "kmeans_pp": K-means++ initialization (diverse landmarks)
    # - "farthest_point": Farthest point sampling (maximum coverage)
    selection_strategy: str = "kmeans_pp"

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class KRRConfig:
    """Configuration for Kernel Ridge Regression.

    KRR is used to learn a mapping from JVP embeddings to gradient space,
    allowing influence propagation from landmarks to all samples.

    C = E_S @ E_L.T @ (E_L @ E_L.T + λI)^{-1}
    """

    # Regularization parameter (λ in the formula)
    # Larger values = more regularization = numerically stable but less accurate
    lambda_reg: float = 1e-4

    # Use Cholesky decomposition for solving (more stable than direct inverse)
    use_cholesky: bool = True

    # Chunk size for memory-efficient computation when N is large
    chunk_size: int = 10000


@dataclass
class InfluenceDistillationConfig:
    """Full configuration for Influence Distillation.

    This is the main config for the landmark-based influence approximation
    algorithm. It combines JVP embeddings, landmark selection, and KRR
    to efficiently estimate influence for large datasets.

    Algorithm:
    1. Compute JVP embeddings E_S for all source samples
    2. Select landmarks L, get E_L
    3. Solve KRR: C = E_S @ E_L.T @ (E_L @ E_L.T + λI)^{-1}
    4. Compute accurate gradients G_L for landmarks only
    5. Cache target gradient g_T from probe set
    6. Propagate: p = C @ (G_L @ g_T)

    Reference: arXiv:2505.19051
    """

    # Component configs
    jvp: JVPEmbeddingConfig = field(default_factory=JVPEmbeddingConfig)
    landmark: LandmarkConfig = field(default_factory=LandmarkConfig)
    krr: KRRConfig = field(default_factory=KRRConfig)

    # Number of warmup training steps before computing embeddings
    # Gradients are unstable early in training
    warmup_steps: int = 100

    # Batch size for gradient computation
    batch_size: int = 32

    # Number of samples per dataset when computing mixture weights
    samples_per_dataset: int = 1000

    # Weight constraints (same as MixtureOptimizationConfig)
    min_weight: float = 0.01
    max_weight: float = 0.90

    # Device placement
    device: str = "cuda"

    # Whether to store embeddings on CPU (saves GPU memory)
    store_embeddings_on_cpu: bool = True

    # Eval set configuration
    # Number of batches to use for eval loss computation (0 = use all)
    eval_batches: int = 0
