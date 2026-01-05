"""Configuration for efficient meta-optimization.

This module provides configuration classes for two complementary methods:
1. LDC-MTL: Objective weight optimization (CE vs DLM vs distillation)
2. ODM: Dataset weight optimization (web vs code vs math)

Both methods are O(1) complexity and require no external dependencies.

References:
- LDC-MTL: https://arxiv.org/abs/2502.08585
- ODM: https://arxiv.org/abs/2312.02406
"""

from dataclasses import dataclass, field


@dataclass
class LDCMTLConfig:
    """LDC-MTL configuration for objective weight optimization.

    LDC-MTL (Loss Discrepancy Control for Multi-Task Learning) uses a small
    router network to learn optimal task weights with O(1) complexity.

    The key insight from the paper is that one gradient term in bi-level
    optimization is empirically ~100x smaller than the other, allowing safe
    simplification to a penalized single-level problem.

    Reference:
        https://arxiv.org/abs/2502.08585

    Attributes:
        enabled: Whether to enable LDC-MTL objective weighting.
        lambda_penalty: Weight for the discrepancy penalty. Higher values
            encourage more balanced weighted losses across objectives.
            Paper recommends 0.1-1.0 range.
        hidden_dim: Hidden layer dimension for the router MLP. A small
            network (32-64) is sufficient as input is just K loss values.
        router_lr: Learning rate for the router optimizer. Should be
            similar to or slightly higher than main model learning rate.
    """

    enabled: bool = True
    lambda_penalty: float = 0.1
    hidden_dim: int = 32
    router_lr: float = 1e-3
    step_interval: int = 1
    """How often to update router weights. Set >1 when using large gradient
    accumulation steps so the router sees more diverse gradients before updating.
    For single-batch training, set equal to gradient_accumulation_steps."""


@dataclass
class ODMConfig:
    """ODM (EXP3) configuration for dataset weight optimization.

    ODM (Online Data Mixing) uses the EXP3 multi-armed bandit algorithm
    to learn optimal dataset sampling probabilities during training.

    Each dataset domain is treated as an "arm" in the bandit. Training
    loss serves as the reward signal - higher loss means more to learn
    from that domain, so it gets higher sampling probability.

    Published results show 19% fewer iterations to reach same perplexity
    with ~0% wall-clock overhead.

    Reference:
        https://arxiv.org/abs/2312.02406

    Attributes:
        enabled: Whether to enable ODM dataset weighting.
        reward_smoothing: EMA coefficient for reward updates (0-1).
            Higher values give more weight to historical rewards,
            providing smoother but slower adaptation. Default 0.9.
        warmup_ratio: Fraction of training to use uniform weights (0-1).
            During warmup, all datasets are sampled equally to gather
            initial statistics. Default 0.01 (1% of training).
        min_weight: Minimum allowed sampling probability per dataset.
            Prevents any dataset from being completely ignored.
        max_weight: Maximum allowed sampling probability per dataset.
            Prevents any single dataset from dominating training.
    """

    enabled: bool = True
    reward_smoothing: float = 0.9
    warmup_ratio: float = 0.01
    min_weight: float = 0.05
    max_weight: float = 0.60


@dataclass
class MetaOptimizationConfig:
    """Unified configuration for meta-optimization.

    Combines LDC-MTL (objective weights) and ODM (dataset weights) into
    a single configuration. Both can be enabled independently.

    Example YAML config:
        meta_optimization:
          enabled: true
          ldc_mtl:
            enabled: true
            lambda_penalty: 0.1
            hidden_dim: 32
            router_lr: 0.001
          odm:
            enabled: true
            reward_smoothing: 0.9
            warmup_ratio: 0.01
            min_weight: 0.05
            max_weight: 0.60
          log_interval: 100

    Attributes:
        enabled: Master switch for meta-optimization
        ldc_mtl: LDC-MTL config for objective weights
        odm: ODM config for dataset weights
        log_interval: Steps between logging meta-optimization metrics
    """

    enabled: bool = False

    # Objective weights (LDC-MTL)
    ldc_mtl: LDCMTLConfig = field(default_factory=LDCMTLConfig)

    # Dataset weights (ODM/EXP3)
    odm: ODMConfig = field(default_factory=ODMConfig)

    # Logging
    log_interval: int = 100
