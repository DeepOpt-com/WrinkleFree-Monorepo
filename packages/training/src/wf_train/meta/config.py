"""Configuration for efficient meta-optimization.

This module provides configuration classes for three complementary methods:
1. LDC-MTL: Objective weight optimization (CE vs DLM vs distillation)
2. ODM: Dataset weight optimization (web vs code vs math)
3. LayerLR: Per-layer learning rate optimization

All methods are O(1) complexity and require no external dependencies.

References:
- LDC-MTL: https://arxiv.org/abs/2502.08585
- ODM: https://arxiv.org/abs/2312.02406
- LayerLR: Inspired by LARS (https://arxiv.org/abs/1708.03888)
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
class LayerLRConfig:
    """Per-layer learning rate optimization config.

    Learns per-layer LR multipliers via direct gradient descent on
    log-scale parameters. Uses gradient norms as signal for logging
    and optional penalty terms.

    Inspired by LARS (Layer-wise Adaptive Rate Scaling) but learned
    dynamically rather than using a fixed formula.

    Reference:
        https://arxiv.org/abs/1708.03888 (LARS)

    Attributes:
        enabled: Whether to enable per-layer LR optimization.
        lr: Learning rate for the multiplier optimizer.
        min_multiplier: Minimum LR multiplier (prevents collapse).
        max_multiplier: Maximum LR multiplier (prevents explosion).
        ema_decay: EMA decay for gradient norm smoothing.
        lambda_mean: Penalty weight for mean deviation from 1.0.
            Keeps geometric mean of multipliers near 1.0.
        warmup_ratio: Fraction of training with multipliers=1.0.
            Skips adaptation during LR warmup when grad stats unreliable.
        step_interval: Update multipliers every N optimizer steps.
    """

    enabled: bool = False
    lr: float = 1e-3
    min_multiplier: float = 0.1
    max_multiplier: float = 10.0
    ema_decay: float = 0.99
    lambda_mean: float = 0.1
    warmup_ratio: float = 0.05
    step_interval: int = 1


@dataclass
class MetaOptimizationConfig:
    """Unified configuration for meta-optimization.

    Combines LDC-MTL (objective weights), ODM (dataset weights), and
    LayerLR (per-layer learning rates) into a single configuration.
    All can be enabled independently.

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
          layer_lr:
            enabled: false
            lr: 0.001
            min_multiplier: 0.1
            max_multiplier: 10.0
          log_interval: 100

    Attributes:
        enabled: Master switch for meta-optimization
        ldc_mtl: LDC-MTL config for objective weights
        odm: ODM config for dataset weights
        layer_lr: LayerLR config for per-layer learning rates
        log_interval: Steps between logging meta-optimization metrics
    """

    enabled: bool = False

    # Objective weights (LDC-MTL)
    ldc_mtl: LDCMTLConfig = field(default_factory=LDCMTLConfig)

    # Dataset weights (ODM/EXP3)
    odm: ODMConfig = field(default_factory=ODMConfig)

    # Per-layer learning rates
    layer_lr: LayerLRConfig = field(default_factory=LayerLRConfig)

    # Logging
    log_interval: int = 100
