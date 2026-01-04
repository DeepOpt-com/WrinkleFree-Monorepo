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
from typing import Literal, Optional


@dataclass
class LDCMTLConfig:
    """LDC-MTL configuration for objective weight optimization.

    LDC-MTL (Loss Discrepancy Control for Multi-Task Learning) uses a small
    router network to learn optimal task weights with O(1) complexity.

    Attributes:
        enabled: Whether to enable LDC-MTL objective weighting
        lambda_penalty: Weight for the discrepancy penalty (higher = more balanced)
        hidden_dim: Hidden layer dimension for the router MLP
        router_lr: Learning rate for the router optimizer
    """

    enabled: bool = True
    lambda_penalty: float = 0.1
    hidden_dim: int = 32
    router_lr: float = 1e-3


@dataclass
class ODMConfig:
    """ODM (EXP3) configuration for dataset weight optimization.

    ODM (Online Data Mixing) uses the EXP3 multi-armed bandit algorithm
    to learn optimal dataset sampling probabilities during training.

    Attributes:
        enabled: Whether to enable ODM dataset weighting
        reward_smoothing: EMA coefficient for reward updates (0-1, higher = more smoothing)
        warmup_ratio: Fraction of training to use uniform weights (0-1)
        min_weight: Minimum allowed sampling probability per dataset
        max_weight: Maximum allowed sampling probability per dataset
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
