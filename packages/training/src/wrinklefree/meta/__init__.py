"""Efficient meta-optimization for PyTorch Lightning trainer.

This module provides O(1) complexity meta-optimization using three complementary
methods that can be enabled independently or together:

1. **LDC-MTL** (Loss Discrepancy Control for Multi-Task Learning):
   - Optimizes objective weights (e.g., CE vs DLM vs distillation)
   - Uses a small router MLP to learn task weights dynamically
   - O(1) complexity via single-level optimization with discrepancy penalty

2. **ODM/EXP3** (Online Data Mixing via multi-armed bandit):
   - Optimizes dataset sampling probabilities (e.g., web vs code vs math)
   - Uses EXP3 algorithm with training loss as reward signal
   - ~0% wall-clock overhead, published 19% iteration reduction

3. **LayerLR** (Per-layer learning rate optimization):
   - Learns per-layer LR multipliers based on gradient norms
   - Bidirectional: increases LR when grads low, decreases when high
   - Inspired by LARS but learned dynamically

All methods are efficient, principled, and require no external dependencies.

Example usage:
    ```python
    from wrinklefree.meta import MetaOptimizationConfig, MetaOptimizerCallback

    config = MetaOptimizationConfig(
        enabled=True,
        ldc_mtl=LDCMTLConfig(enabled=True, lambda_penalty=0.1),
        odm=ODMConfig(enabled=True, warmup_ratio=0.01),
        layer_lr=LayerLRConfig(enabled=True),
    )
    trainer = pl.Trainer(callbacks=[MetaOptimizerCallback(config)])
    ```

References:
    - LDC-MTL: https://arxiv.org/abs/2502.08585
    - ODM: https://arxiv.org/abs/2312.02406
    - LayerLR: Inspired by LARS (https://arxiv.org/abs/1708.03888)
"""

from wrinklefree.meta.config import (
    LayerLRConfig,
    LDCMTLConfig,
    MetaOptimizationConfig,
    ODMConfig,
)
from wrinklefree.meta.ldc_mtl import (
    LDCMTLManager,
    ObjectiveRouter,
    compute_loss_discrepancy,
)
from wrinklefree.meta.layer_lr import LayerLRManager
from wrinklefree.meta.odm import OnlineDataMixer
from wrinklefree.meta.callback import MetaOptimizerCallback

__all__ = [
    # Config classes
    "MetaOptimizationConfig",
    "LDCMTLConfig",
    "ODMConfig",
    "LayerLRConfig",
    # LDC-MTL components
    "LDCMTLManager",
    "ObjectiveRouter",
    "compute_loss_discrepancy",
    # ODM components
    "OnlineDataMixer",
    # LayerLR components
    "LayerLRManager",
    # Callback
    "MetaOptimizerCallback",
]
