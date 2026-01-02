"""PyTorch Lightning integration for WrinkleFree training.

Provides a clean, maintainable training loop with:
- Auto batch size scaling (BatchSizeFinder)
- Built-in DDP/FSDP support
- Checkpointing and logging
- Reusable ObjectiveManager integration
"""

from wrinklefree.lightning.callbacks import (
    GCSCheckpointCallback,
    InfluenceTrackerCallback,
    LambdaWarmupCallback,
    QKClipCallback,
    TokenCountCallback,
    ZClipCallback,
)
from wrinklefree.lightning.datamodule import WrinkleFreeDataModule
from wrinklefree.lightning.module import WrinkleFreeLightningModule

__all__ = [
    "WrinkleFreeLightningModule",
    "WrinkleFreeDataModule",
    "GCSCheckpointCallback",
    "InfluenceTrackerCallback",
    "LambdaWarmupCallback",
    "QKClipCallback",
    "TokenCountCallback",
    "ZClipCallback",
]
