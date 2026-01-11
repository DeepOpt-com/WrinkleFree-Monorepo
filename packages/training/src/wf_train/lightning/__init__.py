"""PyTorch Lightning integration for WrinkleFree training.

Provides a clean, maintainable training loop with:
- Auto batch size scaling (BatchSizeFinder)
- Built-in DDP/FSDP support
- Checkpointing and logging
- Reusable ObjectiveManager integration
"""

from wf_train.lightning.callbacks import (
    DatasetRatioCallback,
    GCSAuthError,
    GCSCheckpointCallback,
    GCSUploadError,
    LambdaWarmupCallback,
    MuonClipInitCallback,
    QKClipCallback,
    RunManagerCallback,
    TokenCountCallback,
    ZClipCallback,
    validate_gcs_auth,
)
from wf_train.lightning.datamodule import WrinkleFreeDataModule
from wf_train.lightning.module import WrinkleFreeLightningModule

__all__ = [
    "WrinkleFreeLightningModule",
    "WrinkleFreeDataModule",
    "DatasetRatioCallback",
    "GCSAuthError",
    "GCSCheckpointCallback",
    "GCSUploadError",
    "LambdaWarmupCallback",
    "MuonClipInitCallback",
    "QKClipCallback",
    "RunManagerCallback",
    "TokenCountCallback",
    "ZClipCallback",
    "validate_gcs_auth",
]
