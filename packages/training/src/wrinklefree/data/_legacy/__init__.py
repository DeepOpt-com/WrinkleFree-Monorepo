"""DEPRECATED: Legacy data loading implementations.

These modules are DEPRECATED and will be removed in a future release.
Please install cheapertraining for the supported data loading interface:

    pip install -e ../WrinkleFree-CheaperTraining

Or with uv:

    uv add -e ../WrinkleFree-CheaperTraining
"""

import warnings

_DEPRECATION_WARNING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ⚠️  DEPRECATED: You are using legacy data loading code!                     ║
║                                                                              ║
║   This code is DEPRECATED and will be removed in a future release.           ║
║   Please install cheapertraining for the supported interface:                ║
║                                                                              ║
║       pip install -e ../WrinkleFree-CheaperTraining                          ║
║                                                                              ║
║   Or from the WrinkleFree root:                                              ║
║                                                                              ║
║       pip install -e ./WrinkleFree-CheaperTraining                           ║
║                                                                              ║
║   Benefits of cheapertraining:                                               ║
║   • Unified data loading interface                                           ║
║   • Influence-based dataset remixing (InfluenceTracker)                      ║
║   • Multi-domain probe support                                               ║
║   • Shared packing utilities                                                 ║
║   • Canonical dataset configs                                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

_warning_shown = False


def _show_deprecation_warning():
    """Show deprecation warning once per session."""
    global _warning_shown
    if not _warning_shown:
        warnings.warn(_DEPRECATION_WARNING, DeprecationWarning, stacklevel=3)
        _warning_shown = True


# Re-export with deprecation warnings
from wrinklefree.data._legacy.pretrain_dataset import (
    PretrainDataset,
    PackedPretrainDataset,
    StreamingPretrainDataset,
    create_pretrain_dataloader,
    create_mixed_dataloader,
)

from wrinklefree.data._legacy.mixed_dataset import (
    MixedPretrainDataset,
    create_probe_dataloader,
)

__all__ = [
    "PretrainDataset",
    "PackedPretrainDataset",
    "StreamingPretrainDataset",
    "MixedPretrainDataset",
    "create_pretrain_dataloader",
    "create_mixed_dataloader",
    "create_probe_dataloader",
    "_show_deprecation_warning",
]
