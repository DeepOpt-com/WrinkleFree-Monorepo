"""Fairy2 model wrapper for complex-valued quantized inference."""

from pathlib import Path
from typing import Any
import logging
import os
import sys

import torch

from wrinklefree_eval.models.hf_model import HuggingFaceModel

logger = logging.getLogger(__name__)


def find_fairy2_path() -> Path | None:
    """Find Fairy2 installation path.

    Checks:
    1. FAIRY2_PATH environment variable
    2. WrinkleFree-Fairy2 relative to this file
    """
    # Check environment variable
    if env_path := os.environ.get("FAIRY2_PATH"):
        path = Path(env_path)
        if path.exists():
            return path

    # Check relative to WrinkleFree project
    current = Path(__file__).resolve()
    for _ in range(5):  # Go up to 5 levels
        current = current.parent
        fairy2_path = current / "WrinkleFree-Fairy2"
        if fairy2_path.exists():
            return fairy2_path

    return None


def ensure_fairy2_installed() -> bool:
    """Ensure Fairy2 package is importable."""
    try:
        import fairy2
        return True
    except ImportError:
        pass

    # Try to add to path
    fairy2_path = find_fairy2_path()
    if fairy2_path:
        src_path = fairy2_path / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
            try:
                import fairy2
                logger.info(f"Added Fairy2 to path from {src_path}")
                return True
            except ImportError:
                pass

    return False


class Fairy2Model(HuggingFaceModel):
    """Fairy2-optimized model wrapper for complex-valued quantized models.

    Supports models trained with Fairy2i algorithm where weights are
    quantized to {+1, -1, +i, -i} (fourth roots of unity).

    Uses table lookup for multiplication-free inference when available.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int | str = "auto",
        fairy2_path: str | None = None,
        use_table_lookup: bool = True,
        fallback_to_hf: bool = True,
        **kwargs,
    ):
        """Initialize Fairy2 model wrapper.

        Args:
            model_path: Path to Fairy2 model checkpoint
            device: Device to run on
            dtype: Model dtype
            batch_size: Batch size for evaluation
            fairy2_path: Path to Fairy2 installation (auto-detected if None)
            use_table_lookup: Use multiplication-free table lookup inference
            fallback_to_hf: Fall back to HuggingFace if Fairy2 not available
            **kwargs: Additional arguments passed to parent
        """
        self._fairy2_path = Path(fairy2_path) if fairy2_path else find_fairy2_path()
        self._fairy2_available = ensure_fairy2_installed()
        self._using_table_lookup = False

        if self._fairy2_available and use_table_lookup:
            try:
                self._init_fairy2(model_path, device, dtype, batch_size, **kwargs)
                self._using_table_lookup = True
                logger.info("Using Fairy2 table lookup inference")
            except Exception as e:
                if fallback_to_hf:
                    logger.warning(f"Fairy2 init failed, falling back to HF: {e}")
                    self._load_fairy2_as_hf(model_path, device, dtype, batch_size, **kwargs)
                else:
                    raise RuntimeError(f"Fairy2 initialization failed: {e}") from e
        else:
            if fallback_to_hf:
                logger.info("Fairy2 table lookup not available, using HuggingFace")
                self._load_fairy2_as_hf(model_path, device, dtype, batch_size, **kwargs)
            else:
                raise RuntimeError(
                    "Fairy2 not found. Install fairy2 package or set fallback_to_hf=True"
                )

    def _load_fairy2_as_hf(
        self,
        model_path: str,
        device: str,
        dtype: str,
        batch_size: int | str,
        **kwargs,
    ):
        """Load Fairy2 model using standard HuggingFace inference.

        This loads the model with Fairy2Linear layers and runs inference
        using the standard PyTorch forward pass (not table lookup).
        """
        # Register Fairy2Linear for safe loading
        if self._fairy2_available:
            from fairy2.models.fairy2_linear import Fairy2Linear
            from fairy2.models.widely_linear import WidelyLinearComplex

            # These are automatically handled by HF when loading

        # Load with HuggingFace
        super().__init__(model_path, device, dtype, batch_size, **kwargs)
        logger.info(f"Loaded Fairy2 model from {model_path} using HuggingFace")

    def _init_fairy2(
        self,
        model_path: str,
        device: str,
        dtype: str,
        batch_size: int | str,
        **kwargs,
    ):
        """Initialize with Fairy2 table lookup inference.

        Table lookup inference uses precomputed lookup tables to avoid
        multiplication entirely during forward pass.
        """
        from fairy2.inference.table_lookup import TableLookupInference

        # Load model and convert to table lookup mode
        self._table_lookup = TableLookupInference.from_checkpoint(
            model_path,
            device=device,
        )
        self._batch_size = batch_size if isinstance(batch_size, int) else 1

        logger.info(f"Loaded Fairy2 model with table lookup inference from {model_path}")

    @property
    def using_table_lookup(self) -> bool:
        """Check if Fairy2 table lookup inference is being used."""
        return self._using_table_lookup


def create_fairy2_model(cfg) -> Fairy2Model:
    """Factory function to create Fairy2 model from Hydra config.

    Args:
        cfg: Hydra config with model settings

    Returns:
        Configured Fairy2Model instance
    """
    return Fairy2Model(
        model_path=cfg.model_path,
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "bfloat16"),
        batch_size=cfg.get("batch_size", "auto"),
        fairy2_path=cfg.model.get("fairy2_path"),
        use_table_lookup=cfg.model.get("use_table_lookup", True),
        fallback_to_hf=cfg.model.get("fallback_to_hf", True),
    )
