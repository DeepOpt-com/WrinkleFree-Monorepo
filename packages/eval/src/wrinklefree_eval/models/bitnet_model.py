"""BitNet model wrapper for optimized ternary inference."""

from pathlib import Path
from typing import Any
import logging
import os

import torch

from wrinklefree_eval.models.hf_model import HuggingFaceModel

logger = logging.getLogger(__name__)


def find_bitnet_path() -> Path | None:
    """Find BitNet installation path.

    Checks:
    1. BITNET_PATH environment variable
    2. WrinkleFree-1.58Quant/extern/BitNet relative to this file
    3. Common installation locations
    """
    # Check environment variable
    if env_path := os.environ.get("BITNET_PATH"):
        path = Path(env_path)
        if path.exists():
            return path

    # Check relative to WrinkleFree project
    current = Path(__file__).resolve()
    for _ in range(5):  # Go up to 5 levels
        current = current.parent
        bitnet_path = current / "WrinkleFree-1.58Quant" / "extern" / "BitNet"
        if bitnet_path.exists():
            return bitnet_path

    # Check common locations
    common_paths = [
        Path.home() / "BitNet",
        Path("/opt/BitNet"),
    ]
    for path in common_paths:
        if path.exists():
            return path

    return None


def check_bitnet_available() -> bool:
    """Check if BitNet kernels are available and compiled."""
    bitnet_path = find_bitnet_path()
    if not bitnet_path:
        return False

    # Check for compiled kernels
    kernel_indicators = [
        bitnet_path / "build",
        bitnet_path / "bitnet.so",
        bitnet_path / "3rdparty" / "llama.cpp" / "build",
    ]
    return any(p.exists() for p in kernel_indicators)


class BitNetModel(HuggingFaceModel):
    """BitNet-optimized model wrapper with ternary kernel support.

    Uses optimized BitNet inference kernels when available, falling back
    to standard HuggingFace inference otherwise.

    BitNet kernels provide significant speedup for ternary (1.58-bit) models
    by using specialized CUDA kernels for {-1, 0, 1} weight operations.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int | str = "auto",
        bitnet_path: str | None = None,
        fallback_to_hf: bool = True,
        **kwargs,
    ):
        """Initialize BitNet model wrapper.

        Args:
            model_path: Path to BitNet model checkpoint
            device: Device to run on
            dtype: Model dtype
            batch_size: Batch size for evaluation
            bitnet_path: Path to BitNet installation (auto-detected if None)
            fallback_to_hf: Fall back to HuggingFace if BitNet not available
            **kwargs: Additional arguments passed to parent
        """
        self._bitnet_path = Path(bitnet_path) if bitnet_path else find_bitnet_path()
        self._bitnet_available = check_bitnet_available()
        self._using_bitnet_kernels = False

        if self._bitnet_available:
            try:
                self._init_bitnet(model_path, device, dtype, batch_size, **kwargs)
                self._using_bitnet_kernels = True
                logger.info("Using BitNet optimized kernels for inference")
            except Exception as e:
                if fallback_to_hf:
                    logger.warning(f"BitNet init failed, falling back to HF: {e}")
                    super().__init__(model_path, device, dtype, batch_size, **kwargs)
                else:
                    raise RuntimeError(f"BitNet initialization failed: {e}") from e
        else:
            if fallback_to_hf:
                logger.info("BitNet kernels not available, using HuggingFace")
                super().__init__(model_path, device, dtype, batch_size, **kwargs)
            else:
                raise RuntimeError(
                    "BitNet kernels not found. Install BitNet or set fallback_to_hf=True"
                )

    def _init_bitnet(
        self,
        model_path: str,
        device: str,
        dtype: str,
        batch_size: int | str,
        **kwargs,
    ):
        """Initialize with BitNet.cpp inference server.

        This method connects to a running BitNet.cpp server for
        optimized ternary weight computation.

        The server can be started via WrinkleFree-Inference-Engine:
            uv run wrinklefree-inference serve -m model.gguf --port 8080
        """
        import os

        # Check for inference server URL
        inference_url = os.environ.get("INFERENCE_URL", "http://localhost:8080")

        try:
            # Import client from WrinkleFree-Inference-Engine
            # First try installed package
            try:
                from wrinklefree_inference.client import BitNetClient
            except ImportError:
                # Fall back to relative import from meta-repo
                import sys
                inference_engine_path = self._bitnet_path.parent.parent / "WrinkleFree-Inference-Engine"
                if inference_engine_path.exists():
                    sys.path.insert(0, str(inference_engine_path / "src"))
                    from wrinklefree_inference.client import BitNetClient
                else:
                    raise ImportError("WrinkleFree-Inference-Engine not found")

            self._client = BitNetClient.from_url(inference_url)

            # Check server health
            if not self._client.health_check():
                raise ConnectionError(
                    f"BitNet inference server not responding at {inference_url}. "
                    "Start the server with: uv run wrinklefree-inference serve -m model.gguf"
                )

            logger.info(f"Connected to BitNet inference server at {inference_url}")

            # Store for generation
            self._inference_url = inference_url
            self._batch_size = batch_size if isinstance(batch_size, int) else 1

        except Exception as e:
            raise RuntimeError(
                f"BitNet kernel integration failed: {e}\n"
                "Make sure the inference server is running at INFERENCE_URL or localhost:8080"
            ) from e

    @property
    def using_bitnet_kernels(self) -> bool:
        """Check if BitNet optimized kernels are being used."""
        return self._using_bitnet_kernels


def create_bitnet_model(cfg) -> BitNetModel:
    """Factory function to create BitNet model from Hydra config.

    Args:
        cfg: Hydra config with model settings

    Returns:
        Configured BitNetModel instance
    """
    return BitNetModel(
        model_path=cfg.model_path,
        device=cfg.get("device", "cuda"),
        dtype=cfg.get("dtype", "bfloat16"),
        batch_size=cfg.get("batch_size", "auto"),
        bitnet_path=cfg.model.get("bitnet_path"),
        fallback_to_hf=cfg.model.get("fallback_to_hf", True),
    )
