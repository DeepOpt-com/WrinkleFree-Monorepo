"""HuggingFace model download and conversion to GGUF format.

This module handles the full pipeline:
1. Download model from HuggingFace Hub
2. Convert to GGUF using BitNet's setup_env.py
3. Validate the output
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for HuggingFace to GGUF conversion."""

    hf_repo: str
    """HuggingFace repository ID (e.g., 'microsoft/BitNet-b1.58-2B-4T')"""

    output_dir: Path = field(default_factory=lambda: Path("models"))
    """Directory to store downloaded and converted models"""

    quant_type: str = "i2_s"
    """Quantization type: 'i2_s' (CPU optimized) or 'tl2' (AVX512)"""

    quant_embd: bool = False
    """Whether to quantize embeddings to f16"""

    use_pretuned: bool = True
    """Use pre-tuned kernel configuration if available"""


class HFToGGUFConverter:
    """
    Convert HuggingFace models to GGUF format using BitNet.

    This wrapper handles:
    - Downloading from HuggingFace Hub
    - Running BitNet's conversion pipeline
    - Validating output files

    Args:
        bitnet_path: Path to BitNet installation (extern/BitNet)
    """

    def __init__(self, bitnet_path: Optional[Path] = None):
        if bitnet_path is None:
            bitnet_path = self._get_default_bitnet_path()
        self.bitnet_path = Path(bitnet_path)

        if not self.bitnet_path.exists():
            raise FileNotFoundError(
                f"BitNet not found at {self.bitnet_path}. "
                "Run 'git submodule update --init' to initialize."
            )

    @staticmethod
    def _get_default_bitnet_path() -> Path:
        """Get default BitNet path relative to this package."""
        return Path(__file__).parent.parent.parent.parent / "extern" / "BitNet"

    def convert(
        self,
        config: ConversionConfig,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        Full conversion pipeline: download from HF, convert to GGUF.

        Args:
            config: Conversion configuration
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the converted GGUF model file
        """
        def log_progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # Step 1: Download model from HuggingFace
        log_progress(f"Downloading {config.hf_repo} from HuggingFace...")
        model_name = config.hf_repo.split("/")[-1]

        # Step 2: Run BitNet setup_env.py to download and convert
        log_progress(f"Converting to GGUF with {config.quant_type} quantization...")
        gguf_path = self._run_bitnet_setup(config, log_progress)

        # Step 3: Validate output
        log_progress("Validating converted model...")
        self._validate_gguf(gguf_path)

        log_progress(f"Conversion complete: {gguf_path}")
        return gguf_path

    def _run_bitnet_setup(
        self,
        config: ConversionConfig,
        log_progress: Callable[[str], None],
    ) -> Path:
        """Run BitNet's setup_env.py to download and convert model."""
        setup_script = self.bitnet_path / "setup_env.py"

        if not setup_script.exists():
            raise FileNotFoundError(f"BitNet setup script not found: {setup_script}")

        cmd = [
            sys.executable,
            str(setup_script),
            "--hf-repo", config.hf_repo,
            "-q", config.quant_type,
        ]

        if config.quant_embd:
            cmd.append("--quant-embd")

        if config.use_pretuned:
            cmd.append("--use-pretuned")

        log_progress(f"Running: {' '.join(cmd)}")

        # Run conversion
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Fast downloads

        result = subprocess.run(
            cmd,
            cwd=str(self.bitnet_path),
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Conversion failed:\n{result.stderr}")
            raise RuntimeError(f"BitNet conversion failed: {result.stderr}")

        # Find output file
        model_name = config.hf_repo.split("/")[-1]
        gguf_path = (
            self.bitnet_path / "models" / model_name /
            f"ggml-model-{config.quant_type}.gguf"
        )

        if not gguf_path.exists():
            # Try alternative naming
            alt_path = self.bitnet_path / "models" / model_name / "ggml-model.gguf"
            if alt_path.exists():
                gguf_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Expected GGUF file not found at {gguf_path}"
                )

        return gguf_path

    def _validate_gguf(self, gguf_path: Path) -> None:
        """Validate the converted GGUF file."""
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        # Check file size (should be non-trivial)
        size_mb = gguf_path.stat().st_size / (1024 * 1024)
        if size_mb < 10:
            raise ValueError(f"GGUF file too small ({size_mb:.1f} MB), conversion may have failed")

        # Check magic bytes
        with open(gguf_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"Invalid GGUF magic bytes: {magic}")

        logger.info(f"GGUF validation passed: {gguf_path} ({size_mb:.1f} MB)")

    def download_only(self, hf_repo: str, output_dir: Optional[Path] = None) -> Path:
        """
        Download model from HuggingFace without conversion.

        Args:
            hf_repo: HuggingFace repository ID
            output_dir: Directory to save model (default: models/<repo_name>)

        Returns:
            Path to downloaded model directory
        """
        model_name = hf_repo.split("/")[-1]
        if output_dir is None:
            output_dir = self.bitnet_path / "models" / model_name

        logger.info(f"Downloading {hf_repo} to {output_dir}")

        local_dir = snapshot_download(
            repo_id=hf_repo,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )

        return Path(local_dir)

    def list_available_models(self) -> list[Path]:
        """List GGUF models available in the models directory."""
        models_dir = self.bitnet_path / "models"
        if not models_dir.exists():
            return []

        gguf_files = list(models_dir.glob("**/ggml-model*.gguf"))
        return sorted(gguf_files)


def convert_hf_model(
    hf_repo: str,
    quant_type: str = "i2_s",
    bitnet_path: Optional[Path] = None,
) -> Path:
    """
    High-level function to convert a HuggingFace model to GGUF.

    Args:
        hf_repo: HuggingFace repository (e.g., 'microsoft/BitNet-b1.58-2B-4T')
        quant_type: Quantization type ('i2_s' for CPU, 'tl2' for AVX512)
        bitnet_path: Path to BitNet installation

    Returns:
        Path to converted GGUF file

    Example:
        >>> gguf_path = convert_hf_model("microsoft/BitNet-b1.58-2B-4T")
        >>> print(f"Model ready at: {gguf_path}")
    """
    converter = HFToGGUFConverter(bitnet_path)
    config = ConversionConfig(
        hf_repo=hf_repo,
        quant_type=quant_type,
    )
    return converter.convert(config)
