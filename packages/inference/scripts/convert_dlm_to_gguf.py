#!/usr/bin/env python3
"""
Convert DLM BitNet checkpoints to GGUF format for native inference.

Usage:
    # Basic conversion with TL2 (recommended for AVX512)
    python convert_dlm_to_gguf.py /path/to/checkpoint -o model.gguf

    # With TL1 quantization (older CPUs)
    python convert_dlm_to_gguf.py /path/to/checkpoint -o model.gguf --quant tl1

    # From GCS
    python convert_dlm_to_gguf.py gs://bucket/checkpoint -o model.gguf
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.resolve()
INFERENCE_DIR = SCRIPT_DIR.parent
BITNET_DIR = INFERENCE_DIR / "extern" / "BitNet"
CONVERTER_SCRIPT = BITNET_DIR / "utils" / "convert-hf-to-gguf-bitnet.py"


def download_from_gcs(gcs_path: str, local_dir: Path) -> Path:
    """Download checkpoint from GCS to local directory."""
    logger.info(f"Downloading from GCS: {gcs_path}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Use gsutil to download
    cmd = ["gsutil", "-m", "cp", "-r", gcs_path.rstrip("/") + "/*", str(local_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download from GCS: {result.stderr}")

    return local_dir


def validate_checkpoint(checkpoint_path: Path) -> dict:
    """Validate checkpoint and return config."""
    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_path}")

    with open(config_file) as f:
        config = json.load(f)

    # Check for required fields
    model_type = config.get("model_type", "").lower()
    if "bitnet" not in model_type:
        logger.warning(f"Model type '{model_type}' may not be supported")

    # Check for DLM-specific fields
    if "bd_size" in config:
        logger.info(f"Detected DLM model with block size: {config['bd_size']}")

    # Check for required model files
    safetensors = checkpoint_path / "model.safetensors"
    pytorch_bin = checkpoint_path / "pytorch_model.bin"
    if not safetensors.exists() and not pytorch_bin.exists():
        raise FileNotFoundError(f"No model weights found in {checkpoint_path}")

    return config


def generate_kernel_config(config: dict, quant_type: str, output_dir: Path):
    """Generate kernel_config.ini for weight preprocessing."""
    # The conversion script needs kernel_config.ini in the include/ directory
    # This defines the tiling parameters for TL1/TL2 kernels

    hidden_size = config.get("hidden_size", 2560)
    intermediate_size = config.get("intermediate_size", 6912)
    num_heads = config.get("num_attention_heads", 20)
    num_kv_heads = config.get("num_key_value_heads", 5)
    head_dim = config.get("head_dim", hidden_size // num_heads)

    # Common weight dimensions for BitNet 2B
    weight_dims = [
        # Q, K, V projections
        (hidden_size, hidden_size),  # Q
        (hidden_size, num_kv_heads * head_dim),  # K
        (hidden_size, num_kv_heads * head_dim),  # V
        (hidden_size, hidden_size),  # O
        # FFN
        (intermediate_size, hidden_size),  # gate, up
        (hidden_size, intermediate_size),  # down
    ]

    include_dir = output_dir / "include"
    include_dir.mkdir(parents=True, exist_ok=True)

    # Write kernel config
    config_file = include_dir / "kernel_config.ini"
    with open(config_file, "w") as f:
        for i, (m, k) in enumerate(weight_dims):
            if quant_type == "tl2":
                # TL2 kernel parameters (AVX512 optimized)
                bm = 32 if m >= 2048 else 16
                bk = 192 if k >= 2048 else 96
                bmm = 32
            else:
                # TL1 kernel parameters (AVX2)
                bm = 32 if m >= 2048 else 16
                bk = 256 if k >= 2048 else 128
                bmm = 32

            f.write(f"[kernel_{i}]\n")
            f.write(f"m = {m}\n")
            f.write(f"k = {k}\n")
            f.write(f"bm = {bm}\n")
            f.write(f"bk = {bk}\n")
            f.write(f"bmm = {bmm}\n")
            f.write("\n")

    logger.info(f"Generated kernel config at {config_file}")
    return include_dir


def convert_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    quant_type: str = "tl2",
    verbose: bool = False,
) -> Path:
    """Convert checkpoint to GGUF format."""

    # Validate checkpoint
    config = validate_checkpoint(checkpoint_path)
    logger.info(f"Converting {checkpoint_path.name} to GGUF ({quant_type})")

    # Generate kernel config in a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate kernel config
        include_dir = generate_kernel_config(config, quant_type, tmpdir)

        # Build conversion command
        cmd = [
            sys.executable,
            str(CONVERTER_SCRIPT),
            str(checkpoint_path),
            "--outfile", str(output_path),
            "--outtype", quant_type,
        ]

        if verbose:
            cmd.append("--verbose")

        # Set environment to find kernel config
        env = os.environ.copy()
        env["BITNET_INCLUDE_DIR"] = str(include_dir.parent)

        # Run conversion with kernel config in include/
        logger.info(f"Running: {' '.join(cmd)}")

        # Change to temp directory where include/ is located
        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Conversion failed:\n{result.stderr}")
            raise RuntimeError("GGUF conversion failed")

        if verbose:
            print(result.stdout)

    logger.info(f"Successfully converted to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert DLM BitNet checkpoints to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert local checkpoint
  python convert_dlm_to_gguf.py ./checkpoint -o model.gguf

  # Convert from GCS with TL1 quantization
  python convert_dlm_to_gguf.py gs://bucket/checkpoint -o model.gguf --quant tl1
        """,
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint directory (local or gs://...)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["tl1", "tl2", "f16", "f32"],
        default="tl2",
        help="Quantization type (default: tl2 for AVX512)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Handle GCS paths
    if args.checkpoint.startswith("gs://"):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = download_from_gcs(args.checkpoint, Path(tmpdir) / "checkpoint")
            convert_checkpoint(local_path, args.output, args.quant, args.verbose)
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        convert_checkpoint(checkpoint_path, args.output, args.quant, args.verbose)


if __name__ == "__main__":
    main()
