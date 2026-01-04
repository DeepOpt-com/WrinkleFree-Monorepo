#!/usr/bin/env python3
"""
Unified GGUF conversion script for DLM/BitNet checkpoints.

This script handles:
- bf16 "online" quantization checkpoints (weights are continuous floats)
- Packed 2-bit "offline" quantization checkpoints (4 values per byte)
- Both sentencepiece and GPT2/BPE tokenizers
- Architecture name variants (BitnetForCausalLM vs BitNetForCausalLM)

Supported output formats:
- i2_s: 2-bit integer, multiply-add (default, works with vanilla llama.cpp)
- tq2_0: Ternary quantization (works with vanilla llama.cpp)
- tl1: LUT-based 1 (requires pre-generated kernel config)
- tl2: LUT-based 2 (requires pre-generated kernel config)
- f32/f16: Reference only (DO NOT USE for inference - 4x larger, slower)

Usage:
    # Basic conversion (I2_S recommended for vanilla llama.cpp)
    python convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf

    # With validation
    python convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf --validate

    # TQ2_0 format (alternative to I2_S)
    python convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf --outtype tq2_0

    # From GCS (auto-downloads, excludes optimizer state)
    python convert_checkpoint_to_gguf.py gs://bucket/checkpoint --outfile model.gguf
"""

import argparse
import json
import logging
import os
import re
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

# Get the repository root (where extern/ is located)
SCRIPT_DIR = Path(__file__).parent.resolve()
INFERENCE_PKG = SCRIPT_DIR.parent
REPO_ROOT = INFERENCE_PKG.parent.parent
BITNET_DIR = REPO_ROOT / "extern" / "BitNet"
CONVERTER_SCRIPT = BITNET_DIR / "utils" / "convert-hf-to-gguf-bitnet.py"

# Model size expectations (approximate, for validation)
MODEL_SIZE_EXPECTATIONS = {
    # (num_params, expected_i2s_size_bytes)
    "2B": (2_000_000_000, 1_100_000_000),  # ~1.1GB
    "135M": (135_000_000, 80_000_000),      # ~80MB
}


def download_from_gcs(gcs_path: str, local_dir: Path) -> Path:
    """Download checkpoint from GCS, excluding optimizer state."""
    logger.info(f"Downloading from GCS: {gcs_path}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download only essential files (skip optimizer_state*)
    patterns = ["*.json", "*.safetensors", "*.jinja", "tokenizer*"]
    for pattern in patterns:
        cmd = [
            "gcloud", "storage", "cp",
            f"{gcs_path.rstrip('/')}/{pattern}",
            str(local_dir) + "/",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Some patterns may not match, that's OK
        if result.returncode != 0 and "No URLs matched" not in result.stderr:
            logger.warning(f"Pattern {pattern}: {result.stderr.strip()}")

    # Verify we got the essentials
    if not (local_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not downloaded from {gcs_path}")

    return local_dir


def fix_architecture_name(checkpoint_path: Path) -> bool:
    """Fix BitNetForCausalLM -> BitnetForCausalLM if needed.

    WrinkleFree training uses BitNetForCausalLM (capital N), but llama.cpp
    expects BitnetForCausalLM (lowercase n). This is a common source of
    conversion failures.
    """
    config_file = checkpoint_path / "config.json"
    with open(config_file) as f:
        content = f.read()

    if "BitNetForCausalLM" in content:
        logger.warning("=" * 60)
        logger.warning("AUTO-FIX: Architecture name correction applied!")
        logger.warning("  Before: BitNetForCausalLM (WrinkleFree training format)")
        logger.warning("  After:  BitnetForCausalLM (llama.cpp expected format)")
        logger.warning("=" * 60)
        content = content.replace("BitNetForCausalLM", "BitnetForCausalLM")
        with open(config_file, "w") as f:
            f.write(content)
        return True
    return False


def validate_checkpoint(checkpoint_path: Path) -> dict:
    """Validate checkpoint and return config."""
    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_path}")

    with open(config_file) as f:
        config = json.load(f)

    # Check model type
    arch = config.get("architectures", [""])[0].lower()
    if "bitnet" not in arch:
        logger.warning(f"Architecture '{arch}' may not be a BitNet model")

    # Check for weight files
    safetensors = list(checkpoint_path.glob("*.safetensors"))
    pytorch_bins = list(checkpoint_path.glob("*.bin"))
    if not safetensors and not pytorch_bins:
        raise FileNotFoundError(f"No model weights found in {checkpoint_path}")

    # Detect quantization mode
    quant_config = config.get("quantization_config", {})
    quant_mode = quant_config.get("quantization_mode", "online")
    logger.info(f"Detected quantization mode: {quant_mode}")

    return config


def estimate_model_params(config: dict) -> int:
    """Estimate number of parameters from config."""
    hidden = config.get("hidden_size", 2560)
    layers = config.get("num_hidden_layers", 30)
    intermediate = config.get("intermediate_size", 6912)
    vocab = config.get("vocab_size", 128000)

    # Rough estimate: embeddings + attention + FFN
    embed_params = vocab * hidden * 2  # input + output
    attn_params = layers * (4 * hidden * hidden)  # Q, K, V, O
    ffn_params = layers * (3 * hidden * intermediate)  # gate, up, down

    return embed_params + attn_params + ffn_params


def generate_kernel_config(config: dict, quant_type: str, output_dir: Path) -> Path:
    """Generate kernel_config.ini for TL1/TL2 weight preprocessing."""
    if quant_type not in ("tl1", "tl2"):
        return output_dir  # No config needed for i2_s

    hidden_size = config.get("hidden_size", 2560)
    intermediate_size = config.get("intermediate_size", 6912)
    num_heads = config.get("num_attention_heads", 20)
    num_kv_heads = config.get("num_key_value_heads", 5)
    head_dim = config.get("head_dim", hidden_size // num_heads)
    num_layers = config.get("num_hidden_layers", 30)

    # All unique weight dimensions
    weight_dims = set()
    # Attention projections
    weight_dims.add((hidden_size, hidden_size))  # Q, O
    weight_dims.add((hidden_size, num_kv_heads * head_dim))  # K, V
    # FFN
    weight_dims.add((intermediate_size, hidden_size))  # gate, up
    weight_dims.add((hidden_size, intermediate_size))  # down

    include_dir = output_dir / "include"
    include_dir.mkdir(parents=True, exist_ok=True)

    config_file = include_dir / "kernel_config.ini"
    with open(config_file, "w") as f:
        for i, (m, k) in enumerate(sorted(weight_dims)):
            if quant_type == "tl2":
                bm = 32 if m >= 2048 else 16
                bk = 192 if k >= 2048 else 96
            else:  # tl1
                bm = 32 if m >= 2048 else 16
                bk = 256 if k >= 2048 else 128
            bmm = 32
            by = 256 // bmm

            f.write(f"[{m}_{k}]\n")
            f.write(f"m = {m}\n")
            f.write(f"k = {k}\n")
            f.write(f"bm = {bm}\n")
            f.write(f"bk = {bk}\n")
            f.write(f"bmm = {bmm}\n\n")

    logger.info(f"Generated kernel config: {config_file}")
    return output_dir


def validate_gguf_output(
    output_path: Path,
    config: dict,
    quant_type: str,
) -> bool:
    """Validate the converted GGUF file."""
    if not output_path.exists():
        logger.error(f"Output file not created: {output_path}")
        return False

    file_size = output_path.stat().st_size
    estimated_params = estimate_model_params(config)

    # Expected sizes based on quantization type
    if quant_type in ("i2_s", "tl1", "tl2", "tq2_0"):
        # ~0.5-0.6 bytes per param for 2-bit quantization
        expected_min = estimated_params * 0.4
        expected_max = estimated_params * 0.8
    else:  # f16
        expected_min = estimated_params * 1.5
        expected_max = estimated_params * 2.5

    if file_size < expected_min:
        logger.error(
            f"File too small: {file_size / 1e9:.2f}GB "
            f"(expected >= {expected_min / 1e9:.2f}GB for {quant_type})"
        )
        return False

    if file_size > expected_max:
        logger.warning(
            f"File larger than expected: {file_size / 1e9:.2f}GB "
            f"(expected <= {expected_max / 1e9:.2f}GB for {quant_type})"
        )

    logger.info(f"Output validation passed: {file_size / 1e9:.2f}GB")
    return True


def convert_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    quant_type: str = "i2_s",
    validate: bool = False,
    verbose: bool = False,
) -> Path:
    """Convert checkpoint to GGUF format."""
    # Validate checkpoint
    config = validate_checkpoint(checkpoint_path)
    logger.info(f"Converting {checkpoint_path.name} to GGUF ({quant_type})")

    # Fix architecture name if needed
    fix_architecture_name(checkpoint_path)

    # Generate kernel config if needed
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        work_dir = generate_kernel_config(config, quant_type, tmpdir)

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

        # Run conversion from work directory (for kernel config)
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Conversion failed:\n{result.stderr}")
            if "sentencepiece" in result.stderr.lower():
                logger.error(
                    "HINT: This may be a tokenizer issue. "
                    "The converter now supports GPT2/BPE tokenizers."
                )
            if "BitnetForCausalLM" in result.stderr:
                logger.error(
                    "HINT: Try fixing the architecture name with:\n"
                    f"  sed -i 's/BitNetForCausalLM/BitnetForCausalLM/g' {checkpoint_path}/config.json"
                )
            raise RuntimeError("GGUF conversion failed")

        if verbose:
            print(result.stdout)

    # Validate output if requested
    if validate:
        if not validate_gguf_output(output_path, config, quant_type):
            raise RuntimeError("Output validation failed")

    logger.info(f"Successfully converted to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1e9:.2f}GB")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert DLM/BitNet checkpoints to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Format Guide:
  i2_s   - 2-bit integer, multiply-add (~1.1GB for 2B model) [DEFAULT, RECOMMENDED]
           Works with vanilla llama.cpp, good balance of speed and compatibility
  tq1_0  - Ternary quantization v1 (~680MB for 2B model)
           Good for bf16 DLM checkpoints, faster than i2_s
  tq2_0  - Ternary quantization v2 (~780MB for 2B model)
           WARNING: Produces garbage for bf16 checkpoints! Use --force-tq2 to override
  tl1    - LUT-based format 1 (~1.1GB)
           Requires pre-generated kernel config, faster on some CPUs
  tl2    - LUT-based format 2 (~1.1GB)
           Requires pre-generated kernel config, optimized for AVX512
  f16    - Float16 (~4.5GB for 2B model)
           DO NOT USE for inference - for reference/debugging only

Examples:
  # Basic conversion with I2_S (recommended)
  python convert_checkpoint_to_gguf.py ./checkpoint --outfile model.gguf

  # With validation
  python convert_checkpoint_to_gguf.py ./checkpoint --outfile model.gguf --validate

  # From GCS (auto-downloads, excludes optimizer state)
  python convert_checkpoint_to_gguf.py gs://bucket/checkpoint --outfile model.gguf

  # TQ2_0 format
  python convert_checkpoint_to_gguf.py ./checkpoint --outfile model.gguf --outtype tq2_0
        """,
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint directory (local or gs://...)",
    )
    parser.add_argument(
        "--outfile", "-o",
        type=Path,
        required=True,
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["i2_s", "tq1_0", "tq2_0", "tl1", "tl2", "f16", "f32"],
        default="i2_s",
        help="Output quantization type (default: i2_s)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output file size after conversion",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--force-tq2",
        action="store_true",
        help="Force TQ2_0 output even for bf16 checkpoints (NOT RECOMMENDED)",
    )

    args = parser.parse_args()

    # TQ2_0 warning for bf16 DLM checkpoints
    if args.outtype == "tq2_0" and not args.force_tq2:
        logger.error("=" * 70)
        logger.error("ERROR: TQ2_0 produces GARBAGE output for bf16 DLM checkpoints!")
        logger.error("")
        logger.error("TQ2_0 applies ternary quantization which corrupts already-ternary")
        logger.error("weights stored in bf16 format. The resulting model will output")
        logger.error("nonsense (e.g., 'GGGGGG...' or random tokens).")
        logger.error("")
        logger.error("RECOMMENDED: Use --outtype i2_s instead (default)")
        logger.error("")
        logger.error("If you REALLY want TQ2_0 (not bf16 checkpoint), use --force-tq2")
        logger.error("=" * 70)
        sys.exit(1)

    # Check converter exists
    if not CONVERTER_SCRIPT.exists():
        logger.error(f"Converter not found: {CONVERTER_SCRIPT}")
        logger.error("Please run: git submodule update --init extern/BitNet")
        sys.exit(1)

    # Handle GCS paths
    if args.checkpoint.startswith("gs://"):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = download_from_gcs(args.checkpoint, Path(tmpdir) / "checkpoint")
            convert_checkpoint(
                local_path,
                args.outfile,
                args.outtype,
                args.validate,
                args.verbose,
            )
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        convert_checkpoint(
            checkpoint_path,
            args.outfile,
            args.outtype,
            args.validate,
            args.verbose,
        )


if __name__ == "__main__":
    main()
