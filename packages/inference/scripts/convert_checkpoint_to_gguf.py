#!/usr/bin/env python3
"""
Unified GGUF conversion script for DLM/BitNet checkpoints.

This script handles:
- bf16 "online" quantization checkpoints (weights are continuous floats)
- Both sentencepiece and GPT2/BPE tokenizers
- Architecture name variants (BitnetForCausalLM vs BitNetForCausalLM)
- SubLN models (with attn_sub_norm/ffn_sub_norm layers)

Supported output formats:
- f16: Float16 (default, universally compatible, works with all model sizes)
- tq1_0: Ternary quantization v1 (~0.5x size, requires hidden_size % 256 == 0)
- tq2_0: Ternary quantization v2 (AVOID for bf16 DLM - corrupts weights!)
- bf16: BFloat16 (same size as f16)
- f32: Float32 (2x size, for debugging only)

Usage:
    # Basic conversion (F16 recommended, works for all models)
    python convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf

    # TQ1_0 for 2B+ models (smaller, faster)
    python convert_checkpoint_to_gguf.py /path/to/checkpoint --outfile model.gguf --outtype tq1_0

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
LLAMA_CPP_DIR = INFERENCE_PKG / "extern" / "llama.cpp"
CONVERTER_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

# Model size expectations (approximate, for validation)
MODEL_SIZE_EXPECTATIONS = {
    # (num_params, expected_f16_size_bytes)
    "2B": (2_000_000_000, 4_500_000_000),  # ~4.5GB for f16
    "135M": (135_000_000, 260_000_000),     # ~260MB for f16
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


def check_tq1_compatibility(config: dict) -> tuple[bool, str]:
    """Check if TQ1_0 format is compatible with this model.

    TQ1_0 requires tensor dimensions to be divisible by 256 (block size).
    Small models like 135M (hidden_size=576) are NOT compatible.

    Returns: (is_compatible, reason)
    """
    hidden_size = config.get("hidden_size", 0)
    intermediate_size = config.get("intermediate_size", 0)

    if hidden_size % 256 != 0:
        return False, f"hidden_size={hidden_size} not divisible by 256"
    if intermediate_size % 256 != 0:
        return False, f"intermediate_size={intermediate_size} not divisible by 256"

    return True, "compatible"


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

    # Expected sizes based on quantization type (bytes per param)
    if quant_type in ("tq1_0", "tq2_0", "i2_s"):
        # ~0.5-0.6 bytes per param for ternary quantization
        expected_min = estimated_params * 0.4
        expected_max = estimated_params * 0.8
    elif quant_type in ("f16", "bf16"):
        # ~2 bytes per param
        expected_min = estimated_params * 1.5
        expected_max = estimated_params * 2.5
    else:  # f32
        # ~4 bytes per param
        expected_min = estimated_params * 3.0
        expected_max = estimated_params * 5.0

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
    quant_type: str = "f16",
    validate: bool = False,
    verbose: bool = False,
) -> Path:
    """Convert checkpoint to GGUF format."""
    # Validate checkpoint
    config = validate_checkpoint(checkpoint_path)
    logger.info(f"Converting {checkpoint_path.name} to GGUF ({quant_type})")

    # Check TQ1_0 compatibility
    if quant_type == "tq1_0":
        compatible, reason = check_tq1_compatibility(config)
        if not compatible:
            logger.warning("=" * 60)
            logger.warning(f"TQ1_0 incompatible: {reason}")
            logger.warning("Falling back to F16 format")
            logger.warning("=" * 60)
            quant_type = "f16"

    # Fix architecture name if needed
    fix_architecture_name(checkpoint_path)

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

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
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
  f16    - Float16 (~260MB for 135M, ~4.5GB for 2B) [DEFAULT]
           Works with ALL model sizes, recommended for DLM inference
  tq1_0  - Ternary quantization v1 (~0.5x size of f16)
           ONLY for models with hidden_size % 256 == 0 (2B+, not 135M)
           Will auto-fallback to f16 if incompatible
  bf16   - BFloat16 (same size as f16)
           Alternative to f16, same compatibility
  f32    - Float32 (~2x size of f16)
           For debugging only

WARNING: NEVER use tq2_0 for bf16 DLM checkpoints - it corrupts weights!

Examples:
  # Basic conversion (F16 - works for all models)
  python convert_checkpoint_to_gguf.py ./checkpoint --outfile model.gguf

  # TQ1_0 for 2B+ models (smaller, faster)
  python convert_checkpoint_to_gguf.py ./checkpoint --outfile model.gguf --outtype tq1_0

  # From GCS (auto-downloads, excludes optimizer state)
  python convert_checkpoint_to_gguf.py gs://bucket/checkpoint --outfile model.gguf
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
        choices=["f16", "tq1_0", "bf16", "f32"],
        default="f16",
        help="Output format (default: f16)",
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

    args = parser.parse_args()

    # Check converter exists
    if not CONVERTER_SCRIPT.exists():
        logger.error(f"Converter not found: {CONVERTER_SCRIPT}")
        logger.error("Please build llama.cpp first - see CLAUDE.md")
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
