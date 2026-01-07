#!/usr/bin/env python3
"""Convert BitNet checkpoint to GGUF with proper ternary quantization.

This script applies the on-the-fly weight quantization that BitLinear uses,
then converts the quantized weights to GGUF format.

The quantization formula (from wf_arch.layers.bitlinear):
    scale = 1.0 / mean(|W|)
    W_ternary = round(W * scale).clamp(-1, 1)  # Ternary: {-1, 0, 1}
    W_quant = W_ternary / scale  # Scaled back

For GGUF I2_S format, we store:
    - W_ternary as packed 2-bit values (0=−1, 1=0, 2=+1)
    - scale as a per-tensor float

Usage:
    python scripts/convert_with_quantization.py \\
        /path/to/checkpoint \\
        --outfile models/model-quantized.gguf
"""

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def weight_quant(w: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, float]:
    """
    Apply BitLinear weight quantization.

    Args:
        w: Weight tensor
        eps: Small constant for numerical stability

    Returns:
        (ternary_weights, scale) where ternary_weights are {-1, 0, 1}
    """
    # Compute scale (1 / absmean)
    absmean = w.abs().mean().clamp(min=eps)
    scale = 1.0 / absmean

    # Quantize to ternary
    w_ternary = (w * scale).round().clamp(-1, 1)

    return w_ternary.to(torch.int8), absmean.item()


def is_quantizable_weight(name: str) -> bool:
    """Check if a weight tensor should be quantized."""
    # Quantize projection weights, not embeddings or norms
    quantizable_patterns = [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
        "attn.qkv.weight",  # Fused attention
        "attn.out_proj.weight",
        "mlp.fc1.weight",
        "mlp.fc2.weight",
    ]

    return any(pattern in name for pattern in quantizable_patterns)


def quantize_checkpoint(
    input_path: Path,
    output_path: Path,
) -> dict:
    """
    Quantize all BitLinear weights in a checkpoint.

    Args:
        input_path: Path to original checkpoint directory
        output_path: Path to save quantized checkpoint

    Returns:
        Dict with quantization statistics
    """
    logger.info(f"Loading checkpoint from {input_path}")

    # Find safetensors files
    safetensor_files = list(input_path.glob("*.safetensors"))
    if not safetensor_files:
        safetensor_files = list(input_path.glob("model*.safetensors"))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "quantized_tensors": 0,
        "total_tensors": 0,
        "scales": {},
    }

    for sf_file in safetensor_files:
        logger.info(f"Processing {sf_file.name}")

        tensors = {}
        with safe_open(sf_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                stats["total_tensors"] += 1

                if is_quantizable_weight(key):
                    # Apply quantization
                    w_ternary, scale = weight_quant(tensor)

                    # Store quantized weights with scale baked in
                    # w_quant = ternary_value * scale (where ternary is -1, 0, 1)
                    w_scaled = w_ternary.float() * scale
                    tensors[key] = w_scaled.to(tensor.dtype)

                    stats["quantized_tensors"] += 1
                    stats["scales"][key] = scale

                    # Log distribution
                    unique_vals = torch.unique(w_ternary)
                    logger.info(
                        f"  {key}: quantized to ternary, "
                        f"scale={scale:.6f}, unique={unique_vals.tolist()}"
                    )
                else:
                    # Keep as-is
                    tensors[key] = tensor

        # Save quantized checkpoint
        out_file = output_path / sf_file.name
        save_safetensors(tensors, out_file)
        logger.info(f"Saved quantized tensors to {out_file}")

    # Copy config and tokenizer files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"]:
        src = input_path / fname
        if src.exists():
            shutil.copy(src, output_path / fname)

    # Update config to mark as quantized
    config_path = output_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config["quantization_config"] = {
            "quant_method": "bitnet_ternary",
            "bits": 2,
            "group_size": 128,  # I2_S block size
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    logger.info(
        f"Quantized {stats['quantized_tensors']}/{stats['total_tensors']} tensors"
    )

    return stats


def convert_to_gguf(
    checkpoint_path: Path,
    gguf_path: Path,
    outtype: str = "f16",
) -> None:
    """Convert quantized checkpoint to GGUF format."""
    import subprocess
    import sys

    script_dir = Path(__file__).parent
    converter_script = (
        script_dir.parent
        / "extern/sglang-bitnet/3rdparty/llama.cpp/convert_hf_to_gguf.py"
    )

    if not converter_script.exists():
        raise FileNotFoundError(f"Converter not found: {converter_script}")

    cmd = [
        sys.executable,
        str(converter_script),
        str(checkpoint_path),
        "--outfile",
        str(gguf_path),
        "--outtype",
        outtype,
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        raise RuntimeError("GGUF conversion failed")

    logger.info(f"Successfully created {gguf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BitNet checkpoint to GGUF with proper ternary quantization"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to original checkpoint directory",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        required=True,
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--outtype",
        choices=["f16", "bf16", "f32"],
        default="f16",
        help="Output tensor type for non-quantized tensors",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary quantized checkpoint",
    )
    args = parser.parse_args()

    # Create temporary directory for quantized checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        if args.keep_temp:
            quantized_path = args.outfile.parent / f"{args.outfile.stem}_quantized"
            quantized_path.mkdir(parents=True, exist_ok=True)
        else:
            quantized_path = Path(tmpdir) / "quantized"

        # Step 1: Quantize weights
        logger.info("=" * 60)
        logger.info("Step 1: Applying BitLinear weight quantization")
        logger.info("=" * 60)
        stats = quantize_checkpoint(args.checkpoint, quantized_path)

        # Step 2: Convert to GGUF
        logger.info("=" * 60)
        logger.info("Step 2: Converting to GGUF format")
        logger.info("=" * 60)
        convert_to_gguf(quantized_path, args.outfile, args.outtype)

        logger.info("=" * 60)
        logger.info("Conversion complete!")
        logger.info(f"Output: {args.outfile}")
        logger.info(
            f"Quantized {stats['quantized_tensors']} tensors to ternary"
        )
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
