#!/usr/bin/env python3
"""
Convert DLM checkpoint from online quantization (bf16) to offline quantization (packed ternary).

BitNet uses ternary weights {-1, 0, 1} stored in packed uint8 format:
- 4 ternary values per byte (2 bits each)
- Packing: 0=-1, 1=0, 2=1
- Separate weight_scale tensor per linear layer

Usage:
    python quantize_dlm_to_offline.py <input_checkpoint> <output_checkpoint>
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def pack_ternary_weights(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize bf16 weights to ternary {-1, 0, 1} and pack into uint8.

    The GGUF converter unpacks with interleaved pattern:
    - byte j contains rows j, j+N/4, j+2*N/4, j+3*N/4 in bits [0-1], [2-3], [4-5], [6-7]

    So for 8 rows: byte 0 has [0,2,4,6], byte 1 has [1,3,5,7]

    Args:
        weights: Input tensor of shape (out_features, in_features) in bf16/fp16/fp32

    Returns:
        packed_weights: Shape (out_features // 4, in_features) in uint8
        weight_scale: Shape (1,) in bf16
    """
    # Convert to float32 for precision
    w = weights.float()

    # Compute scale: mean of absolute values per tensor
    weight_scale = w.abs().mean()

    if weight_scale < 1e-8:
        logger.warning(f"Very small weight scale: {weight_scale}")
        weight_scale = torch.tensor(1.0)

    # Quantize to ternary: round(W / scale) clamped to {-1, 0, 1}
    w_quant = (w / weight_scale).round().clamp(-1, 1).to(torch.int8)

    # Pack 4 values into 1 byte with interleaved pattern
    # Mapping: -1 -> 0, 0 -> 1, 1 -> 2 (so we use 2 bits per value)
    w_packed_vals = (w_quant + 1).to(torch.uint8)  # Now {0, 1, 2}

    # Ensure out_features is divisible by 4
    out_features = w_packed_vals.shape[0]
    in_features = w_packed_vals.shape[1]
    if out_features % 4 != 0:
        # Pad to multiple of 4
        pad_size = 4 - (out_features % 4)
        w_packed_vals = torch.nn.functional.pad(w_packed_vals, (0, 0, 0, pad_size), value=1)
        out_features = w_packed_vals.shape[0]
        logger.warning(f"Padded out_features to {out_features}")

    n_bytes = out_features // 4

    # GGUF converter unpacking:
    #   output[r] = packed[r % n_bytes] >> (2 * (r // n_bytes)) & 3
    # So:
    #   output[j]             = bits 0-1 of packed[j]
    #   output[j + n_bytes]   = bits 2-3 of packed[j]
    #   output[j + 2*n_bytes] = bits 4-5 of packed[j]
    #   output[j + 3*n_bytes] = bits 6-7 of packed[j]
    #
    # Therefore packed[j] should be:
    #   bits 0-1 = row j
    #   bits 2-3 = row j + n_bytes
    #   bits 4-5 = row j + 2*n_bytes
    #   bits 6-7 = row j + 3*n_bytes
    packed = (
        w_packed_vals[:n_bytes] |                    # rows 0..n_bytes-1 -> bits 0-1
        (w_packed_vals[n_bytes:2*n_bytes] << 2) |    # rows n_bytes..2*n_bytes-1 -> bits 2-3
        (w_packed_vals[2*n_bytes:3*n_bytes] << 4) |  # rows 2*n_bytes..3*n_bytes-1 -> bits 4-5
        (w_packed_vals[3*n_bytes:] << 6)             # rows 3*n_bytes.. -> bits 6-7
    )

    return packed, weight_scale.to(torch.bfloat16).unsqueeze(0)


def is_linear_weight(name: str) -> bool:
    """Check if tensor name corresponds to a linear layer weight to quantize."""
    # Quantize projection weights in attention and MLP
    linear_suffixes = [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    ]
    return any(name.endswith(suffix) for suffix in linear_suffixes)


def convert_checkpoint(input_path: Path, output_path: Path) -> None:
    """Convert online-quantization checkpoint to offline-quantization format."""

    # Load config
    config_path = input_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    logger.info(f"Converting checkpoint: {input_path.name}")
    logger.info(f"Model type: {config.get('model_type')}")
    logger.info(f"Hidden size: {config.get('hidden_size')}")
    logger.info(f"Current quantization_mode: {config.get('quantization_config', {}).get('quantization_mode')}")

    # Load weights
    safetensors_path = input_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {input_path}")

    logger.info("Loading weights...")
    weights = load_file(safetensors_path)

    # Process weights
    new_weights = {}
    quantized_count = 0

    for name, tensor in weights.items():
        if is_linear_weight(name):
            # Quantize and pack
            packed, scale = pack_ternary_weights(tensor)
            new_weights[name] = packed
            new_weights[name.replace(".weight", ".weight_scale")] = scale
            quantized_count += 1

            # Log shape change
            orig_shape = tuple(tensor.shape)
            new_shape = tuple(packed.shape)
            logger.debug(f"{name}: {orig_shape} -> {new_shape} (packed)")
        else:
            # Keep as-is
            new_weights[name] = tensor

    logger.info(f"Quantized {quantized_count} linear layers")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save new weights
    output_safetensors = output_path / "model.safetensors"
    logger.info(f"Saving to {output_safetensors}...")
    save_file(new_weights, output_safetensors)

    # Update config
    new_config = config.copy()
    if "quantization_config" in new_config:
        new_config["quantization_config"]["quantization_mode"] = "offline"
    else:
        new_config["quantization_config"] = {
            "quant_method": "bitnet",
            "linear_class": "autobitlinear",
            "quantization_mode": "offline",
        }

    output_config = output_path / "config.json"
    with open(output_config, "w") as f:
        json.dump(new_config, f, indent=2)
    logger.info(f"Updated config with quantization_mode: offline")

    # Copy other files
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "generation_config.json", "tokenizer.model"]:
        src = input_path / fname
        if src.exists():
            shutil.copy(src, output_path / fname)
            logger.info(f"Copied {fname}")

    # Summary
    input_size = safetensors_path.stat().st_size / (1024 ** 3)
    output_size = output_safetensors.stat().st_size / (1024 ** 3)
    logger.info(f"\nConversion complete!")
    logger.info(f"  Input:  {input_size:.2f} GB")
    logger.info(f"  Output: {output_size:.2f} GB")
    logger.info(f"  Ratio:  {output_size/input_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Convert DLM online→offline quantization")
    parser.add_argument("input", type=Path, help="Input checkpoint directory")
    parser.add_argument("output", type=Path, help="Output checkpoint directory")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    convert_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
