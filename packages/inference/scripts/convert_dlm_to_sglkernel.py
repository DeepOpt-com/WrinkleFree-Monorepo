#!/usr/bin/env python3
"""Convert DLM checkpoint (full precision) to sgl-kernel packed format with proper quantization."""

import torch
from safetensors.torch import load_file
import struct
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAGIC = b"SGLBITNT"
VERSION = 1
QK_I2_S = 128  # Block size for SIMD kernels


def pack_ternary_simd(weights: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Quantize full-precision weights to ternary and pack for SIMD kernels.
    Uses vectorized operations for speed.

    Args:
        weights: Input tensor [out_features, in_features] in any dtype

    Returns:
        packed: Packed uint8 tensor [out_features, in_features/4]
        scale: Per-tensor scale factor for dequantization
    """
    M, K = weights.shape
    assert K % QK_I2_S == 0, f"K ({K}) must be multiple of {QK_I2_S}"

    # Convert to float for quantization
    w = weights.float()

    # Compute per-tensor scale (mean absolute value)
    scale = w.abs().mean().item()
    if scale < 1e-8:
        # This shouldn't happen with trained weights - indicates a problem
        scale = 1.0

    # Quantize to ternary: round(W / scale) clamped to {-1, 0, 1}
    w_quant = (w / scale).round().clamp(-1, 1).to(torch.int8)

    # Shift to unsigned: {-1, 0, 1} -> {0, 1, 2}
    w_unsigned = (w_quant + 1).to(torch.uint8)

    # SIMD block-interleaved packing (vectorized)
    # For each 128-element block: 32 packed bytes
    # byte[j] packs weights at positions j, j+32, j+64, j+96 within block
    num_blocks = K // QK_I2_S

    # Reshape to [M, num_blocks, 128] then to [M, num_blocks, 4, 32]
    w_blocks = w_unsigned.view(M, num_blocks, 4, 32)

    # v0 = w_blocks[:, :, 0, :] at positions 0-31
    # v1 = w_blocks[:, :, 1, :] at positions 32-63
    # v2 = w_blocks[:, :, 2, :] at positions 64-95
    # v3 = w_blocks[:, :, 3, :] at positions 96-127
    v0 = w_blocks[:, :, 0, :].to(torch.uint8)
    v1 = w_blocks[:, :, 1, :].to(torch.uint8)
    v2 = w_blocks[:, :, 2, :].to(torch.uint8)
    v3 = w_blocks[:, :, 3, :].to(torch.uint8)

    # Pack: v0 in bits 6-7, v1 in bits 4-5, v2 in bits 2-3, v3 in bits 0-1
    packed_blocks = (v0 << 6) | (v1 << 4) | (v2 << 2) | v3

    # Reshape back to [M, K/4]
    packed = packed_blocks.view(M, -1)

    return packed, scale


def convert_checkpoint(input_dir: str, output_path: str):
    """Convert DLM checkpoint to sgl-kernel binary format."""
    input_dir = Path(input_dir)

    logger.info(f"Loading checkpoint from {input_dir}")
    tensors = load_file(input_dir / "model.safetensors")
    config = json.load(open(input_dir / "config.json"))

    # Prepare output tensors
    out_tensors = []
    linear_count = 0

    for name in sorted(tensors.keys()):
        tensor = tensors[name]
        is_linear = (
            "weight" in name and
            len(tensor.shape) == 2 and
            "norm" not in name and
            "embed" not in name and
            "lm_head" not in name
        )

        if is_linear:
            M, K = tensor.shape

            # Pad K to multiple of 128 if needed
            if K % QK_I2_S != 0:
                pad = QK_I2_S - (K % QK_I2_S)
                tensor = torch.nn.functional.pad(tensor, (0, pad))
                K = tensor.shape[1]
                logger.info(f"Padded {name} K dimension to {K}")

            packed, scale = pack_ternary_simd(tensor)
            out_tensors.append((name, packed, 0, scale))  # dtype 0 = uint8
            linear_count += 1

            if linear_count % 20 == 0:
                logger.info(f"Packed {linear_count} linear layers...")
        else:
            # Non-linear weights: convert to f16 to save memory
            tensor = tensor.half()  # Convert bf16/f32 to f16
            dtype_id = 2  # f16
            out_tensors.append((name, tensor, dtype_id, None))

    logger.info(f"Packed {linear_count} linear layers total")

    # Write binary file
    logger.info(f"Writing to {output_path}")
    with open(output_path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION))

        # Config JSON
        config_json = json.dumps(config).encode('utf-8')
        f.write(struct.pack('<I', len(config_json)))
        f.write(config_json)

        # Number of tensors
        f.write(struct.pack('<I', len(out_tensors)))

        # Tensors
        for name, tensor, dtype_id, scale in out_tensors:
            # Name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            # Dtype
            f.write(struct.pack('<I', dtype_id))

            # Shape
            f.write(struct.pack('<I', len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack('<I', dim))

            # Scale (for packed weights)
            f.write(struct.pack('<I', 1 if scale is not None else 0))
            f.write(struct.pack('<f', scale if scale is not None else 0.0))

            # Data
            data = tensor.numpy().tobytes()
            f.write(struct.pack('<Q', len(data)))
            f.write(data)

    output_size = Path(output_path).stat().st_size
    logger.info(f"Done! Output size: {output_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Convert DLM checkpoint to sgl-kernel format")
    parser.add_argument("input_dir", help="Input checkpoint directory")
    parser.add_argument("output", help="Output .bin file path")
    args = parser.parse_args()

    convert_checkpoint(args.input_dir, args.output)


if __name__ == "__main__":
    main()
