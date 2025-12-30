#!/usr/bin/env python3
"""
Convert DLM/BitNet checkpoint to sgl-kernel binary format.

This produces a simple binary format that the C++ inference engine can load directly.

Format (sgl-kernel .bin):
    [8 bytes]  Magic: "SGLBITNT"
    [4 bytes]  Version: 1
    [4 bytes]  Config JSON length
    [N bytes]  Config JSON
    [4 bytes]  Number of tensors
    For each tensor:
        [4 bytes]  Name length
        [N bytes]  Name (UTF-8)
        [4 bytes]  Dtype (0=uint8, 1=float32, 2=float16, 3=bfloat16)
        [4 bytes]  Number of dimensions
        [dims x 4] Shape
        [4 bytes]  Scale present flag (1 for packed weights)
        [4 bytes]  Scale value (float32, if present)
        [8 bytes]  Data size in bytes
        [N bytes]  Raw tensor data

Weight packing for sgl-kernel (I2_I8 format):
    - Layout: [out_features, in_features/4]
    - 4 ternary values per byte along K (input) dimension
    - Encoding: 00=-1, 01=0, 10=+1
    - Block size: K must be multiple of 128 (QK_I2_S)

Usage:
    python convert_to_sglkernel.py <input_checkpoint> <output.bin>
"""

import argparse
import json
import logging
import struct
from pathlib import Path

import torch
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAGIC = b"SGLBITNT"
VERSION = 1
QK_I2_S = 128  # Block size for sgl-kernel


def pack_ternary_sglkernel(weights: torch.Tensor, simd_mode: bool = True) -> tuple[torch.Tensor, float]:
    """
    Quantize and pack weights for sgl-kernel format.

    SIMD mode (default): Block-interleaved packing for AVX2/AVX512 kernels
    - Block size: 128 elements (QK_I2_S)
    - 32 bytes per block
    - byte[j].bits[6:7] = weight[j+0]   (for activation[j+0])
    - byte[j].bits[4:5] = weight[j+32]  (for activation[j+32])
    - byte[j].bits[2:3] = weight[j+64]  (for activation[j+64])
    - byte[j].bits[0:1] = weight[j+96]  (for activation[j+96])

    Scalar mode: Sequential packing (4 consecutive weights per byte)

    Args:
        weights: Input tensor [out_features, in_features]
        simd_mode: Use SIMD-optimized block-interleaved packing (default True)

    Returns:
        packed: Packed uint8 tensor [out_features, in_features/4]
        scale: Per-tensor scale factor
    """
    M, K = weights.shape

    # K must be multiple of 128 for SIMD block alignment
    assert K % QK_I2_S == 0, f"K ({K}) must be multiple of {QK_I2_S}"

    # Convert to float32 for computation
    w = weights.float()

    # Compute per-tensor scale (mean absolute value)
    scale = w.abs().mean().item()
    if scale < 1e-8:
        logger.warning(f"Very small weight scale: {scale}")
        scale = 1.0

    # Quantize to ternary: round(W / scale) clamped to {-1, 0, 1}
    w_quant = (w / scale).round().clamp(-1, 1).to(torch.int8)

    # Shift to unsigned: {-1, 0, 1} -> {0, 1, 2}
    w_unsigned = (w_quant + 1).to(torch.uint8)

    K_packed = K // 4
    packed = torch.zeros(M, K_packed, dtype=torch.uint8)

    if simd_mode:
        # SIMD Block-interleaved packing
        # For each 128-element block: 32 packed bytes
        # byte[j] packs weights at positions j, j+32, j+64, j+96 within block
        num_blocks = K // QK_I2_S
        for block_idx in range(num_blocks):
            base_w = block_idx * QK_I2_S  # Start of 128-element block in weights
            base_p = block_idx * 32       # Start of 32-byte block in packed
            for j in range(32):
                # Pack 4 weights from positions j, j+32, j+64, j+96 within block
                packed[:, base_p + j] = (
                    (w_unsigned[:, base_w + j + 0] << 6) |   # bits 6-7
                    (w_unsigned[:, base_w + j + 32] << 4) |  # bits 4-5
                    (w_unsigned[:, base_w + j + 64] << 2) |  # bits 2-3
                    (w_unsigned[:, base_w + j + 96] << 0)    # bits 0-1
                )
    else:
        # Sequential packing: 4 consecutive weights per byte
        for byte_idx in range(K_packed):
            k = byte_idx * 4
            packed[:, byte_idx] = (
                (w_unsigned[:, k] << 0) |
                (w_unsigned[:, k + 1] << 2) |
                (w_unsigned[:, k + 2] << 4) |
                (w_unsigned[:, k + 3] << 6)
            )

    return packed, scale


def is_linear_weight(name: str) -> bool:
    """Check if tensor name is a linear layer weight to quantize."""
    linear_suffixes = [
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    ]
    return any(name.endswith(suffix) for suffix in linear_suffixes)


def write_tensor(f, name: str, data: torch.Tensor, dtype_id: int, scale: float = None):
    """Write a single tensor to the binary file."""
    # Name
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)

    # Dtype
    f.write(struct.pack('<I', dtype_id))

    # Shape
    f.write(struct.pack('<I', len(data.shape)))
    for dim in data.shape:
        f.write(struct.pack('<I', dim))

    # Scale (for packed weights)
    if scale is not None:
        f.write(struct.pack('<I', 1))  # Scale present
        f.write(struct.pack('<f', scale))
    else:
        f.write(struct.pack('<I', 0))  # No scale

    # Data
    data_bytes = data.numpy().tobytes()
    f.write(struct.pack('<Q', len(data_bytes)))
    f.write(data_bytes)


def convert_checkpoint(input_path: Path, output_path: Path, pack_lm_head: bool = False) -> None:
    """Convert checkpoint to sgl-kernel binary format.

    Args:
        input_path: Path to input checkpoint directory
        output_path: Path for output .bin file
        pack_lm_head: If True, create a quantized lm_head for faster inference
                      (uses embed_tokens quantized to 2-bit)
    """

    # Load config
    config_path = input_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    logger.info(f"Converting: {input_path.name}")
    logger.info(f"Model type: {config.get('model_type')}")
    logger.info(f"Hidden size: {config.get('hidden_size')}")

    # Load weights
    safetensors_path = input_path / "model.safetensors"
    if not safetensors_path.exists():
        # Try sharded format
        index_path = input_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            # Load all shards
            weights = {}
            shard_files = set(index["weight_map"].values())
            for shard_file in shard_files:
                shard_path = input_path / shard_file
                logger.info(f"Loading shard: {shard_file}")
                shard_weights = load_file(shard_path)
                weights.update(shard_weights)
        else:
            raise FileNotFoundError(f"No model weights found in {input_path}")
    else:
        logger.info("Loading model.safetensors...")
        weights = load_file(safetensors_path)

    # Determine if weights are already packed (offline mode)
    quant_config = config.get("quantization_config", {})
    is_offline = quant_config.get("quantization_mode") == "offline"

    if is_offline:
        logger.info("Checkpoint is offline (pre-packed) - repacking for sgl-kernel format")
    else:
        logger.info("Checkpoint is online (bf16) - quantizing and packing")

    # Process weights
    processed = {}
    scales = {}
    quantized_count = 0

    for name, tensor in weights.items():
        if is_linear_weight(name):
            # Get the weight (handle packed vs unpacked)
            if is_offline:
                # Need to unpack first, then repack in sgl-kernel format
                # Offline format: [out_features/4, in_features] with interleaved rows
                # sgl-kernel format: [out_features, in_features/4] sequential K

                # First unpack the offline format
                packed = tensor.numpy()
                M_packed, K = packed.shape
                M = M_packed * 4

                # Unpack: byte j contains rows j, j+M_packed, j+2*M_packed, j+3*M_packed
                unpacked = torch.zeros(M, K, dtype=torch.int8)
                for i in range(4):
                    unpacked[i*M_packed:(i+1)*M_packed] = ((torch.from_numpy(packed).to(torch.int32) >> (i*2)) & 0x03).to(torch.int8) - 1

                # Get scale
                scale_name = name.replace(".weight", ".weight_scale")
                if scale_name in weights:
                    scale = weights[scale_name].float().item()
                else:
                    scale = 1.0

                # Repack in sgl-kernel format
                packed_new, _ = pack_ternary_sglkernel(unpacked.float() * scale)
                processed[name] = packed_new
                scales[name] = scale
            else:
                # Online format: bf16/fp16/fp32 weights
                packed, scale = pack_ternary_sglkernel(tensor)
                processed[name] = packed
                scales[name] = scale

            quantized_count += 1
            logger.debug(f"Packed {name}: {tensor.shape} -> {processed[name].shape}")
        else:
            # Keep non-linear weights as-is (embeddings, norms, etc.)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            processed[name] = tensor
            scales[name] = None

    logger.info(f"Packed {quantized_count} linear layers")

    # Optionally create quantized lm_head from embed_tokens
    if pack_lm_head:
        embed_name = "model.embed_tokens.weight"
        if embed_name in processed:
            embed_tensor = processed[embed_name]
            if embed_tensor.dtype == torch.float32:
                # Check if lm_head is tied to embeddings
                if config.get("tie_word_embeddings", True):
                    logger.info("Creating quantized lm_head from embed_tokens (tie_word_embeddings=true)")
                    # lm_head needs K (hidden_size) to be multiple of 128 for SIMD
                    hidden_size = config.get("hidden_size", embed_tensor.shape[1])
                    if hidden_size % QK_I2_S == 0:
                        packed_lm, scale_lm = pack_ternary_sglkernel(embed_tensor)
                        processed["lm_head.weight"] = packed_lm
                        scales["lm_head.weight"] = scale_lm
                        logger.info(f"  Quantized lm_head: {embed_tensor.shape} -> {packed_lm.shape}")
                        logger.info(f"  Added ~{packed_lm.numel() / 1e6:.1f}MB for 2x faster inference")
                    else:
                        logger.warning(f"Cannot pack lm_head: hidden_size {hidden_size} not divisible by {QK_I2_S}")
                else:
                    logger.warning("Cannot pack lm_head: tie_word_embeddings is false, would need separate lm_head weights")
        else:
            logger.warning(f"Cannot pack lm_head: {embed_name} not found")

    # Write binary file
    logger.info(f"Writing to {output_path}...")

    with open(output_path, 'wb') as f:
        # Magic
        f.write(MAGIC)

        # Version
        f.write(struct.pack('<I', VERSION))

        # Config JSON
        config_json = json.dumps(config).encode('utf-8')
        f.write(struct.pack('<I', len(config_json)))
        f.write(config_json)

        # Number of tensors
        f.write(struct.pack('<I', len(processed)))

        # Write each tensor
        for name, tensor in processed.items():
            if tensor.dtype == torch.uint8:
                dtype_id = 0
            elif tensor.dtype == torch.float32:
                dtype_id = 1
            elif tensor.dtype == torch.float16:
                dtype_id = 2
            elif tensor.dtype == torch.bfloat16:
                dtype_id = 3
            else:
                logger.warning(f"Unknown dtype {tensor.dtype} for {name}, converting to float32")
                tensor = tensor.float()
                dtype_id = 1

            write_tensor(f, name, tensor.contiguous(), dtype_id, scales.get(name))

    # Summary
    input_size = sum(f.stat().st_size for f in input_path.glob("*.safetensors")) / (1024**3)
    output_size = output_path.stat().st_size / (1024**3)

    logger.info(f"\nConversion complete!")
    logger.info(f"  Input:  {input_size:.2f} GB")
    logger.info(f"  Output: {output_size:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Convert BitNet checkpoint to sgl-kernel format")
    parser.add_argument("input", type=Path, help="Input checkpoint directory")
    parser.add_argument("output", type=Path, help="Output .bin file")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--pack-lm-head", action="store_true",
                        help="Quantize lm_head to 2-bit for faster inference (adds ~82MB, ~2x faster)")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    convert_checkpoint(args.input, args.output, pack_lm_head=args.pack_lm_head)


if __name__ == "__main__":
    main()
