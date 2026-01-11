#!/usr/bin/env python3
"""
Convert HuggingFace BitNet model to GGUF with sequential I2_S layout.

This creates a GGUF file with simple sequential packing that matches
our Rust decoder, bypassing Microsoft's blocked layout complexity.

HF packed format:
- Shape: (out_features/4, in_features)
- Each byte contains 4 output row values at bits 0-1, 2-3, 4-5, 6-7
- Row interleaving: byte[packed_row, col] encodes rows [packed_row, packed_row+N/4, packed_row+N/2, packed_row+3N/4]

Our GGUF I2_S format:
- Sequential row-major layout
- Encoding: 00=-1, 01=0, 10=+1
- 4 weights per byte at bits 0-1, 2-3, 4-5, 6-7 (LSB first)
- Extra 32 bytes: first 4 bytes = F32 weight scale
"""

import sys
import struct
import numpy as np
from pathlib import Path
from safetensors import safe_open


def unpack_hf_ternary(packed_2d: np.ndarray) -> np.ndarray:
    """Unpack HuggingFace packed ternary weights to full matrix.

    HF format: (out_features/4, in_features), each byte contains 4 rows.
    Returns: (out_features, in_features) with values in {-1, 0, +1}
    """
    packed_rows = packed_2d.shape[0]
    in_features = packed_2d.shape[1]
    out_features = packed_rows * 4

    unpacked = np.zeros((out_features, in_features), dtype=np.int8)
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        mask = 3 << (2 * i)
        unpacked[start:end] = ((packed_2d & mask) >> (2 * i)).astype(np.int8) - 1

    return unpacked


def pack_sequential_i2s(ternary_2d: np.ndarray, scale: float) -> bytes:
    """Pack ternary weights to sequential I2_S format.

    Input: (out_features, in_features) with values in {-1, 0, +1}
    Output: Packed bytes with 4 sequential values per byte + 32-byte header

    Encoding: -1->00, 0->01, +1->10
    LSB first: byte = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6)
    """
    flat = ternary_2d.flatten()
    n_elements = len(flat)

    # Convert ternary to 2-bit: -1->0, 0->1, +1->2
    encoded = (flat + 1).astype(np.uint8)

    # Pack 4 values per byte, LSB first
    packed_size = n_elements // 4
    packed = np.zeros(packed_size, dtype=np.uint8)

    for i in range(packed_size):
        base = i * 4
        packed[i] = (
            encoded[base + 0] |
            (encoded[base + 1] << 2) |
            (encoded[base + 2] << 4) |
            (encoded[base + 3] << 6)
        )

    # Create 32-byte extra data (scale as F32 in first 4 bytes)
    extra = bytearray(32)
    extra[0:4] = struct.pack('<f', scale)

    return bytes(packed) + bytes(extra)


def create_gguf_header(tensors_info: list, metadata: dict) -> bytes:
    """Create GGUF file header.

    GGUF format:
    - Magic: "GGUF" (4 bytes)
    - Version: U32 (we use version 3)
    - Tensor count: U64
    - Metadata count: U64
    - Metadata key-value pairs
    - Tensor info entries
    """
    magic = b'GGUF'
    version = struct.pack('<I', 3)
    n_tensors = struct.pack('<Q', len(tensors_info))
    n_metadata = struct.pack('<Q', len(metadata))

    header = magic + version + n_tensors + n_metadata

    # Write metadata
    for key, value in metadata.items():
        # String key
        header += struct.pack('<Q', len(key))
        header += key.encode('utf-8')

        # Value type and data
        if isinstance(value, str):
            header += struct.pack('<I', 8)  # GGUF_TYPE_STRING
            header += struct.pack('<Q', len(value))
            header += value.encode('utf-8')
        elif isinstance(value, int):
            header += struct.pack('<I', 4)  # GGUF_TYPE_UINT32
            header += struct.pack('<I', value)
        elif isinstance(value, float):
            header += struct.pack('<I', 6)  # GGUF_TYPE_FLOAT32
            header += struct.pack('<f', value)

    return header


def write_gguf(output_path: str, hf_path: str):
    """Convert HF safetensors to GGUF with sequential I2_S layout."""

    print(f"Loading HuggingFace model from {hf_path}...")

    with safe_open(hf_path, framework="pt") as f:
        tensor_names = list(f.keys())
        print(f"Found {len(tensor_names)} tensors")

        # Collect tensor data
        tensors = {}
        for name in tensor_names:
            t = f.get_tensor(name)
            tensors[name] = t

    # Build tensor info list and prepare data
    tensor_info = []
    tensor_data = []

    # Prepare metadata
    metadata = {
        "general.architecture": "bitnet",
        "general.name": "BitNet-b1.58-2B-4T-sequential",
    }

    # Map HF names to GGUF names
    def hf_to_gguf_name(hf_name: str) -> str:
        # model.embed_tokens.weight -> token_embd.weight
        # model.layers.N.input_layernorm.weight -> blk.N.attn_norm.weight
        # model.layers.N.mlp.gate_proj.weight -> blk.N.ffn_gate.weight
        # model.layers.N.mlp.up_proj.weight -> blk.N.ffn_up.weight
        # model.layers.N.mlp.down_proj.weight -> blk.N.ffn_down.weight
        # model.layers.N.self_attn.q_proj.weight -> blk.N.attn_q.weight
        # etc.

        if hf_name == "model.embed_tokens.weight":
            return "token_embd.weight"
        if hf_name == "model.norm.weight":
            return "output_norm.weight"
        if hf_name == "lm_head.weight":
            return "output.weight"

        if hf_name.startswith("model.layers."):
            parts = hf_name.split(".")
            layer_idx = parts[2]
            rest = ".".join(parts[3:])

            mapping = {
                "input_layernorm.weight": "attn_norm.weight",
                "post_attention_layernorm.weight": "ffn_norm.weight",
                "mlp.gate_proj.weight": "ffn_gate.weight",
                "mlp.up_proj.weight": "ffn_up.weight",
                "mlp.down_proj.weight": "ffn_down.weight",
                "mlp.gate_proj.weight_scale": "ffn_gate.weight_scale",
                "mlp.up_proj.weight_scale": "ffn_up.weight_scale",
                "mlp.down_proj.weight_scale": "ffn_down.weight_scale",
                "mlp.ffn_sub_norm.weight": "ffn_sub_norm.weight",
                "self_attn.q_proj.weight": "attn_q.weight",
                "self_attn.k_proj.weight": "attn_k.weight",
                "self_attn.v_proj.weight": "attn_v.weight",
                "self_attn.o_proj.weight": "attn_output.weight",
                "self_attn.q_proj.weight_scale": "attn_q.weight_scale",
                "self_attn.k_proj.weight_scale": "attn_k.weight_scale",
                "self_attn.v_proj.weight_scale": "attn_v.weight_scale",
                "self_attn.o_proj.weight_scale": "attn_output.weight_scale",
                "self_attn.attn_sub_norm.weight": "attn_sub_norm.weight",
            }

            if rest in mapping:
                return f"blk.{layer_idx}.{mapping[rest]}"

        return hf_name  # Keep original if no mapping

    current_offset = 0
    alignment = 32

    for hf_name, tensor in tensors.items():
        gguf_name = hf_to_gguf_name(hf_name)

        # Skip weight_scale tensors (they're incorporated into the packed weights)
        if "weight_scale" in hf_name:
            continue

        # Check if this is a packed ternary weight tensor (uint8 dtype)
        import torch
        is_packed = tensor.dtype == torch.uint8 and len(tensor.shape) == 2

        # Get the corresponding scale if available
        scale_name = hf_name.replace(".weight", ".weight_scale")
        scale = 1.0
        if scale_name in tensors:
            scale = float(tensors[scale_name].float().item())

        if is_packed:
            # Unpack HF ternary
            packed_uint8 = tensor.numpy()
            ternary = unpack_hf_ternary(packed_uint8)

            # Pack to sequential I2_S
            data = pack_sequential_i2s(ternary, scale)

            out_features, in_features = ternary.shape
            tensor_info.append({
                "name": gguf_name,
                "dims": [in_features, out_features],  # GGUF convention
                "dtype": 36,  # GGML_TYPE_I2_S
                "offset": current_offset,
            })
            tensor_data.append(data)

            print(f"  {hf_name} -> {gguf_name}: I2_S ({out_features}, {in_features}), scale={scale:.4f}")
        else:
            # Keep as F32
            tensor_np = tensor.float().numpy()
            data = tensor_np.astype(np.float32).tobytes()

            tensor_info.append({
                "name": gguf_name,
                "dims": list(tensor_np.shape),
                "dtype": 0,  # GGML_TYPE_F32
                "offset": current_offset,
            })
            tensor_data.append(data)

            print(f"  {hf_name} -> {gguf_name}: F32 {tensor_np.shape}")

        current_offset += len(tensor_data[-1])
        # Align to 32 bytes
        padding = current_offset % alignment
        if padding != 0:
            tensor_data[-1] += b'\x00' * (alignment - padding)
            current_offset += alignment - padding

    # Write GGUF file
    print(f"\nWriting GGUF to {output_path}...")

    with open(output_path, 'wb') as f:
        # Magic and version
        f.write(b'GGUF')
        f.write(struct.pack('<I', 3))  # Version
        f.write(struct.pack('<Q', len(tensor_info)))
        f.write(struct.pack('<Q', len(metadata)))

        # Metadata
        for key, value in metadata.items():
            f.write(struct.pack('<Q', len(key)))
            f.write(key.encode('utf-8'))
            if isinstance(value, str):
                f.write(struct.pack('<I', 8))  # String type
                f.write(struct.pack('<Q', len(value)))
                f.write(value.encode('utf-8'))
            elif isinstance(value, int):
                f.write(struct.pack('<I', 4))  # U32 type
                f.write(struct.pack('<I', value))
            elif isinstance(value, float):
                f.write(struct.pack('<I', 6))  # F32 type
                f.write(struct.pack('<f', value))

        # Tensor info
        for info in tensor_info:
            f.write(struct.pack('<Q', len(info["name"])))
            f.write(info["name"].encode('utf-8'))
            f.write(struct.pack('<I', len(info["dims"])))
            for dim in info["dims"]:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', info["dtype"]))
            f.write(struct.pack('<Q', info["offset"]))

        # Align to 32 bytes before data
        current_pos = f.tell()
        padding = current_pos % alignment
        if padding != 0:
            f.write(b'\x00' * (alignment - padding))

        # Tensor data
        for data in tensor_data:
            f.write(data)

    print("Done!")


if __name__ == "__main__":
    import torch

    hf_path = "/tmp/bitnet-hf/model.safetensors"
    output_path = "/tmp/bitnet-sequential.gguf"

    write_gguf(output_path, hf_path)
