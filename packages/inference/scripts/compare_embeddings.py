#!/usr/bin/env python3
"""Compare simpler tensors (embeddings, norms) between GGUF and HuggingFace."""

import numpy as np
from safetensors import safe_open
import struct
import mmap


def load_gguf_tensor(gguf_path, tensor_name):
    """Load any tensor from GGUF."""
    def read_string(mm, offset):
        length = struct.unpack_from('<Q', mm, offset)[0]
        s = mm[offset + 8:offset + 8 + length].decode('utf-8')
        return s, 8 + length

    def skip_value(mm, offset, value_type):
        type_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        if value_type == 8:
            _, consumed = read_string(mm, offset)
            return consumed
        elif value_type == 9:
            arr_type = struct.unpack_from('<I', mm, offset)[0]
            arr_len = struct.unpack_from('<Q', mm, offset + 4)[0]
            consumed = 12
            if arr_type == 8:
                for _ in range(arr_len):
                    _, c = read_string(mm, offset + consumed)
                    consumed += c
            else:
                consumed += arr_len * type_sizes.get(arr_type, 0)
            return consumed
        return type_sizes.get(value_type, 0)

    with open(gguf_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        n_tensors = struct.unpack_from('<Q', mm, 8)[0]
        n_kv = struct.unpack_from('<Q', mm, 16)[0]
        offset = 24
        alignment = 32

        for _ in range(n_kv):
            _, consumed = read_string(mm, offset)
            offset += consumed
            vtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            offset += skip_value(mm, offset, vtype)

        tensors = {}
        for _ in range(n_tensors):
            name, consumed = read_string(mm, offset)
            offset += consumed
            n_dims = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack_from('<Q', mm, offset)[0])
                offset += 8
            dtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            rel_offset = struct.unpack_from('<Q', mm, offset)[0]
            offset += 8
            tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': rel_offset}

        padding = offset % alignment
        if padding != 0:
            offset += alignment - padding
        data_offset = offset

        if tensor_name not in tensors:
            print(f"Tensor {tensor_name} not found")
            print(f"Available tensors: {list(tensors.keys())[:20]}...")
            return None

        info = tensors[tensor_name]
        abs_offset = data_offset + info['offset']
        n_elements = 1
        for d in info['dims']:
            n_elements *= d

        dtype_map = {0: 'float32', 1: 'float16', 36: 'i2_s'}
        dtype_name = dtype_map.get(info['dtype'], f'unknown({info["dtype"]})')

        if info['dtype'] == 0:  # F32
            data = np.frombuffer(mm[abs_offset:abs_offset + n_elements * 4], dtype=np.float32).copy()
        elif info['dtype'] == 1:  # F16
            data = np.frombuffer(mm[abs_offset:abs_offset + n_elements * 2], dtype=np.float16).copy()
        elif info['dtype'] == 36:  # I2_S
            packed_size = n_elements // 4
            packed = np.frombuffer(mm[abs_offset:abs_offset + packed_size], dtype=np.uint8).copy()
            data = decode_i2s_lsb(packed, n_elements)
        else:
            print(f"Unknown dtype {info['dtype']}")
            return None

        return {'data': data, 'dims': info['dims'], 'dtype': dtype_name}


def decode_i2s_lsb(packed, n_elements):
    """Decode I2_S with LSB-first."""
    output = np.zeros(n_elements, dtype=np.int8)
    idx = 0
    for byte in packed:
        byte = int(byte)
        for shift in [0, 2, 4, 6]:
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output[idx] = int(val) - 1
            idx += 1
    return output


def main():
    gguf_path = "/tmp/bitnet-gguf-official/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    print("=== COMPARING NON-QUANTIZED TENSORS ===\n")

    # List tensors
    print("HuggingFace tensors:")
    with safe_open(hf_path, framework="pt") as f:
        hf_tensors = list(f.keys())
        for t in hf_tensors[:15]:
            tensor = f.get_tensor(t)
            print(f"  {t}: {tensor.shape} {tensor.dtype}")

    # Try to find matching tensors
    print("\n--- Comparing RMS norm weights ---")

    # GGUF uses different naming
    gguf_norm = load_gguf_tensor(gguf_path, "blk.0.attn_norm.weight")
    if gguf_norm:
        print(f"GGUF blk.0.attn_norm.weight: dims={gguf_norm['dims']}, dtype={gguf_norm['dtype']}")
        print(f"  First 10: {gguf_norm['data'][:10]}")

    with safe_open(hf_path, framework="pt") as f:
        hf_norm = f.get_tensor("model.layers.0.input_layernorm.weight").float().numpy()
        print(f"HF model.layers.0.input_layernorm.weight: shape={hf_norm.shape}")
        print(f"  First 10: {hf_norm[:10]}")

    if gguf_norm is not None:
        # Compare
        gguf_flat = gguf_norm['data'].flatten().astype(np.float32)
        hf_flat = hf_norm.flatten().astype(np.float32)

        if len(gguf_flat) == len(hf_flat):
            diff = np.abs(gguf_flat - hf_flat).max()
            match_pct = (np.isclose(gguf_flat, hf_flat, rtol=1e-3)).mean()
            print(f"  Max diff: {diff:.6f}")
            print(f"  Close match: {match_pct:.1%}")
        else:
            print(f"  Length mismatch: {len(gguf_flat)} vs {len(hf_flat)}")

    # Also compare token embeddings
    print("\n--- Comparing token embeddings ---")
    gguf_emb = load_gguf_tensor(gguf_path, "token_embd.weight")
    if gguf_emb:
        print(f"GGUF token_embd.weight: dims={gguf_emb['dims']}, dtype={gguf_emb['dtype']}")
        print(f"  First 10 of row 0: {gguf_emb['data'].reshape(gguf_emb['dims'][::-1])[0, :10] if len(gguf_emb['dims']) > 1 else gguf_emb['data'][:10]}")

    with safe_open(hf_path, framework="pt") as f:
        hf_emb = f.get_tensor("model.embed_tokens.weight").float().numpy()
        print(f"HF model.embed_tokens.weight: shape={hf_emb.shape}")
        print(f"  First 10 of row 0: {hf_emb[0, :10]}")

    if gguf_emb is not None:
        # GGUF dims are reversed from numpy/torch
        gguf_shape = tuple(gguf_emb['dims'][::-1])  # Reverse dims
        gguf_emb_2d = gguf_emb['data'].reshape(gguf_shape)

        if gguf_emb_2d.shape == hf_emb.shape:
            diff = np.abs(gguf_emb_2d.astype(np.float32) - hf_emb.astype(np.float32)).max()
            match_pct = (np.isclose(gguf_emb_2d.astype(np.float32), hf_emb.astype(np.float32), rtol=1e-3)).mean()
            print(f"  Max diff: {diff:.6f}")
            print(f"  Close match: {match_pct:.1%}")
            if match_pct > 0.99:
                print("  *** EMBEDDINGS MATCH! Same model confirmed. ***")
        else:
            print(f"  Shape mismatch: {gguf_emb_2d.shape} vs {hf_emb.shape}")


if __name__ == "__main__":
    main()
