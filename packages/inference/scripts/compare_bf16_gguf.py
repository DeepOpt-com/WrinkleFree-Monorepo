#!/usr/bin/env python3
"""Compare BF16 master weights with GGUF I2_S."""

import torch
import numpy as np
from safetensors import safe_open
import struct
import mmap


def load_gguf_i2s(gguf_path, tensor_name):
    """Load I2_S tensor from GGUF."""
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
            return None

        info = tensors[tensor_name]
        abs_offset = data_offset + info['offset']
        n_elements = 1
        for d in info['dims']:
            n_elements *= d

        if info['dtype'] == 36:
            packed_size = n_elements // 4
            packed_data = np.frombuffer(mm[abs_offset:abs_offset + packed_size], dtype=np.uint8).copy()
            return {'packed': packed_data, 'dims': info['dims'], 'n_elements': n_elements}

        return None


def decode_lsb(packed, n_elements):
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
    bf16_path = "/tmp/bitnet-bf16/model.safetensors"

    print("=== COMPARING BF16 MASTER WEIGHTS WITH GGUF I2_S ===\n")

    # Load BF16
    print("Loading BF16 model...")
    with safe_open(bf16_path, framework="pt") as f:
        bf16_tensors = list(f.keys())
        print(f"BF16 tensors: {len(bf16_tensors)}")
        for t in bf16_tensors[:10]:
            tensor = f.get_tensor(t)
            print(f"  {t}: {tensor.shape} {tensor.dtype}")

    # Load a specific weight
    print("\n--- Comparing gate_proj weights ---")

    with safe_open(bf16_path, framework="pt") as f:
        bf16_gate = f.get_tensor("model.layers.0.mlp.gate_proj.weight").float().numpy()

    print(f"BF16 gate_proj shape: {bf16_gate.shape}")
    print(f"BF16 first row[:10]: {bf16_gate[0, :10]}")
    print(f"BF16 unique values: {np.unique(bf16_gate)}")

    # Convert BF16 to ternary
    bf16_ternary = np.round(bf16_gate).astype(np.int8)
    print(f"BF16 as ternary first row[:10]: {bf16_ternary[0, :10]}")

    # Load GGUF
    gguf_data = load_gguf_i2s(gguf_path, "blk.0.ffn_gate.weight")
    gguf_decoded = decode_lsb(gguf_data['packed'], gguf_data['n_elements'])

    in_dim = gguf_data['dims'][0]   # 2560
    out_dim = gguf_data['dims'][1]  # 6912

    print(f"\nGGUF dims: [{in_dim}, {out_dim}]")

    # Try different reshape interpretations
    print("\n--- Testing reshape interpretations ---")

    for name, reshape_fn in [
        ("(out, in)", lambda x: x.reshape(out_dim, in_dim)),
        ("(in, out).T", lambda x: x.reshape(in_dim, out_dim).T),
    ]:
        gguf_2d = reshape_fn(gguf_decoded)
        if gguf_2d.shape != bf16_ternary.shape:
            print(f"  {name}: shape {gguf_2d.shape} != BF16 {bf16_ternary.shape}")
            continue

        match = (gguf_2d == bf16_ternary).mean()
        print(f"  {name}: {match:.2%} match")
        print(f"    First row[:10]: {gguf_2d[0, :10]}")

        if match > 0.99:
            print(f"    *** FOUND IT! GGUF {name} matches BF16! ***")

    # Also check sorted row/col sums
    print("\n--- Row sum comparison ---")
    gguf_out_in = gguf_decoded.reshape(out_dim, in_dim)
    gguf_sorted = np.sort(gguf_out_in.sum(axis=1))
    bf16_sorted = np.sort(bf16_ternary.sum(axis=1))
    print(f"Sorted row sums match: {np.array_equal(gguf_sorted, bf16_sorted)}")

    gguf_in_out_T = gguf_decoded.reshape(in_dim, out_dim).T
    gguf_T_sorted = np.sort(gguf_in_out_T.sum(axis=1))
    print(f"Sorted row sums (in,out).T match: {np.array_equal(gguf_T_sorted, bf16_sorted)}")


if __name__ == "__main__":
    main()
