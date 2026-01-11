#!/usr/bin/env python3
"""Debug weight layout difference between GGUF and HuggingFace."""

import torch
import numpy as np
from safetensors import safe_open
import struct
import mmap


def load_gguf_i2s_raw(gguf_path, tensor_name):
    """Load raw I2_S data from GGUF."""
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

        if info['dtype'] == 36:  # I2_S
            packed_size = n_elements // 4
            packed_data = bytes(mm[abs_offset:abs_offset + packed_size])
            scale_bytes = mm[abs_offset + packed_size:abs_offset + packed_size + 4]
            scale = struct.unpack('<f', scale_bytes)[0]
            return {'packed': packed_data, 'dims': info['dims'], 'scale': scale, 'n_elements': n_elements}

        return None


def decode_i2s_method1(data, n_elements):
    """Decode I2_S - Method 1: Sequential (MSB first)."""
    output = np.zeros(n_elements, dtype=np.int8)
    idx = 0
    for byte in data:
        if idx >= n_elements:
            break
        for shift in [6, 4, 2, 0]:  # MSB first
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output[idx] = val - 1  # 00=-1, 01=0, 10=+1
            idx += 1
    return output


def decode_i2s_method2(data, n_elements):
    """Decode I2_S - Method 2: Sequential (LSB first)."""
    output = np.zeros(n_elements, dtype=np.int8)
    idx = 0
    for byte in data:
        if idx >= n_elements:
            break
        for shift in [0, 2, 4, 6]:  # LSB first
            if idx >= n_elements:
                break
            val = (byte >> shift) & 0x03
            output[idx] = val - 1
            idx += 1
    return output


def decode_hf_style(packed_bytes, n_rows, n_cols):
    """Decode using HuggingFace's unpack_weights style."""
    # HF stores packed weights as (out_features/4, in_features)
    # where each uint8 has 4 values packed along the out_features dimension
    packed = np.frombuffer(packed_bytes, dtype=np.uint8)
    packed_rows = n_rows // 4
    packed = packed.reshape(packed_rows, n_cols)

    unpacked = np.zeros((n_rows, n_cols), dtype=np.int8)
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        mask = 3 << (2 * i)
        val = (packed & mask) >> (2 * i)
        unpacked[start:end] = val.astype(np.int8) - 1

    return unpacked


def main():
    gguf_path = "/tmp/bitnet-gguf-official/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf/model.safetensors"

    # Load GGUF raw data
    print("Loading GGUF gate_proj...")
    gguf_data = load_gguf_i2s_raw(gguf_path, "blk.0.ffn_gate.weight")
    dims = gguf_data['dims']
    n_elements = gguf_data['n_elements']
    packed_bytes = gguf_data['packed']

    print(f"GGUF dims: {dims}")  # [2560, 6912] = [in_features, out_features]
    print(f"GGUF n_elements: {n_elements}")
    print(f"GGUF packed bytes: {len(packed_bytes)}")
    print(f"First 8 packed bytes: {list(packed_bytes[:8])}")

    # Load HuggingFace weights
    print("\nLoading HF gate_proj...")
    with safe_open(hf_path, framework="pt") as f:
        W_hf_packed = f.get_tensor("model.layers.0.mlp.gate_proj.weight").numpy()
    print(f"HF packed shape: {W_hf_packed.shape}")  # (1728, 2560) = (out_features/4, in_features)
    print(f"First 8 packed bytes (row 0): {list(W_hf_packed[0, :8])}")

    # Try different decoding methods
    print("\n" + "="*60)
    print("Trying different decode methods...")

    # Method 1: Sequential MSB first
    decoded_m1 = decode_i2s_method1(packed_bytes, n_elements)
    W_m1 = decoded_m1.reshape(dims[1], dims[0])  # Reshape to (out, in)
    print(f"\nMethod 1 (MSB first, reshape to [out, in]):")
    print(f"  Shape: {W_m1.shape}")
    print(f"  First row[:10]: {W_m1[0, :10]}")

    # Method 1b: Sequential MSB first, different reshape
    W_m1b = decoded_m1.reshape(dims[0], dims[1]).T  # Reshape to (in, out) then transpose
    print(f"\nMethod 1b (MSB first, reshape to [in, out].T):")
    print(f"  Shape: {W_m1b.shape}")
    print(f"  First row[:10]: {W_m1b[0, :10]}")

    # Method 2: Sequential LSB first
    decoded_m2 = decode_i2s_method2(packed_bytes, n_elements)
    W_m2 = decoded_m2.reshape(dims[1], dims[0])
    print(f"\nMethod 2 (LSB first, reshape to [out, in]):")
    print(f"  Shape: {W_m2.shape}")
    print(f"  First row[:10]: {W_m2[0, :10]}")

    # Method 3: HF-style unpacking
    # GGUF stores as [in_features, out_features], so we need to adjust
    in_features = dims[0]
    out_features = dims[1]
    print(f"\nMethod 3 (HF-style):")
    print(f"  Trying to unpack as ({out_features}, {in_features})...")
    try:
        W_m3 = decode_hf_style(packed_bytes, out_features, in_features)
        print(f"  Shape: {W_m3.shape}")
        print(f"  First row[:10]: {W_m3[0, :10]}")
    except Exception as e:
        print(f"  Error: {e}")

    # Unpack HF weights for comparison
    print("\n" + "="*60)
    print("HF unpacked weights:")
    packed_shape = W_hf_packed.shape
    original_row_dim = packed_shape[0] * 4
    W_hf = np.zeros((original_row_dim, packed_shape[1]), dtype=np.int8)
    for i in range(4):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)
        W_hf[start:end] = (W_hf_packed & mask) >> (2 * i)
    W_hf = W_hf - 1  # Convert to ternary
    print(f"  Shape: {W_hf.shape}")
    print(f"  First row[:10]: {W_hf[0, :10]}")

    # Compare each method to HF
    print("\n" + "="*60)
    print("Comparison to HF weights:")

    methods = [
        ("Method 1 (MSB, [out,in])", W_m1),
        ("Method 1b (MSB, [in,out].T)", W_m1b),
        ("Method 2 (LSB, [out,in])", W_m2),
    ]
    if 'W_m3' in dir():
        methods.append(("Method 3 (HF-style)", W_m3))

    for name, W in methods:
        if W.shape == W_hf.shape:
            match = (W == W_hf).mean()
            print(f"  {name}: {match:.1%} match")

            # Check transposed version
            if W.T.shape == W_hf.shape:
                match_T = (W.T == W_hf).mean()
                print(f"    Transposed: {match_T:.1%} match")
        else:
            print(f"  {name}: shape mismatch ({W.shape} vs {W_hf.shape})")
            if W.T.shape == W_hf.shape:
                match_T = (W.T == W_hf).mean()
                print(f"    Transposed: {match_T:.1%} match")

    # Try other approaches
    print("\n" + "="*60)
    print("Additional attempts:")

    # Method 4: Treat packed data as (out/4, in) directly
    packed_array = np.frombuffer(packed_bytes, dtype=np.uint8)
    packed_rows = out_features // 4
    print(f"\nMethod 4: Reshape packed as ({packed_rows}, {in_features})")
    try:
        packed_2d = packed_array.reshape(packed_rows, in_features)
        W_m4 = np.zeros((out_features, in_features), dtype=np.int8)
        for i in range(4):
            start = i * packed_rows
            end = start + packed_rows
            mask = 3 << (2 * i)
            W_m4[start:end] = ((packed_2d & mask) >> (2 * i)).astype(np.int8) - 1
        print(f"  Shape: {W_m4.shape}")
        print(f"  First row[:10]: {W_m4[0, :10]}")
        match = (W_m4 == W_hf).mean()
        print(f"  Match to HF: {match:.1%}")
    except Exception as e:
        print(f"  Error: {e}")

    # Method 5: Reshape packed as (in/4, out) - transposed
    print(f"\nMethod 5: Reshape packed as ({in_features // 4}, {out_features})")
    try:
        packed_2d = packed_array.reshape(in_features // 4, out_features)
        W_m5 = np.zeros((in_features, out_features), dtype=np.int8)
        for i in range(4):
            start = i * (in_features // 4)
            end = start + (in_features // 4)
            mask = 3 << (2 * i)
            W_m5[start:end] = ((packed_2d & mask) >> (2 * i)).astype(np.int8) - 1
        W_m5 = W_m5.T  # Transpose to (out, in)
        print(f"  Shape: {W_m5.shape}")
        print(f"  First row[:10]: {W_m5[0, :10]}")
        match = (W_m5 == W_hf).mean()
        print(f"  Match to HF: {match:.1%}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
