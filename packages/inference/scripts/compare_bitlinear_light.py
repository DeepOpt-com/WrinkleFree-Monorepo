#!/usr/bin/env python3
"""Light comparison of BitLinear: load only needed weights from safetensors."""

import torch
import numpy as np
from safetensors import safe_open
import struct
import mmap

def load_gguf_i2s_tensor(gguf_path, tensor_name):
    """Load a single I2_S tensor from GGUF."""
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

    def decode_i2s(data, n_elements):
        output = np.zeros(n_elements, dtype=np.int8)
        idx = 0
        for byte in data:
            if idx >= n_elements:
                break
            for shift in [6, 4, 2, 0]:
                if idx >= n_elements:
                    break
                val = (byte >> shift) & 0x03
                output[idx] = val - 1  # 00=-1, 01=0, 10=+1
                idx += 1
        return output

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
            # Scale is in the extra 32 bytes
            scale_bytes = mm[abs_offset + packed_size:abs_offset + packed_size + 4]
            scale = struct.unpack('<f', scale_bytes)[0]
            ternary = decode_i2s(packed_data, n_elements)
            return {'data': ternary, 'dims': info['dims'], 'scale': scale}

        return None


def main():
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"
    hf_path = "/tmp/bitnet-hf"

    # Load GGUF gate_proj weights
    print("Loading GGUF gate_proj for layer 0...")
    gate_gguf = load_gguf_i2s_tensor(gguf_path, "blk.0.ffn_gate.weight")
    if gate_gguf is None:
        print("ERROR: Could not load gate_proj from GGUF")
        return

    in_features = gate_gguf['dims'][0]
    out_features = gate_gguf['dims'][1]
    print(f"  Shape: ({out_features}, {in_features})")
    print(f"  Scale: {gate_gguf['scale']}")

    W_gguf = gate_gguf['data'].reshape(out_features, in_features)
    print(f"  First row[:10]: {W_gguf[0, :10].tolist()}")

    # Load HuggingFace gate_proj weights from safetensors
    print("\nLoading HuggingFace gate_proj from safetensors...")
    hf_weights_file = f"{hf_path}/model.safetensors"
    try:
        with safe_open(hf_weights_file, framework="pt") as f:
            # Find the gate_proj weights
            keys = list(f.keys())
            gate_key = "model.layers.0.mlp.gate_proj.weight"
            if gate_key in keys:
                W_hf = f.get_tensor(gate_key).numpy().astype(np.float32)
                print(f"  Shape: {W_hf.shape}")
                print(f"  First row[:10]: {W_hf[0, :10].tolist()}")
                print(f"  Unique values: {np.unique(W_hf)[:10]}")
            else:
                print(f"  Key {gate_key} not found. Available keys (first 10):")
                for k in keys[:10]:
                    print(f"    {k}")
                return
    except Exception as e:
        print(f"  Error loading safetensors: {e}")
        return

    # Compare weights
    print("\n=== Weight Comparison ===")
    W_gguf_f = W_gguf.astype(np.float32)
    match = (W_gguf_f == W_hf).sum()
    total = W_hf.size
    print(f"  Matching elements: {match}/{total} ({match/total*100:.1f}%)")

    # If weights are stored transposed
    W_gguf_T = W_gguf_f.T
    match_T = (W_gguf_T == W_hf).sum()
    print(f"  Matching (transposed): {match_T}/{total} ({match_T/total*100:.1f}%)")

    # Check if HF weights are actually ternary
    n_minus1 = (W_hf == -1).sum()
    n_zero = (W_hf == 0).sum()
    n_plus1 = (W_hf == 1).sum()
    print(f"\n  HF Ternary distribution: -1: {n_minus1/total*100:.1f}%, 0: {n_zero/total*100:.1f}%, +1: {n_plus1/total*100:.1f}%")

    n_minus1 = (W_gguf == -1).sum()
    n_zero = (W_gguf == 0).sum()
    n_plus1 = (W_gguf == 1).sum()
    print(f"  GGUF Ternary distribution: -1: {n_minus1/total*100:.1f}%, 0: {n_zero/total*100:.1f}%, +1: {n_plus1/total*100:.1f}%")

    # Test forward pass
    print("\n=== Forward Pass Comparison ===")

    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(in_features).astype(np.float32) * 10

    # HuggingFace forward (simple matmul since weights are ternary)
    # NOTE: HF BitLinear may have additional scaling - let's check
    hf_output_raw = W_hf @ test_input
    print(f"  HF output (raw matmul): min={hf_output_raw.min():.2f}, max={hf_output_raw.max():.2f}")

    # My BitLinear forward with absmax quantization
    max_abs = np.abs(test_input).max()
    input_scale = max_abs / 127.0
    input_quant = np.round(test_input / input_scale).clip(-127, 127).astype(np.int8)

    output_int = W_gguf.astype(np.int32) @ input_quant.astype(np.int32)
    my_output = output_int.astype(np.float32) * input_scale * gate_gguf['scale']
    print(f"  My output (quantized): min={my_output.min():.2f}, max={my_output.max():.2f}")

    # Compare
    print(f"\n  Ratio (my/HF_raw): {my_output.max() / hf_output_raw.max():.4f}x")

    # The key question: does HuggingFace BitLinear apply any scaling?
    # Let me check what the expected output should be
    print("\n=== Investigating HF BitLinear behavior ===")

    # HF BitLinear should apply: weight_scale * (ternary @ quant(x)) * input_scale
    # If HF doesn't quantize input, then: output = ternary @ x
    # If HF quantizes input: output = (ternary @ quant(x)) * input_scale * weight_scale

    # Let's see if raw matmul gives reasonable outputs
    print(f"  If HF doesn't quantize: output = W @ x")
    print(f"    Output range: [{hf_output_raw.min():.2f}, {hf_output_raw.max():.2f}]")

    # Maybe HF applies weight_scale only?
    hf_with_wscale = hf_output_raw * gate_gguf['scale']
    print(f"  If HF applies weight_scale: output = (W @ x) * {gate_gguf['scale']:.4f}")
    print(f"    Output range: [{hf_with_wscale.min():.2f}, {hf_with_wscale.max():.2f}]")

    # What if HF ALSO quantizes but uses a different scaling formula?
    # Let's check without weight_scale
    my_output_no_wscale = output_int.astype(np.float32) * input_scale
    print(f"\n  My output without weight_scale:")
    print(f"    Output range: [{my_output_no_wscale.min():.2f}, {my_output_no_wscale.max():.2f}]")

    ratio_without_wscale = my_output_no_wscale.max() / hf_output_raw.max()
    print(f"    Ratio (my_no_wscale/HF_raw): {ratio_without_wscale:.4f}x")

    # So the question is: is weight_scale supposed to be applied or not?
    # The GGUF I2_S format stores weight_scale, but maybe HF model doesn't use it?
    print(f"\n=== CONCLUSION ===")
    print(f"  GGUF weight_scale: {gate_gguf['scale']:.4f}")
    print(f"  This scale {gate_gguf['scale']:.4f} makes output {gate_gguf['scale']:.4f}x larger")

    # Check what HF actually does in BitLinear
    print("\n=== Checking HF BitLinear implementation ===")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "bitnet_module",
            f"{hf_path}/bitnet.py"
        )
        if spec and spec.loader:
            print("  Found bitnet.py in HF model directory")
            # Let's read it to see the forward pass
            with open(f"{hf_path}/bitnet.py", 'r') as f:
                content = f.read()
                # Look for forward method
                if "def forward" in content:
                    print("  Found forward method in bitnet.py")
                    # Print the forward implementation
                    lines = content.split('\n')
                    in_forward = False
                    forward_lines = []
                    indent_level = 0
                    for line in lines:
                        if 'def forward' in line and 'BitLinear' in ''.join(lines[max(0,lines.index(line)-20):lines.index(line)]):
                            in_forward = True
                            indent_level = len(line) - len(line.lstrip())
                        elif in_forward:
                            if line.strip() and not line.startswith(' ' * (indent_level + 1)):
                                break
                            forward_lines.append(line)
                    if forward_lines:
                        print("\n  BitLinear.forward implementation:")
                        for line in forward_lines[:30]:
                            print(f"    {line}")
    except Exception as e:
        print(f"  Could not load bitnet.py: {e}")


if __name__ == "__main__":
    main()
