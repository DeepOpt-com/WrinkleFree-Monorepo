#!/usr/bin/env python3
"""Compare our Rust inference with HuggingFace reference implementation."""

import torch
import numpy as np
from safetensors import safe_open
import struct
import mmap

def load_gguf_embedding(gguf_path):
    """Load embedding from GGUF file."""
    with open(gguf_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Parse header
        magic = mm[0:4]
        version = struct.unpack_from('<I', mm, 4)[0]
        n_tensors = struct.unpack_from('<Q', mm, 8)[0]
        n_kv = struct.unpack_from('<Q', mm, 16)[0]

        offset = 24
        alignment = 32

        def read_string(off):
            length = struct.unpack_from('<Q', mm, off)[0]
            s = mm[off + 8:off + 8 + length].decode('utf-8')
            return s, 8 + length

        def skip_value(off, vtype):
            type_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
            if vtype == 8:
                _, c = read_string(off)
                return c
            elif vtype == 9:
                arr_type = struct.unpack_from('<I', mm, off)[0]
                arr_len = struct.unpack_from('<Q', mm, off + 4)[0]
                consumed = 12
                if arr_type == 8:
                    for _ in range(arr_len):
                        _, c = read_string(off + consumed)
                        consumed += c
                else:
                    consumed += arr_len * type_sizes.get(arr_type, 0)
                return consumed
            return type_sizes.get(vtype, 0)

        # Skip KV pairs
        for _ in range(n_kv):
            _, consumed = read_string(offset)
            offset += consumed
            vtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            offset += skip_value(offset, vtype)

        # Read tensor info
        tensors = {}
        for _ in range(n_tensors):
            name, consumed = read_string(offset)
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

        # Load embedding (F32)
        emb_info = tensors.get('token_embd.weight')
        if emb_info and emb_info['dtype'] == 0:  # F32
            vocab_size = emb_info['dims'][1]
            hidden_size = emb_info['dims'][0]
            abs_offset = data_offset + emb_info['offset']
            n_bytes = vocab_size * hidden_size * 4
            emb_data = np.frombuffer(mm[abs_offset:abs_offset + n_bytes], dtype=np.float32)
            return emb_data.reshape(vocab_size, hidden_size)

        return None

def main():
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"

    print(f"Loading embeddings from {gguf_path}...")
    embeddings = load_gguf_embedding(gguf_path)
    if embeddings is None:
        print("Failed to load embeddings")
        return

    print(f"Embedding shape: {embeddings.shape}")

    # Test input
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input: '{prompt}'")
    print(f"Token IDs: {input_ids.tolist()}")

    # Get embeddings
    embeddings = model.model.embed_tokens(input_ids)
    print(f"\n=== EMBEDDINGS ===")
    print(f"Shape: {embeddings.shape}")
    print(f"First 8 values: {embeddings[0, 0, :8].tolist()}")
    print(f"Min: {embeddings.min().item():.6f}, Max: {embeddings.max().item():.6f}")

    # Get attention norm weights
    attn_norm = model.model.layers[0].attn_norm.weight.data
    print(f"\n=== ATTN NORM (layer 0) ===")
    print(f"First 8 gamma: {attn_norm[:8].tolist()}")
    print(f"Min: {attn_norm.min().item():.6f}, Max: {attn_norm.max().item():.6f}")

    # Get FFN norm weights
    ffn_norm = model.model.layers[0].ffn_norm.weight.data
    print(f"\n=== FFN NORM (layer 0) ===")
    print(f"First 8 gamma: {ffn_norm[:8].tolist()}")
    print(f"Min: {ffn_norm.min().item():.6f}, Max: {ffn_norm.max().item():.6f}")

    # Apply attention norm manually to verify
    def rms_norm(x, weight, eps=1e-5):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return x_normed * weight

    normed_for_attn = rms_norm(embeddings[0, 0:1, :], attn_norm)
    print(f"\n=== AFTER ATTN NORM (position 0) ===")
    print(f"First 8 values: {normed_for_attn[0, :8].tolist()}")
    print(f"Min: {normed_for_attn.min().item():.6f}, Max: {normed_for_attn.max().item():.6f}")

    # Forward through the full model with hooks to capture intermediate values
    layer0_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                layer0_outputs[name] = output[0].detach()
            else:
                layer0_outputs[name] = output.detach()
        return hook

    # Register hooks
    hooks = []
    hooks.append(model.model.layers[0].register_forward_hook(make_hook("layer0_output")))

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print layer 0 output
    if "layer0_output" in layer0_outputs:
        l0_out = layer0_outputs["layer0_output"]
        print(f"\n=== LAYER 0 OUTPUT ===")
        print(f"Shape: {l0_out.shape}")
        print(f"First 8 values (last position): {l0_out[0, -1, :8].tolist()}")
        print(f"Min: {l0_out.min().item():.6f}, Max: {l0_out.max().item():.6f}")

    # Get logits
    logits = outputs.logits[0, -1, :]  # Last position logits
    print(f"\n=== LOGITS (last position) ===")
    print(f"Shape: {logits.shape}")
    print(f"Min: {logits.min().item():.6f}, Max: {logits.max().item():.6f}")

    # Top 5 predictions
    top5 = torch.topk(logits, 5)
    print(f"\nTop 5 predictions:")
    for i, (idx, val) in enumerate(zip(top5.indices.tolist(), top5.values.tolist())):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. Token {idx} ('{token}'): {val:.4f}")

    # Check token 11 specifically
    print(f"\nToken 11 logit: {logits[11].item():.4f}")
    sorted_indices = torch.argsort(logits, descending=True)
    rank = (sorted_indices == 11).nonzero().item()
    print(f"Token 11 rank: {rank}")

    # Generate output
    print(f"\n=== GENERATION ===")
    generated = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated: {output_text}")


if __name__ == "__main__":
    main()
