#!/usr/bin/env python3
"""Trace layer 0 values through HuggingFace BitNet model for comparison with Rust."""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "microsoft/BitNet-b1.58-2B-4T"

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for comparison
        device_map="cpu"
    )
    model.eval()

    # Test input
    prompt = "Hello"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"\n=== TOKENIZATION ===")
    print(f"Prompt: '{prompt}'")
    print(f"Token IDs: {input_ids.tolist()}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids[0].tolist()]}")

    # Get embeddings
    with torch.no_grad():
        embeddings = model.model.embed_tokens(input_ids)

    print(f"\n=== EMBEDDINGS ===")
    print(f"Shape: {embeddings.shape}")
    print(f"First 8 values (pos 0): {embeddings[0, 0, :8].tolist()}")
    print(f"Min: {embeddings.min().item():.6f}, Max: {embeddings.max().item():.6f}")

    # Manual RMS norm
    def rms_norm(x, weight, eps=1e-5):
        variance = x.pow(2).mean(-1, keepdim=True)
        rms = torch.sqrt(variance + eps)
        x_normed = x / rms
        return x_normed * weight, rms

    # Layer 0 attention norm
    attn_norm_weight = model.model.layers[0].attn_norm.weight.data
    print(f"\n=== ATTN NORM WEIGHT (layer 0) ===")
    print(f"First 8: {attn_norm_weight[:8].tolist()}")
    print(f"Min: {attn_norm_weight.min().item():.6f}, Max: {attn_norm_weight.max().item():.6f}")

    # Apply attention norm to embeddings
    attn_normed, rms = rms_norm(embeddings[0, 0:1, :], attn_norm_weight)
    print(f"\n=== AFTER ATTN NORM (pos 0) ===")
    print(f"RMS: {rms[0, 0].item():.6f}")
    print(f"First 8: {attn_normed[0, :8].tolist()}")
    print(f"Min: {attn_normed.min().item():.6f}, Max: {attn_normed.max().item():.6f}")

    # FFN norm weight
    ffn_norm_weight = model.model.layers[0].ffn_norm.weight.data
    print(f"\n=== FFN NORM WEIGHT (layer 0) ===")
    print(f"First 8: {ffn_norm_weight[:8].tolist()}")
    print(f"Min: {ffn_norm_weight.min().item():.6f}, Max: {ffn_norm_weight.max().item():.6f}")

    # Get layer 0 weights
    layer0 = model.model.layers[0]

    # Check for BitLinear layers
    print(f"\n=== LAYER 0 MLP STRUCTURE ===")
    print(f"gate_proj type: {type(layer0.mlp.gate_proj)}")
    print(f"up_proj type: {type(layer0.mlp.up_proj)}")
    print(f"down_proj type: {type(layer0.mlp.down_proj)}")

    # Check if weights are ternary
    if hasattr(layer0.mlp.gate_proj, 'weight'):
        gate_w = layer0.mlp.gate_proj.weight.data
        print(f"\ngate_proj weight shape: {gate_w.shape}")
        unique_vals = torch.unique(gate_w)
        print(f"Unique values: {unique_vals.tolist()[:20]} (showing first 20)")

        # Count ternary values
        n_neg1 = (gate_w == -1).sum().item()
        n_zero = (gate_w == 0).sum().item()
        n_pos1 = (gate_w == 1).sum().item()
        total = gate_w.numel()
        print(f"Ternary distribution: -1:{n_neg1} ({100*n_neg1/total:.1f}%), 0:{n_zero} ({100*n_zero/total:.1f}%), +1:{n_pos1} ({100*n_pos1/total:.1f}%)")

        # Weight scale
        if hasattr(layer0.mlp.gate_proj, 'weight_scale'):
            print(f"Weight scale: {layer0.mlp.gate_proj.weight_scale}")

    # Forward through layer 0 with hooks
    layer0_intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            layer0_intermediates[name] = {
                'input': input[0].detach() if isinstance(input, tuple) else input.detach(),
                'output': output[0].detach() if isinstance(output, tuple) else output.detach()
            }
        return hook

    # Register hooks
    hooks = []
    hooks.append(layer0.register_forward_hook(make_hook("layer0")))
    hooks.append(layer0.self_attn.register_forward_hook(make_hook("attn")))
    hooks.append(layer0.mlp.register_forward_hook(make_hook("mlp")))
    if hasattr(layer0.mlp, 'gate_proj'):
        hooks.append(layer0.mlp.gate_proj.register_forward_hook(make_hook("gate_proj")))
        hooks.append(layer0.mlp.up_proj.register_forward_hook(make_hook("up_proj")))
        hooks.append(layer0.mlp.down_proj.register_forward_hook(make_hook("down_proj")))

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print intermediates
    for name, data in layer0_intermediates.items():
        inp = data['input']
        out = data['output']
        print(f"\n=== {name.upper()} ===")
        print(f"Input shape: {inp.shape if hasattr(inp, 'shape') else type(inp)}")
        if hasattr(inp, 'shape'):
            print(f"Input first 8 (pos 0): {inp[0, 0, :8].tolist() if len(inp.shape) == 3 else inp[0, :8].tolist()}")
            print(f"Input min: {inp.min().item():.4f}, max: {inp.max().item():.4f}")
        print(f"Output shape: {out.shape if hasattr(out, 'shape') else type(out)}")
        if hasattr(out, 'shape'):
            print(f"Output first 8 (pos 0): {out[0, 0, :8].tolist() if len(out.shape) == 3 else out[0, :8].tolist()}")
            print(f"Output min: {out.min().item():.4f}, max: {out.max().item():.4f}")

    # Get logits
    logits = outputs.logits[0, -1, :]
    print(f"\n=== LOGITS (last position) ===")
    print(f"Shape: {logits.shape}")
    print(f"Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}")

    # Top 5 predictions
    top5 = torch.topk(logits, 5)
    print(f"\nTop 5 predictions:")
    for i, (idx, val) in enumerate(zip(top5.indices.tolist(), top5.values.tolist())):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. Token {idx} ('{token}'): {val:.4f}")

    print(f"\n=== GENERATION ===")
    generated = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated: {output_text}")


if __name__ == "__main__":
    main()
