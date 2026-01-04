#!/usr/bin/env python3
"""Debug weight conversion to verify ternary quantization is correct.

This script compares:
1. Original checkpoint weights
2. What BitLinear.weight_quant() produces
3. What our conversion script produces

Usage:
    python scripts/debug_conversion.py /path/to/checkpoint
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def weight_quant_bitlinear(w: torch.Tensor, eps: float = 1e-5):
    """Exact replica of BitLinear.weight_quant()."""
    scale = 1.0 / w.abs().mean().clamp(min=eps)
    w_quant = (w * scale).round().clamp(-1, 1) / scale
    return w_quant, scale


def weight_quant_conversion(w: torch.Tensor, eps: float = 1e-5):
    """What convert_with_quantization.py does."""
    absmean = w.abs().mean().clamp(min=eps)
    scale = 1.0 / absmean
    w_ternary = (w * scale).round().clamp(-1, 1)
    # Returns ternary (-1, 0, 1) and absmean (the scale factor)
    return w_ternary, absmean.item()


def analyze_weight(name: str, w: torch.Tensor):
    """Analyze a weight tensor's distribution."""
    print(f"\n{'='*60}")
    print(f"Weight: {name}")
    print(f"  Shape: {w.shape}")
    print(f"  Dtype: {w.dtype}")
    print(f"  Range: [{w.min().item():.6f}, {w.max().item():.6f}]")
    print(f"  Mean: {w.mean().item():.6f}")
    print(f"  AbsMean: {w.abs().mean().item():.6f}")
    print(f"  Std: {w.std().item():.6f}")

    # Check unique values
    unique = torch.unique(w)
    if len(unique) <= 10:
        print(f"  Unique values ({len(unique)}): {unique.tolist()}")
    else:
        print(f"  Unique values: {len(unique)} (too many to show)")

    # Apply BitLinear quantization
    w_quant_bl, scale_bl = weight_quant_bitlinear(w)

    print(f"\n  After BitLinear.weight_quant():")
    print(f"    Scale (1/absmean): {scale_bl.item():.6f}")
    print(f"    Range: [{w_quant_bl.min().item():.6f}, {w_quant_bl.max().item():.6f}]")
    unique_quant = torch.unique(w_quant_bl)
    print(f"    Unique values ({len(unique_quant)}): {unique_quant.tolist()[:10]}")

    # Apply conversion quantization
    w_ternary, absmean = weight_quant_conversion(w)
    w_scaled = w_ternary.float() * absmean  # What we store in converted checkpoint

    print(f"\n  After conversion script:")
    print(f"    AbsMean (scale factor): {absmean:.6f}")
    print(f"    Ternary range: [{w_ternary.min().item()}, {w_ternary.max().item()}]")
    print(f"    Scaled range: [{w_scaled.min().item():.6f}, {w_scaled.max().item():.6f}]")
    unique_scaled = torch.unique(w_scaled)
    print(f"    Unique scaled values ({len(unique_scaled)}): {unique_scaled.tolist()[:10]}")

    # Verify they match
    diff = (w_quant_bl - w_scaled).abs().max().item()
    print(f"\n  Difference (BitLinear vs conversion): {diff:.10f}")

    # Show ternary distribution
    ternary_counts = {
        -1: (w_ternary == -1).sum().item(),
        0: (w_ternary == 0).sum().item(),
        1: (w_ternary == 1).sum().item(),
    }
    total = sum(ternary_counts.values())
    print(f"\n  Ternary distribution:")
    for val, count in ternary_counts.items():
        print(f"    {val:+d}: {count:,} ({100*count/total:.1f}%)")

    return diff < 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint directory")
    parser.add_argument("--layer", type=int, default=0, help="Layer number to analyze")
    args = parser.parse_args()

    # Find safetensors files
    sf_files = list(args.checkpoint.glob("*.safetensors"))
    if not sf_files:
        print(f"No safetensors files found in {args.checkpoint}")
        sys.exit(1)

    print(f"Analyzing checkpoint: {args.checkpoint}")
    print(f"Found {len(sf_files)} safetensors files")

    # Patterns to analyze
    patterns = [
        f"model.layers.{args.layer}.self_attn.q_proj.weight",
        f"model.layers.{args.layer}.self_attn.k_proj.weight",
        f"model.layers.{args.layer}.mlp.gate_proj.weight",
        f"model.layers.{args.layer}.mlp.down_proj.weight",
    ]

    all_match = True
    for sf_file in sf_files:
        print(f"\nFile: {sf_file.name}")
        with safe_open(sf_file, framework="pt") as f:
            keys = f.keys()
            for pattern in patterns:
                matching_keys = [k for k in keys if pattern in k or k == pattern]
                for key in matching_keys:
                    tensor = f.get_tensor(key)
                    match = analyze_weight(key, tensor)
                    if not match:
                        all_match = False

    print(f"\n{'='*60}")
    if all_match:
        print("✓ All conversions match BitLinear.weight_quant() exactly")
    else:
        print("✗ Some conversions don't match - there's a bug!")

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
