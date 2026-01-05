#!/usr/bin/env python3
"""Test weight repacking."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from wf_infer.kernels.bitnet_patch import pack_ternary_weights

# Test weight repacking
print("Testing weight repacking...")

weight = torch.randint(-1, 2, (256, 256)).float()
print(f"Original shape: {weight.shape}")
print(f"Original dtype: {weight.dtype}")
print(f"Unique values: {torch.unique(weight).tolist()}")

packed, scale = pack_ternary_weights(weight)
print(f"Packed shape: {packed.shape}")
print(f"Packed dtype: {packed.dtype}")
print(f"Scale: {scale.item()}")

# Verify packing by unpacking
unpacked = torch.zeros_like(weight)
for i in range(4):
    bits = (packed.to(torch.int32) >> (i * 2)) & 0x03
    unpacked[:, i::4] = bits.float() - 1.0

# Check if repacking matches
match = torch.allclose(weight, unpacked)
print(f"Repacking correct: {match}")

if not match:
    diff = (weight - unpacked).abs().sum()
    print(f"Total difference: {diff}")
