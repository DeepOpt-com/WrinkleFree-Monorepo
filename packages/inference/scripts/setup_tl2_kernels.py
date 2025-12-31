#!/usr/bin/env python3
"""
Generate TL2 kernels for any BitNet model.

TL2 (Table Lookup 2) is the fastest lossless quantization format for BitNet models.
It uses lookup tables with 5-bit indices for groups of 3 ternary weights.

Usage:
    # From model config.json
    python setup_tl2_kernels.py --config models/dlm-bitnet-2b/config.json

    # With explicit dimensions
    python setup_tl2_kernels.py --hidden-size 2560 --intermediate-size 6912

    # Just show commands (don't run)
    python setup_tl2_kernels.py --config models/dlm-bitnet-2b/config.json --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_divisor(n: int, candidates: list[int]) -> int:
    """Find the largest divisor of n from candidates list."""
    for c in sorted(candidates, reverse=True):
        if n % c == 0:
            return c
    # If no exact divisor, find one that satisfies constraints
    for c in sorted(candidates, reverse=True):
        if n >= c:
            return c
    return candidates[-1]


def get_optimal_block_params(M: int, K: int) -> tuple[int, int, int]:
    """
    Calculate optimal BM, BK, bm parameters for a matrix of shape [M, K].

    Constraints:
    - M % BM == 0
    - (K % BK) % 32 == 0
    - bm in [32]
    """
    # BM candidates (must divide M)
    bm_candidates = [320, 256, 160, 128, 64, 32]
    BM = find_divisor(M, bm_candidates)

    # BK candidates (K % BK must be divisible by 32, or K % BK == 0)
    # Common values: 96, 192, 128, 64
    bk_candidates = [192, 128, 96, 64, 32]
    for bk in bk_candidates:
        remainder = K % bk
        if remainder == 0 or remainder % 32 == 0:
            BK = bk
            break
    else:
        BK = 96  # Default

    # bm is always 32 for TL2
    bm = 32

    return BM, BK, bm


def get_kernel_shapes(hidden_size: int, intermediate_size: int,
                      num_attention_heads: int = None, num_key_value_heads: int = None,
                      head_dim: int = None) -> list[list[int]]:
    """
    Get the kernel shapes for a BitNet model.

    For a transformer with:
    - hidden_size: dimension of embeddings
    - intermediate_size: dimension of MLP hidden layer
    - num_attention_heads: number of query heads
    - num_key_value_heads: number of KV heads (for GQA)
    - head_dim: dimension per head

    PyTorch stores weights as [out_features, in_features].
    Kernel config expects [M, K] where M=out_dim, K=in_dim.

    The matrices are (in PyTorch shape format [out, in]):
    1. MLP gate/up: [intermediate_size, hidden_size]
    2. MLP down: [hidden_size, intermediate_size]
    3. Attention Q/O: [hidden_size, hidden_size]
    4. Attention K/V: [kv_dim, hidden_size] (if using GQA)
    """
    shapes = [
        [intermediate_size, hidden_size],      # gate_proj, up_proj: hidden → intermediate
        [hidden_size, intermediate_size],      # down_proj: intermediate → hidden
        [hidden_size, hidden_size],            # q_proj, o_proj: hidden → hidden
    ]

    # Add GQA K/V shapes if different from hidden_size
    if num_key_value_heads is not None and head_dim is not None:
        kv_dim = num_key_value_heads * head_dim
        if kv_dim != hidden_size:
            shapes.append([kv_dim, hidden_size])  # k_proj, v_proj: hidden → kv_dim

    return shapes


def main():
    parser = argparse.ArgumentParser(
        description="Generate TL2 kernels for BitNet models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", type=Path, help="Path to model config.json")
    parser.add_argument("--hidden-size", type=int, help="Model hidden size")
    parser.add_argument("--intermediate-size", type=int, help="Model intermediate size")
    parser.add_argument("--bitnet-dir", type=Path,
                        default=Path(__file__).parent.parent.parent.parent / "extern" / "BitNet",
                        help="Path to BitNet directory")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without running")
    parser.add_argument("--model-name", type=str, default="custom_model",
                        help="Name for the model (used in codegen)")

    args = parser.parse_args()

    # Get dimensions from config or args
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        num_attention_heads = config.get("num_attention_heads")
        num_key_value_heads = config.get("num_key_value_heads")
        head_dim = config.get("head_dim")
        print(f"Loaded config from {args.config}")
    elif args.hidden_size and args.intermediate_size:
        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size
        num_attention_heads = None
        num_key_value_heads = None
        head_dim = None
    else:
        parser.error("Must provide either --config or both --hidden-size and --intermediate-size")

    print(f"Model dimensions: hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    if num_key_value_heads and head_dim:
        print(f"GQA: num_kv_heads={num_key_value_heads}, head_dim={head_dim}, kv_dim={num_key_value_heads * head_dim}")

    # Calculate kernel shapes and parameters
    kernel_shapes = get_kernel_shapes(
        hidden_size, intermediate_size,
        num_attention_heads, num_key_value_heads, head_dim
    )

    BM_list = []
    BK_list = []
    bm_list = []

    print("\nKernel configurations:")
    print("-" * 60)
    for i, (M, K) in enumerate(kernel_shapes):
        BM, BK, bm = get_optimal_block_params(M, K)
        BM_list.append(BM)
        BK_list.append(BK)
        bm_list.append(bm)

        # Verify constraints
        assert M % BM == 0, f"M={M} not divisible by BM={BM}"
        remainder = K % BK
        assert remainder == 0 or remainder % 32 == 0, f"(K={K} % BK={BK}) % 32 != 0"

        print(f"  Shape [{M:5d}, {K:5d}]: BM={BM:3d}, BK={BK:3d}, bm={bm}")

    # Build codegen command
    BM_str = ",".join(str(x) for x in BM_list)
    BK_str = ",".join(str(x) for x in BK_list)
    bm_str = ",".join(str(x) for x in bm_list)

    # We need to modify the codegen script to accept custom shapes
    # For now, add the model to the ModelShapeDict manually

    codegen_script = args.bitnet_dir / "utils" / "codegen_tl2.py"

    print("\n" + "=" * 60)
    print("TL2 KERNEL GENERATION")
    print("=" * 60)

    # Create a custom ModelShapeDict entry
    shapes_str = str(kernel_shapes).replace(" ", "")

    print(f"""
To generate TL2 kernels, add this model to {codegen_script}:

    "{args.model_name}": {kernel_shapes},

Then run:
    cd {args.bitnet_dir}
    python utils/codegen_tl2.py --model {args.model_name} --BM {BM_str} --BK {BK_str} --bm {bm_str}

Or use the one-liner below to patch and run:
""")

    # Generate a one-liner that patches the script and runs it
    patch_cmd = f"""cd {args.bitnet_dir} && \\
python -c "
import sys
sys.path.insert(0, 'utils')
from codegen_tl2 import *

kernel_shapes = {kernel_shapes}
BM_list = [{BM_str}]
BK_list = [{BK_str}]
bm_list = [{bm_str}]

k_list = []
for i in range(len(kernel_shapes)):
    k_list.append(get_three_k_two_k(kernel_shapes[i][1], BK_list[i]))

tbl_impl_code = []
for i in range(len(kernel_shapes)):
    tbl_impl_code.append(
        gen_tbl_impl('{{}}_{{}}'.format(kernel_shapes[i][0], kernel_shapes[i][1]), BM_list[i], BK_list[i], bm_list[i], k_list[i])
    )

ctor_code = gen_ctor_code()
api_code = gen_top_api(kernel_shapes, k_list)
trans_code = gen_transform_code(kernel_shapes)

import os
output_dir = 'include'
with open(os.path.join(output_dir, 'bitnet-lut-kernels.h'), 'w') as f:
    f.write('#if defined(GGML_BITNET_X86_TL2)')
    f.write(ctor_code)
    for code in tbl_impl_code:
        f.write(code)
    f.write(api_code)
    f.write(trans_code)
    f.write('#endif')

from configparser import ConfigParser
config = ConfigParser()
for i in range(len(kernel_shapes)):
    config.add_section('Kernels_{{}}'.format(i))
    config.set('Kernels_{{}}'.format(i), 'M', str(kernel_shapes[i][0]))
    config.set('Kernels_{{}}'.format(i), 'K', str(kernel_shapes[i][1]))
    config.set('Kernels_{{}}'.format(i), 'BM', str(BM_list[i]))
    config.set('Kernels_{{}}'.format(i), 'BK', str(BK_list[i]))
    config.set('Kernels_{{}}'.format(i), 'bmm', str(bm_list[i]))
with open(os.path.join(output_dir, 'kernel_config.ini'), 'w') as f:
    config.write(f)

print('Generated: include/bitnet-lut-kernels.h')
print('Generated: include/kernel_config.ini')
"
"""

    if args.dry_run:
        print("DRY RUN - would execute:")
        print(patch_cmd)
    else:
        print("Generating kernels...")
        result = subprocess.run(patch_cmd, shell=True, cwd=args.bitnet_dir)
        if result.returncode == 0:
            print("\nKernels generated successfully!")
            print(f"\nNext steps:")
            print(f"  1. Rebuild llama.cpp with TL2 support:")
            print(f"     cd {args.bitnet_dir} && cmake -B build -DGGML_BITNET_X86_TL2=ON && cmake --build build -j4")
            print(f"  2. Convert model to TL2 format:")
            print(f"     python utils/convert-hf-to-gguf-bitnet.py <checkpoint> --outtype tl2 --outfile model.gguf")
        else:
            print(f"Error generating kernels (exit code {result.returncode})")
            sys.exit(1)


if __name__ == "__main__":
    main()
