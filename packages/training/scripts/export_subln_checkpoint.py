#!/usr/bin/env python3
"""Export WrinkleFree SubLN checkpoints to llama.cpp-compatible format.

SubLN (Sub-Layer Normalization) in WrinkleFree wraps projections in nn.Sequential:
    o_proj = nn.Sequential(subln, linear)  # .0 = SubLN, .1 = Linear

llama.cpp expects separate named tensors:
    attn_sub_norm.weight  (SubLN before o_proj)
    o_proj.weight         (the projection itself)

This script renames tensors to match llama.cpp's expected format.

Usage:
    uv run --package wrinklefree python scripts/export_subln_checkpoint.py \
        --checkpoint /path/to/checkpoint.pt \
        --output-dir /path/to/output \
        --config-path /path/to/config.json  # optional
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


# Tensor renaming rules for SubLN
SUBLN_RENAMES = {
    # Attention SubLN: o_proj.0 -> attn_sub_norm
    ".self_attn.o_proj.0.weight": ".self_attn.attn_sub_norm.weight",
    ".self_attn.o_proj.1.weight": ".self_attn.o_proj.weight",
    ".self_attn.o_proj.1.bias": ".self_attn.o_proj.bias",
    # FFN SubLN: down_proj.0 -> ffn_sub_norm
    ".mlp.down_proj.0.weight": ".mlp.ffn_sub_norm.weight",
    ".mlp.down_proj.1.weight": ".mlp.down_proj.weight",
    ".mlp.down_proj.1.bias": ".mlp.down_proj.bias",
}


def rename_subln_tensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename SubLN tensors from WrinkleFree format to llama.cpp format.

    WrinkleFree format (nn.Sequential wrapping):
        model.layers.{bid}.self_attn.o_proj.0.weight  -> SubLN scale [n_embd]
        model.layers.{bid}.self_attn.o_proj.1.weight  -> o_proj weights [n_embd, n_embd]
        model.layers.{bid}.mlp.down_proj.0.weight     -> SubLN scale [n_ff]
        model.layers.{bid}.mlp.down_proj.1.weight     -> down_proj weights [n_embd, n_ff]

    llama.cpp format:
        model.layers.{bid}.self_attn.attn_sub_norm.weight  -> [n_embd]
        model.layers.{bid}.self_attn.o_proj.weight         -> [n_embd, n_embd]
        model.layers.{bid}.mlp.ffn_sub_norm.weight         -> [n_ff]
        model.layers.{bid}.mlp.down_proj.weight            -> [n_embd, n_ff]
    """
    renamed = {}
    rename_count = 0

    for key, value in state_dict.items():
        new_key = key

        # Apply SubLN renaming rules
        for pattern, replacement in SUBLN_RENAMES.items():
            if pattern in key:
                new_key = key.replace(pattern, replacement)
                rename_count += 1
                break

        renamed[new_key] = value

    if rename_count > 0:
        print(f"Renamed {rename_count} SubLN tensors")
    else:
        print("No SubLN tensors found (model may not use SubLN)")

    return renamed


def handle_tied_embeddings(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove lm_head if it shares memory with embed_tokens (tied embeddings).

    For BITNET architecture, llama.cpp doesn't support separate lm_head tensors
    (MODEL_TENSOR.OUTPUT not in BITNET arch). When embeddings are tied,
    llama.cpp uses embed_tokens for both, so we just remove lm_head.
    """
    lm_head_key = "lm_head.weight"
    embed_key = "model.embed_tokens.weight"

    if lm_head_key in state_dict and embed_key in state_dict:
        if state_dict[lm_head_key].data_ptr() == state_dict[embed_key].data_ptr():
            print("Detected tied embeddings, removing lm_head.weight (llama.cpp uses embed_tokens)")
            del state_dict[lm_head_key]
        else:
            # Embeddings not tied - llama.cpp BITNET arch doesn't support this
            print("WARNING: lm_head not tied to embed_tokens, but BITNET arch doesn't support OUTPUT tensor")
            print("         Removing lm_head.weight - model may not work correctly!")
            del state_dict[lm_head_key]

    return state_dict


def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
) -> None:
    """Export a WrinkleFree checkpoint to llama.cpp-compatible format.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint (.pt file)
        output_dir: Directory to save the exported model
        config_path: Optional path to config.json (will be copied to output)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict (handle different checkpoint formats)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    print(f"Found {len(state_dict)} tensors")

    # Check for SubLN patterns
    subln_keys = [k for k in state_dict.keys() if ".0.weight" in k or ".1.weight" in k]
    if subln_keys:
        print(f"Detected SubLN architecture ({len(subln_keys)} Sequential tensors)")
        for k in subln_keys[:4]:
            print(f"  - {k}")
        if len(subln_keys) > 4:
            print(f"  ... and {len(subln_keys) - 4} more")

    # Rename SubLN tensors
    state_dict = rename_subln_tensors(state_dict)

    # Handle tied embeddings
    state_dict = handle_tied_embeddings(state_dict)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as safetensors
    output_file = output_dir / "model.safetensors"
    print(f"Saving to {output_file}")
    save_file(state_dict, str(output_file))

    # Copy config.json if provided
    if config_path and config_path.exists():
        dest_config = output_dir / "config.json"
        shutil.copy(config_path, dest_config)
        print(f"Copied config.json to {dest_config}")

    # Verify output
    print("\nExported tensors (sample):")
    for i, key in enumerate(sorted(state_dict.keys())):
        if i < 10 or "sub_norm" in key:
            shape = list(state_dict[key].shape)
            print(f"  {key}: {shape}")
        elif i == 10:
            print(f"  ... and {len(state_dict) - 10} more tensors")

    # Check for SubLN tensors in output
    subln_output = [k for k in state_dict.keys() if "sub_norm" in k]
    if subln_output:
        print(f"\nSubLN tensors in output ({len(subln_output)}):")
        for k in subln_output[:6]:
            print(f"  - {k}: {list(state_dict[k].shape)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export WrinkleFree SubLN checkpoints to llama.cpp format"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PyTorch checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for exported model",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to config.json (optional, will be copied to output)",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    export_checkpoint(args.checkpoint, args.output_dir, args.config_path)
    print("\nExport complete!")


if __name__ == "__main__":
    main()
