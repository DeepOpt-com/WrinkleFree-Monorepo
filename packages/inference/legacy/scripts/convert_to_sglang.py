#!/usr/bin/env python3
"""Convert HuggingFace BitNet model to sglang-bitnet format.

The packed weights are already in the correct format in the HuggingFace model.
We just need to ensure proper loading and serving.

Usage:
    uv run python scripts/convert_to_sglang.py --model microsoft/bitnet-b1.58-2B-4T

Outputs:
    models/<model-name>/sglang/ - Model ready for sglang-bitnet serving
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file, load_file


def convert_for_sglang(
    model_id: str,
    output_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Convert HuggingFace BitNet model to sglang-bitnet format.

    The microsoft/bitnet-b1.58-2B-4T model already has packed 1.58-bit weights.
    We just need to organize them for sglang serving.

    Args:
        model_id: HuggingFace model ID (e.g., microsoft/bitnet-b1.58-2B-4T)
        output_dir: Output directory (default: models/<model-name>/sglang/)
        cache_dir: HuggingFace cache directory

    Returns:
        Path to converted model directory
    """
    print(f"Converting {model_id} for sglang-bitnet...")

    # Download model
    print("Downloading model from HuggingFace...")
    model_path = Path(snapshot_download(
        model_id,
        cache_dir=cache_dir,
    ))

    # Set up output directory
    model_name = model_id.split("/")[-1]
    if output_dir is None:
        output_dir = Path("models") / model_name / "sglang"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Copy config files
    for config_file in ["config.json", "tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json", "generation_config.json"]:
        src = model_path / config_file
        if src.exists():
            shutil.copy(src, output_dir / config_file)
            print(f"  Copied {config_file}")

    # Load and verify weights
    weights_file = model_path / "model.safetensors"
    if not weights_file.exists():
        raise FileNotFoundError(f"No model.safetensors found in {model_path}")

    print("Loading weights...")
    state_dict = load_file(weights_file)

    # Analyze weight format
    print("\nWeight analysis:")
    packed_count = 0
    scale_count = 0

    for name, tensor in state_dict.items():
        if "weight_scale" in name:
            scale_count += 1
        elif tensor.dtype == torch.uint8:
            packed_count += 1
            if packed_count <= 3:
                print(f"  {name}: {tensor.shape} (packed uint8)")

    print(f"\n  Found {packed_count} packed weight tensors")
    print(f"  Found {scale_count} scale tensors")

    # Add quantization config for sglang-bitnet
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Add quantization config that sglang-bitnet expects
    config["quantization_config"] = {
        "quant_method": "bitnet",
        "block_size": 128,
        "activation_bits": 8,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("  Updated config.json with quantization_config")

    # Save weights (direct copy since format is already correct)
    output_weights = output_dir / "model.safetensors"
    shutil.copy(weights_file, output_weights)
    print(f"  Copied weights to {output_weights}")

    print(f"\nConversion complete: {output_dir}")
    print("\nTo serve with sglang-bitnet:")
    print(f"  python -m sglang.launch_server --model-path {output_dir}")

    return output_dir


def verify_weights(model_dir: Path) -> bool:
    """Verify the converted weights are in correct format."""
    weights_file = model_dir / "model.safetensors"
    state_dict = load_file(weights_file)

    # Check for packed weights and scales
    has_packed = any(t.dtype == torch.uint8 for t in state_dict.values())
    has_scales = any("weight_scale" in k for k in state_dict.keys())

    print(f"Verification:")
    print(f"  Has packed uint8 weights: {has_packed}")
    print(f"  Has weight scales: {has_scales}")

    return has_packed and has_scales


def main():
    parser = argparse.ArgumentParser(description="Convert BitNet model for sglang-bitnet")
    parser.add_argument(
        "--model",
        default="microsoft/bitnet-b1.58-2B-4T",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: models/<model-name>/sglang/)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing conversion"
    )

    args = parser.parse_args()

    if args.verify_only:
        if args.output_dir is None:
            model_name = args.model.split("/")[-1]
            args.output_dir = Path("models") / model_name / "sglang"
        verify_weights(args.output_dir)
    else:
        output_dir = convert_for_sglang(args.model, args.output_dir)
        verify_weights(output_dir)


if __name__ == "__main__":
    main()
