"""Convert HuggingFace BitNet model to packed format for caching."""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# Weight patterns that should be packed (BitNet linear layers)
BITNET_WEIGHT_PATTERNS = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
]


def _pack_ternary_weights(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack float ternary weights {-1, 0, +1} to uint8 (4 per byte).

    Matches the packing in bitnet.py:_pack_ternary_weights() exactly.

    Input: [out_features, in_features] float with values in {-1, 0, +1}
    Output: (packed [out_features, in_features/4] uint8, scale tensor)

    Packing is BLOCKED (matching kernel's AVX2 layout):
    For packed byte k (k=0..31 in each block):
    - bits 6-7: weight for input k
    - bits 4-5: weight for input k+32
    - bits 2-3: weight for input k+64
    - bits 0-1: weight for input k+96
    """
    out_features, in_features = weight.shape

    if in_features % 4 != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by 4")

    # Round to nearest ternary value and clamp
    ternary = torch.round(weight.float()).clamp(-1, 1)

    # Encode: -1 -> 0, 0 -> 1, +1 -> 2
    encoded = (ternary + 1).to(torch.uint8)

    # Pack 4 values per byte using BLOCKED layout
    block_size = 32
    packed = torch.zeros(
        out_features, in_features // 4, dtype=torch.uint8, device=weight.device
    )

    num_blocks = in_features // (block_size * 4)
    if num_blocks == 0:
        block_size = in_features // 4
        num_blocks = 1

    for b in range(num_blocks):
        base = b * block_size * 4
        for k in range(block_size):
            packed[:, b * block_size + k] = (
                (encoded[:, base + k] << 6)
                | (encoded[:, base + k + block_size] << 4)
                | (encoded[:, base + k + block_size * 2] << 2)
                | (encoded[:, base + k + block_size * 3])
            )

    scale = weight.abs().max().item()
    if scale < 1e-6:
        scale = 1.0

    return packed, torch.tensor([scale], dtype=torch.float32, device=weight.device)


def _is_ternary_float(weight: torch.Tensor) -> bool:
    """Check if float weight tensor contains only ternary values {-1, 0, +1}."""
    if weight.ndim != 2:
        return False

    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False

    # Check that values round to {-1, 0, +1}
    rounded = torch.round(weight.float())
    unique = torch.unique(rounded)
    if len(unique) > 3:
        return False
    if unique.abs().max() > 1.0:
        return False

    # Check in_features divisibility
    if weight.shape[1] % 4 != 0:
        return False

    return True


def convert_and_save_bitnet(
    source_model_path: str,
    output_path: Path,
    revision: Optional[str] = None,
) -> Path:
    """Convert HuggingFace BitNet model to packed format.

    Args:
        source_model_path: HuggingFace model ID or local path
        output_path: Where to save converted model
        revision: Model revision (for HF models)

    Returns:
        Path to converted model
    """
    from huggingface_hub import snapshot_download

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download or locate model
    if Path(source_model_path).exists():
        model_dir = Path(source_model_path)
    else:
        model_dir = Path(
            snapshot_download(
                source_model_path,
                revision=revision,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            )
        )

    logger.info(f"Converting model from {model_dir}")

    # Copy config and tokenizer files
    for config_file in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "generation_config.json",
    ]:
        src = model_dir / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)

    # Process weight files
    weight_files = list(model_dir.glob("*.safetensors"))

    for wf in weight_files:
        logger.info(f"Processing {wf.name}")
        weights = load_file(str(wf))
        converted_weights = {}

        for name, tensor in weights.items():
            is_bitnet_weight = any(p in name for p in BITNET_WEIGHT_PATTERNS)

            if is_bitnet_weight and _is_ternary_float(tensor):
                logger.debug(f"Packing {name}: {tensor.shape}")
                packed, scale = _pack_ternary_weights(tensor)

                # Store with naming convention that matches SGLang loading
                base_name = name.replace(".weight", "")
                converted_weights[f"{base_name}.qweight"] = packed
                converted_weights[f"{base_name}.weight_scale"] = scale
            else:
                # Keep non-BitNet weights as-is
                converted_weights[name] = tensor

        # Save converted weights
        output_file = output_path / wf.name
        save_file(converted_weights, str(output_file))
        logger.info(f"Saved {output_file}")

    # Add marker file for cache format
    cache_meta = {
        "format_version": "packed_bitnet_v1",
        "source_model": source_model_path,
        "revision": revision,
    }
    with open(output_path / "cache_metadata.json", "w") as f:
        json.dump(cache_meta, f, indent=2)

    return output_path
