"""Model conversion utilities for BitNet architecture.

Provides utilities for:
- Converting pre-trained models to BitNet architecture
- Auto-detecting if a model is already BitNet
- On-the-fly conversion during training
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from bitnet_arch.layers.bitlinear import BitLinear
from bitnet_arch.layers.subln import SubLN

logger = logging.getLogger(__name__)


def is_bitnet_model(model: nn.Module) -> bool:
    """
    Check if a model is already a BitNet model.

    A model is considered BitNet if it contains at least one BitLinear layer.

    Args:
        model: The model to check

    Returns:
        True if the model contains BitLinear layers, False otherwise
    """
    for module in model.modules():
        if isinstance(module, BitLinear):
            return True
    return False


def auto_convert_if_needed(
    model: nn.Module,
    hidden_size: int,
    intermediate_size: int,
    exclude_layers: Optional[list[str]] = None,
) -> nn.Module:
    """
    Convert model to BitNet on-the-fly if not already converted.

    This enables a simplified training flow where users don't need to
    explicitly run Stage 1 - conversion happens automatically.

    Args:
        model: The model to potentially convert
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        exclude_layers: Layer names to exclude from conversion

    Returns:
        The model (converted if needed, unchanged if already BitNet)
    """
    if is_bitnet_model(model):
        logger.info("Model already contains BitLinear layers, skipping conversion")
        return model

    logger.info("Model is not BitNet, converting automatically...")
    return convert_model_to_bitnet(
        model,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        exclude_layers=exclude_layers,
    )


def insert_subln_before_projection(
    module: nn.Module,
    projection_name: str,
    hidden_size: int,
) -> None:
    """
    Insert SubLN layer before a projection (e.g., o_proj, down_proj).

    This is done by wrapping the projection in a sequential module
    with SubLN first.

    Args:
        module: Parent module containing the projection
        projection_name: Name of the projection attribute
        hidden_size: Hidden size for SubLN
    """
    if hasattr(module, projection_name):
        original_proj = getattr(module, projection_name)

        # Determine input size for SubLN
        if isinstance(original_proj, nn.Linear):
            subln_size = original_proj.in_features
        else:
            subln_size = hidden_size

        # Create SubLN-wrapped projection
        subln = SubLN(subln_size)

        # Create new projection (BitLinear for quantized training)
        if isinstance(original_proj, BitLinear):
            new_proj = original_proj
        else:
            new_proj = BitLinear(
                original_proj.in_features,
                original_proj.out_features,
                bias=original_proj.bias is not None,
            )
            new_proj.weight.data.copy_(original_proj.weight.data)
            if original_proj.bias is not None:
                new_proj.bias.data.copy_(original_proj.bias.data)

        # Wrap in Sequential
        wrapped = nn.Sequential(subln, new_proj)
        setattr(module, projection_name, wrapped)


def convert_attention_layer(
    attention_module: nn.Module,
    hidden_size: int,
    exclude_layers: list[str],
) -> None:
    """
    Convert attention layer to BitNet with SubLN.

    Inserts SubLN before o_proj (output projection).

    Args:
        attention_module: The attention module to convert
        hidden_size: Model hidden size
        exclude_layers: Layer names to exclude from conversion
    """
    # Convert Q, K, V projections to BitLinear
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        if hasattr(attention_module, proj_name) and proj_name not in exclude_layers:
            proj = getattr(attention_module, proj_name)
            if isinstance(proj, nn.Linear) and not isinstance(proj, BitLinear):
                new_proj = BitLinear(
                    proj.in_features,
                    proj.out_features,
                    bias=proj.bias is not None,
                )
                new_proj.weight.data.copy_(proj.weight.data)
                if proj.bias is not None:
                    new_proj.bias.data.copy_(proj.bias.data)
                setattr(attention_module, proj_name, new_proj)

    # Insert SubLN before o_proj
    if hasattr(attention_module, "o_proj") and "o_proj" not in exclude_layers:
        o_proj = attention_module.o_proj
        o_proj_in = o_proj.in_features if isinstance(o_proj, nn.Linear) else hidden_size

        subln = SubLN(o_proj_in)

        if isinstance(o_proj, nn.Linear) and not isinstance(o_proj, BitLinear):
            new_o_proj = BitLinear(
                o_proj.in_features,
                o_proj.out_features,
                bias=o_proj.bias is not None,
            )
            new_o_proj.weight.data.copy_(o_proj.weight.data)
            if o_proj.bias is not None:
                new_o_proj.bias.data.copy_(o_proj.bias.data)
        else:
            new_o_proj = o_proj

        # Wrap SubLN + projection in Sequential so SubLN is called in forward pass
        attention_module.o_proj = nn.Sequential(subln, new_o_proj)


def convert_mlp_layer(
    mlp_module: nn.Module,
    hidden_size: int,
    exclude_layers: list[str],
) -> None:
    """
    Convert MLP/FFN layer to BitNet with SubLN.

    Inserts SubLN before down_proj.

    Args:
        mlp_module: The MLP module to convert
        hidden_size: Model hidden size
        exclude_layers: Layer names to exclude from conversion
    """
    # Convert gate and up projections
    for proj_name in ["gate_proj", "up_proj"]:
        if hasattr(mlp_module, proj_name) and proj_name not in exclude_layers:
            proj = getattr(mlp_module, proj_name)
            if isinstance(proj, nn.Linear) and not isinstance(proj, BitLinear):
                new_proj = BitLinear(
                    proj.in_features,
                    proj.out_features,
                    bias=proj.bias is not None,
                )
                new_proj.weight.data.copy_(proj.weight.data)
                if proj.bias is not None:
                    new_proj.bias.data.copy_(proj.bias.data)
                setattr(mlp_module, proj_name, new_proj)

    # Insert SubLN before down_proj
    if hasattr(mlp_module, "down_proj") and "down_proj" not in exclude_layers:
        down_proj = mlp_module.down_proj
        down_proj_in = down_proj.in_features if isinstance(down_proj, nn.Linear) else hidden_size

        subln = SubLN(down_proj_in)

        if isinstance(down_proj, nn.Linear) and not isinstance(down_proj, BitLinear):
            new_down_proj = BitLinear(
                down_proj.in_features,
                down_proj.out_features,
                bias=down_proj.bias is not None,
            )
            new_down_proj.weight.data.copy_(down_proj.weight.data)
            if down_proj.bias is not None:
                new_down_proj.bias.data.copy_(down_proj.bias.data)
        else:
            new_down_proj = down_proj

        # Wrap SubLN + projection in Sequential so SubLN is called in forward pass
        mlp_module.down_proj = nn.Sequential(subln, new_down_proj)


def convert_model_to_bitnet(
    model: nn.Module,
    hidden_size: int,
    intermediate_size: int,
    exclude_layers: Optional[list[str]] = None,
) -> nn.Module:
    """
    Convert a HuggingFace model to BitNet architecture with SubLN.

    This performs Stage 1 of BitDistill:
    1. Replace nn.Linear with BitLinear (except embeddings/LM head)
    2. Insert SubLN before output projections in attention and FFN

    Args:
        model: Pre-trained HuggingFace model
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        exclude_layers: Layer names to exclude from conversion

    Returns:
        Converted model
    """
    exclude_layers = exclude_layers or ["embed_tokens", "lm_head", "embed_positions"]

    # Find and convert transformer layers
    # This handles common architectures (LLaMA, Mistral, Qwen, etc.)
    for name, module in model.named_modules():
        # Skip excluded layers
        if any(excl in name for excl in exclude_layers):
            continue

        # Convert attention layers
        if "self_attn" in name or "attention" in name:
            if hasattr(module, "q_proj"):  # LLaMA-style attention
                convert_attention_layer(module, hidden_size, exclude_layers)

        # Convert MLP layers
        if "mlp" in name or "feed_forward" in name:
            if hasattr(module, "gate_proj"):  # LLaMA-style MLP
                convert_mlp_layer(module, intermediate_size, exclude_layers)

    logger.info("Converted model to BitNet architecture")
    logger.info("  - Replaced Linear -> BitLinear")
    logger.info("  - Inserted SubLN before o_proj and down_proj")

    return model


def run_stage1(
    pretrained_model_name: str,
    output_dir: Path,
    hidden_size: int,
    intermediate_size: int,
    exclude_layers: Optional[list[str]] = None,
    save_format: str = "safetensors",
) -> tuple[nn.Module, "AutoTokenizer"]:  # noqa: F821
    """
    Run Stage 1: Convert pre-trained model to BitNet with SubLN.

    Args:
        pretrained_model_name: HuggingFace model name or path
        output_dir: Directory to save converted model
        hidden_size: Model hidden dimension
        intermediate_size: FFN intermediate dimension
        exclude_layers: Layers to exclude from conversion
        save_format: Save format ("safetensors" or "pytorch")

    Returns:
        Tuple of (converted model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Stage 1: Loading pre-trained model {pretrained_model_name}")

    # Load model and tokenizer
    # Use bfloat16 to save memory while maintaining numerical precision
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce memory by 2x
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name,
        trust_remote_code=True,
    )

    logger.info("Stage 1: Converting model to BitNet architecture")

    # Convert to BitNet
    model = convert_model_to_bitnet(
        model,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        exclude_layers=exclude_layers,
    )

    # Save converted model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_format == "safetensors":
        from safetensors.torch import save_file

        state_dict = model.state_dict()

        # Handle shared tensors (e.g. tied embeddings)
        # safetensors doesn't support shared tensors, so we clone them
        if "lm_head.weight" in state_dict and "model.embed_tokens.weight" in state_dict:
            if state_dict["lm_head.weight"].data_ptr() == state_dict["model.embed_tokens.weight"].data_ptr():
                logger.info("Cloning shared lm_head.weight for safetensors saving")
                state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()

        save_file(state_dict, output_dir / "model.safetensors")
    else:
        torch.save(model.state_dict(), output_dir / "model.pt")

    tokenizer.save_pretrained(output_dir)

    logger.info(f"Stage 1: Saved converted model to {output_dir}")

    return model, tokenizer
