"""Model conversion utilities for BitNet.cpp inference.

Supports:
- Dense models (standard LLaMA-style)
- MoE models (Mixtral-style with K-of-N routing)
- Fake MoE conversion for testing
"""

import logging
import re
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BitNetGGUFConverter:
    """
    Convert WrinkleFree trained model to GGUF format for BitNet.cpp inference.

    GGUF (GPT-Generated Unified Format) is the format used by llama.cpp
    and BitNet.cpp for efficient inference.

    Args:
        model: Trained BitNet model
        output_path: Path for output GGUF file
        model_name: Name for the model metadata
    """

    def __init__(
        self,
        model: nn.Module,
        output_path: Path,
        model_name: str = "bitnet",
        is_moe: bool = False,
    ):
        self.model = model
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.is_moe = is_moe or self._detect_moe(model)

    def _detect_moe(self, model: nn.Module) -> bool:
        """Detect if model has MoE architecture."""
        state_dict = model.state_dict()
        for name in state_dict.keys():
            # Check for expert patterns
            if "experts." in name or "block_sparse_moe" in name:
                return True
            # Check for router patterns
            if "router" in name and "gate" in name:
                return True
        return False

    def convert(self, quant_type: str = "i2_s") -> Path:
        """
        Convert model to GGUF format.

        Args:
            quant_type: Quantization type ("i2_s", "tl1", "tl2")

        Returns:
            Path to output GGUF file
        """
        logger.info(f"Converting model to GGUF format ({quant_type})")

        # Extract model config
        config = self._extract_config()

        # Get state dict
        state_dict = self.model.state_dict()

        # Convert tensors
        tensors = {}
        for name, tensor in state_dict.items():
            converted_name = self._convert_tensor_name(name)
            if converted_name:
                tensors[converted_name] = self._convert_tensor(tensor, name, quant_type)

        # Write GGUF file
        self._write_gguf(tensors, config, quant_type)

        logger.info(f"Saved GGUF model to {self.output_path}")
        return self.output_path

    def _extract_config(self) -> dict:
        """Extract model configuration."""
        # Try to get config from model
        if hasattr(self.model, "config"):
            config = self.model.config
            extracted = {
                "vocab_size": getattr(config, "vocab_size", 128256),
                "hidden_size": getattr(config, "hidden_size", 4096),
                "intermediate_size": getattr(config, "intermediate_size", 11008),
                "num_hidden_layers": getattr(config, "num_hidden_layers", 32),
                "num_attention_heads": getattr(config, "num_attention_heads", 32),
                "num_kv_heads": getattr(config, "num_kv_heads", 8),
                "head_dim": getattr(config, "head_dim", 128),
                "rope_theta": getattr(config, "rope_theta", 500000.0),
            }
            # MoE config
            if self.is_moe:
                extracted["num_experts"] = getattr(config, "num_local_experts", 8)
                extracted["num_experts_per_tok"] = getattr(config, "num_experts_per_tok", 2)
            return extracted

        # Infer from model structure
        logger.warning("Could not extract config, inferring from model structure")
        return self._infer_config()

    def _infer_config(self) -> dict:
        """Infer configuration from model structure."""
        config = {}
        num_experts = 0
        expert_indices = set()

        for name, param in self.model.named_parameters():
            if "embed_tokens" in name:
                config["vocab_size"] = param.shape[0]
                config["hidden_size"] = param.shape[1]
            elif "q_proj" in name and "weight" in name:
                config["hidden_size"] = param.shape[1]
                # Estimate num_heads
                if "num_attention_heads" not in config:
                    config["num_attention_heads"] = 32
                    config["head_dim"] = param.shape[0] // config["num_attention_heads"]
            elif "gate_proj" in name and "weight" in name and "experts" not in name:
                # Only use non-expert gate_proj for intermediate_size
                config["intermediate_size"] = param.shape[0]

            # Detect MoE structure
            if "experts." in name:
                # Extract expert index: experts.0.gate_proj.weight -> 0
                match = re.search(r"experts\.(\d+)\.", name)
                if match:
                    expert_indices.add(int(match.group(1)))
                # Get intermediate size from expert
                if "gate_proj" in name and "weight" in name:
                    config["intermediate_size"] = param.shape[0]

        # Count layers
        layer_count = 0
        for name in self.model.state_dict().keys():
            if "layers." in name:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                layer_count = max(layer_count, layer_num + 1)
        config["num_hidden_layers"] = layer_count

        # MoE config
        if expert_indices:
            config["num_experts"] = len(expert_indices)
            config["num_experts_per_tok"] = 2  # Default, common for Mixtral-style

        # Defaults
        config.setdefault("num_kv_heads", 8)
        config.setdefault("rope_theta", 500000.0)

        return config

    def _convert_tensor_name(self, name: str) -> Optional[str]:
        """Convert PyTorch tensor name to BitNet GGUF convention.

        BitNet GGUF format (from extern/BitNet/3rdparty/llama.cpp):
        - token_embd.weight
        - output_norm.weight
        - output.weight (lm_head)
        - blk.{n}.attn_norm.weight
        - blk.{n}.attn_q.weight, attn_k.weight, attn_v.weight, attn_output.weight
        - blk.{n}.attn_sub_norm.weight (SubLN)
        - blk.{n}.ffn_norm.weight
        - blk.{n}.ffn_gate.weight, ffn_up.weight, ffn_down.weight
        - blk.{n}.ffn_sub_norm.weight (SubLN)

        MoE extensions:
        - blk.{n}.ffn_gate_inp.weight (router)
        - blk.{n}.ffn_gate_exps.weight (expert gates, 3D tensor)
        - blk.{n}.ffn_up_exps.weight (expert up, 3D tensor)
        - blk.{n}.ffn_down_exps.weight (expert down, 3D tensor)
        """
        # Skip optimizer states, buffers, etc.
        if "optimizer" in name or "scheduler" in name:
            return None
        if "_indices" in name or "_values" in name:
            return None

        # Strip common prefixes
        name = name.replace("model.", "")
        name = name.replace("transformer.", "")

        # Global tensors (not per-block)
        name = name.replace("embed_tokens", "token_embd")
        name = name.replace("lm_head", "output")
        if "norm" in name and "layers" not in name and "blk" not in name:
            name = name.replace("norm", "output_norm")

        # Layer mappings: layers.{n} -> blk.{n}
        name = name.replace("layers.", "blk.")

        # Attention mappings
        name = name.replace("self_attn.", "")
        name = name.replace("input_layernorm", "attn_norm")
        name = name.replace("q_proj", "attn_q")
        name = name.replace("k_proj", "attn_k")
        name = name.replace("v_proj", "attn_v")
        name = name.replace("o_proj", "attn_output")

        # SubLN normalization (BitNet-specific)
        name = name.replace("attn_sub_layernorm", "attn_sub_norm")
        name = name.replace("ffn_sub_layernorm", "ffn_sub_norm")

        # Post-attention norm -> ffn_norm
        name = name.replace("post_attention_layernorm", "ffn_norm")

        # Handle MoE expert weights
        if self.is_moe and "experts." in name:
            # Pattern: mlp.experts.{n}.gate_proj -> ffn_gate_exps (packed later)
            # For now, keep individual expert names; they get merged in _write_gguf
            match = re.search(r"experts\.(\d+)\.(\w+)", name)
            if match:
                expert_idx = match.group(1)
                proj_name = match.group(2)
                proj_map = {
                    "gate_proj": "ffn_gate_exps",
                    "up_proj": "ffn_up_exps",
                    "down_proj": "ffn_down_exps",
                    "w1": "ffn_up_exps",
                    "w2": "ffn_down_exps",
                    "w3": "ffn_gate_exps",
                }
                if proj_name in proj_map:
                    # Store with expert index for later merging
                    name = re.sub(
                        r"(mlp\.|block_sparse_moe\.)?experts\.\d+\.\w+",
                        f"{proj_map[proj_name]}.{expert_idx}",
                        name
                    )

            # Handle router gate -> ffn_gate_inp
            if "router" in name and "gate" in name:
                name = re.sub(r"(mlp\.|block_sparse_moe\.)?router\.gate", "ffn_gate_inp", name)
        else:
            # Dense model FFN
            name = name.replace("mlp.", "")
            name = name.replace("gate_proj", "ffn_gate")
            name = name.replace("up_proj", "ffn_up")
            name = name.replace("down_proj", "ffn_down")

        return name

    def _convert_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        quant_type: str,
    ) -> np.ndarray:
        """Convert tensor to appropriate format."""
        tensor = tensor.detach().cpu()

        # Check if this is a quantized weight
        is_weight = "weight" in name and "norm" not in name and "embed" not in name

        if is_weight and quant_type in ["i2_s", "tl1", "tl2"]:
            # Ternary quantize to {-1, 0, 1}
            return self._quantize_ternary(tensor)

        # Keep as float for non-quantized tensors
        return tensor.numpy().astype(np.float16)

    def _quantize_ternary(self, tensor: torch.Tensor) -> np.ndarray:
        """Quantize tensor to ternary values {-1, 0, 1}."""
        # Compute scale
        scale = tensor.abs().mean()

        # Quantize
        tensor_scaled = tensor / (scale + 1e-5)
        tensor_quant = tensor_scaled.round().clamp(-1, 1)

        # Pack into int8 for storage (values are -1, 0, 1)
        return tensor_quant.numpy().astype(np.int8)

    def _write_gguf(
        self,
        tensors: dict[str, np.ndarray],
        config: dict,
        quant_type: str,
    ) -> None:
        """Write GGUF format file for BitNet.

        Uses BitNet architecture type and metadata format:
        - Architecture: "bitnet" (standard) or "bitnet-25" (MoE)
        - Metadata: bitnet.* namespace for hyperparameters
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from gguf import GGUFWriter

            # Use bitnet architecture type
            arch = "bitnet"
            if self.is_moe:
                arch = "bitnet"  # MoE is still under bitnet arch in GGUF

            writer = GGUFWriter(str(self.output_path), arch)

            # Write metadata (BitNet format)
            writer.add_name(self.model_name)
            writer.add_architecture(arch)
            writer.add_context_length(config.get("context_length", 4096))
            writer.add_embedding_length(config["hidden_size"])
            writer.add_block_count(config["num_hidden_layers"])
            writer.add_feed_forward_length(config["intermediate_size"])
            writer.add_head_count(config["num_attention_heads"])
            writer.add_head_count_kv(config.get("num_kv_heads", config["num_attention_heads"]))
            writer.add_rope_freq_base(config.get("rope_theta", 500000.0))

            # BitNet-specific metadata
            writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5)

            # MoE metadata
            if self.is_moe:
                num_experts = config.get("num_experts", 8)
                num_experts_per_tok = config.get("num_experts_per_tok", 2)
                writer.add_uint32(f"{arch}.expert_count", num_experts)
                writer.add_uint32(f"{arch}.expert_used_count", num_experts_per_tok)
                logger.info(f"MoE config: {num_experts} experts, top-{num_experts_per_tok}")

            # Write tensors
            for name, tensor in tensors.items():
                writer.add_tensor(name, tensor)

            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            logger.info(f"Wrote BitNet GGUF to {self.output_path}")

        except ImportError:
            logger.warning("gguf package not installed, writing raw format instead")
            self._write_raw_format(tensors, config)

    def _write_raw_format(self, tensors: dict, config: dict) -> None:
        """Write raw format as fallback."""
        # Save as safetensors instead
        from safetensors.torch import save_file

        # Convert back to torch tensors
        torch_tensors = {
            name: torch.from_numpy(tensor)
            for name, tensor in tensors.items()
        }

        save_file(torch_tensors, self.output_path.with_suffix(".safetensors"))
        logger.info("Saved as safetensors (GGUF conversion requires gguf package)")


def convert_to_gguf(
    model_path: Path,
    output_path: Path,
    quant_type: str = "i2_s",
    model_name: str = "bitnet",
) -> Path:
    """
    High-level function to convert a saved model to GGUF.

    Args:
        model_path: Path to model checkpoint
        output_path: Path for output GGUF file
        quant_type: Quantization type
        model_name: Model name for metadata

    Returns:
        Path to converted file
    """
    from safetensors.torch import load_file

    # Load model
    if model_path.suffix == ".safetensors":
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    # Create a simple container for the state dict
    class ModelContainer(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self._state_dict = state_dict

        def state_dict(self):
            return self._state_dict

    container = ModelContainer(state_dict)

    # Convert
    converter = BitNetGGUFConverter(container, output_path, model_name)
    return converter.convert(quant_type)


def convert_moe_to_gguf(
    model: nn.Module,
    output_path: Path,
    quant_type: str = "i2_s",
    model_name: str = "bitnet-moe",
) -> Path:
    """
    Convert an MoE model to GGUF format.

    Args:
        model: MoE model (e.g., from FakeMoEConverter)
        output_path: Path for output GGUF file
        quant_type: Quantization type ("i2_s", "tl1", "tl2")
        model_name: Model name for metadata

    Returns:
        Path to converted file
    """
    converter = BitNetGGUFConverter(
        model, output_path, model_name, is_moe=True
    )
    return converter.convert(quant_type)


def convert_dense_to_fake_moe_gguf(
    model: nn.Module,
    output_path: Path,
    num_experts: int = 8,
    top_k: int = 2,
    quant_type: str = "i2_s",
    model_name: str = "bitnet-fake-moe",
) -> Path:
    """
    Convert a dense model to Fake MoE and then to GGUF.

    This creates an MoE model where all experts share weights with the
    original dense model, using IdentityRouter to route all tokens to
    expert 0. The output should be functionally identical to the dense model.

    Args:
        model: Dense BitNet model
        output_path: Path for output GGUF file
        num_experts: Total number of experts (N)
        top_k: Number of active experts (K)
        quant_type: Quantization type
        model_name: Model name for metadata

    Returns:
        Path to converted file
    """
    from wf_train._experimental.moe import create_fake_moe_from_dense

    # Convert to Fake MoE
    logger.info(f"Converting dense model to Fake MoE ({num_experts} experts, top-{top_k})")
    moe_model = create_fake_moe_from_dense(
        model, num_experts=num_experts, top_k=top_k, use_identity_router=True
    )

    # Convert to GGUF
    return convert_moe_to_gguf(moe_model, output_path, quant_type, model_name)
