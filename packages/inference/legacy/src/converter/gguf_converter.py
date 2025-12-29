"""Model conversion utilities for BitNet.cpp inference.

This module handles conversion of trained PyTorch models to GGUF format.
For HuggingFace model download and conversion, see hf_to_gguf.py.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BitNetGGUFConverter:
    """
    Convert WrinkleFree trained model to GGUF format for BitNet.cpp inference.

    GGUF (GPT-Generated Unified Format) is the format used by llama.cpp
    and BitNet.cpp for efficient inference.

    Args:
        model: Trained BitNet model (nn.Module)
        output_path: Path for output GGUF file
        model_name: Name for the model metadata
    """

    def __init__(
        self,
        model,  # nn.Module - avoid import for optional torch dep
        output_path: Path,
        model_name: str = "bitnet",
    ):
        self.model = model
        self.output_path = Path(output_path)
        self.model_name = model_name

    def convert(self, quant_type: str = "i2_s") -> Path:
        """
        Convert model to GGUF format.

        Args:
            quant_type: Quantization type ("i2_s", "tl1", "tl2")

        Returns:
            Path to output GGUF file
        """
        import torch

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
        if hasattr(self.model, "config"):
            config = self.model.config
            return {
                "vocab_size": getattr(config, "vocab_size", 128256),
                "hidden_size": getattr(config, "hidden_size", 4096),
                "intermediate_size": getattr(config, "intermediate_size", 11008),
                "num_hidden_layers": getattr(config, "num_hidden_layers", 32),
                "num_attention_heads": getattr(config, "num_attention_heads", 32),
                "num_kv_heads": getattr(config, "num_kv_heads", 8),
                "head_dim": getattr(config, "head_dim", 128),
                "rope_theta": getattr(config, "rope_theta", 500000.0),
            }

        logger.warning("Could not extract config, inferring from model structure")
        return self._infer_config()

    def _infer_config(self) -> dict:
        """Infer configuration from model structure."""
        config = {}

        for name, param in self.model.named_parameters():
            if "embed_tokens" in name:
                config["vocab_size"] = param.shape[0]
                config["hidden_size"] = param.shape[1]
            elif "q_proj" in name and "weight" in name:
                config["hidden_size"] = param.shape[1]
                if "num_attention_heads" not in config:
                    config["num_attention_heads"] = 32
                    config["head_dim"] = param.shape[0] // config["num_attention_heads"]
            elif "gate_proj" in name and "weight" in name:
                config["intermediate_size"] = param.shape[0]

        # Count layers
        layer_count = 0
        for name in self.model.state_dict().keys():
            if "layers." in name:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                layer_count = max(layer_count, layer_num + 1)
        config["num_hidden_layers"] = layer_count

        # Defaults
        config.setdefault("num_kv_heads", 8)
        config.setdefault("rope_theta", 500000.0)

        return config

    def _convert_tensor_name(self, name: str) -> Optional[str]:
        """Convert PyTorch tensor name to GGUF convention."""
        if "optimizer" in name or "scheduler" in name:
            return None

        # Common mappings
        name = name.replace("model.", "")
        name = name.replace("transformer.", "")

        # Layer mappings for LLaMA-style
        name = name.replace("embed_tokens", "token_embd")
        name = name.replace("layers.", "blk.")
        name = name.replace("self_attn.", "attn_")
        name = name.replace("mlp.", "ffn_")
        name = name.replace("input_layernorm", "attn_norm")
        name = name.replace("post_attention_layernorm", "ffn_norm")
        name = name.replace("q_proj", "attn_q")
        name = name.replace("k_proj", "attn_k")
        name = name.replace("v_proj", "attn_v")
        name = name.replace("o_proj", "attn_output")
        name = name.replace("gate_proj", "ffn_gate")
        name = name.replace("up_proj", "ffn_up")
        name = name.replace("down_proj", "ffn_down")
        name = name.replace("norm.", "output_norm.")
        name = name.replace("lm_head", "output")

        return name

    def _convert_tensor(
        self,
        tensor,  # torch.Tensor
        name: str,
        quant_type: str,
    ) -> np.ndarray:
        """Convert tensor to appropriate format."""
        tensor = tensor.detach().cpu()

        is_weight = "weight" in name and "norm" not in name and "embed" not in name

        if is_weight and quant_type in ["i2_s", "tl1", "tl2"]:
            return self._quantize_ternary(tensor)

        return tensor.numpy().astype(np.float16)

    def _quantize_ternary(self, tensor) -> np.ndarray:
        """Quantize tensor to ternary values {-1, 0, 1}."""
        scale = tensor.abs().mean()
        tensor_scaled = tensor / (scale + 1e-5)
        tensor_quant = tensor_scaled.round().clamp(-1, 1)
        return tensor_quant.numpy().astype(np.int8)

    def _write_gguf(
        self,
        tensors: dict[str, np.ndarray],
        config: dict,
        quant_type: str,
    ) -> None:
        """Write GGUF format file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from gguf import GGUFWriter

            writer = GGUFWriter(str(self.output_path), "llama")

            writer.add_name(self.model_name)
            writer.add_architecture("llama")
            writer.add_context_length(4096)
            writer.add_embedding_length(config["hidden_size"])
            writer.add_block_count(config["num_hidden_layers"])
            writer.add_feed_forward_length(config["intermediate_size"])
            writer.add_head_count(config["num_attention_heads"])
            writer.add_head_count_kv(config.get("num_kv_heads", config["num_attention_heads"]))
            writer.add_rope_freq_base(config.get("rope_theta", 500000.0))

            for name, tensor in tensors.items():
                writer.add_tensor(name, tensor)

            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

        except ImportError:
            logger.warning("gguf package not installed, writing raw format instead")
            self._write_raw_format(tensors, config)

    def _write_raw_format(self, tensors: dict, config: dict) -> None:
        """Write raw format as fallback."""
        import torch
        from safetensors.torch import save_file

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
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file

    if model_path.suffix == ".safetensors":
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    class ModelContainer(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self._state_dict = state_dict

        def state_dict(self):
            return self._state_dict

    container = ModelContainer(state_dict)

    converter = BitNetGGUFConverter(container, output_path, model_name)
    return converter.convert(quant_type)
