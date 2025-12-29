"""Local teacher model wrappers for in-process distillation."""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


class LocalTeacher(nn.Module):
    """
    Wrapper for frozen teacher model loaded in-process.

    Handles teacher model loading, freezing, and optional CPU offloading.

    Args:
        model_name_or_path: HuggingFace model name or local path
        device: Device to load model on
        load_in_fp16: Load in BF16 for memory efficiency
        offload_to_cpu: Offload to CPU between forward passes
        load_in_4bit: Load in 4-bit NF4 quantization for memory efficiency
        use_eager_attention: Use eager attention instead of SDPA/Flash
            (required for attention distillation, as SDPA doesn't return attention weights)
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        load_in_fp16: bool = True,
        offload_to_cpu: bool = False,
        load_in_4bit: bool = False,
        use_eager_attention: bool = True,
    ):
        super().__init__()

        logger.info(f"Loading teacher model: {model_name_or_path}")

        dtype = torch.bfloat16 if load_in_fp16 else torch.float32

        # Build model loading kwargs
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }

        # Add 4-bit quantization if requested
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Using 4-bit NF4 quantization for teacher")
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to BF16")

        # Use eager attention if attention distillation is needed
        # SDPA/Flash attention doesn't return attention weights
        if use_eager_attention:
            model_kwargs["attn_implementation"] = "eager"
            logger.info("Using eager attention for teacher (required for attention distillation)")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        # Freeze teacher
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.offload_to_cpu = offload_to_cpu

        if not offload_to_cpu:
            self.model = self.model.to(device)

        logger.info(f"Teacher model loaded (frozen, {'BF16' if load_in_fp16 else 'FP32'})")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        """
        Forward pass through teacher model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            output_attentions: Whether to return attention weights

        Returns:
            Dictionary with logits and optionally attentions
        """
        if self.offload_to_cpu:
            self.model = self.model.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        result = {
            "logits": outputs.logits,
            "attentions": outputs.attentions if output_attentions else None,
        }

        if self.offload_to_cpu:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

        return result


class HiddenStateTeacher(nn.Module):
    """
    Teacher wrapper that extracts per-layer hidden states.

    Extends LocalTeacher to return hidden states from all transformer
    layers for layer-wise distillation.

    Args:
        model_name_or_path: HuggingFace model name or local path
        device: Device to load model on
        load_in_fp16: Load model in BF16 for memory efficiency
        offload_to_cpu: Offload model to CPU between forward passes
        load_in_4bit: Load model in 4-bit NF4 quantization for 3x memory reduction
        use_flash_attention: Use Flash Attention 2 for faster attention
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        load_in_fp16: bool = True,
        offload_to_cpu: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        logger.info(f"Loading teacher model for hidden state extraction: {model_name_or_path}")

        # Use bfloat16 for better training stability (matches student dtype)
        dtype = torch.bfloat16 if load_in_fp16 else torch.float32

        # Build model loading kwargs
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }

        # Add 4-bit quantization if requested (3x memory reduction)
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Using 4-bit NF4 quantization for teacher (3x memory reduction)")
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to BF16")

        # Add Flash Attention 2 if requested (15-25% speedup)
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2 for teacher")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        # Freeze teacher
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.offload_to_cpu = offload_to_cpu

        if not offload_to_cpu:
            self.model = self.model.to(device)

        logger.info(
            f"Teacher model loaded (frozen, {'BF16' if load_in_fp16 else 'FP32'}, "
            f"offload={'enabled' if offload_to_cpu else 'disabled'})"
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        """
        Forward pass returning hidden states from all layers.

        Args:
            input_ids: Input token IDs (batch, seq)
            attention_mask: Attention mask (batch, seq)
            output_attentions: Whether to return attention weights

        Returns:
            Dictionary with:
                - hidden_states: Tuple of hidden states per layer
                - logits: Output logits
                - attentions: Optional attention weights
        """
        if self.offload_to_cpu:
            self.model = self.model.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Always extract hidden states
            output_attentions=output_attentions,
        )

        result = {
            "hidden_states": outputs.hidden_states,
            "logits": outputs.logits,
            "attentions": outputs.attentions if output_attentions else None,
        }

        if self.offload_to_cpu:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

        return result
