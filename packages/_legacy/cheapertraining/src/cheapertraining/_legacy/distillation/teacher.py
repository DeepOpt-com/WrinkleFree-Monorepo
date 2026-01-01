"""Teacher model wrapper for knowledge distillation.

Handles loading and inference of teacher models for distillation.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


class TeacherWrapper(nn.Module):
    """Wrapper for teacher model in knowledge distillation.

    Features:
    - Frozen parameters (no gradients)
    - Inference mode optimization
    - Optional logit caching for efficiency
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize teacher wrapper.

        Args:
            model: Teacher model
            device: Device to place model on
            dtype: Data type for model
        """
        super().__init__()
        self.model = model
        self.device = device
        self.dtype = dtype

        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Move to device and dtype if specified
        if device is not None:
            self.model = self.model.to(device)
        if dtype is not None:
            self.model = self.model.to(dtype)

    @torch.no_grad()
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Get teacher logits.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            **kwargs: Additional arguments for model

        Returns:
            Teacher logits (batch, seq, vocab)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )

        if isinstance(outputs, dict):
            return outputs.get("logits", outputs.get("last_hidden_state"))
        elif hasattr(outputs, "logits"):
            return outputs.logits
        else:
            return outputs

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "TeacherWrapper":
        """Load teacher from pretrained model.

        Args:
            model_name_or_path: HuggingFace model name or path
            device: Device to load on
            dtype: Data type
            **kwargs: Additional arguments for AutoModelForCausalLM

        Returns:
            TeacherWrapper instance
        """
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if device is None else None,
            trust_remote_code=True,
            **kwargs,
        )

        return cls(model, device=device, dtype=dtype)


class CachedTeacher:
    """Teacher with logit caching for repeated queries.

    Useful when the same data is used multiple times.
    """

    def __init__(
        self,
        teacher: TeacherWrapper,
        cache_size: int = 1000,
    ):
        """Initialize cached teacher.

        Args:
            teacher: TeacherWrapper instance
            cache_size: Maximum cache size (in samples)
        """
        self.teacher = teacher
        self.cache_size = cache_size
        self.cache = {}

    def get_logits(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        cache_key: Optional[str] = None,
    ) -> Tensor:
        """Get teacher logits with caching.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            cache_key: Optional cache key (if None, caching is disabled)

        Returns:
            Teacher logits
        """
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]

        logits = self.teacher(input_ids, attention_mask)

        if cache_key is not None:
            # Manage cache size
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = logits.detach()

        return logits

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
