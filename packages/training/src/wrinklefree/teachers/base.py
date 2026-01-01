"""Base teacher protocol for distillation."""

from typing import Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class BaseTeacher(Protocol):
    """
    Protocol for teacher models in distillation.

    All teacher implementations should follow this interface,
    allowing interchangeable use of local models, vLLM servers,
    or cached teacher outputs.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        """
        Get teacher outputs.

        Args:
            input_ids: Input token IDs (batch, seq)
            attention_mask: Optional attention mask (batch, seq)
            output_attentions: Whether to return attention weights

        Returns:
            Dictionary with:
                - logits: Output logits (batch, seq, vocab)
                - attentions: Optional tuple of attention weights per layer,
                    each (batch, heads, seq, seq). None if not supported.
        """
        ...

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        """Allow calling as a module."""
        return self.forward(input_ids, attention_mask, output_attentions)
