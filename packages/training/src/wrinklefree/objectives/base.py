"""Base objective class for training objectives.

Provides the abstract interface that all objectives must implement,
including hooks for objectives that need to modify inputs (like DLM).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class ObjectiveOutput:
    """Output from an objective's forward pass.

    Attributes:
        loss: The loss value for this objective
        metrics: Dictionary of metrics to log (will be prefixed with objective name)
        ce_loss: Optional pure cross-entropy loss (for unweighted CE logging)
    """

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    ce_loss: Optional[torch.Tensor] = None  # For train/loss_unweighted_ce


class Objective(nn.Module, ABC):
    """Abstract base class for training objectives.

    All objectives must implement:
    - name: Unique identifier for the objective
    - forward: Compute loss and metrics

    Objectives that modify inputs (like DLM) should:
    - Set modifies_input = True
    - Override preprocess_batch to apply modifications

    Objectives that need teacher model outputs should:
    - Set requires_teacher = True

    Objectives that need hidden states should:
    - Set requires_hidden_states = True
    """

    # Class attributes - override in subclasses as needed
    requires_teacher: bool = False
    requires_hidden_states: bool = False
    requires_attentions: bool = False  # For attention distillation
    modifies_input: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this objective."""
        ...

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Hook: Modify batch BEFORE model forward pass.

        Override this for objectives that need to modify inputs,
        like DLM which applies block masking.

        Args:
            batch: The input batch dictionary with at least:
                - input_ids: Token IDs (batch, seq)
                - attention_mask: Attention mask (batch, seq)
                - labels: Target labels (batch, seq)

        Returns:
            Modified batch dictionary. Can add new keys (prefixed with _)
            to pass information to forward().
        """
        return batch  # Default: no modification

    @abstractmethod
    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """Compute loss and metrics for this objective.

        Args:
            model_outputs: Dictionary containing at least:
                - logits: Model logits (batch, seq, vocab)
                - hidden_states: Optional tuple of hidden states per layer
            batch: The input batch (possibly modified by preprocess_batch)
            teacher_outputs: Optional teacher model outputs (if requires_teacher)

        Returns:
            ObjectiveOutput with loss and metrics
        """
        ...

    def extra_repr(self) -> str:
        """String representation for debugging."""
        flags = []
        if self.requires_teacher:
            flags.append("requires_teacher")
        if self.requires_hidden_states:
            flags.append("requires_hidden_states")
        if self.requires_attentions:
            flags.append("requires_attentions")
        if self.modifies_input:
            flags.append("modifies_input")
        return f"name={self.name}, " + ", ".join(flags) if flags else f"name={self.name}"
