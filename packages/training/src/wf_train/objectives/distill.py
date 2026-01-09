"""Unified distillation objective combining all distillation types.

Consolidates:
- Hidden states distillation (layerwise)
- Logits distillation (KL divergence, full or sparse TCS)
- Attention distillation (relation or block-wise)
- LRC reconstruction (post-quantization correction)

Each component is independently configurable via YAML.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn.functional as F

from wf_train.objectives.base import Objective, ObjectiveOutput

logger = logging.getLogger(__name__)


class HiddenLossType(Enum):
    """Loss types for hidden state distillation."""

    MSE = "mse"
    MSE_NORMALIZED = "mse_normalized"
    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"


class LogitsMode(Enum):
    """Logits distillation modes."""

    FULL = "full"  # Standard KL on all logits
    SPARSE = "sparse"  # Top-K TCS style


class AttentionMode(Enum):
    """Attention distillation modes."""

    RELATION = "relation"  # Standard A*A^T relation matching


@dataclass
class LayerWiseConfig:
    """Unified config for Hidden States and LRC layer-wise alignment.

    Used for both hidden state distillation and LRC reconstruction loss.
    The `name` field controls metric naming (e.g., "hidden" or "lrc").
    """

    name: str = "hidden"  # "hidden" or "lrc" - controls metric naming
    enabled: bool = False
    weight: float = 1.0
    loss_type: str = "mse"  # mse, mse_normalized, cosine, inner_product
    layer_weights: Optional[str | list[float]] = None  # null, progressive, exponential, or list
    temperature: float = 1.0  # Applied at end (1.0 = no scaling for hidden)
    normalize: bool = False


# Backward compatibility aliases (deprecated)
HiddenConfig = LayerWiseConfig
LRCConfig = LayerWiseConfig


@dataclass
class LogitsConfig:
    """Configuration for logits distillation component."""

    enabled: bool = False
    weight: float = 10.0
    temperature: float = 5.0
    mode: str = "full"  # "full" (standard KL) or "sparse" (top-K TCS)
    top_k: int = 100  # Only used when mode="sparse"
    shift_labels: bool = True  # Shift labels for next-token prediction
    ignore_index: int = -100


@dataclass
class AttentionConfig:
    """Configuration for attention distillation component."""

    enabled: bool = False
    weight: float = 1.0e-5
    distill_layer: int = -1  # Which layer to distill (-1 = last)
    mode: str = "relation"  # "relation" (standard A*A^T matching)
    temperature: float = 1.0


class DistillObjective(Objective):
    """Unified distillation objective combining all distillation types.

    Supports four components, each independently configurable:
    - hidden: Hidden state alignment (layerwise distillation)
    - logits: KL divergence on teacher/student logits (full or sparse TCS)
    - attention: Attention pattern matching (relation-based)
    - lrc: Low-rank correction post-quantization recovery

    Dynamic requirements based on enabled components:
    - requires_teacher: True if any component enabled
    - requires_hidden_states: True if hidden or lrc enabled
    - requires_attentions: True if attention enabled

    Args:
        hidden: Hidden states distillation config
        logits: Logits distillation config
        attention: Attention distillation config
        lrc: LRC reconstruction config
        ignore_index: Index to ignore in labels (default: -100)
    """

    modifies_input = False  # Never modifies input

    def __init__(
        self,
        hidden: Optional[LayerWiseConfig | dict] = None,
        logits: Optional[LogitsConfig | dict] = None,
        attention: Optional[AttentionConfig | dict] = None,
        lrc: Optional[LayerWiseConfig | dict] = None,
        ignore_index: int = -100,
    ):
        super().__init__()

        # Convert dicts to dataclasses if needed
        # For hidden: default name="hidden", loss_type="mse_normalized", normalize=True
        if isinstance(hidden, dict):
            hidden = LayerWiseConfig(
                name=hidden.get("name", "hidden"),
                enabled=hidden.get("enabled", False),
                weight=hidden.get("weight", 1.0),
                loss_type=hidden.get("loss_type", "mse_normalized"),
                layer_weights=hidden.get("layer_weights"),
                temperature=hidden.get("temperature", 1.0),
                normalize=hidden.get("normalize", True),
            )
        if isinstance(logits, dict):
            logits = LogitsConfig(**logits)
        if isinstance(attention, dict):
            attention = AttentionConfig(**attention)
        # For lrc: default name="lrc", loss_type="mse", normalize=False
        if isinstance(lrc, dict):
            lrc = LayerWiseConfig(
                name=lrc.get("name", "lrc"),
                enabled=lrc.get("enabled", False),
                weight=lrc.get("weight", 1.0),
                loss_type=lrc.get("loss_type", "mse"),
                layer_weights=lrc.get("layer_weights"),
                temperature=lrc.get("temperature", 1.0),
                normalize=lrc.get("normalize", False),
            )

        # Set defaults with appropriate names for each component
        self.hidden_config = hidden or LayerWiseConfig(
            name="hidden", loss_type="mse_normalized", normalize=True
        )
        self.logits_config = logits or LogitsConfig()
        self.attention_config = attention or AttentionConfig()
        self.lrc_config = lrc or LayerWiseConfig(name="lrc")
        self.ignore_index = ignore_index

    @property
    def name(self) -> str:
        return "distill"

    @property
    def requires_teacher(self) -> bool:
        """Any enabled component requires teacher."""
        return (
            self.hidden_config.enabled
            or self.logits_config.enabled
            or self.attention_config.enabled
            or self.lrc_config.enabled
        )

    @property
    def requires_hidden_states(self) -> bool:
        """Hidden or LRC components need hidden states."""
        return self.hidden_config.enabled or self.lrc_config.enabled

    @property
    def requires_attentions(self) -> bool:
        """Only attention component needs attention weights."""
        return self.attention_config.enabled

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """Compute combined distillation loss from all enabled components."""
        if teacher_outputs is None and self.requires_teacher:
            raise ValueError("DistillObjective requires teacher_outputs")

        device = model_outputs["logits"].device
        dtype = model_outputs["logits"].dtype

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        metrics = {}

        # Hidden states component
        if self.hidden_config.enabled:
            hidden_loss, hidden_metrics = self._compute_layerwise_loss(
                self.hidden_config, model_outputs, teacher_outputs, batch
            )
            total_loss = total_loss + self.hidden_config.weight * hidden_loss
            metrics.update(hidden_metrics)
            metrics["hidden_loss"] = hidden_loss.detach()

        # Logits component
        if self.logits_config.enabled:
            logits_loss, logits_metrics = self._compute_logits_loss(
                model_outputs, batch, teacher_outputs
            )
            total_loss = total_loss + self.logits_config.weight * logits_loss
            metrics.update({f"logits_{k}": v for k, v in logits_metrics.items()})
            metrics["logits_loss"] = logits_loss.detach()

        # Attention component
        if self.attention_config.enabled:
            attn_loss, attn_metrics = self._compute_attention_loss(
                model_outputs, batch, teacher_outputs
            )
            total_loss = total_loss + self.attention_config.weight * attn_loss
            metrics.update({f"attention_{k}": v for k, v in attn_metrics.items()})
            metrics["attention_loss"] = attn_loss.detach()

        # LRC component
        if self.lrc_config.enabled:
            lrc_loss, lrc_metrics = self._compute_layerwise_loss(
                self.lrc_config, model_outputs, teacher_outputs, batch
            )
            total_loss = total_loss + self.lrc_config.weight * lrc_loss
            metrics.update(lrc_metrics)
            metrics["lrc_loss"] = lrc_loss.detach()

        return ObjectiveOutput(loss=total_loss, metrics=metrics)

    # =========================================================================
    # Unified Layer-wise Alignment (Hidden States + LRC)
    # =========================================================================

    def _compute_layerwise_loss(
        self,
        config: LayerWiseConfig,
        model_outputs: dict[str, Any],
        teacher_outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Unified layer-wise alignment loss for hidden states or LRC.

        Args:
            config: LayerWiseConfig with name, loss_type, layer_weights, temperature, normalize
            model_outputs: Student model outputs containing "hidden_states"
            teacher_outputs: Teacher model outputs containing "hidden_states"
            batch: Batch dict with optional "attention_mask"

        Returns:
            Tuple of (loss, metrics_dict) where metrics are prefixed with config.name
        """
        student_hidden = model_outputs.get("hidden_states")
        teacher_hidden = teacher_outputs.get("hidden_states")

        if student_hidden is None or teacher_hidden is None:
            raise ValueError(
                "Both student and teacher must output hidden_states. "
                "Set output_hidden_states=True in model config."
            )

        # Skip embedding layer (index 0), use transformer layers only
        student_hidden = student_hidden[1:]
        teacher_hidden = teacher_hidden[1:]

        if len(student_hidden) != len(teacher_hidden):
            raise ValueError(
                f"Layer count mismatch: student={len(student_hidden)}, "
                f"teacher={len(teacher_hidden)}"
            )

        num_layers = len(student_hidden)
        if num_layers == 0:
            device = model_outputs["logits"].device
            return (
                torch.tensor(0.0, device=device),
                {f"{config.name}_mean_layer_loss": torch.tensor(0.0, device=device)},
            )

        attention_mask = batch.get("attention_mask")
        weights = self._get_layer_weights(num_layers, config.layer_weights)
        device = student_hidden[0].device

        layer_losses = []
        total_loss = torch.tensor(0.0, device=device, dtype=student_hidden[0].dtype)

        loss_type = HiddenLossType(config.loss_type)

        for idx, (s_hidden, t_hidden) in enumerate(zip(student_hidden, teacher_hidden)):
            layer_loss = self._compute_hidden_layer_loss(
                s_hidden, t_hidden, attention_mask, loss_type, config.normalize
            )
            layer_losses.append(layer_loss.detach())
            total_loss = total_loss + weights[idx] * layer_loss

        # Apply temperature scaling (1.0 = no effect for hidden, configurable for LRC)
        total_loss = total_loss / config.temperature

        mean_layer_loss = torch.stack(layer_losses).mean()

        # Prefix all metrics with config.name for distinction
        metrics = {
            f"{config.name}_mean_layer_loss": mean_layer_loss,
            f"{config.name}_num_layers": torch.tensor(float(num_layers), device=device),
        }
        if config.temperature != 1.0:
            metrics[f"{config.name}_temperature"] = torch.tensor(
                config.temperature, device=device
            )

        return total_loss, metrics

    def _compute_hidden_layer_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        loss_type: HiddenLossType,
        normalize: bool,
    ) -> torch.Tensor:
        """Compute loss for a single layer."""
        if loss_type == HiddenLossType.MSE:
            return self._mse_loss(student_h, teacher_h, attention_mask, normalize=False)
        elif loss_type == HiddenLossType.MSE_NORMALIZED:
            return self._mse_loss(student_h, teacher_h, attention_mask, normalize=True)
        elif loss_type == HiddenLossType.COSINE:
            return self._cosine_loss(student_h, teacher_h, attention_mask)
        elif loss_type == HiddenLossType.INNER_PRODUCT:
            return self._inner_product_loss(student_h, teacher_h, attention_mask)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    # =========================================================================
    # Logits Component (from LogitsDistillationObjective + TCSDistillationObjective)
    # =========================================================================

    def _compute_logits_loss(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Logits distillation - full KL or sparse TCS."""
        mode = LogitsMode(self.logits_config.mode)
        if mode == LogitsMode.SPARSE:
            return self._compute_sparse_logits_loss(model_outputs, batch, teacher_outputs)
        else:
            return self._compute_full_logits_loss(model_outputs, batch, teacher_outputs)

    def _compute_full_logits_loss(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Full KL divergence (from LogitsDistillationObjective)."""
        student_logits = model_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]

        # Get labels for masking (use original labels if preprocessing modified them)
        labels = batch.get("_original_labels", batch["labels"])
        ignore_index = self.logits_config.ignore_index

        # Shift for next-token prediction (AR models)
        if self.logits_config.shift_labels:
            student_logits = student_logits[..., :-1, :].contiguous()
            teacher_logits = teacher_logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        # Create mask for valid positions
        mask = (labels != ignore_index).float()

        # Temperature-scaled probabilities
        temperature = self.logits_config.temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # KL divergence
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        kl_div = kl_div.sum(dim=-1)

        # Apply mask and normalize
        kl_div = kl_div * mask
        num_valid = mask.sum().clamp(min=1)

        # Temperature^2 scaling (standard for distillation)
        loss = (kl_div.sum() / num_valid) * (temperature**2)

        return loss, {
            "kl_div": (kl_div.sum() / num_valid).detach(),
            "temperature": torch.tensor(temperature, device=loss.device),
        }

    def _compute_sparse_logits_loss(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sparse TCS-style logits distillation (from TCSDistillationObjective)."""
        student_logits = model_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]

        labels = batch["labels"]
        ignore_index = self.logits_config.ignore_index

        # Response mask: positions where we compute loss
        response_mask = labels != ignore_index

        batch_size, seq_len, vocab_size = student_logits.shape

        # Get teacher's top-K predictions
        top_k = min(self.logits_config.top_k, vocab_size)
        teacher_topk_logits, topk_indices = torch.topk(teacher_logits, k=top_k, dim=-1)

        # Gather student logits at the same indices
        student_topk_logits = torch.gather(student_logits, dim=-1, index=topk_indices)

        # Temperature-scaled probabilities
        temperature = self.logits_config.temperature
        teacher_probs = F.softmax(teacher_topk_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_topk_logits / temperature, dim=-1)

        # KL divergence
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

        # Mask to response positions only
        kl = kl * response_mask.float()
        num_valid = response_mask.sum().clamp(min=1)

        # Temperature^2 scaling
        loss = (kl.sum() / num_valid) * (temperature**2)

        return loss, {
            "tcs_kl": (kl.sum() / num_valid).detach(),
            "num_masked": response_mask.sum().detach().float(),
        }

    # =========================================================================
    # Attention Component (from AttentionRelationDistillationObjective)
    # =========================================================================

    def _validate_and_extract_attention(
        self,
        model_outputs: dict[str, Any],
        teacher_outputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Validate attention outputs and extract specified layer.

        Args:
            model_outputs: Student model outputs containing "attentions"
            teacher_outputs: Teacher model outputs containing "attentions"

        Returns:
            Tuple of (student_attn, teacher_attn, normalized_layer_idx)

        Raises:
            ValueError: If attentions are missing or None at specified layer
        """
        student_attentions = model_outputs.get("attentions")
        teacher_attentions = teacher_outputs.get("attentions")

        if student_attentions is None or len(student_attentions) == 0:
            raise ValueError(
                "Student model must return attention weights. "
                "Use output_attentions=True and eager attention implementation."
            )
        if teacher_attentions is None or len(teacher_attentions) == 0:
            raise ValueError(
                "Teacher model must return attention weights. "
                "Load teacher with attn_implementation='eager'."
            )

        layer_idx = self.attention_config.distill_layer
        student_attn = student_attentions[layer_idx]
        teacher_attn = teacher_attentions[layer_idx]

        if student_attn is None:
            raise ValueError(
                f"Student attention at layer {layer_idx} is None. "
                "Ensure output_attentions=True and use eager attention."
            )
        if teacher_attn is None:
            raise ValueError(
                f"Teacher attention at layer {layer_idx} is None. "
                "Load teacher with attn_implementation='eager'."
            )

        # Normalize negative index for metrics
        normalized_idx = layer_idx % len(student_attentions)

        return student_attn, teacher_attn, normalized_idx

    def _compute_attention_loss(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Attention distillation using relation matching."""
        return self._compute_relation_attention_loss(model_outputs, batch, teacher_outputs)

    def _compute_relation_attention_loss(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Attention relation distillation (from AttentionRelationDistillationObjective)."""
        student_attn, teacher_attn, layer_idx = self._validate_and_extract_attention(
            model_outputs, teacher_outputs
        )

        attention_mask = batch.get("attention_mask")

        # Compute relation matrices: R = Softmax(A · A^T / sqrt(d_r))
        d_r = student_attn.shape[-1]
        scale = math.sqrt(d_r)
        temperature = self.attention_config.temperature

        # Student relations
        student_aat = torch.matmul(student_attn, student_attn.transpose(-2, -1))
        student_R = F.softmax(student_aat / (scale * temperature), dim=-1)

        # Teacher relations
        teacher_aat = torch.matmul(teacher_attn, teacher_attn.transpose(-2, -1))
        teacher_R = F.softmax(teacher_aat / (scale * temperature), dim=-1)

        # Handle head count mismatch
        if student_R.shape[1] != teacher_R.shape[1]:
            logger.warning(
                f"Head count mismatch: student={student_R.shape[1]}, teacher={teacher_R.shape[1]}. "
                "Averaging over heads."
            )
            student_R = student_R.mean(dim=1, keepdim=True)
            teacher_R = teacher_R.mean(dim=1, keepdim=True)

        # Numerical stability
        eps = 1e-8
        student_R = student_R.clamp(min=eps)
        teacher_R = teacher_R.clamp(min=eps)

        # KL divergence
        kl = teacher_R * (teacher_R.log() - student_R.log())
        kl = kl.sum(dim=-1)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).float()
            kl = kl * mask
            valid_count = mask.sum() * kl.shape[1]
            loss = kl.sum() / valid_count.clamp(min=1)
        else:
            loss = kl.mean()

        return loss, {
            "attention_kl": loss.detach(),
            "distill_layer": torch.tensor(layer_idx, device=loss.device),
        }

    # =========================================================================
    # Shared Helper Methods
    # =========================================================================

    def _get_layer_weights(
        self, num_layers: int, config: Optional[str | list[float]]
    ) -> list[float]:
        """Get normalized layer weights."""
        if config is None:
            return [1.0 / num_layers] * num_layers
        elif config == "progressive":
            weights = [(i + 1) for i in range(num_layers)]
            total = sum(weights)
            return [w / total for w in weights]
        elif config == "exponential":
            weights = [2**i for i in range(num_layers)]
            total = sum(weights)
            return [w / total for w in weights]
        elif isinstance(config, list):
            if len(config) != num_layers:
                raise ValueError(
                    f"layer_weights length ({len(config)}) must match num_layers ({num_layers})"
                )
            total = sum(config)
            if total <= 0:
                raise ValueError(f"layer_weights must sum to > 0, got {total}")
            return [w / total for w in config]
        else:
            raise ValueError(f"Unknown layer_weights config: {config}")

    def _mse_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        normalize: bool = False,
    ) -> torch.Tensor:
        """MSE loss with optional L2 normalization."""
        if normalize:
            student_h = F.normalize(student_h, p=2, dim=-1)
            teacher_h = F.normalize(teacher_h.detach(), p=2, dim=-1)
        else:
            teacher_h = teacher_h.detach()

        mse = (student_h - teacher_h).pow(2).mean(dim=-1)

        if attention_mask is not None:
            mse = mse * attention_mask.to(mse.dtype)
            return mse.sum() / attention_mask.sum().clamp(min=1)
        return mse.mean()

    def _cosine_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cosine distance loss (1 - cosine_similarity)."""
        cos_sim = F.cosine_similarity(student_h, teacher_h.detach(), dim=-1)
        loss = 1.0 - cos_sim

        if attention_mask is not None:
            loss = loss * attention_mask.to(loss.dtype)
            return loss.sum() / attention_mask.sum().clamp(min=1)
        return loss.mean()

    def _inner_product_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Negative inner product loss (normalized)."""
        student_h = F.normalize(student_h, p=2, dim=-1)
        teacher_h = F.normalize(teacher_h.detach(), p=2, dim=-1)

        inner = -(student_h * teacher_h).sum(dim=-1)

        if attention_mask is not None:
            inner = inner * attention_mask.to(inner.dtype)
            return inner.sum() / attention_mask.sum().clamp(min=1)
        return inner.mean()
