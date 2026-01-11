"""QK-Clipping for attention stability in large-scale training.

Based on Kimi K2: If max attention score > threshold, rescale W_q and W_k.
Reference: https://arxiv.org/abs/2507.20534

Usage:
    from wf_data.training.qk_clip import apply_qk_clip

    # After optimizer.step():
    stats = apply_qk_clip(model, threshold=50.0, alpha=0.5)
    if stats["was_clipped"]:
        print(f"QK clipped: max_score={stats['max_score']:.2f}")
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class QKClipStats:
    """Statistics from QK clipping operation."""

    max_score: float  # Maximum attention score proxy
    was_clipped: bool  # Whether any clipping occurred
    scale_factor: float  # Applied scale factor (1.0 = no clipping)
    num_clipped: int  # Number of layers that were clipped


def apply_qk_clip(
    model: nn.Module,
    threshold: float = 50.0,
    alpha: float = 0.5,
    enabled: bool = True,
) -> QKClipStats:
    """Apply QK-Clipping to attention layers after optimizer step.

    From Kimi K2: If max attention score > threshold, rescale W_q and W_k.
    Uses spectral norm as a proxy for potential max attention scores.

    Args:
        model: The transformer model
        threshold: Max allowed attention score (tau). Default 50.0 (more aggressive than Kimi K2's 100)
        alpha: Balance factor for rescaling (0.5 = equal split between Q and K)
        enabled: Whether clipping is enabled (for easy toggle)

    Returns:
        QKClipStats with clipping statistics
    """
    if not enabled:
        return QKClipStats(max_score=0.0, was_clipped=False, scale_factor=1.0, num_clipped=0)

    max_score = 0.0
    was_clipped = False
    scale_factor = 1.0
    num_clipped = 0

    # Find Q and K projection layers
    for name, module in model.named_modules():
        # Common naming patterns for attention projections
        if any(k in name.lower() for k in ["q_proj", "k_proj", "query", "key"]):
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                if w.ndim >= 2:
                    with torch.no_grad():
                        # Use power iteration for spectral norm estimate
                        # This approximates the max attention score for unit-norm Q/K
                        u = torch.randn(w.size(0), device=w.device, dtype=w.dtype)
                        for _ in range(3):  # Few iterations sufficient
                            v = w.T @ u
                            v = v / (v.norm() + 1e-7)
                            u = w @ v
                            u = u / (u.norm() + 1e-7)
                        spectral_norm = (u @ w @ v).abs().item()

                        if spectral_norm > threshold:
                            scale = threshold / spectral_norm
                            # Apply scaling with alpha balance
                            if "q" in name.lower():
                                module.weight.data.mul_(scale**alpha)
                            else:  # k
                                module.weight.data.mul_(scale ** (1 - alpha))
                            was_clipped = True
                            scale_factor = min(scale_factor, scale)
                            num_clipped += 1

                        max_score = max(max_score, spectral_norm)

    return QKClipStats(
        max_score=max_score,
        was_clipped=was_clipped,
        scale_factor=scale_factor,
        num_clipped=num_clipped,
    )


class QKClipMonitor:
    """Monitor for attention logits and QK clipping.

    Registers forward hooks on attention layers to track actual attention scores
    and provides logging utilities for WandB.
    """

    def __init__(
        self,
        threshold: float = 50.0,
        alpha: float = 0.5,
        enabled: bool = True,
    ):
        self.threshold = threshold
        self.alpha = alpha
        self.enabled = enabled
        self.hooks = []
        self.attention_stats = {}  # layer_name -> max_attn_logit

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on attention layers."""
        if not self.enabled:
            return

        for name, module in model.named_modules():
            # Look for attention score computation patterns
            if "attention" in name.lower() and hasattr(module, "forward"):
                hook = module.register_forward_hook(self._attention_hook(name))
                self.hooks.append(hook)

    def _attention_hook(self, layer_name: str):
        """Create a forward hook for tracking attention scores."""

        def hook(module, inputs, outputs):
            # Try to extract attention weights from outputs
            # This varies by model architecture
            if isinstance(outputs, tuple) and len(outputs) > 1:
                # Many models return (hidden_states, attn_weights)
                attn_weights = outputs[1]
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    max_attn = attn_weights.max().item()
                    self.attention_stats[layer_name] = max_attn

        return hook

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_max_attention_logit(self) -> float:
        """Get the maximum attention logit across all monitored layers."""
        if not self.attention_stats:
            return 0.0
        return max(self.attention_stats.values())

    def reset_stats(self) -> None:
        """Reset attention statistics."""
        self.attention_stats.clear()

    def apply_clipping(self, model: nn.Module) -> QKClipStats:
        """Apply QK clipping and return stats."""
        return apply_qk_clip(
            model,
            threshold=self.threshold,
            alpha=self.alpha,
            enabled=self.enabled,
        )

    def get_wandb_metrics(self, stats: QKClipStats) -> dict:
        """Get metrics formatted for WandB logging."""
        metrics = {
            "qk_clip/max_spectral_norm": stats.max_score,
            "qk_clip/was_clipped": 1.0 if stats.was_clipped else 0.0,
            "qk_clip/scale_factor": stats.scale_factor,
            "qk_clip/num_clipped_layers": stats.num_clipped,
        }

        # Add real attention stats if available
        max_attn = self.get_max_attention_logit()
        if max_attn > 0:
            metrics["qk_clip/max_attention_logit"] = max_attn

        return metrics
