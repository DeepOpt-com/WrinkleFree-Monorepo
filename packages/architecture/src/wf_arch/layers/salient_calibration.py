"""Calibration utilities for BitLinearSalient.

Implements AWQ-style activation-aware saliency scoring:

    saliency[col] = mean(|activation[:, col]|) * ||weight[:, col]||_2

Based on AWQ: https://arxiv.org/abs/2306.00978

The key insight is that we should refer to the activation distribution,
not just the weight distribution, when identifying important columns.
Weight columns corresponding to larger activation magnitudes are more
salient since they process more important features.

Usage:
    1. Create calibrator: calibrator = SalientCalibrator(model, salient_ratio=0.01)
    2. Run calibration: saliency_scores = calibrator.calibrate(dataloader, num_samples=128)
    3. Get indices: salient_indices = calibrator.get_salient_indices(saliency_scores)
    4. Apply to model: use convert_bitlinear_to_salient(model, salient_indices=salient_indices)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from wf_arch.layers.bitlinear import BitLinear
from wf_arch.layers.bitlinear_salient import BitLinearSalient

logger = logging.getLogger(__name__)


class ActivationCollector:
    """Collects activation statistics for saliency scoring.

    Maintains a running sum of absolute activation values per column
    across all forward passes. Used to compute mean absolute activation
    for AWQ-style saliency scoring.
    """

    def __init__(self):
        self.activation_sum: Optional[torch.Tensor] = None
        self.activation_count: int = 0

    def update(self, x: torch.Tensor) -> None:
        """Update running sum of absolute activations.

        Args:
            x: Activation tensor of shape (..., in_features)
        """
        # Flatten to (total_tokens, in_features)
        x_flat = x.reshape(-1, x.shape[-1])

        # Accumulate absolute activations per column
        abs_sum = x_flat.abs().sum(dim=0)

        if self.activation_sum is None:
            self.activation_sum = abs_sum
        else:
            self.activation_sum = self.activation_sum + abs_sum

        self.activation_count += x_flat.shape[0]

    def get_mean_abs(self) -> torch.Tensor:
        """Get mean absolute activation per column.

        Returns:
            Tensor of shape (in_features,) with mean absolute activation per column
        """
        if self.activation_count == 0:
            raise RuntimeError("No activations collected. Run calibration first.")
        return self.activation_sum / self.activation_count

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.activation_sum = None
        self.activation_count = 0


class SalientCalibrator:
    """Calibrates BitLinearSalient layers using activation statistics.

    Collects activation magnitudes from calibration data to compute
    AWQ-style saliency scores:

        saliency[col] = mean(|activation[:, col]|) * ||weight[:, col]||_2

    The top salient_ratio% columns are marked as "salient" and kept in FP16.

    Args:
        model: Model containing BitLinear layers to calibrate
        salient_ratio: Fraction of columns to keep in FP16 per layer (default 0.01 = 1%)
    """

    def __init__(
        self,
        model: nn.Module,
        salient_ratio: float = 0.01,
    ):
        self.model = model
        self.salient_ratio = salient_ratio
        self.collectors: Dict[str, ActivationCollector] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Find all BitLinear layers (not yet converted to Salient)
        self._layer_names: List[str] = []
        for name, module in model.named_modules():
            if isinstance(module, BitLinear) and not isinstance(
                module, BitLinearSalient
            ):
                self._layer_names.append(name)
                self.collectors[name] = ActivationCollector()

        logger.info(f"SalientCalibrator: Found {len(self._layer_names)} BitLinear layers")

    def _get_hook(self, name: str):
        """Create forward hook for collecting activations.

        The hook captures the input to each BitLinear layer and updates
        the corresponding ActivationCollector with absolute activation values.
        """

        def hook(module, input, output):
            if len(input) > 0 and isinstance(input[0], torch.Tensor):
                # Detach to avoid holding onto computation graph
                self.collectors[name].update(input[0].detach())

        return hook

    def _register_hooks(self) -> None:
        """Register forward hooks on all BitLinear layers."""
        for name, module in self.model.named_modules():
            if name in self.collectors:
                handle = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    @torch.no_grad()
    def calibrate(
        self,
        dataloader,
        num_samples: int = 128,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run calibration to collect activation statistics.

        Runs the model on calibration data to collect activation magnitudes
        at each BitLinear layer. Uses these to compute AWQ-style saliency scores.

        Args:
            dataloader: DataLoader providing calibration samples.
                Should yield dicts with "input_ids" or tuples with input tensors.
            num_samples: Number of samples to use for calibration
            device: Device to run calibration on. If None, uses model's device.

        Returns:
            Dict mapping layer names to saliency scores (shape: in_features per layer)
        """
        print("[CALIB DEBUG] calibrate() called", flush=True)
        if device is None:
            device = next(self.model.parameters()).device
        print(f"[CALIB DEBUG] device={device}", flush=True)

        # Save and restore training mode
        was_training = self.model.training
        self.model.eval()
        print("[CALIB DEBUG] model.eval() done", flush=True)
        self._register_hooks()
        print(f"[CALIB DEBUG] hooks registered ({len(self.hooks)} hooks)", flush=True)

        try:
            samples_processed = 0
            batch_count = 0
            # Don't wrap streaming dataloader with tqdm - iterate manually
            # This prevents tqdm from prefetching infinitely from streaming datasets
            pbar = tqdm(
                desc="Calibrating salient columns",
                total=num_samples,
                unit="samples",
            )
            print("[CALIB DEBUG] tqdm created", flush=True)

            dataloader_iter = iter(dataloader)
            print("[CALIB DEBUG] dataloader iter created", flush=True)
            while samples_processed < num_samples:
                print(f"[CALIB DEBUG] getting batch {batch_count}...", flush=True)
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    logger.warning("Dataloader exhausted before reaching num_samples")
                    break
                print(f"[CALIB DEBUG] got batch {batch_count}", flush=True)

                # Handle different batch formats
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids", batch.get("input"))
                elif isinstance(batch, (tuple, list)):
                    input_ids = batch[0]
                else:
                    input_ids = batch

                if input_ids is None:
                    logger.warning("Batch has no input_ids, skipping")
                    continue

                input_ids = input_ids.to(device)
                batch_size = input_ids.shape[0]
                print(f"[CALIB DEBUG] batch {batch_count}: shape={input_ids.shape}", flush=True)

                # Forward pass to collect activations
                try:
                    print(f"[CALIB DEBUG] batch {batch_count}: forward pass...", flush=True)
                    _ = self.model(input_ids)
                    print(f"[CALIB DEBUG] batch {batch_count}: forward done", flush=True)
                except Exception as e:
                    print(f"[CALIB ERROR] Forward pass failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raise  # Don't silently continue - surface the error

                samples_processed += batch_size
                batch_count += 1
                pbar.update(batch_size)
                pbar.set_postfix({"batches": batch_count})

            pbar.close()
            print(f"[CALIB DEBUG] loop done, {samples_processed} samples", flush=True)

        finally:
            self._remove_hooks()
            # Restore original training mode
            if was_training:
                self.model.train()

        if samples_processed == 0:
            raise RuntimeError("No samples processed during calibration")

        logger.info(f"Calibration complete: processed {samples_processed} samples")

        # Compute saliency scores for each layer
        saliency_scores = {}
        for name, collector in self.collectors.items():
            # Get mean absolute activation per column
            mean_abs_act = collector.get_mean_abs()

            # Get weight norm per column (L2 norm)
            module = dict(self.model.named_modules())[name]
            weight = module.weight.data
            weight_norm = weight.norm(dim=0)  # L2 norm per column

            # AWQ-style saliency: activation magnitude * weight magnitude
            saliency = mean_abs_act * weight_norm.to(mean_abs_act.device)
            saliency_scores[name] = saliency

            logger.debug(
                f"Layer {name}: max_saliency={saliency.max():.4f}, "
                f"min_saliency={saliency.min():.4f}, "
                f"mean_saliency={saliency.mean():.4f}"
            )

        return saliency_scores

    def get_salient_indices(
        self,
        saliency_scores: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Get indices of salient columns for each layer.

        Selects the top salient_ratio% of columns based on saliency scores.

        Args:
            saliency_scores: Dict mapping layer names to saliency score tensors

        Returns:
            Dict mapping layer names to salient column indices (sorted ascending)
        """
        salient_indices = {}

        for name, scores in saliency_scores.items():
            num_salient = max(1, int(len(scores) * self.salient_ratio))
            _, top_indices = torch.topk(scores, num_salient)

            # Sort indices for deterministic behavior
            salient_indices[name] = top_indices.sort().values

            logger.info(
                f"Layer {name}: {num_salient} salient columns "
                f"({self.salient_ratio * 100:.1f}%)"
            )

        return salient_indices


def calibrate_salient_columns(
    model: nn.Module,
    dataloader,
    salient_ratio: float = 0.01,
    num_samples: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Convenience function to run full calibration pipeline.

    Creates a SalientCalibrator, runs calibration, and returns both
    saliency scores and salient indices ready for use with
    convert_bitlinear_to_salient().

    Args:
        model: Model with BitLinear layers to calibrate
        dataloader: DataLoader providing calibration samples
        salient_ratio: Fraction of columns to mark as salient (default 0.01 = 1%)
        num_samples: Number of calibration samples to use
        device: Device to run calibration on

    Returns:
        Tuple of (saliency_scores, salient_indices):
            - saliency_scores: Dict mapping layer names to full saliency score tensors
            - salient_indices: Dict mapping layer names to salient column index tensors

    Example:
        >>> saliency_scores, salient_indices = calibrate_salient_columns(
        ...     model, calibration_dataloader, salient_ratio=0.01
        ... )
        >>> model = convert_bitlinear_to_salient(
        ...     model, salient_indices=salient_indices, saliency_scores=saliency_scores
        ... )
    """
    calibrator = SalientCalibrator(model, salient_ratio=salient_ratio)
    saliency_scores = calibrator.calibrate(dataloader, num_samples, device)
    salient_indices = calibrator.get_salient_indices(saliency_scores)
    return saliency_scores, salient_indices
