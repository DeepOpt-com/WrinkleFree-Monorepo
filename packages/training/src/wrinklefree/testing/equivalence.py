"""Equivalence testing utilities for comparing training code versions.

This module provides utilities to verify that optimized training code produces
functionally equivalent outputs to baseline implementations.

Usage:
    from wrinklefree.testing.equivalence import compare_logits_cosine, compare_gradients

    # Compare two model outputs
    similarity = compare_logits_cosine(model_a, model_b, input_batch)
    assert similarity > 0.99, f"Models diverged: {similarity}"

    # Compare gradient updates
    grad_result = compare_gradients(model_a, model_b, input_batch, loss_fn)
    assert grad_result.cosine_similarity > 0.99
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EquivalenceResult:
    """Results from an equivalence comparison."""

    # Similarity metrics (0-1, higher is more similar)
    logit_cosine_similarity: float = 0.0
    gradient_cosine_similarity: float = 0.0
    hidden_state_cosine_similarity: float = 0.0

    # Difference metrics
    logit_max_diff: float = float("inf")
    logit_mean_diff: float = float("inf")
    gradient_max_diff: float = float("inf")
    weight_max_diff: float = float("inf")

    # Status
    passed: bool = False
    error_message: Optional[str] = None

    # Per-layer breakdown (optional)
    per_layer_similarities: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"EquivalenceResult({status}): "
            f"logit_cos={self.logit_cosine_similarity:.4f}, "
            f"grad_cos={self.gradient_cosine_similarity:.4f}, "
            f"logit_max_diff={self.logit_max_diff:.2e}"
        )


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors.

    Args:
        a: First tensor
        b: Second tensor (must have same shape as a)

    Returns:
        Cosine similarity as a float in range [-1, 1]
    """
    a_flat = a.detach().float().flatten()
    b_flat = b.detach().float().flatten()

    # Handle zero vectors
    a_norm = a_flat.norm()
    b_norm = b_flat.norm()

    if a_norm < 1e-8 or b_norm < 1e-8:
        # If both are near-zero, consider them equivalent
        if a_norm < 1e-8 and b_norm < 1e-8:
            return 1.0
        return 0.0

    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))


def compare_logits_cosine(
    model_a: nn.Module,
    model_b: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compare output logits from two models using cosine similarity.

    Args:
        model_a: First model
        model_b: Second model
        input_ids: Input token IDs (batch, seq)
        attention_mask: Optional attention mask

    Returns:
        Cosine similarity of output logits (0-1, higher is more similar)
    """
    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        outputs_a = model_a(input_ids=input_ids, attention_mask=attention_mask)
        outputs_b = model_b(input_ids=input_ids, attention_mask=attention_mask)

    # Extract logits (handle dict and ModelOutput)
    logits_a = outputs_a["logits"] if isinstance(outputs_a, dict) else outputs_a.logits
    logits_b = outputs_b["logits"] if isinstance(outputs_b, dict) else outputs_b.logits

    return cosine_similarity(logits_a, logits_b)


def compare_hidden_states(
    model_a: nn.Module,
    model_b: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Compare hidden states from all layers between two models.

    Args:
        model_a: First model
        model_b: Second model
        input_ids: Input token IDs
        attention_mask: Optional attention mask

    Returns:
        Dict mapping layer names to cosine similarities
    """
    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        outputs_a = model_a(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        outputs_b = model_b(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # Extract hidden states
    if isinstance(outputs_a, dict):
        hidden_a = outputs_a.get("hidden_states", [])
        hidden_b = outputs_b.get("hidden_states", [])
    else:
        hidden_a = outputs_a.hidden_states or []
        hidden_b = outputs_b.hidden_states or []

    results = {}
    for i, (h_a, h_b) in enumerate(zip(hidden_a, hidden_b)):
        layer_name = f"layer_{i}"
        results[layer_name] = cosine_similarity(h_a, h_b)

    return results


def compare_gradients(
    model_a: nn.Module,
    model_b: nn.Module,
    input_ids: torch.Tensor,
    loss_fn: Callable,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> EquivalenceResult:
    """Compare gradient updates between two models after one forward-backward pass.

    Args:
        model_a: First model (will compute gradients)
        model_b: Second model (will compute gradients)
        input_ids: Input token IDs
        loss_fn: Loss function that takes (logits, labels) or (outputs, batch)
        attention_mask: Optional attention mask
        labels: Labels for loss computation (defaults to shifted input_ids)

    Returns:
        EquivalenceResult with gradient comparison metrics
    """
    model_a.train()
    model_b.train()

    # Zero gradients
    model_a.zero_grad()
    model_b.zero_grad()

    # Default labels to shifted input_ids (language modeling)
    if labels is None:
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token

    # Forward pass - model A
    outputs_a = model_a(input_ids=input_ids, attention_mask=attention_mask)
    logits_a = outputs_a["logits"] if isinstance(outputs_a, dict) else outputs_a.logits

    # Forward pass - model B
    outputs_b = model_b(input_ids=input_ids, attention_mask=attention_mask)
    logits_b = outputs_b["logits"] if isinstance(outputs_b, dict) else outputs_b.logits

    # Compute loss
    try:
        # Try calling loss_fn with (logits, labels)
        loss_a = loss_fn(logits_a.view(-1, logits_a.size(-1)), labels.view(-1))
        loss_b = loss_fn(logits_b.view(-1, logits_b.size(-1)), labels.view(-1))
    except TypeError:
        # Fall back to cross entropy
        loss_a = F.cross_entropy(
            logits_a.view(-1, logits_a.size(-1)), labels.view(-1), ignore_index=-100
        )
        loss_b = F.cross_entropy(
            logits_b.view(-1, logits_b.size(-1)), labels.view(-1), ignore_index=-100
        )

    # Backward pass
    loss_a.backward()
    loss_b.backward()

    # Collect all gradients
    grads_a = []
    grads_b = []
    per_layer_sims = {}

    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        if param_a.grad is not None and param_b.grad is not None:
            grads_a.append(param_a.grad.flatten())
            grads_b.append(param_b.grad.flatten())

            # Per-layer similarity
            layer_sim = cosine_similarity(param_a.grad, param_b.grad)
            per_layer_sims[name_a] = layer_sim

    if not grads_a:
        return EquivalenceResult(
            passed=False, error_message="No gradients found in models"
        )

    # Concatenate all gradients
    all_grads_a = torch.cat(grads_a)
    all_grads_b = torch.cat(grads_b)

    # Compute metrics
    grad_cosine = cosine_similarity(all_grads_a, all_grads_b)
    grad_diff = (all_grads_a - all_grads_b).abs()
    logit_diff = (logits_a - logits_b).abs()

    result = EquivalenceResult(
        logit_cosine_similarity=cosine_similarity(logits_a, logits_b),
        gradient_cosine_similarity=grad_cosine,
        logit_max_diff=float(logit_diff.max()),
        logit_mean_diff=float(logit_diff.mean()),
        gradient_max_diff=float(grad_diff.max()),
        per_layer_similarities=per_layer_sims,
        passed=grad_cosine > 0.99,
    )

    return result


def run_n_steps_and_compare(
    trainer_factory_a: Callable,
    trainer_factory_b: Callable,
    n_steps: int = 10,
    seed: int = 42,
) -> EquivalenceResult:
    """Run N training steps on two trainer instances and compare final states.

    Args:
        trainer_factory_a: Callable that returns (model, optimizer, dataloader) for version A
        trainer_factory_b: Callable that returns (model, optimizer, dataloader) for version B
        n_steps: Number of training steps to run
        seed: Random seed for reproducibility

    Returns:
        EquivalenceResult comparing the two training runs
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    model_a, optimizer_a, dataloader_a = trainer_factory_a()

    torch.manual_seed(seed)
    model_b, optimizer_b, dataloader_b = trainer_factory_b()

    # Run training steps
    model_a.train()
    model_b.train()

    iter_a = iter(dataloader_a)
    iter_b = iter(dataloader_b)

    for step in range(n_steps):
        # Get batches (should be identical due to same seed)
        batch_a = next(iter_a)
        batch_b = next(iter_b)

        # Move to device
        device = next(model_a.parameters()).device
        batch_a = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_a.items()}
        batch_b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_b.items()}

        # Forward pass
        outputs_a = model_a(**{k: v for k, v in batch_a.items() if k != "labels"})
        outputs_b = model_b(**{k: v for k, v in batch_b.items() if k != "labels"})

        logits_a = outputs_a["logits"] if isinstance(outputs_a, dict) else outputs_a.logits
        logits_b = outputs_b["logits"] if isinstance(outputs_b, dict) else outputs_b.logits

        # Compute loss
        labels = batch_a.get("labels", batch_a["input_ids"])
        loss_a = F.cross_entropy(
            logits_a.view(-1, logits_a.size(-1)), labels.view(-1), ignore_index=-100
        )
        loss_b = F.cross_entropy(
            logits_b.view(-1, logits_b.size(-1)), labels.view(-1), ignore_index=-100
        )

        # Backward and step
        loss_a.backward()
        loss_b.backward()

        optimizer_a.step()
        optimizer_b.step()

        optimizer_a.zero_grad()
        optimizer_b.zero_grad()

    # Compare final model weights
    weight_diffs = []
    weight_cosines = []

    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        diff = (param_a - param_b).abs().max().item()
        cos = cosine_similarity(param_a, param_b)
        weight_diffs.append(diff)
        weight_cosines.append(cos)

    avg_weight_cosine = sum(weight_cosines) / len(weight_cosines) if weight_cosines else 0
    max_weight_diff = max(weight_diffs) if weight_diffs else float("inf")

    # Get a sample batch for logit comparison
    model_a.eval()
    model_b.eval()

    torch.manual_seed(seed + 999)  # Different seed for eval batch
    eval_batch = next(iter(dataloader_a))
    device = next(model_a.parameters()).device
    eval_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}

    with torch.no_grad():
        outputs_a = model_a(**{k: v for k, v in eval_batch.items() if k != "labels"})
        outputs_b = model_b(**{k: v for k, v in eval_batch.items() if k != "labels"})

    logits_a = outputs_a["logits"] if isinstance(outputs_a, dict) else outputs_a.logits
    logits_b = outputs_b["logits"] if isinstance(outputs_b, dict) else outputs_b.logits

    logit_cosine = cosine_similarity(logits_a, logits_b)
    logit_diff = (logits_a - logits_b).abs()

    result = EquivalenceResult(
        logit_cosine_similarity=logit_cosine,
        gradient_cosine_similarity=avg_weight_cosine,  # Use weight similarity as proxy
        logit_max_diff=float(logit_diff.max()),
        logit_mean_diff=float(logit_diff.mean()),
        weight_max_diff=max_weight_diff,
        passed=logit_cosine > 0.99 and avg_weight_cosine > 0.99,
    )

    return result


def assert_models_equivalent(
    model_a: nn.Module,
    model_b: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    cosine_threshold: float = 0.99,
) -> None:
    """Assert that two models produce equivalent outputs.

    Args:
        model_a: First model
        model_b: Second model
        input_ids: Input token IDs
        attention_mask: Optional attention mask
        rtol: Relative tolerance for torch.allclose
        atol: Absolute tolerance for torch.allclose
        cosine_threshold: Minimum cosine similarity required

    Raises:
        AssertionError: If models are not equivalent
    """
    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        outputs_a = model_a(input_ids=input_ids, attention_mask=attention_mask)
        outputs_b = model_b(input_ids=input_ids, attention_mask=attention_mask)

    logits_a = outputs_a["logits"] if isinstance(outputs_a, dict) else outputs_a.logits
    logits_b = outputs_b["logits"] if isinstance(outputs_b, dict) else outputs_b.logits

    # Check cosine similarity
    cos_sim = cosine_similarity(logits_a, logits_b)
    assert cos_sim >= cosine_threshold, (
        f"Models not equivalent: cosine similarity {cos_sim:.4f} < {cosine_threshold}"
    )

    # Check absolute/relative tolerance
    if not torch.allclose(logits_a, logits_b, rtol=rtol, atol=atol):
        max_diff = (logits_a - logits_b).abs().max().item()
        logger.warning(
            f"Models pass cosine check but fail allclose: max_diff={max_diff:.2e}, "
            f"rtol={rtol}, atol={atol}"
        )


def create_equivalence_checkpoint(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Create a checkpoint of model state for later comparison.

    Args:
        model: Model to checkpoint
        input_ids: Input to run through model
        attention_mask: Optional attention mask

    Returns:
        Dict with model outputs and state for comparison
    """
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    hidden_states = (
        outputs.get("hidden_states", [])
        if isinstance(outputs, dict)
        else (outputs.hidden_states or [])
    )

    return {
        "logits": logits.cpu().clone(),
        "hidden_states": [h.cpu().clone() for h in hidden_states],
        "input_ids": input_ids.cpu().clone(),
        "state_dict_hash": hash(
            tuple(
                (k, v.sum().item())
                for k, v in sorted(model.state_dict().items())
            )
        ),
    }


def compare_to_checkpoint(
    model: nn.Module,
    checkpoint: dict[str, Any],
    attention_mask: Optional[torch.Tensor] = None,
) -> EquivalenceResult:
    """Compare current model outputs to a saved checkpoint.

    Args:
        model: Model to compare
        checkpoint: Checkpoint from create_equivalence_checkpoint
        attention_mask: Optional attention mask

    Returns:
        EquivalenceResult comparing model to checkpoint
    """
    model.eval()
    device = next(model.parameters()).device

    input_ids = checkpoint["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    hidden_states = (
        outputs.get("hidden_states", [])
        if isinstance(outputs, dict)
        else (outputs.hidden_states or [])
    )

    # Compare logits
    checkpoint_logits = checkpoint["logits"].to(device)
    logit_cosine = cosine_similarity(logits, checkpoint_logits)
    logit_diff = (logits - checkpoint_logits).abs()

    # Compare hidden states
    hidden_cosines = {}
    checkpoint_hidden = checkpoint["hidden_states"]
    for i, (h_curr, h_ckpt) in enumerate(zip(hidden_states, checkpoint_hidden)):
        h_ckpt = h_ckpt.to(device)
        hidden_cosines[f"layer_{i}"] = cosine_similarity(h_curr, h_ckpt)

    avg_hidden_cosine = (
        sum(hidden_cosines.values()) / len(hidden_cosines) if hidden_cosines else 0
    )

    return EquivalenceResult(
        logit_cosine_similarity=logit_cosine,
        hidden_state_cosine_similarity=avg_hidden_cosine,
        logit_max_diff=float(logit_diff.max()),
        logit_mean_diff=float(logit_diff.mean()),
        per_layer_similarities=hidden_cosines,
        passed=logit_cosine > 0.99,
    )
