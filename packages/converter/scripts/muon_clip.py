#!/usr/bin/env python
"""MuonClip Optimizer - Muon with QK-Clipping for stable large-scale training.

Based on:
- Muon: Newton-Schulz orthogonalization for faster convergence
- QK-Clip: Attention score clipping from Kimi K2 (arXiv:2507.20534)

References:
- https://github.com/KellerJordan/Muon
- https://arxiv.org/abs/2502.16982 (Moonshot Muon paper)
- https://fireworks.ai/blog/muonclip
"""

import math
import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Newton-Schulz iteration to compute orthogonalization of G.

    Uses a quintic iteration with coefficients selected to maximize slope at zero.
    Produces something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5).

    From Keller Jordan's Muon implementation.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonClip(torch.optim.Optimizer):
    """Muon optimizer with optional per-group orthogonalization.

    Combines:
    - Muon (Newton-Schulz orthogonalization) for 2D+ params
    - Standard AdamW-style updates for other params (when use_muon=False)

    QK-Clipping is handled separately via apply_qk_clip() function.

    Args:
        params: Iterable of parameters or param groups
        lr: Learning rate (default: 1e-3 for Muon, use 5e-5 for AdamW group)
        betas: Adam-style betas (default: (0.9, 0.95))
        eps: Numerical stability epsilon
        weight_decay: Weight decay coefficient
        ns_steps: Newton-Schulz iteration steps (default: 5)
        use_muon: Whether to apply Muon orthogonalization (can be per-group)

    Example:
        optimizer = MuonClip([
            {"params": muon_params, "lr": 1e-3, "use_muon": True},
            {"params": adam_params, "lr": 5e-5, "use_muon": False},
        ])
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        use_muon: bool = True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", True)
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Update momentum and squared gradient (Adam style)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute update direction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                update = exp_avg / denom

                # Apply Muon orthogonalization if enabled and param is 2D+
                if use_muon and update.ndim >= 2:
                    update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])

                # Apply update
                p.add_(update, alpha=-step_size)

                # Apply decoupled weight decay
                if group["weight_decay"] != 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

        return loss


def apply_qk_clip(
    model: torch.nn.Module,
    threshold: float = 100.0,
    alpha: float = 0.5,
    sample_input_ids: torch.Tensor = None,
) -> dict:
    """Apply QK-Clipping to attention layers after optimizer step.

    From Kimi K2: If max attention score > threshold, rescale W_q and W_k.

    Args:
        model: The transformer model
        threshold: Max allowed attention score (τ = 100 in Kimi K2)
        alpha: Balance factor for rescaling (0.5 = equal split between Q and K)
        sample_input_ids: Optional input for computing attention scores

    Returns:
        Dict with clipping stats (max_score, was_clipped, scale_factor)

    Note: This is a simplified version. Full implementation would need:
    - Access to attention scores during forward pass
    - Hook-based score collection
    For now, we just apply spectral norm constraints to Q/K weights.
    """
    stats = {"max_score": 0.0, "was_clipped": False, "scale_factor": 1.0}

    # Find Q and K projection layers
    for name, module in model.named_modules():
        # Common naming patterns for attention projections
        if any(k in name.lower() for k in ["q_proj", "k_proj", "query", "key"]):
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                if w.ndim >= 2:
                    # Compute spectral norm as proxy for potential max attention score
                    # This is a conservative approximation
                    with torch.no_grad():
                        # Use power iteration for spectral norm estimate
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
                                module.weight.data.mul_(scale ** alpha)
                            else:  # k
                                module.weight.data.mul_(scale ** (1 - alpha))
                            stats["was_clipped"] = True
                            stats["scale_factor"] = min(stats["scale_factor"], scale)

                        stats["max_score"] = max(stats["max_score"], spectral_norm)

    return stats


def get_muon_param_groups(
    model: torch.nn.Module,
    lr_muon: float = 1e-3,
    lr_adam: float = 5e-5,
    weight_decay: float = 0.01,
) -> list:
    """Split model parameters into Muon and AdamW groups.

    Muon group: 2D+ params (Linear weights) excluding embeddings
    Adam group: Embeddings, LayerNorms, biases, 1D params

    Args:
        model: The model to partition
        lr_muon: Learning rate for Muon group
        lr_adam: Learning rate for Adam group
        weight_decay: Weight decay for both groups

    Returns:
        List of param group dicts for MuonClip optimizer
    """
    muon_params = []
    adam_params = []

    muon_names = []
    adam_names = []

    # Build a map of parameter id -> module type for accurate classification
    param_to_module = {}
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param_to_module[id(param)] = (type(module).__name__, full_name)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Get module type if available
        module_type, _ = param_to_module.get(id(param), ("Unknown", name))

        # Check if this is an embedding layer (by name OR module type)
        is_embedding = (
            any(k in name.lower() for k in ["embed", "token", "wte", "wpe", "emb", "lm_head"])
            or module_type == "Embedding"
        )

        # Check if this is a normalization layer (by name OR module type)
        is_norm = (
            any(k in name.lower() for k in ["norm", "ln_", "layernorm", "rmsnorm"])
            or module_type in ["LayerNorm", "RMSNorm", "BatchNorm1d", "BatchNorm2d"]
        )

        # Check if this is a bias
        is_bias = name.endswith(".bias")

        # Muon: 2D+ params that are not embeddings/norms/biases
        if param.ndim >= 2 and not is_embedding and not is_norm:
            muon_params.append(param)
            muon_names.append(name)
        else:
            adam_params.append(param)
            adam_names.append(name)

    print(f"MuonClip param split:")
    print(f"  Muon ({len(muon_params)} params, lr={lr_muon}): {muon_names[:3]}...")
    print(f"  Adam ({len(adam_params)} params, lr={lr_adam}): {adam_names[:3]}...")

    return [
        {
            "params": muon_params,
            "lr": lr_muon,
            "use_muon": True,
            "weight_decay": weight_decay,
        },
        {
            "params": adam_params,
            "lr": lr_adam,
            "use_muon": False,
            "weight_decay": weight_decay,
        },
    ]
