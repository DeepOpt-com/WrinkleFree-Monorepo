"""MuonClip Optimizer - Muon with separate AdamW for non-2D params.

Based on:
- Muon: Newton-Schulz orthogonalization for faster convergence
- Local implementation that doesn't require distributed training

References:
- https://github.com/KellerJordan/Muon
- https://arxiv.org/abs/2502.16982 (Moonshot Muon paper)
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

    This implementation does NOT require distributed training.

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
