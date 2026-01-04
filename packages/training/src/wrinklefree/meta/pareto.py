"""Pareto gradient solver for multi-objective optimization.

Implements MGDA and EPO algorithms following LibMOON methodology.

References:
- LibMOON (NeurIPS 2024): https://arxiv.org/abs/2409.02969
  GitHub: https://github.com/xzhang2523/libmoon
- MGDA: Multiple Gradient Descent Algorithm
- EPO: Exact Pareto Optimal
"""

import logging
from typing import Literal, Optional

import torch
from torch import Tensor

from wrinklefree.meta.config import ParetoConfig

logger = logging.getLogger(__name__)


class ParetoGradientSolver:
    """Gradient-based multi-objective optimization solver.

    Finds Pareto-optimal update directions that improve all objectives
    or achieve specified preference trade-offs.

    Implements two algorithms from LibMOON:
    1. MGDA (Multiple Gradient Descent Algorithm): Finds the minimum-norm
       point in the convex hull of objective gradients.
    2. EPO (Exact Pareto Optimal): Finds the Pareto-optimal point matching
       specified preferences.

    Reference: LibMOON (https://github.com/xzhang2523/libmoon)
    """

    def __init__(self, config: Optional[ParetoConfig] = None):
        """Initialize solver.

        Args:
            config: Pareto configuration (uses defaults if None)
        """
        self.config = config or ParetoConfig()

    def solve(
        self,
        gradients: list[Tensor],
        preferences: Optional[list[float]] = None,
    ) -> Tensor:
        """Solve for Pareto-optimal gradient direction.

        Dispatches to appropriate solver based on config.

        Args:
            gradients: List of gradient tensors, one per objective
            preferences: Optional preference weights (required for EPO)

        Returns:
            Pareto-optimal update direction
        """
        if len(gradients) == 0:
            raise ValueError("At least one gradient required")

        if len(gradients) == 1:
            # Single objective: just return the gradient
            return gradients[0]

        if self.config.normalize_gradients:
            gradients = self._normalize_gradients(gradients)

        if self.config.method == "mgda":
            return self.solve_mgda(gradients)
        elif self.config.method == "epo":
            prefs = preferences or self.config.preferences
            if prefs is None:
                # Default to uniform preferences
                prefs = [1.0 / len(gradients)] * len(gradients)
            return self.solve_epo(gradients, prefs)
        elif self.config.method == "linear":
            prefs = preferences or self.config.preferences
            if prefs is None:
                prefs = [1.0 / len(gradients)] * len(gradients)
            return self.solve_linear(gradients, prefs)
        else:
            raise ValueError(f"Unknown Pareto method: {self.config.method}")

    def _normalize_gradients(self, gradients: list[Tensor]) -> list[Tensor]:
        """Normalize gradients to unit norm for balanced optimization."""
        normalized = []
        for g in gradients:
            norm = g.norm()
            if norm > 1e-8:
                normalized.append(g / norm)
            else:
                normalized.append(g)
        return normalized

    def solve_mgda(self, gradients: list[Tensor]) -> Tensor:
        """Solve MGDA for Pareto-optimal gradient.

        Finds weights alpha = [alpha_1, ..., alpha_k] such that:
            min_{alpha} || sum_i alpha_i * g_i ||^2
            s.t. sum_i alpha_i = 1, alpha_i >= 0

        This gives the steepest descent direction that improves all objectives.
        If the minimum norm is 0, we're at a Pareto-optimal point.

        Algorithm: Frank-Wolfe on the dual problem.

        Reference: Désidéri, J.A. (2012) "Multiple-gradient descent algorithm"

        Args:
            gradients: List of gradient vectors for each objective

        Returns:
            Pareto-optimal update direction
        """
        k = len(gradients)
        device = gradients[0].device
        dtype = gradients[0].dtype

        # Stack gradients: G = [g_1 | g_2 | ... | g_k], shape [d, k]
        G = torch.stack([g.flatten() for g in gradients], dim=1)

        # Gram matrix: M_ij = g_i^T g_j
        M = G.T @ G  # [k, k]

        # Initialize with uniform weights
        alpha = torch.ones(k, device=device, dtype=dtype) / k

        for iteration in range(self.config.max_iter):
            # Gradient of objective: nabla_alpha ||G @ alpha||^2 = 2 * M @ alpha
            grad_alpha = 2 * M @ alpha

            # Find direction of steepest descent (argmin over simplex vertices)
            # The vertices of the simplex are the standard basis vectors
            min_idx = grad_alpha.argmin()

            # Direction: from current alpha toward vertex e_{min_idx}
            direction = torch.zeros_like(alpha)
            direction[min_idx] = 1.0

            # Check convergence: if we're already at the minimum vertex
            gap = (alpha - direction).abs().max()
            if gap < self.config.tol:
                break

            # Line search: find optimal step size
            # f(alpha + t*(d - alpha)) where d is the target vertex
            # This is a 1D quadratic, so we can solve analytically
            step = self._line_search_mgda(alpha, direction, M)

            # Update alpha
            alpha = alpha + step * (direction - alpha)

        # Ensure valid simplex point
        alpha = alpha.clamp(min=0)
        alpha = alpha / alpha.sum()

        # Return combined gradient
        result = G @ alpha

        # Log alpha for debugging
        logger.debug(f"MGDA converged in {iteration + 1} iterations, alpha={alpha.tolist()}")

        return result.reshape(gradients[0].shape)

    def _line_search_mgda(
        self,
        alpha: Tensor,
        direction: Tensor,
        M: Tensor,
    ) -> float:
        """Exact line search for MGDA.

        Minimizes f(t) = ||G @ (alpha + t * (direction - alpha))||^2
        over t in [0, 1].

        This is a quadratic in t, so we find the minimum analytically.
        """
        d = direction - alpha
        Md = M @ d
        Ma = M @ alpha

        # f(t) = (alpha + t*d)^T M (alpha + t*d)
        #      = alpha^T M alpha + 2*t * d^T M alpha + t^2 * d^T M d
        # f'(t) = 2 * d^T M alpha + 2*t * d^T M d = 0
        # t* = -d^T M alpha / (d^T M d)

        dMd = d @ Md
        dMa = d @ Ma

        if dMd.abs() < 1e-10:
            # Degenerate case: direction is in null space
            return 0.0

        t_opt = -dMa / dMd
        # Clamp to valid range [0, 1]
        t_opt = max(0.0, min(1.0, t_opt.item()))

        return t_opt

    def solve_epo(
        self,
        gradients: list[Tensor],
        preferences: list[float],
    ) -> Tensor:
        """Solve EPO for preference-weighted Pareto-optimal gradient.

        EPO (Exact Pareto Optimal) finds the point on the Pareto front
        that matches the specified preference vector.

        This implementation uses a simplified approach: we first solve MGDA
        to find the Pareto-optimal direction, then adjust based on preferences.

        Reference: Mahapatra & Rajan (2020) "Multi-Task Learning with EPO"

        Args:
            gradients: List of gradient vectors for each objective
            preferences: Preference weights for each objective (sum to 1)

        Returns:
            Preference-weighted Pareto-optimal update direction
        """
        k = len(gradients)
        device = gradients[0].device
        dtype = gradients[0].dtype

        # Normalize preferences
        prefs = torch.tensor(preferences, device=device, dtype=dtype)
        prefs = prefs / prefs.sum()

        # Stack gradients
        G = torch.stack([g.flatten() for g in gradients], dim=1)

        # Gram matrix
        M = G.T @ G

        # EPO finds alpha such that the improvement ratios match preferences
        # We use a simplified approach: weighted combination biased by preferences
        # then project to ensure Pareto improvement

        # Start with preference-weighted combination
        alpha = prefs.clone()

        # Iterate to adjust for Pareto improvement
        for iteration in range(self.config.max_iter):
            # Combined gradient
            combined = G @ alpha

            # Check improvement on each objective
            # Improvement_i = -g_i^T @ combined (negative because we descend)
            improvements = -(G.T @ combined)

            # If any improvement is negative (i.e., we'd hurt that objective),
            # adjust alpha to fix it
            if (improvements < 0).any():
                # Reduce weight on objectives we're hurting
                # Increase weight on objectives we're helping too much
                adjustment = torch.zeros_like(alpha)
                for i in range(k):
                    if improvements[i] < 0:
                        # We're hurting objective i, increase its weight
                        adjustment[i] = -improvements[i].abs()

                # Apply adjustment
                alpha = alpha + 0.1 * adjustment
                alpha = alpha.clamp(min=0)
                alpha = alpha / alpha.sum()
            else:
                # All objectives improve, we're done
                break

        # Return combined gradient
        result = G @ alpha

        logger.debug(f"EPO converged in {iteration + 1} iterations, alpha={alpha.tolist()}")

        return result.reshape(gradients[0].shape)

    def solve_linear(
        self,
        gradients: list[Tensor],
        weights: list[float],
    ) -> Tensor:
        """Simple linear scalarization (weighted sum).

        This is the simplest approach but doesn't guarantee Pareto optimality.
        Useful as a baseline or when speed is critical.

        Args:
            gradients: List of gradient vectors for each objective
            weights: Weights for each objective

        Returns:
            Weighted sum of gradients
        """
        device = gradients[0].device
        dtype = gradients[0].dtype

        w = torch.tensor(weights, device=device, dtype=dtype)
        w = w / w.sum()

        G = torch.stack([g.flatten() for g in gradients], dim=1)
        result = G @ w

        return result.reshape(gradients[0].shape)

    def compute_pareto_weights(
        self,
        gradients: list[Tensor],
    ) -> Tensor:
        """Compute just the Pareto weights (alpha) without returning the gradient.

        Useful for logging or analysis.

        Args:
            gradients: List of gradient vectors for each objective

        Returns:
            Tensor of weights alpha for each objective
        """
        if len(gradients) == 0:
            return torch.tensor([])

        if len(gradients) == 1:
            return torch.tensor([1.0])

        k = len(gradients)
        device = gradients[0].device
        dtype = gradients[0].dtype

        if self.config.normalize_gradients:
            gradients = self._normalize_gradients(gradients)

        G = torch.stack([g.flatten() for g in gradients], dim=1)
        M = G.T @ G
        alpha = torch.ones(k, device=device, dtype=dtype) / k

        for iteration in range(self.config.max_iter):
            grad_alpha = 2 * M @ alpha
            min_idx = grad_alpha.argmin()
            direction = torch.zeros_like(alpha)
            direction[min_idx] = 1.0

            gap = (alpha - direction).abs().max()
            if gap < self.config.tol:
                break

            step = self._line_search_mgda(alpha, direction, M)
            alpha = alpha + step * (direction - alpha)

        alpha = alpha.clamp(min=0)
        alpha = alpha / alpha.sum()

        return alpha
