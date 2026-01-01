"""Gradient extraction utilities for influence calculation.

Implements discriminative layer selection from AutoMixer paper:
only uses embedding and output layers for efficiency.

Reference: AutoMixer (ACL 2025) - Discriminative Layer Selection
"""

from typing import Dict, Optional, List

import torch
import torch.nn as nn
from torch import Tensor

from data_handler.influence.config import InfluenceConfig, InfluenceTarget


class DiscriminativeGradientExtractor:
    """Extracts gradients from embedding and output layers only.

    Following AutoMixer's discriminative layer selection strategy,
    only uses embed_tokens and lm_head layers for influence calculation.
    This reduces computation from O(d_model * n_layers) to O(d_vocab * d_embed).

    Reference: AutoMixer paper - "Discriminative Layer Selection"
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InfluenceConfig] = None,
    ):
        """Initialize gradient extractor.

        Args:
            model: MobileLLM model instance with embed_tokens and lm_head attributes
            config: Influence configuration (uses defaults if None)
        """
        self.model = model
        self.config = config or InfluenceConfig()

        # Find embed_tokens and lm_head (handle multiple architectures)
        # LLaMA/Mistral: model.model.embed_tokens, model.lm_head
        # Direct models: model.embed_tokens, model.lm_head
        # GPT-2/GPT-Neo: model.transformer.wte, model.lm_head
        if hasattr(model, "embed_tokens"):
            self.embed_tokens = model.embed_tokens
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            self.embed_tokens = model.model.embed_tokens
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            self.embed_tokens = model.transformer.wte
        elif hasattr(model, "wte"):
            self.embed_tokens = model.wte
        else:
            raise ValueError(
                "Model must have 'embed_tokens' or 'wte' attribute "
                "(supported: LLaMA, GPT-2, GPT-Neo)"
            )

        # Find lm_head (output projection)
        if hasattr(model, "lm_head"):
            self.lm_head = model.lm_head
        elif hasattr(model, "cls") and hasattr(model.cls, "predictions"):
            # BERT-style models
            self.lm_head = model.cls.predictions.decoder
        else:
            raise ValueError("Model must have 'lm_head' attribute")

        # Check for weight sharing
        self.weight_sharing = self._check_weight_sharing()

        # Cache for gradient dimension
        self._grad_dim: Optional[int] = None

    def _check_weight_sharing(self) -> bool:
        """Check if embedding and output layers share weights."""
        try:
            return self.lm_head.weight.data_ptr() == self.embed_tokens.weight.data_ptr()
        except AttributeError:
            return False

    def get_target_parameters(self) -> Dict[str, nn.Parameter]:
        """Get parameters to compute gradients for.

        Returns:
            Dictionary mapping parameter names to Parameter objects
        """
        params = {}
        target = self.config.target_layers

        if target in [InfluenceTarget.EMBEDDING_ONLY, InfluenceTarget.EMBEDDING_AND_OUTPUT]:
            params["embed_tokens"] = self.embed_tokens.weight

        if target in [InfluenceTarget.OUTPUT_ONLY, InfluenceTarget.EMBEDDING_AND_OUTPUT]:
            # Handle weight sharing - only add lm_head if weights are different
            if not self.weight_sharing:
                params["lm_head"] = self.lm_head.weight

        return params

    def get_gradient_dimension(self) -> int:
        """Get the total dimension of flattened gradients.

        Returns:
            Total number of parameters in target layers
        """
        if self._grad_dim is not None:
            return self._grad_dim

        total = 0
        for param in self.get_target_parameters().values():
            total += param.numel()

        self._grad_dim = total
        return total

    def compute_per_sample_gradient(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute gradient for a single sample.

        Args:
            input_ids: Input token IDs [seq_len] or [1, seq_len]
            labels: Target token IDs (defaults to input_ids for LM)
            attention_mask: Optional attention mask

        Returns:
            Dictionary mapping parameter names to gradients
        """
        # Ensure model is in eval mode for consistent gradients
        was_training = self.model.training
        self.model.eval()

        # Clear any existing gradients
        self.model.zero_grad()

        # Handle input dimensions
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if labels is None:
            labels = input_ids.clone()
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        # Enable gradients for target parameters
        target_params = self.get_target_parameters()
        original_requires_grad = {}
        for name, param in target_params.items():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad_(True)

        try:
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs["logits"]

            # Compute loss (same as PretrainStage)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Backward to compute gradients
            loss.backward()

            # Extract target gradients
            gradients = {}
            for name, param in target_params.items():
                if param.grad is not None:
                    grad = param.grad.clone().detach()
                    # Optional gradient clipping
                    if self.config.max_grad_norm > 0:
                        grad_norm = grad.norm()
                        if grad_norm > self.config.max_grad_norm:
                            grad = grad * (self.config.max_grad_norm / grad_norm)
                    gradients[name] = grad
                else:
                    # Return zeros if no gradient (shouldn't happen normally)
                    gradients[name] = torch.zeros_like(param)

        finally:
            # Restore original requires_grad state
            for name, param in target_params.items():
                param.requires_grad_(original_requires_grad[name])

            # Clear gradients
            self.model.zero_grad()

            # Restore training mode
            if was_training:
                self.model.train()

        return gradients

    def flatten_gradients(self, gradients: Dict[str, Tensor]) -> Tensor:
        """Flatten gradient dictionary into a single vector.

        Args:
            gradients: Dictionary mapping names to gradient tensors

        Returns:
            Flattened gradient tensor [total_params]
        """
        flat_grads = []
        for name in sorted(gradients.keys()):  # Sort for consistency
            flat_grads.append(gradients[name].flatten())
        return torch.cat(flat_grads)

    def compute_batch_gradients(
        self,
        batch: Dict[str, Tensor],
        return_individual: bool = False,
    ) -> Tensor:
        """Compute flattened gradient vectors for a batch.

        This is less efficient than per-sample because we need
        to compute gradients one sample at a time.

        Args:
            batch: Batch dictionary with input_ids, optional attention_mask and labels
            return_individual: If True, returns dict with separate gradients

        Returns:
            Flattened gradient tensor [batch_size, total_params]
        """
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        attention_mask = batch.get("attention_mask")

        batch_size = input_ids.size(0)
        batch_grads = []

        for i in range(batch_size):
            mask = attention_mask[i] if attention_mask is not None else None
            grads = self.compute_per_sample_gradient(
                input_ids[i],
                labels[i],
                mask,
            )
            flat_grad = self.flatten_gradients(grads)
            batch_grads.append(flat_grad)

        return torch.stack(batch_grads)

    def compute_aggregated_gradient(
        self,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        """Compute the mean gradient for a batch (more efficient).

        Uses standard batch backward which computes mean gradient.

        Args:
            batch: Batch dictionary

        Returns:
            Mean gradient tensor [total_params]
        """
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad()

        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        attention_mask = batch.get("attention_mask")

        # Move to model device
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        target_params = self.get_target_parameters()
        for param in target_params.values():
            param.requires_grad_(True)

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs["logits"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            loss.backward()

            gradients = {}
            for name, param in target_params.items():
                if param.grad is not None:
                    gradients[name] = param.grad.clone().detach()
                else:
                    gradients[name] = torch.zeros_like(param)

        finally:
            self.model.zero_grad()
            if was_training:
                self.model.train()

        return self.flatten_gradients(gradients)

