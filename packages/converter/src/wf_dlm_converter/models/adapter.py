"""Block Diffusion Adapter for BitNet models.

Adapts BitNet models for block diffusion inference by modifying:
1. Attention masks to use block-wise causal pattern
2. Adding noise embedding for diffusion timesteps
3. Implementing token shift mechanism

Based on Fast-dLLM v2 architecture modifications.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


class BlockDiffusionAdapter:
    """Adapts a BitNet model for block diffusion inference.

    The adapter modifies the model's attention mechanism to support
    block-wise generation, enabling parallel token prediction within
    each block while maintaining causal dependencies between blocks.

    Block-wise causal attention pattern:
        Standard causal:     Block causal (block_size=2):
        [1,0,0,0]           [1,1,0,0]  <- Block 1 sees all of block 1
        [1,1,0,0]           [1,1,0,0]
        [1,1,1,0]           [1,1,1,1]  <- Block 2 sees blocks 1 and 2
        [1,1,1,1]           [1,1,1,1]

    Example:
        >>> adapter = BlockDiffusionAdapter(block_size=32, num_diffusion_steps=8)
        >>> adapted_model = adapter.adapt_model(bitnet_model)
        >>> # Now model supports block diffusion inference
    """

    def __init__(
        self,
        block_size: int = 32,
        num_diffusion_steps: int = 8,
        preserve_bitlinear: bool = True,
        noise_schedule: str = "cosine",
    ):
        """Initialize the adapter.

        Args:
            block_size: Number of tokens per block for parallel generation
            num_diffusion_steps: Number of denoising steps per block
            preserve_bitlinear: Keep BitLinear layers unchanged (recommended)
            noise_schedule: Noise schedule type ('cosine' or 'linear')
        """
        self.block_size = block_size
        self.num_diffusion_steps = num_diffusion_steps
        self.preserve_bitlinear = preserve_bitlinear
        self.noise_schedule = noise_schedule

    def adapt_model(self, model: nn.Module) -> nn.Module:
        """Apply block diffusion modifications to BitNet model.

        This modifies the model in-place to support block diffusion:
        1. Registers block causal mask generation
        2. Adds noise embedding layer
        3. Wraps attention for block-wise processing

        Args:
            model: The BitNet model to adapt

        Returns:
            The adapted model (modified in-place)
        """
        logger.info(
            f"Adapting model for block diffusion: "
            f"block_size={self.block_size}, steps={self.num_diffusion_steps}"
        )

        hidden_size = model.config.hidden_size

        # Add noise embedding layer
        noise_embed = NoiseEmbedding(
            num_steps=self.num_diffusion_steps,
            hidden_size=hidden_size,
        )
        model.register_module("noise_embedding", noise_embed)

        # Add block diffusion config to model
        model.config.block_size = self.block_size
        model.config.num_diffusion_steps = self.num_diffusion_steps
        model.config.is_dlm = True

        # Store adapter reference
        model._dlm_adapter = self

        # Register forward hook for attention mask modification
        self._register_attention_hooks(model)

        logger.info("Model adapted for block diffusion")
        return model

    def _register_attention_hooks(self, model: nn.Module):
        """Register hooks to modify attention masks during forward pass."""
        # Find attention layers and register hooks
        for name, module in model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                # Store original forward
                original_forward = module.forward

                # Create wrapper that modifies attention mask
                def make_hook(orig_fwd, mod_name):
                    def hooked_forward(*args, **kwargs):
                        # Modify attention mask if provided
                        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                            kwargs["attention_mask"] = self._modify_attention_mask(
                                kwargs["attention_mask"]
                            )
                        return orig_fwd(*args, **kwargs)

                    return hooked_forward

                module.forward = make_hook(original_forward, name)

    def _modify_attention_mask(
        self, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Convert standard attention mask to block causal mask."""
        # If already block causal, return as-is
        if hasattr(attention_mask, "_is_block_causal"):
            return attention_mask

        batch_size, seq_len = attention_mask.shape[:2]

        # Create block causal mask
        block_mask = self.create_block_mask(
            seq_len=seq_len,
            block_size=self.block_size,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        # Combine with padding mask if needed
        if attention_mask.dim() == 2:
            # Expand padding mask: [B, L] -> [B, 1, L, L]
            padding_mask = attention_mask[:, None, None, :].expand(-1, -1, seq_len, -1)
            combined = block_mask.unsqueeze(0) * padding_mask
        else:
            combined = block_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)

        combined._is_block_causal = True
        return combined

    def create_block_mask(
        self,
        seq_len: int,
        block_size: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create block-wise causal attention mask.

        Tokens can attend to:
        - All tokens within their own block
        - All tokens in previous blocks

        Args:
            seq_len: Sequence length
            block_size: Size of each block
            device: Target device
            dtype: Data type

        Returns:
            Block causal mask of shape [1, seq_len, seq_len]
        """
        # Calculate block indices for each position
        block_ids = torch.arange(seq_len, device=device) // block_size

        # Create mask: position i can attend to j if block[j] <= block[i]
        # This gives block-wise causal pattern
        row_blocks = block_ids.unsqueeze(1)  # [L, 1]
        col_blocks = block_ids.unsqueeze(0)  # [1, L]

        mask = (col_blocks <= row_blocks).to(dtype)

        return mask.unsqueeze(0)  # [1, L, L]

    def create_complementary_mask(
        self,
        seq_len: int,
        block_size: int,
        step: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Create complementary training mask for alternating token supervision.

        During training, we alternate which tokens are masked to ensure
        all positions receive supervision across steps.

        Args:
            seq_len: Sequence length
            block_size: Block size
            step: Current training step (determines mask pattern)
            device: Target device

        Returns:
            Boolean mask indicating which tokens to predict
        """
        # Create alternating pattern based on step
        positions = torch.arange(seq_len, device=device)

        # Different patterns for different steps
        pattern = step % self.num_diffusion_steps

        # Mask tokens where (position % num_steps) == pattern
        mask = (positions % self.num_diffusion_steps) == pattern

        return mask

    def get_noise_schedule(
        self, num_steps: int, device: torch.device = None
    ) -> torch.Tensor:
        """Get noise schedule for diffusion steps.

        Args:
            num_steps: Number of diffusion steps
            device: Target device

        Returns:
            Tensor of noise levels for each step
        """
        if self.noise_schedule == "cosine":
            # Cosine schedule (smoother)
            steps = torch.linspace(0, 1, num_steps + 1, device=device)
            alphas = torch.cos(steps * math.pi / 2) ** 2
            betas = 1 - alphas[1:] / alphas[:-1]
            return torch.clamp(betas, max=0.999)
        else:
            # Linear schedule
            return torch.linspace(0.0001, 0.02, num_steps, device=device)


class NoiseEmbedding(nn.Module):
    """Embedding layer for diffusion timestep/noise level.

    Maps discrete timestep indices to hidden dimension embeddings
    that are added to token representations during diffusion.
    """

    def __init__(
        self,
        num_steps: int,
        hidden_size: int,
        max_period: int = 10000,
    ):
        """Initialize noise embedding.

        Args:
            num_steps: Number of diffusion steps
            hidden_size: Hidden dimension of the model
            max_period: Maximum period for sinusoidal embedding
        """
        super().__init__()
        self.num_steps = num_steps
        self.hidden_size = hidden_size

        # Learnable embedding for each step
        self.step_embedding = nn.Embedding(num_steps, hidden_size)

        # MLP to project embedding
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        nn.init.normal_(self.step_embedding.weight, std=0.02)
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get noise embeddings for given timesteps.

        Args:
            timesteps: Tensor of timestep indices [B] or [B, L]

        Returns:
            Noise embeddings [B, H] or [B, L, H]
        """
        embed = self.step_embedding(timesteps)
        return self.proj(embed)


class TokenShift(nn.Module):
    """Token shift mechanism for block diffusion.

    Implements the token shift from Fast-dLLM that allows
    predicting masked tokens from the preceding token's logits.
    This maintains some autoregressive properties within blocks.
    """

    def __init__(self, shift_amount: int = 1):
        """Initialize token shift.

        Args:
            shift_amount: Number of positions to shift (usually 1)
        """
        super().__init__()
        self.shift_amount = shift_amount

    def forward(
        self,
        logits: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply token shift to logits.

        Args:
            logits: Model logits [B, L, V]
            target_positions: Optional mask of positions to shift for

        Returns:
            Shifted logits where position i uses logits from position i-1
        """
        # Shift logits by prepending zeros and removing last
        shifted = F.pad(logits[:, :-1], (0, 0, 1, 0))

        if target_positions is not None:
            # Only use shifted logits at target positions
            result = torch.where(
                target_positions.unsqueeze(-1),
                shifted,
                logits,
            )
            return result

        return shifted
