"""MobileLLM model implementation.

Clean room implementation based on:
- MobileLLM-R1 paper (arXiv:2509.24945)
- LLaMA architecture patterns

Features:
- Input/output embedding weight sharing
- RMSNorm
- Pre-normalization transformer blocks
- Grouped Query Attention
- QK-norm for training stability
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from data_handler._legacy.models.attention import RMSNorm
from data_handler._legacy.models.config import MobileLLMConfig
from data_handler._legacy.models.transformer import TransformerDecoder


class MobileLLM(nn.Module):
    """MobileLLM language model.

    A decoder-only transformer model based on the MobileLLM-R1 architecture.
    Designed for efficient training and inference at various scales.

    Reference: https://arxiv.org/abs/2509.24945
    """

    def __init__(self, config: MobileLLMConfig):
        """Initialize MobileLLM model.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim)

        # Transformer decoder
        self.decoder = TransformerDecoder(config)

        # Final layer norm
        self.final_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)

        # Language model head (output projection to vocab)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight sharing between input and output embeddings
        if config.use_weight_sharing:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize model weights.

        Uses normal distribution with std=initializer_range for linear layers.
        Embeddings are initialized separately.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding):
        """Set input embedding layer."""
        self.embed_tokens = embeddings

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[Tensor, ...], dict]:
        """Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Optional attention mask (batch, seq_len)
            position_ids: Optional position indices (batch, seq_len)
            past_key_values: Optional cached KV from previous steps
            use_cache: Whether to return updated KV cache
            return_dict: Whether to return a dict or tuple

        Returns:
            If return_dict=True: dict with 'logits', 'past_key_values', 'hidden_states'
            If return_dict=False: tuple of (logits, past_key_values)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create position IDs if not provided
        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                # For incremental decoding, start from past sequence length
                past_len = past_key_values[0][0].size(1)
                position_ids = torch.arange(
                    past_len, past_len + seq_len, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(
                    seq_len, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)

        # Create causal attention mask if needed
        # For efficiency, skip mask creation if all positions are attended to
        # (common case in pretraining with packed sequences)
        causal_mask = None
        if attention_mask is not None:
            # Check if mask has any zeros (padding tokens)
            if not attention_mask.all():
                # Convert 2D attention mask to 4D causal mask
                causal_mask = self._prepare_attention_mask(
                    attention_mask, hidden_states.dtype, past_key_values
                )
            # If all ones, causal_mask stays None and attention uses is_causal=True

        # Pass through transformer decoder
        hidden_states, new_kv_caches = self.decoder(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Language model head
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {
                "logits": logits,
                "past_key_values": new_kv_caches,
                "hidden_states": hidden_states,
            }
        else:
            return logits, new_kv_caches

    def _prepare_attention_mask(
        self,
        attention_mask: Tensor,
        dtype: torch.dtype,
        past_key_values: Optional[list] = None,
    ) -> Tensor:
        """Prepare 4D causal attention mask from 2D mask.

        Args:
            attention_mask: 2D attention mask (batch, seq_len)
            dtype: Data type for mask
            past_key_values: Cached KV to determine full sequence length

        Returns:
            4D attention mask (batch, 1, seq_len, kv_seq_len)
        """
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device
        min_val = torch.finfo(dtype).min

        # Determine full key/value sequence length
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].size(1)
            kv_seq_len = past_len + seq_len
        else:
            past_len = 0
            kv_seq_len = seq_len

        # Create causal mask efficiently using tril
        # Shape: (1, 1, seq_len, kv_seq_len)
        causal_mask = torch.triu(
            torch.full((seq_len, kv_seq_len), min_val, dtype=dtype, device=device),
            diagonal=past_len + 1,
        ).unsqueeze(0).unsqueeze(0)

        # Expand to batch size
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, kv_seq_len)

        # Apply padding mask (mask padded positions with -inf)
        if past_key_values is not None and len(past_key_values) > 0:
            # Extend attention mask to include past positions
            extended_mask = torch.ones(
                (batch_size, kv_seq_len),
                dtype=attention_mask.dtype,
                device=device,
            )
            extended_mask[:, -seq_len:] = attention_mask
            attention_mask = extended_mask

        # Padding mask: shape (batch, 1, 1, kv_seq_len)
        padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)) * min_val

        # Combine causal and padding masks
        # Need to clone since expand doesn't allocate new memory
        combined_mask = causal_mask.clone()
        combined_mask = combined_mask + padding_mask

        return combined_mask

    def compute_loss(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        """Compute language modeling loss.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            labels: Target token IDs (batch, seq_len), -100 for ignored positions
            attention_mask: Optional attention mask
            **kwargs: Additional arguments passed to forward

        Returns:
            Tuple of (loss, metrics_dict)
        """
        outputs = self.forward(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        logits = outputs["logits"]

        # Shift logits and labels for next-token prediction
        # logits: (batch, seq-1, vocab), labels: (batch, seq-1)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten for cross-entropy
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Compute perplexity (for logging)
        with torch.no_grad():
            # Only count non-ignored tokens
            valid_tokens = (shift_labels != -100).sum()
            perplexity = torch.exp(loss) if valid_tokens > 0 else torch.tensor(float("inf"))

        metrics = {
            "loss": loss.item(),
            "perplexity": perplexity.item(),
            "num_tokens": valid_tokens.item(),
        }

        return loss, metrics

    @classmethod
    def from_config(cls, config: Union[MobileLLMConfig, dict, str]) -> "MobileLLM":
        """Create model from configuration.

        Args:
            config: MobileLLMConfig, dict of config params, or config name string

        Returns:
            Initialized MobileLLM model
        """
        from data_handler._legacy.models.config import get_config

        if isinstance(config, str):
            config = get_config(config)
        elif isinstance(config, dict):
            config = MobileLLMConfig(**config)

        return cls(config)

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Count number of parameters.

        Args:
            only_trainable: Only count trainable parameters

        Returns:
            Number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def gradient_checkpointing_enable(self, mode: str = "quantized"):
        """Enable gradient checkpointing for memory efficiency.

        Args:
            mode: Checkpointing mode - "standard" (PyTorch native) or "quantized" (INT8)
                  Quantized mode provides additional ~2x memory savings on top of standard.
        """
        if mode not in ("standard", "quantized"):
            raise ValueError(f"mode must be 'standard' or 'quantized', got '{mode}'")

        for layer in self.decoder.layers:
            layer.gradient_checkpointing = True
            layer.checkpointing_mode = mode

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        for layer in self.decoder.layers:
            layer.gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Check if gradient checkpointing is enabled."""
        if len(self.decoder.layers) > 0:
            return self.decoder.layers[0].gradient_checkpointing
        return False
