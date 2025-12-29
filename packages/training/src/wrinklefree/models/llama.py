"""LLaMA-style BitNet model implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrinklefree.models.config import BitNetConfig
from wrinklefree.models.transformer import BitNetTransformer


class BitNetLlama(nn.Module):
    """
    LLaMA-style language model with BitNet quantization.

    This is the main model class that includes:
    - Token embeddings (full precision)
    - Positional embeddings via RoPE (in attention)
    - Stack of BitNet transformer layers
    - Language modeling head (optionally tied to embeddings)

    Args:
        config: BitNetConfig with model hyperparameters
    """

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config

        # Token embeddings (kept in full precision)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.transformer = BitNetTransformer(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            attention_dropout=config.attention_dropout,
            hidden_act=config.hidden_act,
            use_flash_attention=config.use_flash_attention,
        )

        # LM head (can be tied to embeddings)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for the model."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set input embedding layer."""
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """Get output embedding layer (LM head)."""
        if self.config.tie_word_embeddings:
            return self.embed_tokens
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            position_ids: Position IDs of shape (batch, seq_len)
            labels: Labels for LM loss of shape (batch, seq_len)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states

        Returns:
            Dictionary with:
                - logits: Language modeling logits
                - loss: LM loss (if labels provided)
                - hidden_states: Hidden states per layer (if requested)
                - attentions: Attention weights per layer (if requested)
        """
        batch_size, seq_len = input_ids.shape

        # Create causal attention mask
        if attention_mask is not None:
            # Convert 1D mask to 4D causal mask
            causal_mask = self._make_causal_mask(seq_len, input_ids.device)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Use model's dtype for attention mask (matches training precision)
            mask_dtype = self.embed_tokens.weight.dtype
            attention_mask = (1.0 - attention_mask.to(mask_dtype)) * torch.finfo(mask_dtype).min
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = self._make_causal_mask(seq_len, input_ids.device)

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Transformer layers
        outputs = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs["last_hidden_state"]

        # LM head
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }

    def _make_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            do_sample: Whether to sample (False = greedy)
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs of shape (batch, seq_len + num_generated)
        """
        eos_token_id = eos_token_id or self.config.eos_token_id

        for _ in range(max_new_tokens):
            # Get logits for last position
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :]  # (batch, vocab)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids


class BitNetLlamaForSequenceClassification(nn.Module):
    """
    BitNet LLaMA model for sequence classification.

    Useful for fine-tuning on classification tasks during Stage 3 distillation.

    Args:
        config: BitNetConfig with model hyperparameters
        num_labels: Number of classification labels
    """

    def __init__(self, config: BitNetConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        # Base model (without LM head)
        self.model = BitNetLlama(config)
        self.model.lm_head = None  # Remove LM head

        # Classification head
        self.score = nn.Linear(config.hidden_size, num_labels, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list]:
        """
        Forward pass for classification.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Classification labels
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states

        Returns:
            Dictionary with logits, loss, etc.
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs["hidden_states"][-1] if outputs["hidden_states"] else None
        if hidden_states is None:
            # Re-run to get hidden states
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs["hidden_states"][-1]

        # Pool: use last token (like GPT-style)
        if attention_mask is not None:
            # Find last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        else:
            pooled = hidden_states[:, -1]

        logits = self.score(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }
