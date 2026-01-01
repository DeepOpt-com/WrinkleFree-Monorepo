"""Fast-dLLM v2 inference with DualCache for sub-block re-use.

Implements the DualCache mechanism from Fast-dLLM v2 (arXiv:2509.26328) which:
- Maintains both prefix and suffix KV caches for partially decoded blocks
- Enables efficient recomputation as additional tokens are revealed
- Supports iterative, selective decoding with confidence-aware refinement

Example:
    >>> from wf_dlm_converter.inference import generate_with_dualcache
    >>> output = generate_with_dualcache(model, tokenizer, "Hello, world!")
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Fast-dLLM v2 constants
FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645
DEFAULT_MASK_TOKEN = "|<MASK>|"


@dataclass
class GenerationResult:
    """Result from generate_with_dualcache."""

    text: str
    tokens_generated: int
    elapsed_seconds: float
    tokens_per_second: float
    nfe: int  # Number of Forward Evaluations
    used_dualcache: bool


def generate_with_dualcache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    block_size: int = 32,
    small_block_size: int = 8,
    use_block_cache: bool = True,
    threshold: float = 0.95,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> GenerationResult:
    """Generate text using Fast-dLLM v2 with DualCache for sub-block re-use.

    DualCache maintains both prefix and suffix KV caches for partially decoded
    blocks, enabling efficient recomputation as additional tokens are revealed.
    This hierarchical caching supports iterative, selective decoding used in
    confidence-aware refinement.

    Args:
        model: HuggingFace model with Fast-dLLM support
        tokenizer: Tokenizer with mask token added
        prompt: Input prompt
        block_size: Tokens per block (bd_size, default: 32)
        small_block_size: Sub-block size for iterative denoising (default: 8)
        use_block_cache: Enable DualCache for sub-block re-use (default: True)
        threshold: Confidence threshold for unmasking (default: 0.95)
        max_new_tokens: Maximum tokens to generate (default: 128)
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Top-p sampling parameter

    Returns:
        GenerationResult with generated text and statistics
    """
    device = next(model.parameters()).device
    model.eval()

    # Get mask token ID from tokenizer or config
    mask_id = _get_mask_id(tokenizer, model)
    stop_token = tokenizer.eos_token_id or FAST_DLLM_STOP_TOKEN

    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = torch.tensor([input_ids.shape[1]], device=device)

    start_time = time.time()
    nfe = 0

    # Check if model has batch_sample method (Fast-dLLM)
    if hasattr(model, "batch_sample"):
        # Use Fast-dLLM's native batch_sample with DualCache
        outputs = model.batch_sample(
            input_ids=input_ids,
            tokenizer=tokenizer,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
            small_block_size=small_block_size,
            min_len=input_ids.shape[1],
            seq_len=seq_len,
            mask_id=mask_id,
            threshold=threshold,
            stop_token=stop_token,
            use_block_cache=use_block_cache,
            top_p=top_p,
            temperature=temperature,
        )
        # batch_sample returns dict {sample_idx: tensor}
        output_ids = outputs[0] if isinstance(outputs, dict) else outputs
        nfe = -1  # NFE tracked internally
    else:
        # Fallback: manual block-wise generation with DualCache
        output_ids, nfe = _generate_block_diffusion(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            block_size=block_size,
            small_block_size=small_block_size,
            max_new_tokens=max_new_tokens,
            mask_id=mask_id,
            stop_token=stop_token,
            threshold=threshold,
            use_block_cache=use_block_cache,
            temperature=temperature,
            top_p=top_p,
        )

    elapsed = time.time() - start_time

    # Decode output
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    prompt_len = input_ids.shape[1]
    tokens_generated = len(output_ids) - prompt_len if len(output_ids) > prompt_len else 0

    return GenerationResult(
        text=generated_text,
        tokens_generated=tokens_generated,
        elapsed_seconds=elapsed,
        tokens_per_second=tokens_generated / elapsed if elapsed > 0 else 0,
        nfe=nfe,
        used_dualcache=use_block_cache,
    )


def _get_mask_id(tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> int:
    """Get mask token ID from tokenizer or model config."""
    # Try tokenizer first
    if DEFAULT_MASK_TOKEN in tokenizer.get_vocab():
        return tokenizer.encode(DEFAULT_MASK_TOKEN, add_special_tokens=False)[0]

    # Check model config
    if hasattr(model.config, "mask_token_id"):
        return model.config.mask_token_id

    # Check for bd_size (indicates DLM model)
    if hasattr(model.config, "bd_size"):
        logger.warning(f"Model has bd_size but no mask token found, using default {FAST_DLLM_MASK_ID}")
        return FAST_DLLM_MASK_ID

    raise ValueError(
        f"Model does not have mask token. Add '{DEFAULT_MASK_TOKEN}' to tokenizer "
        "or set model.config.mask_token_id"
    )


@torch.no_grad()
def _generate_block_diffusion(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    mask_id: int,
    stop_token: int,
    threshold: float,
    use_block_cache: bool,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, int]:
    """Generate using block diffusion with optional DualCache.

    This implements the core Fast-dLLM v2 generation loop with:
    - Block-wise parallel decoding
    - Sub-block iteration with confidence thresholding
    - DualCache for KV cache reuse within blocks

    Returns:
        (output_ids, nfe): Generated token IDs and number of forward evaluations
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    num_blocks = max_new_tokens // block_size

    nfe = 0
    past_key_values = None
    num_small_blocks = block_size // small_block_size

    # Initial forward pass for prefix
    prompt_len = input_ids.shape[1]
    if prompt_len > block_size:
        aligned_len = (prompt_len // block_size) * block_size
        output = model(
            input_ids=input_ids[:, :aligned_len],
            use_cache=True,
        )
        past_key_values = output.past_key_values
        nfe += 1

    x_t = input_ids.clone()

    for block_idx in range(num_blocks):
        # Check for stop token
        if stop_token in x_t[:, prompt_len:]:
            break

        # Pad to block boundary with mask tokens
        current_len = x_t.shape[1]
        pad_len = block_size - (current_len % block_size)
        if pad_len < block_size:
            mask_pad = torch.full((batch_size, pad_len), mask_id, device=device, dtype=torch.long)
            x_t = torch.cat([x_t, mask_pad], dim=1)

        block_past_key_values = None

        # Iteratively unmask tokens in the current block
        while True:
            mask_idx = x_t[:, -block_size:] == mask_id
            if mask_idx.sum() == 0:
                break

            # Process each small block
            for small_block_idx in range(num_small_blocks):
                start_idx = small_block_idx * small_block_size
                end_idx = start_idx + small_block_size

                start = -block_size + start_idx
                end = None if end_idx == block_size else -block_size + end_idx

                while True:
                    mask_idx = x_t[:, -block_size:] == mask_id
                    if mask_idx[:, start_idx:end_idx].sum() == 0:
                        break

                    # DualCache: reuse cache if first token in small block is not masked
                    if use_block_cache and block_past_key_values is not None:
                        if not (x_t[:, -block_size + start_idx] == mask_id).any():
                            # Reuse cache - only compute for current small block
                            output = model(
                                input_ids=x_t[:, start:end] if end else x_t[:, start:],
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                            logits = output.logits
                        else:
                            output = model(
                                input_ids=x_t[:, -block_size:],
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                            logits = output.logits[:, start_idx:end_idx]
                            block_past_key_values = output.past_key_values
                    else:
                        output = model(
                            input_ids=x_t[:, -block_size:],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        logits = output.logits[:, start_idx:end_idx]
                        if use_block_cache:
                            block_past_key_values = output.past_key_values

                    nfe += 1

                    # Shift logits for next-token prediction
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                    # Sample with temperature/top_p
                    x_1, p_1t = _sample_with_top_p(logits, top_p=top_p, temperature=temperature)

                    # Get confidence scores
                    x1_p = torch.gather(p_1t, dim=-1, index=x_1.unsqueeze(-1)).squeeze(-1)

                    # Mask out non-masked positions
                    x1_p = torch.where(mask_idx[:, start_idx:end_idx], x1_p, -torch.inf)

                    # Unmask high-confidence tokens
                    unmask_idx = x1_p > threshold

                    # Always unmask at least the most confident token
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(batch_size, device=device), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start_idx:end_idx]

                    # Update tokens
                    if end is None:
                        x_t[:, start:][unmask_idx] = x_1[unmask_idx]
                    else:
                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

        # Update cache for next block
        output = model(
            input_ids=x_t[:, -block_size:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        nfe += 1

        # Generate first token of next block
        next_token = output.logits[:, -1:, :].argmax(dim=-1)
        x_t = torch.cat([x_t, next_token], dim=1)

    return x_t[0], nfe


def _sample_with_top_p(
    logits: torch.Tensor, top_p: float = 0.95, temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from logits with top-p (nucleus) sampling."""
    if temperature == 0:
        # Greedy sampling
        probs = torch.softmax(logits, dim=-1)
        tokens = logits.argmax(dim=-1)
        return tokens, probs

    # Temperature scaling
    logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)

    # Sort by probability
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask for top-p
    mask = cumsum_probs - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample
    tokens = torch.multinomial(sorted_probs.view(-1, sorted_probs.shape[-1]), 1)
    tokens = tokens.view(*sorted_probs.shape[:-1])

    # Map back to original indices
    tokens = torch.gather(sorted_indices, -1, tokens.unsqueeze(-1)).squeeze(-1)

    return tokens, probs


def load_dlm_model(
    model_path: Union[str, Path],
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a DLM model for inference.

    Args:
        model_path: Path to DLM checkpoint
        device: Device to load model on (default: auto)
        torch_dtype: Data type for model weights

    Returns:
        (model, tokenizer) tuple ready for generation
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading DLM model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    # Verify mask token exists
    if DEFAULT_MASK_TOKEN not in tokenizer.get_vocab():
        logger.warning(f"Model missing mask token '{DEFAULT_MASK_TOKEN}', adding it")
        tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_MASK_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
