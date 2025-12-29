"""DLM (Diffusion Language Model) wrapper for lm-evaluation-harness.

Based on LLaDA's evaluation approach:
https://github.com/ML-GSAI/LLaDA/blob/main/eval_llada.py
https://github.com/ML-GSAI/LLaDA/blob/main/get_log_likelihood.py

Key difference from AR models:
- Loglikelihood computed via masked token prediction + Monte Carlo averaging
- Uses mask token for diffusion-style evaluation

This wrapper is designed for Fast-dLLM v2 models:
https://github.com/NVlabs/Fast-dLLM
https://arxiv.org/abs/2509.26328
"""

from typing import Any, Optional, Union
import logging

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default mask token for Fast-dLLM v2
DEFAULT_MASK_TOKEN = "|<MASK>|"
FAST_DLLM_MASK_ID = 151665


class DLMEvalHarness(LM):
    """lm-eval wrapper for Fast-dLLM v2 / DLM models.

    This wrapper implements DLM-style loglikelihood computation using
    Monte Carlo masking, which is fundamentally different from standard
    autoregressive evaluation.

    For classification tasks (like GLUE), this computes log P(answer | prompt)
    using the diffusion model's masked prediction capabilities.
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 1,
        mc_iterations: int = 128,
        mask_token: str = DEFAULT_MASK_TOKEN,
        mask_token_id: Optional[int] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        """Initialize the DLM model wrapper.

        Args:
            pretrained: HuggingFace model ID or local path
            device: Device to run on (cuda, cpu)
            dtype: Model dtype (float16, bfloat16, float32)
            batch_size: Batch size for evaluation
            mc_iterations: Number of Monte Carlo iterations for loglikelihood
            mask_token: Mask token string
            mask_token_id: Explicit mask token ID (auto-detected if None)
            trust_remote_code: Trust remote code in model config
        """
        super().__init__()

        self._device = device
        self._batch_size = batch_size
        self.mc_iterations = mc_iterations
        self.mask_token = mask_token

        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        logger.info(f"Loading DLM model from {pretrained}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=self.torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # Get mask token ID
        if mask_token_id is not None:
            self.mask_id = mask_token_id
        else:
            self.mask_id = self._get_mask_id()

        # Get vocab size
        self.vocab_size = len(self.tokenizer)

        logger.info(f"  Device: {device}, Dtype: {dtype}")
        logger.info(f"  Mask token: '{self.mask_token}' (ID: {self.mask_id})")
        logger.info(f"  MC iterations: {self.mc_iterations}")
        logger.info(f"  Vocab size: {self.vocab_size}")

    def _get_mask_id(self) -> int:
        """Get mask token ID from tokenizer or model config."""
        # Try tokenizer first
        if self.mask_token in self.tokenizer.get_vocab():
            ids = self.tokenizer.encode(self.mask_token, add_special_tokens=False)
            if ids:
                return ids[0]

        # Check model config
        if hasattr(self.model.config, "mask_token_id"):
            return self.model.config.mask_token_id

        # Check for bd_size (indicates DLM model)
        if hasattr(self.model.config, "bd_size"):
            logger.warning(
                f"Model has bd_size but no mask token found, using default {FAST_DLLM_MASK_ID}"
            )
            return FAST_DLLM_MASK_ID

        raise ValueError(
            f"Model does not have mask token. Add '{self.mask_token}' to tokenizer "
            "or set mask_token_id parameter"
        )

    @property
    def eot_token_id(self) -> int:
        """Return end-of-text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Return maximum sequence length."""
        return getattr(self.model.config, "max_position_embeddings", 4096)

    @property
    def max_gen_toks(self) -> int:
        """Return maximum generation tokens."""
        return 256

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self._batch_size

    @property
    def device(self) -> str:
        """Return device."""
        return self._device

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        """Tokenize a string."""
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        """Decode tokens to string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _forward_process(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        p_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply forward diffusion process by randomly masking tokens.

        For each sample in the batch, masks a proportion p_mask of the
        non-prompt tokens (target tokens) with the mask token.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            prompt_len: Length of the prompt (tokens to keep unmasked)
            p_mask: Masking probability for each sample [batch_size]

        Returns:
            Masked input_ids tensor
        """
        batch_size, seq_len = input_ids.shape
        target_len = seq_len - prompt_len

        # Clone input to avoid modifying original
        masked_ids = input_ids.clone()

        # For each sample, mask p_mask proportion of target tokens
        for i in range(batch_size):
            n_mask = int(p_mask[i].item() * target_len)
            if n_mask > 0:
                # Random positions in target region
                perm = torch.randperm(target_len, device=input_ids.device)
                mask_positions = prompt_len + perm[:n_mask]
                masked_ids[i, mask_positions] = self.mask_id

        return masked_ids

    @torch.no_grad()
    def get_loglikelihood(
        self,
        prefix_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """Compute DLM-style loglikelihood via Monte Carlo sampling.

        This implements the key insight from LLaDA: for diffusion models,
        we compute loglikelihood by:
        1. Randomly masking portions of the target
        2. Predicting masked tokens
        3. Importance weighting by mask probability
        4. Averaging over Monte Carlo iterations

        Args:
            prefix_ids: Context/prompt token IDs [seq_len]
            target_ids: Continuation/answer token IDs [target_len]

        Returns:
            Log likelihood of target given prefix
        """
        device = next(self.model.parameters()).device

        # Concatenate prefix and target
        full_ids = torch.cat([prefix_ids, target_ids])
        seq_len = len(full_ids)
        prompt_len = len(prefix_ids)
        target_len = len(target_ids)

        if target_len == 0:
            return 0.0

        # Create batch for Monte Carlo iterations
        # Each iteration uses a different masking rate
        total_loss = 0.0

        for mc_iter in range(self.mc_iterations):
            # Sample masking probability uniformly from (0, 1]
            # Using (mc_iter + 1) / mc_iterations ensures we cover the range
            p_mask = torch.tensor(
                [(mc_iter + 1) / self.mc_iterations],
                device=device,
            )

            # Create masked input
            input_batch = full_ids.unsqueeze(0).to(device)
            masked_input = self._forward_process(input_batch, prompt_len, p_mask)

            # Forward pass
            outputs = self.model(masked_input)
            logits = outputs.logits  # [1, seq_len, vocab_size]

            # Compute cross-entropy loss on masked positions
            # Shift logits for next-token prediction
            shift_logits = logits[0, prompt_len - 1 : seq_len - 1]  # [target_len, vocab]
            shift_labels = full_ids[prompt_len:].to(device)  # [target_len]

            # Find which positions were masked
            mask_idx = masked_input[0, prompt_len:] == self.mask_id

            if mask_idx.sum() > 0:
                # Compute loss only on masked positions
                masked_logits = shift_logits[mask_idx]
                masked_labels = shift_labels[mask_idx]

                # Cross-entropy loss (negative log likelihood)
                loss = F.cross_entropy(masked_logits, masked_labels, reduction="sum")

                # Importance weighting: divide by mask probability
                # This corrects for the fact that we only observe masked positions
                weighted_loss = loss / p_mask[0]

                total_loss += weighted_loss.item()

        # Average over MC iterations and return negative loss (log likelihood)
        avg_loss = total_loss / self.mc_iterations
        return -avg_loss

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute loglikelihood for a list of requests.

        This is the main lm-eval API method for classification tasks.

        Args:
            requests: List of Instance objects with (context, continuation) pairs

        Returns:
            List of (loglikelihood, is_greedy) tuples
        """
        results = []

        for request in tqdm(requests, desc="DLM loglikelihood"):
            context, continuation = request.args

            # Tokenize
            context_ids = torch.tensor(
                self.tok_encode(context),
                dtype=torch.long,
            )
            continuation_ids = torch.tensor(
                self.tok_encode(continuation),
                dtype=torch.long,
            )

            # Compute loglikelihood
            ll = self.get_loglikelihood(context_ids, continuation_ids)

            # For DLM, we don't have a greedy check (would require generation)
            # Set is_greedy to False
            results.append((ll, False))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute rolling loglikelihood (for perplexity).

        For DLM models, we use the same Monte Carlo approach.
        """
        results = []

        for request in tqdm(requests, desc="DLM rolling loglikelihood"):
            (text,) = request.args

            # Tokenize full text
            token_ids = torch.tensor(
                self.tok_encode(text),
                dtype=torch.long,
            )

            # Use empty prefix, compute LL for whole text
            # This is a simplification - could do sliding window
            prefix = token_ids[:1]  # Keep first token as "context"
            target = token_ids[1:]

            ll = self.get_loglikelihood(prefix, target)
            results.append((ll, False))

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text until stop sequence.

        Uses Fast-dLLM v2 block diffusion for generation.
        For now, falls back to greedy autoregressive generation.

        TODO: Implement proper block diffusion generation from
        wf_dlm_converter.inference.generate_with_dualcache
        """
        results = []

        for request in tqdm(requests, desc="DLM generation"):
            context, gen_kwargs = request.args

            # Get stop sequences
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            # Tokenize context
            input_ids = self.tokenizer.encode(
                context,
                return_tensors="pt",
            ).to(self.device)

            # Generate using standard HuggingFace generate (fallback)
            # TODO: Use block diffusion generation for speed
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_gen_toks,
                    do_sample=False,  # Greedy
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            # Decode only generated tokens
            generated_ids = outputs[0, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Truncate at stop sequences
            for stop in until:
                if stop in generated_text:
                    generated_text = generated_text[:generated_text.index(stop)]

            results.append(generated_text)

        return results


def create_dlm_model(
    model_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    mc_iterations: int = 128,
    **kwargs,
) -> DLMEvalHarness:
    """Factory function to create DLM model wrapper.

    Args:
        model_path: HuggingFace model ID or local path
        device: Device to run on
        dtype: Model dtype
        mc_iterations: Monte Carlo iterations for loglikelihood

    Returns:
        Configured DLMEvalHarness instance
    """
    return DLMEvalHarness(
        pretrained=model_path,
        device=device,
        dtype=dtype,
        mc_iterations=mc_iterations,
        **kwargs,
    )
