"""Tokenizer wrapper for MobileLLM training.

Uses LLaMA3.2 tokenizer (128k vocabulary) as specified in the paper.
Reference: MobileLLM-R1 paper (arXiv:2509.24945)
"""

from typing import Optional, Union, List

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


class TokenizerWrapper:
    """Wrapper around HuggingFace tokenizer with training utilities.

    Provides:
    - Consistent tokenization interface
    - Sequence packing utilities
    - Chat template support for SFT
    """

    def __init__(
        self,
        tokenizer_path: str = "meta-llama/Llama-3.2-1B",
        max_length: int = 2048,
        padding_side: str = "right",
        truncation_side: str = "right",
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        """Initialize tokenizer.

        Args:
            tokenizer_path: HuggingFace tokenizer path
            max_length: Maximum sequence length
            padding_side: Side to pad on ('left' or 'right')
            truncation_side: Side to truncate on ('left' or 'right')
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = padding_side
        self.tokenizer.truncation_side = truncation_side

        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS
            max_length: Max length (uses self.max_length if None)
            truncation: Whether to truncate

        Returns:
            List of token IDs
        """
        max_length = max_length or self.max_length

        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
        )

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self,
        texts: Union[str, List[str]],
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> dict:
        """Tokenize texts.

        Args:
            texts: Text or list of texts to tokenize
            padding: Padding strategy
            truncation: Whether to truncate
            max_length: Max length
            return_tensors: Return type ('pt' for PyTorch)

        Returns:
            Dictionary with 'input_ids', 'attention_mask'
        """
        max_length = max_length or self.max_length

        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

    def apply_chat_template(
        self,
        messages: List[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        max_length: Optional[int] = None,
    ) -> Union[str, List[int]]:
        """Apply chat template for SFT.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tokenize: Whether to tokenize the result
            add_generation_prompt: Whether to add generation prompt
            max_length: Max length for tokenization

        Returns:
            Templated string or token IDs
        """
        max_length = max_length or self.max_length

        if hasattr(self.tokenizer, "apply_chat_template"):
            result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            if tokenize and max_length:
                result = result[:max_length]
            return result
        else:
            # Fallback for tokenizers without chat template
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|system|>\n{content}\n"
                elif role == "user":
                    text += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}\n"

            if add_generation_prompt:
                text += "<|assistant|>\n"

            if tokenize:
                return self.encode(text, max_length=max_length)
            return text

    def create_sft_labels(
        self,
        input_ids: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        """Create labels for SFT with masked prompt.

        Sets prompt tokens to -100 (ignored in loss computation).

        Args:
            input_ids: Input token IDs
            prompt_length: Number of prompt tokens to mask

        Returns:
            Labels tensor with prompt masked
        """
        labels = input_ids.clone()
        labels[:, :prompt_length] = -100
        return labels
