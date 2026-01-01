"""WrinkleFree DLM Converter - Convert BitNet models to Diffusion LLMs.

This library converts BitNet 1.58-bit quantized models to Diffusion Language
Models (DLMs) for faster parallel inference using the Fast-dLLM approach.

Example:
    >>> from wf_dlm_converter import convert, validate
    >>> result = convert(model="qwen3_4b", checkpoint="hf://repo/ckpt")
    >>> validate(model_path=result.output_path)
"""

from wf_dlm_converter.core import (
    convert,
    validate,
    logs,
    cancel,
)

__all__ = [
    "convert",
    "validate",
    "logs",
    "cancel",
]

__version__ = "0.1.0"
