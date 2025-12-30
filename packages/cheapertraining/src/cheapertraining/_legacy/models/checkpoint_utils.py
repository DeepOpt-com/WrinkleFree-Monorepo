"""Gradient checkpointing utilities with optional INT8 quantization.

Provides memory-efficient gradient checkpointing for transformer training:
- Standard mode: Uses PyTorch's native checkpoint with BF16/FP32 activations
- Quantized mode: Stores activations in INT8 for additional ~2x memory savings

Based on:
- PyTorch torch.utils.checkpoint (https://pytorch.org/docs/stable/checkpoint.html)
- COAT Framework (https://arxiv.org/abs/2410.19313)
"""

from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as pytorch_checkpoint


def quantize_activation(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Quantize tensor to INT8 with per-tensor scaling.

    Args:
        x: Input tensor in floating point format

    Returns:
        Tuple of (quantized_tensor, scale)
    """
    # Compute per-tensor scale
    abs_max = x.abs().max()
    scale = abs_max / 127.0
    scale = scale.clamp(min=1e-8)  # Avoid division by zero

    # Quantize to INT8
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)

    return x_int8, scale


def dequantize_activation(x_int8: Tensor, scale: Tensor, dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Dequantize INT8 tensor back to floating point.

    Args:
        x_int8: Quantized tensor in INT8
        scale: Per-tensor scale factor
        dtype: Target dtype for dequantized tensor

    Returns:
        Dequantized tensor in target dtype
    """
    return x_int8.to(dtype) * scale


class QuantizedCheckpointFunction(torch.autograd.Function):
    """Custom autograd function for quantized activation checkpointing.

    Stores input activations in INT8 format during forward pass and
    dequantizes them during backward pass for gradient computation.
    """

    @staticmethod
    def forward(ctx, run_function: Callable, preserve_rng_state: bool, *args) -> Any:
        """Forward pass with quantized activation storage.

        Args:
            ctx: Autograd context
            run_function: Function to checkpoint
            preserve_rng_state: Whether to save and restore RNG state
            *args: Arguments to run_function

        Returns:
            Output of run_function
        """
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        # Store quantized inputs
        ctx.quantized_inputs = []
        ctx.input_dtypes = []
        ctx.input_devices = []

        for arg in args:
            if isinstance(arg, Tensor) and arg.is_floating_point():
                x_int8, scale = quantize_activation(arg.detach())
                ctx.quantized_inputs.append((x_int8, scale, True))
                ctx.input_dtypes.append(arg.dtype)
                ctx.input_devices.append(arg.device)
            elif isinstance(arg, Tensor):
                # Non-floating point tensor, store as-is
                ctx.quantized_inputs.append((arg.detach(), None, False))
                ctx.input_dtypes.append(arg.dtype)
                ctx.input_devices.append(arg.device)
            else:
                # Non-tensor argument
                ctx.quantized_inputs.append((arg, None, False))
                ctx.input_dtypes.append(None)
                ctx.input_devices.append(None)

        # Save RNG state if requested
        if preserve_rng_state:
            ctx.cpu_rng_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.cuda_rng_state = torch.cuda.get_rng_state()

        # Run forward pass without gradient tracking
        with torch.no_grad():
            outputs = run_function(*args)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs) -> Tuple[None, None, ...]:
        """Backward pass with activation recomputation.

        Dequantizes stored activations and recomputes the forward pass
        to obtain intermediate activations needed for gradient computation.
        """
        # Dequantize inputs
        inputs = []
        for i, (stored, scale, is_quantized) in enumerate(ctx.quantized_inputs):
            if is_quantized:
                dtype = ctx.input_dtypes[i]
                x = dequantize_activation(stored, scale, dtype)
                x.requires_grad_(True)
                inputs.append(x)
            elif ctx.input_dtypes[i] is not None:
                # Non-quantized tensor
                inputs.append(stored.requires_grad_(stored.is_floating_point()))
            else:
                # Non-tensor
                inputs.append(stored)

        # Restore RNG state if it was saved
        if ctx.preserve_rng_state:
            torch.set_rng_state(ctx.cpu_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(ctx.cuda_rng_state)

        # Recompute forward pass with gradients
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)

        # Handle single or multiple outputs
        if isinstance(outputs, Tensor):
            outputs = (outputs,)

        # Compute gradients
        outputs_with_grad = []
        grad_outputs_filtered = []
        for out, grad in zip(outputs, grad_outputs):
            if isinstance(out, Tensor) and out.requires_grad:
                outputs_with_grad.append(out)
                grad_outputs_filtered.append(grad)

        if len(outputs_with_grad) == 0:
            return (None, None) + tuple(None for _ in inputs)

        # Get input tensors that require grad
        inputs_with_grad = [inp for inp in inputs if isinstance(inp, Tensor) and inp.requires_grad]

        if len(inputs_with_grad) == 0:
            return (None, None) + tuple(None for _ in inputs)

        grads = torch.autograd.grad(
            outputs_with_grad,
            inputs_with_grad,
            grad_outputs_filtered,
            allow_unused=True,
        )

        # Map gradients back to original input positions
        grad_iter = iter(grads)
        result_grads = []
        for inp in inputs:
            if isinstance(inp, Tensor) and inp.requires_grad:
                result_grads.append(next(grad_iter))
            else:
                result_grads.append(None)

        return (None, None) + tuple(result_grads)


def checkpoint_fn(
    function: Callable,
    *args,
    mode: str = "standard",
    use_reentrant: bool = False,
    preserve_rng_state: bool = True,
) -> Any:
    """Apply gradient checkpointing with optional quantization.

    This is the main entry point for gradient checkpointing. It wraps a function
    such that intermediate activations are not stored during forward pass, but
    recomputed during backward pass.

    Args:
        function: Function to checkpoint (typically a transformer block's forward)
        *args: Arguments to pass to function
        mode: Checkpointing mode - "standard" or "quantized"
        use_reentrant: Whether to use reentrant checkpointing (False recommended)
        preserve_rng_state: Whether to save and restore RNG state

    Returns:
        Output of function

    Example:
        >>> # Standard checkpointing
        >>> output = checkpoint_fn(block.forward, x, mask, mode="standard")
        >>>
        >>> # Quantized checkpointing (saves more memory)
        >>> output = checkpoint_fn(block.forward, x, mask, mode="quantized")
    """
    if mode == "quantized":
        return QuantizedCheckpointFunction.apply(function, preserve_rng_state, *args)
    else:
        # Standard PyTorch checkpointing
        return pytorch_checkpoint(
            function,
            *args,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve_rng_state,
        )


def estimate_memory_savings(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    mode: str = "standard",
) -> dict:
    """Estimate memory savings from gradient checkpointing.

    Args:
        num_layers: Number of transformer layers
        batch_size: Batch size
        seq_len: Sequence length
        embed_dim: Model embedding dimension
        mode: Checkpointing mode ("none", "standard", or "quantized")

    Returns:
        Dictionary with estimated memory usage in GB
    """
    # Estimate activation memory per layer (rough approximation)
    # Main activations: hidden states, attention scores, FFN intermediates
    bytes_per_element = 2  # BF16

    # Hidden states: batch * seq * embed * 2 (input + output)
    hidden_mem = batch_size * seq_len * embed_dim * 2 * bytes_per_element

    # Attention: batch * heads * seq * seq (for each layer)
    # Assuming heads = embed_dim / 64
    num_heads = embed_dim // 64
    attn_mem = batch_size * num_heads * seq_len * seq_len * bytes_per_element

    # FFN intermediate: batch * seq * (4 * embed) * 2 (SwiGLU)
    ffn_mem = batch_size * seq_len * (4 * embed_dim) * 2 * bytes_per_element

    per_layer_mem = hidden_mem + attn_mem + ffn_mem

    if mode == "none":
        total_mem = num_layers * per_layer_mem
    elif mode == "standard":
        # Checkpoint reduces to sqrt(n) layers stored
        import math
        total_mem = math.sqrt(num_layers) * per_layer_mem
    else:  # quantized
        # Quantized adds ~2x savings on checkpointed portion
        import math
        total_mem = math.sqrt(num_layers) * per_layer_mem / 2

    return {
        "mode": mode,
        "estimated_activation_memory_gb": total_mem / (1024 ** 3),
        "per_layer_memory_mb": per_layer_mem / (1024 ** 2),
    }
