"""Quantization utilities for BitNet 1.58-bit models."""

from wrinklefree.quantization.activation_quant import (
    activation_quantization_absmean,
    activation_quantization_per_tensor,
    activation_quantization_per_token,
)
from wrinklefree.quantization.activation_sparse import (
    block_sparsify_nm,
    detach_sparsify,
    topk_sparsify,
)
from wrinklefree.quantization.fp8_gemm import (
    FP8Capability,
    FP8Config,
    detect_fp8_capability,
    log_fp8_config,
    should_use_fp8_for_layer,
)
from wrinklefree.quantization.lambda_warmup import (
    LambdaWarmup,
    get_current_lambda,
    get_global_lambda_warmup,
    set_global_lambda_warmup,
)
from wrinklefree.quantization.saliency_curriculum import SaliencyCurriculum
from wrinklefree.quantization.sparsity_warmup import (
    SparsityWarmup,
    get_current_sparsity,
    get_global_sparsity_warmup,
    set_global_sparsity_warmup,
)
from wrinklefree.quantization.ste import detach_quantize, ste_quantize
from wrinklefree.quantization.weight_quant import (
    compute_weight_scale,
    ternary_weight_quantization,
    ternary_weight_quantization_no_scale,
)

__all__ = [
    # STE
    "ste_quantize",
    "detach_quantize",
    # Ternary weight quantization
    "ternary_weight_quantization",
    "ternary_weight_quantization_no_scale",
    "compute_weight_scale",
    # Activation quantization
    "activation_quantization_per_token",
    "activation_quantization_per_tensor",
    "activation_quantization_absmean",
    # Activation sparsity (Q-Sparse)
    "topk_sparsify",
    "block_sparsify_nm",
    "detach_sparsify",
    # Sparsity warmup
    "SparsityWarmup",
    "get_current_sparsity",
    "get_global_sparsity_warmup",
    "set_global_sparsity_warmup",
    # Curriculum
    "SaliencyCurriculum",
    # Lambda warmup
    "LambdaWarmup",
    "get_current_lambda",
    "get_global_lambda_warmup",
    "set_global_lambda_warmup",
    # FP8 GEMM (DeepSeek-V3 style)
    "FP8Config",
    "FP8Capability",
    "detect_fp8_capability",
    "should_use_fp8_for_layer",
    "log_fp8_config",
]
