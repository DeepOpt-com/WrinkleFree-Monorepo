"""Run naming utilities for WandB tracking.

Generates human-readable run names from Hydra/OmegaConf configs.

Training format: {model}-{stage}-{optimizer}-lr{lr}-bs{batch}-{suffix}
Example: qwen3_4b-s2-muon-lr2.4e3-bs64-a3f

MoE format: {model}-moe{N}k{K}-{stage}-{optimizer}-lr{lr}-bs{batch}-{suffix}
Example: qwen3_4b-moe8k2-s2-muon-lr2.4e3-bs64-a3f

Benchmark format: {model}-{quant}-ctx{ctx}-t{threads}-{suffix}
Example: bitnet2b-i2s-ctx4096-t16-a3f
"""

import ast
import hashlib
import time
from typing import Any, Union

ConfigType = Union[dict, Any]  # Any covers OmegaConf DictConfig


def _parse_string_dict(value: Any) -> Any:
    """Parse a string that looks like a dict into an actual dict."""
    if isinstance(value, str) and value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _get_nested(config: ConfigType, *keys: str, default: Any = None) -> Any:
    """Safely get nested config value, supporting both dict and OmegaConf.

    Also handles configs where nested dicts are stored as strings.
    """
    value = config
    for key in keys:
        try:
            if hasattr(value, "get"):
                value = value.get(key, default)
            elif hasattr(value, key):
                value = getattr(value, key, default)
            else:
                return default
            if value is None:
                return default
            # Parse string dicts
            value = _parse_string_dict(value)
        except (KeyError, AttributeError, TypeError):
            return default
    return value


def _format_lr(lr: float) -> str:
    """Format learning rate for run name (e.g., 2.4e-3 -> lr2.4e3)."""
    if lr is None:
        return "lr?"
    formatted = f"{lr:.1e}"  # "2.4e-03"
    parts = formatted.split("e")
    if len(parts) == 2:
        mantissa, exp = parts
        exp_val = int(exp)
        return f"lr{mantissa}e{abs(exp_val)}"
    return f"lr{formatted}"


def _get_stage_short(stage: str) -> str:
    """Convert stage name to short form."""
    stage_map = {
        "subln_insertion": "s1",
        "layerwise_distillation": "s1.9",
        "continue_pretrain": "s2",
        "distillation": "s3",
    }
    return stage_map.get(stage, f"s{stage}" if stage else "s?")


def _get_optimizer_short(opt_type: str) -> str:
    """Convert optimizer type to short form."""
    if not opt_type:
        return "opt?"
    opt_map = {
        "muonclip": "muon",
        "muon": "muon",
        "adamw": "adam",
        "adamw_8bit": "ad8b",
        "sgd": "sgd",
    }
    return opt_map.get(opt_type.lower(), opt_type[:4])


def _generate_suffix(length: int = 3) -> str:
    """Generate short unique suffix from timestamp hash."""
    hash_input = f"{time.time()}-{time.perf_counter_ns()}"
    hash_bytes = hashlib.md5(hash_input.encode()).hexdigest()
    return hash_bytes[:length]


def generate_run_name(
    config: ConfigType,
    max_length: int = 64,
    suffix_length: int = 3,
) -> str:
    """Generate a human-readable run name from config.

    Args:
        config: Hydra/OmegaConf config dict or DictConfig
        max_length: Maximum length for wandb compatibility (default: 64)
        suffix_length: Length of unique suffix (default: 3)

    Returns:
        Run name string, e.g., "smollm2-s2-muon-lr2.4e3-bs64-a3f"
    """
    # Extract model name - try multiple paths
    model_name = (
        _get_nested(config, "model", "name") or
        _get_nested(config, "model_name") or
        _get_nested(config, "model")
    )
    if isinstance(model_name, dict) or hasattr(model_name, "keys"):
        model_name = _get_nested(model_name, "name")
    if not model_name or not isinstance(model_name, str):
        model_name = "model"
    model_name = str(model_name).replace("_", "")  # smollm2_135m -> smollm2135m

    # Extract stage - try both flat and nested paths
    stage = (
        _get_nested(config, "stage") or
        _get_nested(config, "training", "stage") or
        "?"
    )
    stage_short = _get_stage_short(stage)

    # Extract optimizer type - try both flat and nested paths
    opt_type = (
        _get_nested(config, "optimizer", "type") or
        _get_nested(config, "training", "optimizer", "type")
    )
    opt_short = _get_optimizer_short(opt_type)

    # Extract learning rate - try both flat and nested paths
    lr = (
        _get_nested(config, "optimizer", "lr") or
        _get_nested(config, "training", "optimizer", "lr")
    )
    lr_str = _format_lr(lr) if lr else "lr?"

    # Extract batch size - try both flat and nested paths
    batch_size = (
        _get_nested(config, "batch_size") or
        _get_nested(config, "training", "batch_size") or
        0
    )
    grad_accum = (
        _get_nested(config, "gradient_accumulation_steps") or
        _get_nested(config, "training", "gradient_accumulation_steps") or
        1
    )
    try:
        effective_batch = int(batch_size) * int(grad_accum) if batch_size else 0
    except (ValueError, TypeError):
        effective_batch = 0
    bs_str = f"bs{effective_batch}" if effective_batch else "bs?"

    # Generate unique suffix
    suffix = _generate_suffix(suffix_length)

    # Check for MoE config
    num_experts = _get_nested(config, "model", "num_experts", default=0)
    top_k = _get_nested(config, "model", "top_k", default=0)
    moe_str = ""
    if num_experts and num_experts > 0:
        moe_str = f"-moe{num_experts}k{top_k}" if top_k else f"-moe{num_experts}"

    # Assemble name
    name = f"{model_name}{moe_str}-{stage_short}-{opt_short}-{lr_str}-{bs_str}-{suffix}"

    # Truncate if needed (preserve suffix)
    if len(name) > max_length:
        available = max_length - suffix_length - 1
        name = f"{name[:available]}-{suffix}"

    return name


def generate_benchmark_name(
    model_name: str,
    quant_type: str = "i2_s",
    context_size: int = 4096,
    num_threads: int = 0,
    batch_size: int = 1,
    suffix_length: int = 3,
) -> str:
    """Generate a human-readable benchmark run name.

    Args:
        model_name: Model identifier (e.g., "bitnet-2b", "qwen3-4b")
        quant_type: Quantization type (e.g., "i2_s", "tl2")
        context_size: Context window size
        num_threads: Number of inference threads (0 = auto)
        batch_size: Batch size for continuous batching
        suffix_length: Length of unique suffix

    Returns:
        Run name string, e.g., "bitnet2b-i2s-ctx4096-t16-a3f"
    """
    # Clean model name
    model_short = model_name.lower().replace("-", "").replace("_", "")
    model_short = model_short.replace("bitnet", "bn").replace("microsoft/", "")

    # Format quant type
    quant_short = quant_type.replace("_", "")

    # Build components
    ctx_str = f"ctx{context_size}"
    threads_str = f"t{num_threads}" if num_threads > 0 else "tauto"
    batch_str = f"b{batch_size}" if batch_size > 1 else ""

    # Generate suffix
    suffix = _generate_suffix(suffix_length)

    # Assemble
    parts = [model_short, quant_short, ctx_str, threads_str]
    if batch_str:
        parts.append(batch_str)
    parts.append(suffix)

    return "-".join(parts)


def generate_moe_benchmark_name(
    model_name: str,
    num_experts: int,
    top_k: int,
    quant_type: str = "i2_s",
    context_size: int = 4096,
    num_threads: int = 0,
    suffix_length: int = 3,
) -> str:
    """Generate a benchmark run name for MoE models.

    Args:
        model_name: Model identifier
        num_experts: Total number of experts (N)
        top_k: Number of active experts (K)
        quant_type: Quantization type
        context_size: Context window size
        num_threads: Number of inference threads
        suffix_length: Length of unique suffix

    Returns:
        Run name string, e.g., "bn2b-moe8k2-i2s-ctx4096-t16-a3f"
    """
    # Clean model name
    model_short = model_name.lower().replace("-", "").replace("_", "")
    model_short = model_short.replace("bitnet", "bn").replace("microsoft/", "")

    # MoE suffix
    moe_str = f"moe{num_experts}k{top_k}"

    # Format quant type
    quant_short = quant_type.replace("_", "")

    # Build components
    ctx_str = f"ctx{context_size}"
    threads_str = f"t{num_threads}" if num_threads > 0 else "tauto"

    # Generate suffix
    suffix = _generate_suffix(suffix_length)

    return f"{model_short}-{moe_str}-{quant_short}-{ctx_str}-{threads_str}-{suffix}"
