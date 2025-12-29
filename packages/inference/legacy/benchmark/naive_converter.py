"""
Naive ternary converter for BitNet benchmarking.

Converts FP16/BF16 model weights to ternary (-1, 0, 1) format.
This produces LOW QUALITY outputs - only for cost/speed benchmarking.

The naive conversion:
1. Computes scale = mean(|weights|) for each tensor
2. Divides weights by scale
3. Rounds to nearest integer
4. Clamps to [-1, 0, 1]

This is NOT a proper BitNet quantization (which requires training from scratch).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for naive ternary conversion."""

    model_id: str
    output_dir: Path
    architecture: str = "llama"  # "llama" or "moe"
    use_gpu: bool = True
    batch_size: int = 1  # Process layers in batches
    save_intermediate: bool = False
    verbose: bool = True


@dataclass
class ConversionResult:
    """Result from a conversion operation."""

    success: bool
    model_id: str
    output_path: Optional[Path] = None
    original_size_gb: float = 0.0
    converted_size_gb: float = 0.0
    compression_ratio: float = 0.0
    num_layers: int = 0
    error: Optional[str] = None


class NaiveConverter:
    """
    Converts pretrained models to naive ternary format.

    Warning: The resulting models will have very poor quality.
    This is only intended for benchmarking BitNet.cpp inference speed and cost.
    """

    def __init__(self, config: ConversionConfig):
        self.config = config
        self.progress_callback: Optional[Callable[[str, float], None]] = None

    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback for progress updates. Callback receives (message, progress_pct)."""
        self.progress_callback = callback

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress to callback if set."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        if self.config.verbose:
            logger.info(f"[{progress:.1f}%] {message}")

    def convert(self) -> ConversionResult:
        """
        Convert model to naive ternary format.

        Returns:
            ConversionResult with conversion details
        """
        try:
            # Lazy import torch to avoid loading it if not needed
            import torch
            from transformers import AutoModelForCausalLM, AutoConfig
            from safetensors.torch import save_file

        except ImportError as e:
            return ConversionResult(
                success=False,
                model_id=self.config.model_id,
                error=f"Missing dependencies: {e}. Install with: uv sync --extra convert",
            )

        self._report_progress(f"Loading model config: {self.config.model_id}", 0)

        try:
            # Load config to check architecture
            config = AutoConfig.from_pretrained(self.config.model_id)
            is_moe = hasattr(config, "num_local_experts") or "moe" in str(type(config)).lower()

            if is_moe and self.config.architecture != "moe":
                logger.warning("Detected MoE architecture, adjusting config")
                self.config.architecture = "moe"

        except Exception as e:
            return ConversionResult(
                success=False,
                model_id=self.config.model_id,
                error=f"Failed to load model config: {e}",
            )

        self._report_progress("Loading model weights (this may take a while for large models)", 5)

        try:
            # Load model with minimal memory footprint
            device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"

            # Load in bfloat16 to reduce memory
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
            )

        except Exception as e:
            return ConversionResult(
                success=False,
                model_id=self.config.model_id,
                error=f"Failed to load model: {e}",
            )

        self._report_progress("Converting weights to ternary", 20)

        # Get original size
        original_size_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        original_size_gb = original_size_bytes / (1024**3)

        # Convert each parameter
        converted_state_dict = {}
        scales = {}
        param_names = list(model.state_dict().keys())
        total_params = len(param_names)

        for i, name in enumerate(param_names):
            param = model.state_dict()[name]

            # Skip non-weight tensors (biases are usually small)
            if param.numel() < 1024:
                converted_state_dict[name] = param
                continue

            # Naive ternary conversion
            ternary_param, scale = self._ternary_quantize(param)
            converted_state_dict[name] = ternary_param
            scales[name] = scale.item() if hasattr(scale, "item") else scale

            progress = 20 + (i / total_params) * 60
            if i % 10 == 0:
                self._report_progress(f"Converted {i}/{total_params} tensors", progress)

        self._report_progress("Saving converted weights", 85)

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save converted model
        model_name = self.config.model_id.replace("/", "_")
        output_path = output_dir / f"{model_name}_naive_ternary.safetensors"

        try:
            # Convert to float16 for saving (ternary values: -1, 0, 1 as fp16)
            save_dict = {k: v.to(torch.float16).contiguous() for k, v in converted_state_dict.items()}
            save_file(save_dict, output_path)

        except Exception as e:
            return ConversionResult(
                success=False,
                model_id=self.config.model_id,
                error=f"Failed to save converted model: {e}",
            )

        # Save scales for reference
        scales_path = output_dir / f"{model_name}_scales.json"
        with open(scales_path, "w") as f:
            json.dump(scales, f, indent=2)

        # Calculate converted size (ternary = 1.58 bits per weight)
        total_weights = sum(p.numel() for p in model.parameters())
        converted_size_gb = (total_weights * 1.58 / 8) / (1024**3)

        self._report_progress("Conversion complete", 100)

        return ConversionResult(
            success=True,
            model_id=self.config.model_id,
            output_path=output_path,
            original_size_gb=original_size_gb,
            converted_size_gb=converted_size_gb,
            compression_ratio=original_size_gb / converted_size_gb if converted_size_gb > 0 else 0,
            num_layers=len([n for n in param_names if "layers" in n or "block" in n]),
        )

    def _ternary_quantize(self, tensor):
        """
        Naive ternary quantization.

        For each tensor:
        1. Compute scale = mean(|tensor|)
        2. Normalize: tensor / scale
        3. Round to integers
        4. Clamp to [-1, 0, 1]

        Returns (quantized_tensor, scale)
        """
        import torch

        # Compute scale as mean of absolute values
        scale = tensor.abs().mean()

        if scale == 0:
            return torch.zeros_like(tensor), torch.tensor(1.0)

        # Normalize, round, and clamp
        normalized = tensor / scale
        quantized = torch.round(normalized)
        ternary = torch.clamp(quantized, -1, 1)

        return ternary, scale

    def estimate_memory_requirements(self) -> dict:
        """
        Estimate memory requirements for conversion.

        Returns dict with estimated memory in GB.
        """
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.config.model_id)

            # Estimate parameters
            if hasattr(config, "num_parameters"):
                num_params = config.num_parameters
            else:
                # Rough estimate based on hidden size and layers
                hidden = getattr(config, "hidden_size", 4096)
                layers = getattr(config, "num_hidden_layers", 32)
                vocab = getattr(config, "vocab_size", 32000)
                num_params = hidden * hidden * layers * 4 + vocab * hidden * 2

            # Memory estimates
            bf16_size_gb = (num_params * 2) / (1024**3)  # bfloat16
            working_memory_gb = bf16_size_gb * 1.5  # Additional working memory

            return {
                "model_bf16_gb": round(bf16_size_gb, 2),
                "working_memory_gb": round(working_memory_gb, 2),
                "total_recommended_gb": round(bf16_size_gb + working_memory_gb, 2),
                "estimated_params": num_params,
            }

        except Exception as e:
            return {"error": str(e)}


def convert_to_gguf(
    safetensors_path: Path,
    output_path: Path,
    model_arch: str = "llama",
) -> Path:
    """
    Convert safetensors to GGUF format for BitNet.cpp.

    This is a placeholder - actual conversion requires BitNet's convert utilities.

    Args:
        safetensors_path: Path to converted safetensors file
        output_path: Output GGUF path
        model_arch: Model architecture

    Returns:
        Path to GGUF file
    """
    # TODO: Integrate with BitNet's GGUF conversion utilities
    # For now, this is a placeholder that documents the expected interface

    raise NotImplementedError(
        "GGUF conversion requires BitNet's convert utilities. "
        "Use the naive_to_bitnet.py script which wraps BitNet's setup_env.py"
    )
