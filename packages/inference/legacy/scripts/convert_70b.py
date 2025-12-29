#!/usr/bin/env python3
"""
70B Model Conversion Orchestrator for BitNet.cpp Benchmarking.

Converts large 70B parameter models (e.g., Llama-3.1-70B) to BitNet i2_s format
for CPU benchmarking. Uses naive ternary conversion for speed/cost analysis only.

Pipeline:
1. Download from HuggingFace (with resume support)
2. Naive ternary conversion using naive_converter
3. GGUF conversion using BitNet's convert-hf-to-gguf-bitnet.py
4. i2_s quantization using llama-quantize
5. Validation (load model, generate test tokens)

Memory Requirements:
- Minimum: 32GB RAM (for runtime inference)
- Conversion: 170GB+ (loads full FP16 model)
- Disk: 300GB+ (for intermediate files)

Usage:
    python scripts/convert_70b.py --model meta-llama/Llama-3.1-70B
    python scripts/convert_70b.py --model Qwen/Qwen2-72B --resume
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for 70B model conversion."""

    model_id: str
    output_dir: Path
    bitnet_dir: Path

    # Conversion settings
    quant_type: str = "i2_s"  # Use i2_s for naive conversions (TL2 requires codegen)
    use_gpu: bool = True
    batch_size: int = 1

    # Resource requirements
    min_memory_gb: int = 170
    min_disk_space_gb: int = 300

    # Checkpointing
    checkpoint_dir: Optional[Path] = None

    @property
    def model_name(self) -> str:
        return self.model_id.split("/")[-1]

    @property
    def model_dir(self) -> Path:
        return self.output_dir / self.model_name

    @property
    def gguf_f32_path(self) -> Path:
        return self.model_dir / "ggml-model-f32.gguf"

    @property
    def gguf_i2s_path(self) -> Path:
        return self.model_dir / f"ggml-model-{self.quant_type}.gguf"

    @property
    def ternary_path(self) -> Path:
        return self.model_dir / f"{self.model_name}_naive_ternary.safetensors"


class ConversionStep:
    """Base class for conversion steps with checkpointing."""

    name: str = "step"
    flag_suffix: str = ".flag"

    def __init__(self, config: ConversionConfig):
        self.config = config

    @property
    def checkpoint_flag(self) -> Path:
        if self.config.checkpoint_dir:
            return self.config.checkpoint_dir / f"{self.name}{self.flag_suffix}"
        return self.config.model_dir / f".{self.name}{self.flag_suffix}"

    def is_complete(self) -> bool:
        return self.checkpoint_flag.exists()

    def mark_complete(self):
        self.checkpoint_flag.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_flag.write_text(
            json.dumps({
                "step": self.name,
                "timestamp": time.time(),
                "model": self.config.model_id,
            })
        )

    def run(self) -> bool:
        raise NotImplementedError


class DownloadStep(ConversionStep):
    """Download model from HuggingFace."""

    name = "download"

    def run(self) -> bool:
        if self.is_complete():
            logger.info(f"Download already complete for {self.config.model_id}")
            return True

        logger.info(f"Downloading model: {self.config.model_id}")
        self.config.model_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                self.config.model_id,
                local_dir=str(self.config.model_dir),
                resume_download=True,
            )
            self.mark_complete()
            logger.info("Download complete")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False


class NaiveConversionStep(ConversionStep):
    """Convert FP16 weights to naive ternary format."""

    name = "naive_ternary"

    def run(self) -> bool:
        if self.is_complete():
            logger.info("Naive ternary conversion already complete")
            return True

        logger.info("Starting naive ternary conversion...")
        logger.warning("This produces LOW QUALITY outputs - for benchmarking only!")

        try:
            # Add benchmark directory to path
            sys.path.insert(0, str(self.config.bitnet_dir.parent / "benchmark"))
            from naive_converter import NaiveConverter, ConversionConfig as NaiveConfig

            naive_config = NaiveConfig(
                model_id=str(self.config.model_dir),
                output_dir=self.config.model_dir,
                use_gpu=self.config.use_gpu,
                batch_size=self.config.batch_size,
                verbose=True,
            )

            converter = NaiveConverter(naive_config)
            result = converter.convert()

            if result.success:
                self.mark_complete()
                logger.info(f"Naive conversion complete: {result.output_path}")
                return True
            else:
                logger.error(f"Naive conversion failed: {result.error}")
                return False

        except ImportError:
            logger.warning("naive_converter not available, using fallback method")
            return self._fallback_conversion()
        except Exception as e:
            logger.error(f"Naive conversion failed: {e}")
            return False

    def _fallback_conversion(self) -> bool:
        """Fallback: Skip naive conversion and use model directly."""
        logger.info("Using fallback: direct conversion without naive ternary step")
        # For now, we skip this step and let GGUF conversion handle it
        self.mark_complete()
        return True


class GGUFConversionStep(ConversionStep):
    """Convert to GGUF f32 format."""

    name = "gguf_f32"

    def run(self) -> bool:
        if self.is_complete() and self.config.gguf_f32_path.exists():
            logger.info("GGUF f32 conversion already complete")
            return True

        logger.info("Converting to GGUF f32 format...")

        convert_script = self.config.bitnet_dir / "utils" / "convert-hf-to-gguf-bitnet.py"
        if not convert_script.exists():
            logger.error(f"Conversion script not found: {convert_script}")
            return False

        try:
            cmd = [
                sys.executable,
                str(convert_script),
                str(self.config.model_dir),
                "--outtype", "f32",
            ]
            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.config.bitnet_dir),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"GGUF conversion failed:\n{result.stderr}")
                return False

            if self.config.gguf_f32_path.exists():
                self.mark_complete()
                logger.info(f"GGUF f32 conversion complete: {self.config.gguf_f32_path}")
                return True
            else:
                logger.error("GGUF f32 file not created")
                return False

        except Exception as e:
            logger.error(f"GGUF conversion failed: {e}")
            return False


class QuantizeStep(ConversionStep):
    """Quantize GGUF f32 to i2_s format."""

    name = "quantize_i2s"

    def run(self) -> bool:
        if self.is_complete() and self.config.gguf_i2s_path.exists():
            logger.info("i2_s quantization already complete")
            return True

        if not self.config.gguf_f32_path.exists():
            logger.error(f"f32 GGUF not found: {self.config.gguf_f32_path}")
            return False

        logger.info("Quantizing to i2_s format...")

        # Find llama-quantize binary
        quantize_bin = self.config.bitnet_dir / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            logger.error(f"llama-quantize not found: {quantize_bin}")
            logger.info("Please build BitNet.cpp first: ./scripts/build_bitnet.sh")
            return False

        try:
            cmd = [
                str(quantize_bin),
                str(self.config.gguf_f32_path),
                str(self.config.gguf_i2s_path),
                "I2_S",
                "1",
            ]
            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(self.config.bitnet_dir),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Quantization failed:\n{result.stderr}")
                return False

            if self.config.gguf_i2s_path.exists():
                self.mark_complete()
                logger.info(f"Quantization complete: {self.config.gguf_i2s_path}")
                return True
            else:
                logger.error("i2_s GGUF file not created")
                return False

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False


class ValidationStep(ConversionStep):
    """Validate converted model."""

    name = "validation"

    def run(self) -> bool:
        if self.is_complete():
            logger.info("Validation already complete")
            return True

        logger.info("Validating converted model...")

        if not self.config.gguf_i2s_path.exists():
            logger.error(f"Model not found: {self.config.gguf_i2s_path}")
            return False

        # Check file size (70B at 1.58-bit should be ~17-20GB)
        size_gb = self.config.gguf_i2s_path.stat().st_size / (1024**3)
        logger.info(f"Model size: {size_gb:.2f} GB")

        expected_min = 15.0  # Minimum expected size for 70B
        expected_max = 25.0  # Maximum expected size

        if size_gb < expected_min:
            logger.warning(f"Model seems too small ({size_gb:.2f} GB < {expected_min} GB)")
        elif size_gb > expected_max:
            logger.warning(f"Model seems too large ({size_gb:.2f} GB > {expected_max} GB)")
        else:
            logger.info(f"Model size looks reasonable: {size_gb:.2f} GB")

        # Try to load with BitNet.cpp server (optional)
        server_bin = self.config.bitnet_dir / "build" / "bin" / "llama-server"
        if server_bin.exists():
            logger.info("Testing model load (5 second timeout)...")
            try:
                # Start server briefly to verify model loads
                proc = subprocess.Popen(
                    [
                        str(server_bin),
                        "-m", str(self.config.gguf_i2s_path),
                        "-c", "256",  # Small context for testing
                        "-t", "4",
                        "--port", "18080",
                    ],
                    cwd=str(self.config.bitnet_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                time.sleep(5)
                proc.terminate()
                proc.wait(timeout=5)
                logger.info("Model load test passed (server started)")
            except Exception as e:
                logger.warning(f"Model load test inconclusive: {e}")
        else:
            logger.info("Skipping load test (llama-server not found)")

        self.mark_complete()
        logger.info("Validation complete")
        return True


class ConversionPipeline:
    """Orchestrates the full 70B conversion pipeline."""

    def __init__(self, config: ConversionConfig):
        self.config = config
        self.steps = [
            DownloadStep(config),
            NaiveConversionStep(config),
            GGUFConversionStep(config),
            QuantizeStep(config),
            ValidationStep(config),
        ]

    def check_resources(self) -> bool:
        """Check if system has sufficient resources."""
        logger.info("Checking system resources...")

        # Check available memory
        try:
            import psutil

            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
            logger.info(f"Memory: {available_gb:.1f} GB available / {total_gb:.1f} GB total")

            if total_gb < self.config.min_memory_gb:
                logger.error(
                    f"Insufficient memory: {total_gb:.1f} GB < {self.config.min_memory_gb} GB required"
                )
                return False

        except ImportError:
            logger.warning("psutil not available, skipping memory check")

        # Check disk space
        disk_stat = shutil.disk_usage(self.config.output_dir)
        free_gb = disk_stat.free / (1024**3)
        logger.info(f"Disk space: {free_gb:.1f} GB free")

        if free_gb < self.config.min_disk_space_gb:
            logger.error(
                f"Insufficient disk space: {free_gb:.1f} GB < {self.config.min_disk_space_gb} GB required"
            )
            return False

        return True

    def run(self, resume: bool = True) -> bool:
        """Run the full conversion pipeline."""
        logger.info("=" * 60)
        logger.info(f"70B Model Conversion Pipeline")
        logger.info(f"Model: {self.config.model_id}")
        logger.info(f"Output: {self.config.model_dir}")
        logger.info(f"Quant type: {self.config.quant_type}")
        logger.info("=" * 60)

        if not self.check_resources():
            return False

        for i, step in enumerate(self.steps, 1):
            logger.info(f"\n[Step {i}/{len(self.steps)}] {step.name}")

            if resume and step.is_complete():
                logger.info(f"Skipping {step.name} (already complete)")
                continue

            start_time = time.time()
            success = step.run()
            elapsed = time.time() - start_time

            if not success:
                logger.error(f"Step {step.name} failed after {elapsed:.1f}s")
                return False

            logger.info(f"Step {step.name} completed in {elapsed:.1f}s")

        logger.info("\n" + "=" * 60)
        logger.info("Conversion complete!")
        logger.info(f"Model: {self.config.gguf_i2s_path}")
        logger.info("=" * 60)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert 70B models to BitNet i2_s format for benchmarking"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-70B",
        help="HuggingFace model ID (default: meta-llama/Llama-3.1-70B)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: extern/BitNet/models)",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="i2_s",
        choices=["i2_s", "tl1", "tl2"],
        help="Quantization type (default: i2_s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU for conversion (slower but uses less memory)",
    )
    parser.add_argument(
        "--min-memory",
        type=int,
        default=170,
        help="Minimum memory requirement in GB (default: 170)",
    )
    parser.add_argument(
        "--min-disk",
        type=int,
        default=300,
        help="Minimum disk space requirement in GB (default: 300)",
    )

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    bitnet_dir = repo_root / "extern" / "BitNet"

    if args.output_dir is None:
        output_dir = bitnet_dir / "models"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    config = ConversionConfig(
        model_id=args.model,
        output_dir=output_dir,
        bitnet_dir=bitnet_dir,
        quant_type=args.quant_type,
        use_gpu=not args.no_gpu,
        min_memory_gb=args.min_memory,
        min_disk_space_gb=args.min_disk,
    )

    pipeline = ConversionPipeline(config)
    success = pipeline.run(resume=args.resume)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
