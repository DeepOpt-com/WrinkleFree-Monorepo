#!/usr/bin/env python3
"""Convert trained model to GGUF format for BitNet.cpp inference."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wf_train.serving import convert_to_gguf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """
    Convert trained BitNet model to GGUF format.

    Usage:
        # Basic conversion
        uv run python scripts/convert_to_gguf.py --model outputs/model.safetensors

        # Specify output path and quantization type
        uv run python scripts/convert_to_gguf.py \\
            --model outputs/checkpoint.pt \\
            --output outputs/model.gguf \\
            --quant-type i2_s \\
            --model-name my-bitnet-7b
    """
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")

    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path to input model file (safetensors or pt checkpoint)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output GGUF file path (default: input path with .gguf extension)",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="i2_s",
        choices=["i2_s", "tl1", "tl2"],
        help="Quantization type (default: i2_s)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bitnet",
        help="Model name for metadata",
    )

    args = parser.parse_args()

    # Validate input
    if not args.model.exists():
        logger.error(f"Input model not found: {args.model}")
        sys.exit(1)

    # Set output path
    output_path = args.output or args.model.with_suffix(".gguf")

    logger.info(f"Input model: {args.model}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Quantization type: {args.quant_type}")
    logger.info(f"Model name: {args.model_name}")

    # Convert
    try:
        result_path = convert_to_gguf(
            model_path=args.model,
            output_path=output_path,
            quant_type=args.quant_type,
            model_name=args.model_name,
        )
        logger.info(f"Conversion complete: {result_path}")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
