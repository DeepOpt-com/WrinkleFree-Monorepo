#!/usr/bin/env python3
"""Launch BitNet.cpp inference server."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrinklefree.serving import BitNetServer, convert_to_gguf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """
    Launch BitNet.cpp inference server.

    Usage:
        # Serve a GGUF model
        uv run python scripts/serve.py --model outputs/model.gguf

        # Serve with model conversion
        uv run python scripts/serve.py --model outputs/model.safetensors --convert

        # Specify port and threads
        uv run python scripts/serve.py --model outputs/model.gguf --port 8080 --threads 8
    """
    parser = argparse.ArgumentParser(description="Launch BitNet inference server")

    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path to model file (GGUF or safetensors/pt for conversion)",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert model to GGUF before serving",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="i2_s",
        choices=["i2_s", "tl1", "tl2"],
        help="Quantization type for conversion",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=4,
        help="Number of CPU threads",
    )
    parser.add_argument(
        "--bitnet-path",
        type=Path,
        default=Path(__file__).parent.parent.parent / "extern" / "BitNet",
        help="Path to BitNet.cpp installation (at meta-repo root)",
    )

    args = parser.parse_args()

    # Handle model conversion
    model_path = args.model
    if args.convert or model_path.suffix != ".gguf":
        logger.info(f"Converting model to GGUF format...")
        gguf_path = model_path.with_suffix(".gguf")
        model_path = convert_to_gguf(
            model_path=model_path,
            output_path=gguf_path,
            quant_type=args.quant_type,
        )
        logger.info(f"Converted model saved to {model_path}")

    # Check BitNet installation
    if not args.bitnet_path.exists():
        logger.error(f"BitNet.cpp not found at {args.bitnet_path}")
        logger.error("Please run: git submodule update --init --recursive")
        sys.exit(1)

    # Start server
    logger.info(f"Starting BitNet server on port {args.port}...")
    logger.info(f"Model: {model_path}")
    logger.info(f"Threads: {args.threads}")

    server = BitNetServer(
        bitnet_path=args.bitnet_path,
        model_path=model_path,
        port=args.port,
        num_threads=args.threads,
    )

    try:
        server.start(wait_for_ready=True)
        logger.info(f"Server running at http://localhost:{args.port}")
        logger.info("Press Ctrl+C to stop")

        # Keep running until interrupted
        import time
        while server.is_running():
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
