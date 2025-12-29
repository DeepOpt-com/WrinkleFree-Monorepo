#!/usr/bin/env python3
"""Script to start the BitNet inference server."""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrinklefree_inference.server.bitnet_server import BitNetServer, get_default_bitnet_path


def main():
    parser = argparse.ArgumentParser(description="Start BitNet inference server")
    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=0,
        help="Number of threads (0=auto, default: 0)",
    )
    parser.add_argument(
        "--context-size", "-c",
        type=int,
        default=4096,
        help="Context size / KV cache (default: 4096)",
    )
    parser.add_argument(
        "--bitnet-path",
        type=Path,
        default=None,
        help="Path to BitNet installation",
    )

    args = parser.parse_args()

    bitnet_path = args.bitnet_path or get_default_bitnet_path()

    print(f"Starting BitNet inference server")
    print(f"  Model: {args.model}")
    print(f"  Port: {args.port}")
    print(f"  Context size: {args.context_size}")
    print(f"  BitNet path: {bitnet_path}")

    server = BitNetServer(
        bitnet_path=bitnet_path,
        model_path=args.model,
        port=args.port,
        host=args.host,
        num_threads=args.threads,
        context_size=args.context_size,
        continuous_batching=True,
    )

    try:
        server.start(wait_for_ready=True, timeout=120)
        print(f"\nServer running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")

        import time
        while server.is_running():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop()
        print("Server stopped")


if __name__ == "__main__":
    main()
