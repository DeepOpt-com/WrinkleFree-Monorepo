#!/usr/bin/env python3
"""Script to validate KV cache behavior."""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrinklefree_inference.kv_cache.validator import run_kv_cache_validation


def main():
    parser = argparse.ArgumentParser(description="Validate KV cache behavior")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Inference server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )

    args = parser.parse_args()

    print(f"Validating KV cache at {args.url}")
    print()

    metrics = run_kv_cache_validation(args.url, args.timeout)

    print("Results:")
    print(f"  Prefix Speedup:           {metrics.prefix_speedup:.2f}x")
    print(f"  First Request Latency:    {metrics.first_request_latency_ms:.1f}ms")
    print(f"  Second Request Latency:   {metrics.second_request_latency_ms:.1f}ms")
    print(f"  Context Limit Handled:    {metrics.context_limit_handled}")
    print(f"  Concurrent Success Rate:  {metrics.concurrent_success_rate*100:.0f}%")

    if metrics.errors:
        print("\nErrors:")
        for error in metrics.errors:
            print(f"  - {error}")
        sys.exit(1)

    # Check pass criteria
    if metrics.concurrent_success_rate < 0.8:
        print("\nFAIL: Concurrent success rate too low")
        sys.exit(1)

    print("\nPASS: KV cache validation successful")


if __name__ == "__main__":
    main()
