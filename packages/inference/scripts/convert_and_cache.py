#!/usr/bin/env python3
"""Convert and cache BitNet model to GCS.

Usage:
    uv run python scripts/convert_and_cache.py microsoft/bitnet-b1.58-2B-4T
    uv run python scripts/convert_and_cache.py /path/to/local/model --skip-upload
"""

import argparse
import logging

from wrinklefree_inference.cache import GCSModelCache, compute_cache_key, get_cached_or_convert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert and cache BitNet model")
    parser.add_argument("model_path", help="HuggingFace model ID or local path")
    parser.add_argument("--revision", help="Model revision", default=None)
    parser.add_argument("--skip-upload", action="store_true", help="Skip GCS upload")
    args = parser.parse_args()

    cache = GCSModelCache()

    local_path = get_cached_or_convert(
        args.model_path,
        revision=args.revision,
        cache=cache,
        skip_gcs=args.skip_upload,
    )

    print(f"\nCached model ready at: {local_path}")
    if not args.skip_upload:
        cache_key = compute_cache_key(args.model_path, args.revision)
        print(f"GCS path: gs://{cache.bucket_name}/{cache.cache_prefix}/{cache_key}/")


if __name__ == "__main__":
    main()
