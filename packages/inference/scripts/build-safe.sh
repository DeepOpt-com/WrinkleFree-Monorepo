#!/bin/bash
# Safe build wrapper - prevents system freeze
#
# Usage: ./scripts/build-safe.sh <command>
#
# Examples:
#   ./scripts/build-safe.sh cargo build --release
#   ./scripts/build-safe.sh cmake --build build
#   ./scripts/build-safe.sh make -C extern/BitNet
#
# Hard limits:
#   - 4 parallel build jobs (CMake, Cargo, Make)
#   - CPU affinity to cores 0-7 (8 cores max)

set -euo pipefail

# Hard-code build parallelism limits
export CMAKE_BUILD_PARALLEL_LEVEL=4
export CARGO_BUILD_JOBS=4
export MAKEFLAGS="-j4"

# Run with CPU affinity to prevent system freeze
exec taskset -c 0-7 "$@"
