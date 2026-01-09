#!/bin/bash
# Cleanup script for WrinkleFree inference package
set -e

BASE="/home/lev/code/WrinkleFreeDevWrapper/MonoRepoClean--LRCFEAT/packages/inference"
cd "$BASE"

echo "=== Cleaning up inference package ==="

# Delete Python package and tests
echo "Removing Python package..."
rm -rf src/wf_infer tests

# Clean scripts - keep only convert_checkpoint_to_gguf.py and build-safe.sh
echo "Cleaning scripts..."
cd scripts
rm -f benchmark_*.py launch_*.sh serve*.sh setup-cpu.sh cache_builds.sh \
      quantize_dlm_to_offline.py start_server.sh skypilot_common_setup.sh
rm -rf _legacy
cd ..

# Remove extraneous directories
echo "Removing extra directories..."
rm -rf demo deploy docker research results benchmark_results configs skypilot

# Move sgl-kernel to _legacy
echo "Moving sgl-kernel to _legacy..."
cd extern/sglang-bitnet
mkdir -p _legacy
mv sgl-kernel _legacy/ 2>/dev/null || echo "sgl-kernel already moved or doesn't exist"

# Clean Rust gateway extras
echo "Cleaning Rust extras..."
cd sgl-model-gateway
rm -rf pytest.ini Makefile bench_native.yaml

echo ""
echo "=== Local cleanup complete ==="
echo ""
echo "Now building wf_server..."
cargo build --release --bin wf_server --features native-inference

echo ""
echo "=== Build complete ==="
ls -la target/release/wf_server
