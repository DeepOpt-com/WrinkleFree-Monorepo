#!/bin/bash
# build_bitnet.sh - Optimized build script for BitNet.cpp
#
# This script builds BitNet.cpp with full optimization flags for high-performance
# CPU inference. It supports ccache for faster rebuilds and architecture-specific
# optimizations.
#
# Environment Variables:
#   BITNET_USE_CCACHE=1              Enable ccache (default: 1)
#   BITNET_OPTIMIZATION_LEVEL=native Optimization level: basic|native|aggressive (default: native)
#   CMAKE_BUILD_PARALLEL_LEVEL=      Parallel jobs (default: CPU count)
#   BITNET_KERNEL_TYPE=              Kernel type: i2_s|tl1|tl2|auto (default: auto)
#   GCS_CACHE_BUCKET=                GCS bucket for build cache (optional)
#   CACHE_VERSION=                   Cache version key (default: v1)
#
# Usage:
#   ./scripts/build_bitnet.sh
#   BITNET_OPTIMIZATION_LEVEL=aggressive ./scripts/build_bitnet.sh

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BITNET_DIR="$REPO_ROOT/extern/BitNet"

# Configuration with defaults
USE_CCACHE="${BITNET_USE_CCACHE:-1}"
OPT_LEVEL="${BITNET_OPTIMIZATION_LEVEL:-native}"
PARALLEL_JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}"
KERNEL_TYPE="${BITNET_KERNEL_TYPE:-auto}"
GCS_BUCKET="${GCS_CACHE_BUCKET:-}"
CACHE_VERSION="${CACHE_VERSION:-v1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect architecture
detect_arch() {
    local arch
    arch=$(uname -m)
    case "$arch" in
        x86_64|amd64|AMD64)
            echo "x86_64"
            ;;
        aarch64|arm64|ARM64)
            echo "arm64"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
}

# Detect and setup ccache
setup_ccache() {
    if [[ "$USE_CCACHE" != "1" ]]; then
        log_info "ccache disabled"
        return 1
    fi

    if command -v ccache &>/dev/null; then
        log_info "ccache found: $(ccache --version | head -1)"

        # Configure ccache for optimal performance
        ccache --max-size=10G 2>/dev/null || true
        ccache --set-config=compression=true 2>/dev/null || true
        ccache --set-config=compression_level=6 2>/dev/null || true

        return 0
    else
        log_warn "ccache not found. Installing..."
        if command -v apt-get &>/dev/null; then
            sudo apt-get update && sudo apt-get install -y ccache
        elif command -v yum &>/dev/null; then
            sudo yum install -y ccache
        elif command -v brew &>/dev/null; then
            brew install ccache
        else
            log_warn "Could not install ccache automatically. Proceeding without it."
            return 1
        fi
        return 0
    fi
}

# Get optimization flags based on level
get_opt_flags() {
    case "$OPT_LEVEL" in
        basic)
            echo "-O2"
            ;;
        native)
            echo "-march=native -mtune=native -O3"
            ;;
        aggressive)
            echo "-march=native -mtune=native -O3 -ffast-math"
            ;;
        *)
            log_error "Unknown optimization level: $OPT_LEVEL (use: basic, native, aggressive)"
            exit 1
            ;;
    esac
}

# Get kernel type based on architecture
get_kernel_cmake_flag() {
    local arch="$1"
    local kernel="$KERNEL_TYPE"

    if [[ "$kernel" == "auto" ]]; then
        case "$arch" in
            x86_64) kernel="tl2" ;;
            arm64)  kernel="tl1" ;;
        esac
    fi

    case "$kernel" in
        tl2)
            echo "-DBITNET_X86_TL2=ON"
            ;;
        tl1)
            echo "-DBITNET_ARM_TL1=ON"
            ;;
        i2_s)
            echo ""  # No extra flag needed for i2_s
            ;;
        *)
            log_error "Unknown kernel type: $kernel (use: i2_s, tl1, tl2, auto)"
            exit 1
            ;;
    esac
}

# Check for GCS cache and restore if available
try_restore_from_cache() {
    if [[ -z "$GCS_BUCKET" ]]; then
        return 1
    fi

    if ! command -v gsutil &>/dev/null; then
        log_warn "gsutil not found, skipping cache restore"
        return 1
    fi

    local arch
    arch=$(detect_arch)
    local cache_key="${arch}_bitnet_${OPT_LEVEL}_${CACHE_VERSION}.tar.gz"
    local cache_path="$GCS_BUCKET/$cache_key"

    log_info "Checking GCS cache: $cache_path"

    if gsutil -q stat "$cache_path" 2>/dev/null; then
        log_info "Cache hit! Restoring build artifacts..."
        local tmp_file="/tmp/$cache_key"
        gsutil cp "$cache_path" "$tmp_file"

        cd "$BITNET_DIR"
        tar -xzf "$tmp_file" -C .
        rm "$tmp_file"

        if [[ -f "build/bin/llama-server" ]] || [[ -f "build/bin/llama-cli" ]]; then
            log_info "Build artifacts restored from cache"
            return 0
        else
            log_warn "Cache restored but binaries not found, rebuilding..."
            return 1
        fi
    else
        log_info "No cache found, will build from scratch"
        return 1
    fi
}

# Upload build artifacts to GCS cache
upload_to_cache() {
    if [[ -z "$GCS_BUCKET" ]]; then
        return 0
    fi

    if ! command -v gsutil &>/dev/null; then
        log_warn "gsutil not found, skipping cache upload"
        return 0
    fi

    local arch
    arch=$(detect_arch)
    local cache_key="${arch}_bitnet_${OPT_LEVEL}_${CACHE_VERSION}.tar.gz"
    local cache_path="$GCS_BUCKET/$cache_key"
    local tmp_file="/tmp/$cache_key"

    log_info "Uploading build artifacts to cache: $cache_path"

    cd "$BITNET_DIR"
    tar -czf "$tmp_file" build/bin

    if gsutil cp "$tmp_file" "$cache_path"; then
        log_info "Build artifacts cached successfully"
    else
        log_warn "Failed to upload cache (non-fatal)"
    fi

    rm -f "$tmp_file"
}

# Main build function
build_bitnet() {
    local arch
    arch=$(detect_arch)
    log_info "Building BitNet.cpp for $arch"
    log_info "Optimization level: $OPT_LEVEL"
    log_info "Parallel jobs: $PARALLEL_JOBS"

    # Try to restore from cache first
    if try_restore_from_cache; then
        log_info "Using cached build, skipping compilation"
        return 0
    fi

    cd "$BITNET_DIR"

    # Setup ccache
    local ccache_flags=()
    if setup_ccache; then
        ccache_flags=(
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache"
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
        )
        log_info "ccache enabled"
    fi

    # Get optimization flags
    local opt_flags
    opt_flags=$(get_opt_flags)
    log_info "Optimization flags: $opt_flags"

    # Get kernel flag
    local kernel_flag
    kernel_flag=$(get_kernel_cmake_flag "$arch")
    if [[ -n "$kernel_flag" ]]; then
        log_info "Kernel flag: $kernel_flag"
    fi

    # Detect compiler
    local c_compiler="clang"
    local cxx_compiler="clang++"

    if ! command -v clang &>/dev/null; then
        log_warn "clang not found, trying gcc"
        c_compiler="gcc"
        cxx_compiler="g++"
    fi

    log_info "Using compiler: $c_compiler / $cxx_compiler"

    # Clean previous build if requested
    if [[ "${CLEAN_BUILD:-0}" == "1" ]]; then
        log_info "Cleaning previous build..."
        rm -rf build
    fi

    # Configure with CMake
    log_info "Configuring with CMake..."
    local cmake_args=(
        -B build
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER="$c_compiler"
        -DCMAKE_CXX_COMPILER="$cxx_compiler"
        -DCMAKE_C_FLAGS="$opt_flags"
        -DCMAKE_CXX_FLAGS="$opt_flags"
    )

    # Add ccache flags
    cmake_args+=("${ccache_flags[@]}")

    # Add kernel flag if set
    if [[ -n "$kernel_flag" ]]; then
        cmake_args+=("$kernel_flag")
    fi

    cmake "${cmake_args[@]}"

    # Build
    log_info "Building with $PARALLEL_JOBS parallel jobs..."
    cmake --build build --config Release -j "$PARALLEL_JOBS"

    # Verify build
    if [[ -f "build/bin/llama-server" ]] || [[ -f "build/bin/llama-cli" ]]; then
        log_info "Build completed successfully!"

        # Show ccache stats if enabled
        if [[ "$USE_CCACHE" == "1" ]] && command -v ccache &>/dev/null; then
            log_info "ccache statistics:"
            ccache --show-stats 2>/dev/null || true
        fi

        # Upload to cache
        upload_to_cache
    else
        log_error "Build failed: binaries not found"
        exit 1
    fi
}

# Show CPU info
show_cpu_info() {
    log_info "=== CPU Information ==="
    if command -v lscpu &>/dev/null; then
        lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core|Socket|Cache|AVX|Flags)" | head -20 || true
    elif [[ -f /proc/cpuinfo ]]; then
        grep -E "(model name|cpu cores|flags)" /proc/cpuinfo | head -10 || true
    fi
    echo ""
}

# Print configuration summary
print_config() {
    echo ""
    log_info "=== Build Configuration ==="
    echo "  Architecture:      $(detect_arch)"
    echo "  Optimization:      $OPT_LEVEL"
    echo "  Kernel type:       $KERNEL_TYPE"
    echo "  Parallel jobs:     $PARALLEL_JOBS"
    echo "  ccache enabled:    $USE_CCACHE"
    echo "  GCS cache bucket:  ${GCS_BUCKET:-none}"
    echo "  Cache version:     $CACHE_VERSION"
    echo "  BitNet directory:  $BITNET_DIR"
    echo ""
}

# Main entry point
main() {
    print_config
    show_cpu_info
    build_bitnet
}

main "$@"
