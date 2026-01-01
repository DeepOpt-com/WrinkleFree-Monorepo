#!/bin/bash
# Run all Lightning smoke test combinations on L40 GPUs
#
# Usage:
#   cd packages/deployer
#   source credentials/.env
#   ./scripts/run_lightning_smoke_tests.sh [--parallel] [--combo <combo>]
#
# Options:
#   --parallel: Run all tests in parallel (uses more cloud credits)
#   --combo:    Run only specified combo (ce_only, dlm, distill, bitdistill, lrc)
#
# Examples:
#   ./scripts/run_lightning_smoke_tests.sh                    # Run all sequentially
#   ./scripts/run_lightning_smoke_tests.sh --parallel         # Run all in parallel
#   ./scripts/run_lightning_smoke_tests.sh --combo dlm        # Run only DLM test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYER_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DEPLOYER_DIR"

# All objective combinations to test
ALL_COMBOS=(ce_only dlm distill bitdistill lrc)

# Parse arguments
PARALLEL=false
SINGLE_COMBO=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --parallel)
      PARALLEL=true
      shift
      ;;
    --combo)
      SINGLE_COMBO="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Determine which combos to run
if [[ -n "$SINGLE_COMBO" ]]; then
  COMBOS=("$SINGLE_COMBO")
else
  COMBOS=("${ALL_COMBOS[@]}")
fi

echo "================================================"
echo "Lightning Smoke Test Suite"
echo "================================================"
echo "Combos to test: ${COMBOS[*]}"
echo "Parallel mode: $PARALLEL"
echo ""

# Function to launch a single test
launch_test() {
  local combo=$1
  local cluster="lightning-smoke-${combo}"

  echo "[Launch] Starting $combo test on cluster: $cluster"

  sky launch skypilot/smoke_test_lightning.yaml \
    -y \
    --cluster "$cluster" \
    --env OBJECTIVE_COMBO="$combo"
}

# Function to wait for and check a test
check_test() {
  local combo=$1
  local cluster="lightning-smoke-${combo}"

  echo "[Wait] Waiting for $cluster to complete..."

  # Wait for job to finish
  while true; do
    status=$(sky status "$cluster" --refresh 2>/dev/null | grep "$cluster" | awk '{print $NF}')
    if [[ "$status" == "STOPPED" ]] || [[ -z "$status" ]]; then
      break
    fi
    sleep 30
  done

  # Get logs and check for success
  echo "[Logs] Final output from $cluster:"
  sky logs "$cluster" --tail 50 2>/dev/null || true

  echo ""
}

# Clean up function
cleanup() {
  echo ""
  echo "[Cleanup] Tearing down clusters..."
  for combo in "${COMBOS[@]}"; do
    cluster="lightning-smoke-${combo}"
    sky down "$cluster" -y 2>/dev/null || true
  done
}

# Set up trap for cleanup on exit
trap cleanup EXIT

if $PARALLEL; then
  # Launch all tests in parallel
  echo "[Parallel] Launching all tests..."
  for combo in "${COMBOS[@]}"; do
    launch_test "$combo" &
  done
  wait

  echo ""
  echo "[Parallel] All tests launched. Waiting for completion..."

  # Check all tests
  for combo in "${COMBOS[@]}"; do
    check_test "$combo"
  done
else
  # Run tests sequentially
  for combo in "${COMBOS[@]}"; do
    echo ""
    echo "================================================"
    echo "Testing: $combo"
    echo "================================================"

    launch_test "$combo"
    check_test "$combo"

    # Clean up this cluster before moving to next
    cluster="lightning-smoke-${combo}"
    echo "[Cleanup] Stopping $cluster"
    sky down "$cluster" -y 2>/dev/null || true
  done
fi

echo ""
echo "================================================"
echo "All smoke tests completed!"
echo "================================================"
