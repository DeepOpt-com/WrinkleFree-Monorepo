#!/usr/bin/env bash
# WrinkleFree Smoke Test
# Runs Stage 1 + Stage 1.9 training on RunPod, uploads to GCS
#
# Usage:
#   ./scripts/run_smoke_test.sh          # Run and wait for completion
#   ./scripts/run_smoke_test.sh --async  # Run in background
#
# Prerequisites:
#   - SkyPilot configured with RunPod
#   - GCS credentials in credentials/.env (GOOGLE_APPLICATION_CREDENTIALS_JSON)
#   - W&B API key in credentials/.env (WANDB_API_KEY)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYER_DIR="$(dirname "$SCRIPT_DIR")"

cd "$DEPLOYER_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}WrinkleFree Smoke Test${NC}"
echo -e "${GREEN}=========================================${NC}"

# Check for required files
if [ ! -f credentials/.env ]; then
    echo -e "${RED}Error: credentials/.env not found${NC}"
    echo "Create it with WANDB_API_KEY and GOOGLE_APPLICATION_CREDENTIALS_JSON"
    exit 1
fi

if [ ! -f skypilot/smoke_test.yaml ]; then
    echo -e "${RED}Error: skypilot/smoke_test.yaml not found${NC}"
    exit 1
fi

# Load credentials
source credentials/.env

# Check required env vars
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${YELLOW}Warning: WANDB_API_KEY not set, W&B logging will be disabled${NC}"
fi

# Activate venv if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Check SkyPilot is available
if ! command -v sky &> /dev/null; then
    echo -e "${RED}Error: SkyPilot not found. Install with: pip install skypilot[runpod]${NC}"
    exit 1
fi

ASYNC_MODE=false
if [ "$1" == "--async" ]; then
    ASYNC_MODE=true
fi

echo ""
echo "Configuration:"
echo "  - Model: smollm2_135m"
echo "  - Stages: Stage 1 (SubLN) + Stage 1.9 (layer-wise distill)"
echo "  - Max steps: 100"
echo "  - GCS bucket: wrinklefree-checkpoints"
echo "  - Cloud: RunPod RTX4090"
echo ""

# Generate unique cluster name
CLUSTER_NAME="smoke-$(date +%Y%m%d-%H%M%S)"

echo -e "${YELLOW}Launching SkyPilot cluster: ${CLUSTER_NAME}${NC}"
echo ""

if [ "$ASYNC_MODE" = true ]; then
    # Async mode - detach and return
    sky launch skypilot/smoke_test.yaml \
        -y \
        --cluster "$CLUSTER_NAME" \
        --detach-run \
        --down \
        --idle-minutes-to-autostop 5

    echo ""
    echo -e "${GREEN}Smoke test launched in background${NC}"
    echo ""
    echo "Monitor with:"
    echo "  sky logs $CLUSTER_NAME"
    echo ""
    echo "Check status:"
    echo "  sky status"
    echo ""
else
    # Sync mode - wait for completion
    sky launch skypilot/smoke_test.yaml \
        -y \
        --cluster "$CLUSTER_NAME" \
        --down \
        --idle-minutes-to-autostop 5

    # Check job status
    JOB_STATUS=$(sky queue "$CLUSTER_NAME" 2>/dev/null | grep -E "^\s*1\s+" | awk '{print $3}' || echo "UNKNOWN")

    if [ "$JOB_STATUS" == "SUCCEEDED" ]; then
        echo ""
        echo -e "${GREEN}=========================================${NC}"
        echo -e "${GREEN}Smoke Test PASSED!${NC}"
        echo -e "${GREEN}=========================================${NC}"
        echo ""
        echo "Checkpoints uploaded to:"
        echo "  gs://wrinklefree-checkpoints/checkpoints/smoke-test/"
        echo ""
        echo "View in console:"
        echo "  https://console.cloud.google.com/storage/browser/wrinklefree-checkpoints/checkpoints/smoke-test"
        exit 0
    else
        echo ""
        echo -e "${RED}=========================================${NC}"
        echo -e "${RED}Smoke Test FAILED (status: $JOB_STATUS)${NC}"
        echo -e "${RED}=========================================${NC}"
        echo ""
        echo "Check logs with:"
        echo "  sky logs $CLUSTER_NAME"
        exit 1
    fi
fi
