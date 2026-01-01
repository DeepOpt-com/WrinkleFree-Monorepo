#!/bin/bash
# ============================================================================
# BitNet Deployment Script
# ============================================================================
#
# Deploy BitNet model to GCP using SkyPilot.
#
# Usage:
#   ./deploy_bitnet.sh [--server-type TYPE] [--down] [--status]
#
# Server types:
#   batch    - Rust batch_server with RadixAttention (default)
#   llamacpp - llama.cpp llama-server
#
# Examples:
#   ./deploy_bitnet.sh                    # Launch batch server
#   ./deploy_bitnet.sh --server-type llamacpp
#   ./deploy_bitnet.sh --status           # Check status
#   ./deploy_bitnet.sh --down             # Tear down
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYPILOT_DIR="${SCRIPT_DIR}/../skypilot"

# Default values
SERVER_TYPE="batch"
ACTION="launch"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server-type)
            SERVER_TYPE="$2"
            shift 2
            ;;
        --down)
            ACTION="down"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --logs)
            ACTION="logs"
            shift
            ;;
        --ssh)
            ACTION="ssh"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--server-type TYPE] [--down] [--status] [--logs] [--ssh]"
            echo ""
            echo "Server types:"
            echo "  batch    - Rust batch_server with RadixAttention (default, ~11 tok/s)"
            echo "  llamacpp - llama.cpp llama-server (~17 tok/s)"
            echo ""
            echo "Actions:"
            echo "  (default) - Launch server"
            echo "  --status  - Check server status"
            echo "  --down    - Tear down server"
            echo "  --logs    - View server logs"
            echo "  --ssh     - SSH into server"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Map server type to config file and cluster name
case $SERVER_TYPE in
    batch)
        CONFIG_FILE="${SKYPILOT_DIR}/bitnet_batch_server.yaml"
        CLUSTER_NAME="bitnet-batch"
        ;;
    llamacpp)
        CONFIG_FILE="${SKYPILOT_DIR}/dlm_llamacpp_8vcpu.yaml"
        CLUSTER_NAME="dlm-llamacpp"
        ;;
    *)
        echo "Unknown server type: $SERVER_TYPE"
        echo "Valid types: batch, llamacpp"
        exit 1
        ;;
esac

# Execute action
case $ACTION in
    launch)
        echo "=== Launching BitNet Server (${SERVER_TYPE}) ==="
        echo "Config: ${CONFIG_FILE}"
        echo "Cluster: ${CLUSTER_NAME}"
        echo ""

        if [[ ! -f "$CONFIG_FILE" ]]; then
            echo "Error: Config file not found: ${CONFIG_FILE}"
            exit 1
        fi

        sky launch -y "${CONFIG_FILE}"

        echo ""
        echo "=== Deployment Complete ==="
        echo "Server: http://$(sky status ${CLUSTER_NAME} --endpoint 30000 2>/dev/null || echo 'pending'):30000"
        echo ""
        echo "Test with:"
        echo '  curl -s http://IP:30000/v1/chat/completions \'
        echo '    -H "Content-Type: application/json" \'
        echo '    -d '\''{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'\'''
        ;;

    status)
        echo "=== BitNet Server Status ==="
        sky status
        ;;

    down)
        echo "=== Tearing Down BitNet Server ==="
        sky down -y "${CLUSTER_NAME}"
        ;;

    logs)
        echo "=== BitNet Server Logs ==="
        sky logs "${CLUSTER_NAME}"
        ;;

    ssh)
        echo "=== SSH to BitNet Server ==="
        sky ssh "${CLUSTER_NAME}"
        ;;
esac
