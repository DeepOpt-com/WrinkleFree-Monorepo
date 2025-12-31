#!/bin/bash
# Deploy BitNet inference to Vultr High Frequency instances.
#
# Uses the Rust server with native C++ inference (NO PYTHON).
# Compiles with -march=native for maximum SIMD optimization (AVX512).
#
# Usage:
#   ./scripts/deploy_vultr.sh <instance-ip> [ssh-key-path]
#
# Prerequisites:
#   - Vultr instance running Ubuntu 24.04
#   - SSH access configured
#   - Instance has at least 8GB RAM for compilation
#
# The script will:
#   1. Install build dependencies (Rust, clang, cmake)
#   2. Clone/sync the repository
#   3. Build llama.cpp with -march=native (AVX512)
#   4. Build Rust gateway with native-inference feature
#   5. Download and convert model to GGUF format
#   6. Start server with pm2

set -euo pipefail

INSTANCE_IP="${1:-}"
SSH_KEY="${2:-$HOME/.ssh/id_ed25519}"

if [[ -z "$INSTANCE_IP" ]]; then
    echo "Usage: $0 <instance-ip> [ssh-key-path]"
    echo ""
    echo "Example:"
    echo "  $0 45.76.123.45"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MONOREPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY"
SSH_CMD="ssh $SSH_OPTS root@$INSTANCE_IP"
SCP_CMD="scp $SSH_OPTS"
REMOTE_DIR="/opt/wrinklefree"

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

log "=== Vultr Deployment (Rust Server) ==="
log "Instance: $INSTANCE_IP"
log "Remote dir: $REMOTE_DIR"

# 1. Sync code to remote (with submodules)
log "Syncing code to remote..."
cd "$MONOREPO_ROOT"

# Ensure submodules are initialized locally
git submodule update --init --recursive packages/inference/extern/sglang-bitnet

# Use rsync to sync (faster than scp for subsequent syncs)
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude 'target' \
    --exclude 'node_modules' \
    --exclude 'models' \
    --exclude '*.bin' \
    --exclude '*.gguf' \
    -e "ssh $SSH_OPTS" \
    "$MONOREPO_ROOT/" "root@$INSTANCE_IP:$REMOTE_DIR/"

# 2. Run remote setup script
log "Running remote setup..."
$SSH_CMD "chmod +x $REMOTE_DIR/packages/inference/deploy/setup_remote_rust.sh && $REMOTE_DIR/packages/inference/deploy/setup_remote_rust.sh"

# 3. Start services with pm2
log "Starting services..."
$SSH_CMD "cd $REMOTE_DIR && pm2 delete all 2>/dev/null || true && pm2 start packages/inference/deploy/ecosystem_rust.config.js && pm2 save"

# 4. Test the API
log "Waiting for server to start..."
sleep 10

log "Testing API..."
RESPONSE=$($SSH_CMD "curl -s http://localhost:30000/health" || echo "failed")
if [[ "$RESPONSE" == *"ok"* ]]; then
    log "Server is healthy!"
else
    log "WARNING: Server may not be ready yet. Check logs with: ssh root@$INSTANCE_IP 'pm2 logs'"
fi

log ""
log "=== Deployment Complete ==="
log "API endpoint: http://$INSTANCE_IP:30000/v1/chat/completions"
log "Streamlit UI: http://$INSTANCE_IP:7860"
log ""
log "Commands:"
log "  View logs:    ssh root@$INSTANCE_IP 'pm2 logs'"
log "  Check status: ssh root@$INSTANCE_IP 'pm2 list'"
log "  Test API:     curl http://$INSTANCE_IP:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
