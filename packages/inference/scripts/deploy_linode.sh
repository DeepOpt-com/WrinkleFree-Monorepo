#!/bin/bash
# Deploy BitNet inference to Linode G8 Dedicated instance
# Usage: ./deploy_linode.sh [instance-name]

set -euo pipefail

# Configuration
INSTANCE_NAME="${1:-bitnet-inference}"
INSTANCE_TYPE="g8-dedicated-8-4"  # 4 vCPU Zen5 AVX512, 8GB RAM, $0.14/hr
REGION="${LINODE_REGION:-us-ord}"  # Chicago (has G8)
IMAGE="linode/ubuntu24.04"
INSTALL_DIR="/opt/wrinklefree"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
step() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$INFERENCE_DIR")")"

# Check linode-cli is installed
if ! command -v linode-cli &> /dev/null; then
    err "linode-cli not found. Install with: pip install linode-cli && linode-cli configure"
fi

step "Creating Linode Instance"
log "Instance: $INSTANCE_NAME"
log "Type: $INSTANCE_TYPE (4 vCPU Zen5 AVX512, 8GB RAM)"
log "Region: $REGION"
log "Image: $IMAGE"

# Check if instance already exists
EXISTING=$(linode-cli linodes list --label "$INSTANCE_NAME" --json 2>/dev/null | jq -r '.[0].id // empty')
if [ -n "$EXISTING" ]; then
    warn "Instance '$INSTANCE_NAME' already exists (ID: $EXISTING)"
    read -p "Delete and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Deleting existing instance..."
        linode-cli linodes delete "$EXISTING"
        sleep 5
    else
        log "Using existing instance..."
        INSTANCE_ID="$EXISTING"
    fi
fi

# Create instance if needed
if [ -z "${INSTANCE_ID:-}" ]; then
    log "Creating new instance..."

    # Get SSH key
    SSH_KEY=""
    for key_file in ~/.ssh/id_ed25519.pub ~/.ssh/id_rsa.pub; do
        if [ -f "$key_file" ]; then
            SSH_KEY=$(cat "$key_file")
            break
        fi
    done
    [ -z "$SSH_KEY" ] && err "No SSH public key found in ~/.ssh/"

    # Generate random root password (Linode requires this even with SSH keys)
    ROOT_PASS=$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)

    # Create instance
    RESULT=$(linode-cli linodes create \
        --type "$INSTANCE_TYPE" \
        --region "$REGION" \
        --image "$IMAGE" \
        --label "$INSTANCE_NAME" \
        --root_pass "$ROOT_PASS" \
        --authorized_keys "$SSH_KEY" \
        --json)

    INSTANCE_ID=$(echo "$RESULT" | jq -r '.[0].id')
    log "Created instance ID: $INSTANCE_ID"
    log "Root password: $ROOT_PASS (use SSH key instead)"
fi

step "Waiting for Instance to Boot"
while true; do
    STATUS=$(linode-cli linodes view "$INSTANCE_ID" --json | jq -r '.[0].status')
    log "Status: $STATUS"
    if [ "$STATUS" = "running" ]; then
        break
    fi
    sleep 5
done

# Get IP address
PUBLIC_IP=$(linode-cli linodes view "$INSTANCE_ID" --json | jq -r '.[0].ipv4[0]')
log "Public IP: $PUBLIC_IP"

step "Waiting for SSH"
while ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@"$PUBLIC_IP" exit 2>/dev/null; do
    log "Waiting for SSH..."
    sleep 5
done
log "SSH ready!"

step "Syncing Code"
log "Syncing repository to $INSTALL_DIR..."

# Use rsync with git-tracked files
cd "$REPO_ROOT"
git ls-files -z | rsync -avz --files-from=- --from0 \
    -e "ssh -o StrictHostKeyChecking=no" \
    . "root@${PUBLIC_IP}:${INSTALL_DIR}/"

log "Synced $(git ls-files | wc -l) files"

step "Running Remote Setup"
ssh -o StrictHostKeyChecking=no "root@${PUBLIC_IP}" "INSTALL_DIR=$INSTALL_DIR bash $INSTALL_DIR/packages/inference/deploy/setup_remote.sh"

step "Deployment Complete!"
echo ""
echo -e "${GREEN}=== Access Your Services ===${NC}"
echo -e "Streamlit UI:  ${CYAN}http://${PUBLIC_IP}:7860${NC}"
echo -e "API Endpoint:  ${CYAN}http://${PUBLIC_IP}:30000/v1/chat/completions${NC}"
echo ""
echo -e "${GREEN}=== SSH Access ===${NC}"
echo -e "ssh root@${PUBLIC_IP}"
echo ""
echo -e "${GREEN}=== Useful Commands ===${NC}"
echo "pm2 status           - Check service status"
echo "pm2 logs             - View all logs"
echo "pm2 restart all      - Restart services"
echo ""
echo -e "${YELLOW}Cost: ~\$0.14/hour${NC}"
echo -e "${YELLOW}Remember to delete when done: linode-cli linodes delete $INSTANCE_ID${NC}"
