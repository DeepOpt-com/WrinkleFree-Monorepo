#!/usr/bin/env bash
# Nebius Cloud Setup for WrinkleFree
#
# This script guides you through setting up Nebius credentials for SkyPilot.
# Nebius offers H100 GPUs at $1.99/hr (cheapest available).
#
# Based on: https://docs.skypilot.co/en/latest/cloud-setup/cloud-permissions/nebius.html
#
# Prerequisites:
#   - Nebius account (https://nebius.com)
#   - Browser access for OAuth login

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Nebius Cloud Setup for WrinkleFree${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# Check if nebius CLI is installed
if ! command -v nebius &> /dev/null; then
    if [ -f "$HOME/.nebius/bin/nebius" ]; then
        export PATH="$HOME/.nebius/bin:$PATH"
    else
        echo -e "${YELLOW}Installing Nebius CLI...${NC}"
        curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
        export PATH="$HOME/.nebius/bin:$PATH"
        echo ""
    fi
fi

# ============================================================================
# Step 1: Login to Nebius
# ============================================================================
echo -e "${BLUE}Step 1: Login to Nebius${NC}"
echo "This will open a browser for OAuth login."
echo ""

if [ ! -f "$HOME/.nebius/config.yaml" ]; then
    nebius profile create
else
    echo "Already logged in. To re-login: nebius profile create"
fi

# ============================================================================
# Step 2: Get Tenant ID
# ============================================================================
echo ""
echo -e "${BLUE}Step 2: Get Tenant ID${NC}"
echo ""

TENANT_ID=$(nebius iam whoami --format json 2>/dev/null | jq -r '.user_profile.tenants[0].tenant_id' || echo "")
if [ -z "$TENANT_ID" ] || [ "$TENANT_ID" = "null" ]; then
    # Try alternate method
    TENANT_ID=$(nebius iam tenant list --format json 2>/dev/null | jq -r '.[0].metadata.id' || echo "")
fi

if [ -n "$TENANT_ID" ] && [ "$TENANT_ID" != "null" ]; then
    echo "Tenant ID: $TENANT_ID"
else
    echo -e "${YELLOW}Could not auto-detect tenant ID${NC}"
    echo "Run: nebius iam whoami"
    read -p "Enter Tenant ID: " TENANT_ID
fi

# Save tenant ID for SkyPilot
mkdir -p ~/.nebius
echo "$TENANT_ID" > ~/.nebius/NEBIUS_TENANT_ID.txt
echo "Saved to ~/.nebius/NEBIUS_TENANT_ID.txt"

# ============================================================================
# Step 3: Get Project ID
# ============================================================================
echo ""
echo -e "${BLUE}Step 3: Get Project ID${NC}"
echo ""

PROJECT_ID=$(nebius iam project list --parent-id "$TENANT_ID" --format json 2>/dev/null | jq -r '.[0].metadata.id' || echo "")
if [ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "null" ]; then
    echo "Project ID: $PROJECT_ID"
else
    echo -e "${YELLOW}Could not auto-detect project ID${NC}"
    echo "Run: nebius iam project list --parent-id $TENANT_ID"
    read -p "Enter Project ID: " PROJECT_ID
fi

# ============================================================================
# Step 4: Create Service Account
# ============================================================================
echo ""
echo -e "${BLUE}Step 4: Create Service Account${NC}"
echo ""

SA_NAME="skypilot-sa"

# Check if SA already exists
SA_ID=$(nebius iam service-account get-by-name --parent-id "$PROJECT_ID" --name "$SA_NAME" --format json 2>/dev/null | jq -r '.metadata.id' || echo "")

if [ -z "$SA_ID" ] || [ "$SA_ID" = "null" ]; then
    echo "Creating service account: $SA_NAME"
    SA_RESULT=$(nebius iam service-account create --parent-id "$PROJECT_ID" --name "$SA_NAME" --format json 2>&1)
    SA_ID=$(echo "$SA_RESULT" | jq -r '.metadata.id' || echo "")

    if [ -z "$SA_ID" ] || [ "$SA_ID" = "null" ]; then
        echo -e "${RED}Failed to create service account${NC}"
        echo "$SA_RESULT"
        echo ""
        echo "You may need to create it manually in the Nebius Console:"
        echo "  https://console.eu-north1.nebius.cloud/folders/$PROJECT_ID/iam/service-accounts"
        read -p "Enter Service Account ID: " SA_ID
    fi
else
    echo "Service account already exists: $SA_NAME"
fi
echo "Service Account ID: $SA_ID"

# ============================================================================
# Step 5: Generate Auth Key (for SkyPilot)
# ============================================================================
echo ""
echo -e "${BLUE}Step 5: Generate Auth Public Key${NC}"
echo ""

mkdir -p ~/.nebius

# Use auth-public-key generate (correct command for SkyPilot)
echo "Generating credentials for SkyPilot..."
nebius iam auth-public-key generate \
    --service-account-id "$SA_ID" \
    --output ~/.nebius/credentials.json \
    2>&1 || {
        echo -e "${RED}Failed to generate auth key${NC}"
        echo ""
        echo "This may be a permissions issue. Try:"
        echo "1. Go to Nebius Console → IAM → Service Accounts"
        echo "2. Select '$SA_NAME'"
        echo "3. Add role: 'iam.serviceAccounts.tokenCreator' to your user"
        echo ""
        echo "Or use IAM token method instead (temporary, expires in 12h):"
        echo "  nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt"
        exit 1
    }

echo "Credentials saved to ~/.nebius/credentials.json"

# ============================================================================
# Step 6: Verify Setup
# ============================================================================
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Nebius Setup Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Files created:"
echo "  ~/.nebius/credentials.json     (SkyPilot auth)"
echo "  ~/.nebius/NEBIUS_TENANT_ID.txt (Tenant ID)"
echo ""
echo "Verify with:"
echo "  sky check nebius"
echo ""
echo "To use Nebius in SkyPilot:"
echo "  resources:"
echo "    cloud: nebius"
echo "    accelerators: H100:1"
echo ""
echo -e "${YELLOW}Note: Service Accounts are region-specific.${NC}"
echo "Default region: eu-north1"

# ============================================================================
# Alternative: IAM Token Method (if auth-public-key fails)
# ============================================================================
echo ""
echo -e "${BLUE}Alternative: Quick Setup with IAM Token${NC}"
echo "If the above failed, you can use a temporary token (expires in 12h):"
echo ""
echo "  nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt"
echo "  sky check nebius"
