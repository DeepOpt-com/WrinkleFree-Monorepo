#!/bin/bash
# Setup GCS credentials for gsutil
# This script configures gsutil to use the service account credentials
# Run once after cloning the repo, or when credentials change

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYER_DIR="$(dirname "$SCRIPT_DIR")"
CREDENTIALS_DIR="$DEPLOYER_DIR/credentials"
SERVICE_ACCOUNT_FILE="$CREDENTIALS_DIR/gcp-service-account.json"
BOTO_FILE="$HOME/.boto"

# Check if service account exists
if [[ ! -f "$SERVICE_ACCOUNT_FILE" ]]; then
    echo "❌ Service account not found: $SERVICE_ACCOUNT_FILE"
    echo "   Please create the service account file first."
    echo "   See: packages/deployer/credentials/README.md"
    exit 1
fi

# Create or update .boto file
echo "📝 Configuring gsutil credentials..."
cat > "$BOTO_FILE" << EOF
[Credentials]
gs_service_key_file = $SERVICE_ACCOUNT_FILE

[Boto]
https_validate_certificates = True

[GSUtil]
default_project_id = wrinklefree-481904
EOF

echo "✅ Created $BOTO_FILE"

# Verify it works
echo "🔍 Testing GCS access..."
if gsutil ls gs://wrinklefree-checkpoints/ > /dev/null 2>&1; then
    echo "✅ GCS access verified!"
else
    echo "❌ GCS access failed. Check service account permissions."
    exit 1
fi

echo ""
echo "🎉 GCS setup complete! gsutil is now configured."
