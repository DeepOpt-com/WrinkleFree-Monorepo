#!/bin/bash
# Build and push WrinkleFree training Docker image to Google Artifact Registry (GAR)
#
# Usage:
#   ./scripts/build-image.sh           # Build and push with auto-generated tag
#   ./scripts/build-image.sh --no-push # Build only, don't push
#   ./scripts/build-image.sh v1.0.0    # Build with specific tag

set -e

PROJECT_ID="wrinklefree-481904"
GAR_REGION="us"
GAR_REPO="wf-train"
IMAGE_NAME="${GAR_REGION}-docker.pkg.dev/${PROJECT_ID}/${GAR_REPO}/wf-train"

# Parse arguments
NO_PUSH=false
CUSTOM_TAG=""
for arg in "$@"; do
    case $arg in
        --no-push)
            NO_PUSH=true
            ;;
        *)
            CUSTOM_TAG="$arg"
            ;;
    esac
done

# Generate tag: YYYYMMDD-<git-short-hash> or use custom
if [ -n "$CUSTOM_TAG" ]; then
    TAG="$CUSTOM_TAG"
else
    TAG=$(date +%Y%m%d)-$(git rev-parse --short HEAD 2>/dev/null || echo "local")
fi

# Navigate to deployer directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Building WrinkleFree training image..."
echo "  Registry: Google Artifact Registry (GAR)"
echo "  Image: ${IMAGE_NAME}:${TAG}"
echo "  Context: $(pwd)"

# Build the image
docker build \
    -f docker/Dockerfile.train \
    -t "${IMAGE_NAME}:${TAG}" \
    -t "${IMAGE_NAME}:latest" \
    --build-arg WF_VERSION="${TAG}" \
    .

echo ""
echo "Build complete!"
echo "  Tagged: ${IMAGE_NAME}:${TAG}"
echo "  Tagged: ${IMAGE_NAME}:latest"

if [ "$NO_PUSH" = true ]; then
    echo ""
    echo "Skipping push (--no-push specified)"
    echo "To push manually:"
    echo "  docker push ${IMAGE_NAME}:${TAG}"
    echo "  docker push ${IMAGE_NAME}:latest"
    exit 0
fi

# Push to GAR
echo ""
echo "Pushing to Google Artifact Registry..."

# Check if authenticated
if ! gcloud auth print-access-token &>/dev/null; then
    echo "Error: Not authenticated with GCP. Run:"
    echo "  gcloud auth login"
    echo "  gcloud auth configure-docker ${GAR_REGION}-docker.pkg.dev"
    exit 1
fi

# Configure Docker for GAR (idempotent)
gcloud auth configure-docker ${GAR_REGION}-docker.pkg.dev --quiet

docker push "${IMAGE_NAME}:${TAG}"
docker push "${IMAGE_NAME}:latest"

echo ""
echo "Successfully pushed to GAR:"
echo "  ${IMAGE_NAME}:${TAG}"
echo "  ${IMAGE_NAME}:latest"
echo ""
echo "Your SkyPilot YAMLs are already configured to use this image."
