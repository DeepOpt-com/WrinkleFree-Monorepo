#!/bin/bash
# Build and push WrinkleFree training Docker image to GCR
#
# Usage:
#   ./scripts/build-image.sh           # Build and push with auto-generated tag
#   ./scripts/build-image.sh --no-push # Build only, don't push
#   ./scripts/build-image.sh v1.0.0    # Build with specific tag

set -e

PROJECT_ID="wrinklefree-481904"
IMAGE_NAME="gcr.io/${PROJECT_ID}/wf-train"

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

# Navigate to monorepo root (parent of WrinkleFree-Deployer)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Building WrinkleFree training image..."
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

# Push to GCR
echo ""
echo "Pushing to GCR..."

# Check if authenticated
if ! gcloud auth print-access-token &>/dev/null; then
    echo "Error: Not authenticated with GCP. Run:"
    echo "  gcloud auth login"
    echo "  gcloud auth configure-docker"
    exit 1
fi

docker push "${IMAGE_NAME}:${TAG}"
docker push "${IMAGE_NAME}:latest"

echo ""
echo "Successfully pushed:"
echo "  ${IMAGE_NAME}:${TAG}"
echo "  ${IMAGE_NAME}:latest"
echo ""
echo "Update train.yaml to use this image:"
echo "  resources:"
echo "    image_id: docker:${IMAGE_NAME}:${TAG}"
