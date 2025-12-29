#!/bin/bash
# Wait for inference server to be healthy
#
# Usage:
#   ./scripts/wait_healthy.sh http://localhost:8080
#   ./scripts/wait_healthy.sh http://localhost:8080 120  # 120s timeout

set -e

URL="${1:-http://localhost:8080}"
TIMEOUT="${2:-120}"
INTERVAL=5

echo "Waiting for $URL to be healthy (timeout: ${TIMEOUT}s)..."

start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))

    if [ $elapsed -ge $TIMEOUT ]; then
        echo "ERROR: Timeout waiting for $URL after ${TIMEOUT}s"
        exit 1
    fi

    response=$(curl -s -o /dev/null -w "%{http_code}" "$URL/health" 2>/dev/null || echo "000")
    if [ "$response" = "200" ]; then
        echo "Server is healthy after ${elapsed}s"
        exit 0
    fi

    echo "Waiting... (Status: $response, ${elapsed}s elapsed)"
    sleep $INTERVAL
done
