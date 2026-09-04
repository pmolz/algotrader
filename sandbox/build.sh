#!/usr/bin/env bash
# Build the sandbox image. Run from the repo root (build context must include
# the algotrader package). Re-run whenever algotrader/ changes.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${SANDBOX_IMAGE:-algotrader-sandbox:latest}"
echo "Building $IMAGE ..."
docker build -f sandbox/Dockerfile -t "$IMAGE" .
echo "Done. Image: $IMAGE"
