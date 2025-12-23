#!/bin/bash
# RunPod Influence Function Test Setup
# Run with: curl -fsSL https://raw.githubusercontent.com/DeepOpt-com/WrinkleFree/main/runpod_influence_test.sh | bash

set -e
echo "=================================================================="
echo " WrinkleFree Influence Function Test"
echo " Testing influence-based data selection with mixed dataset"
echo "=================================================================="

# Install uv
if ! command -v uv &> /dev/null; then
    echo "[1/7] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/7] uv already installed ✓"
fi

# Clone repo
cd /workspace
if [ ! -d "WrinkleFree" ]; then
    echo "[2/7] Cloning WrinkleFree repository..."
    git clone --recursive https://github.com/DeepOpt-com/WrinkleFree.git 2>/dev/null || \
    git clone --recursive https://github.com/levelsup/WrinkleFree.git
else
    echo "[2/7] WrinkleFree repo exists ✓"
fi

cd WrinkleFree

# Update submodules
echo "[3/7] Updating submodules..."
git submodule update --init --recursive 2>&1 | grep -v "Submodule" || true

# Pull latest changes
echo "[4/7] Pulling latest code with debug logging..."
cd WrinkleFree-1.58Quant
git pull origin master --quiet 2>&1 | tail -3
cd ../WrinkleFree-CheaperTraining
git pull origin main --quiet 2>&1 | tail -3
cd ..

# Install dependencies
echo "[5/7] Installing Python dependencies (this may take 2-3 minutes)..."
cd WrinkleFree-1.58Quant
uv sync --quiet
echo "  - WrinkleFree-1.58Quant dependencies installed ✓"

cd ../WrinkleFree-CheaperTraining
uv sync --quiet
echo "  - WrinkleFree-CheaperTraining dependencies installed ✓"

cd ../WrinkleFree-1.58Quant

echo "[6/7] Environment setup complete ✓"
echo ""
echo "=================================================================="
echo " Starting Stage 2 Training"
echo "=================================================================="
echo " Config: SmolLM2-135M + Mixed Dataset (5 sources)"
echo " Influence: Update every 10 steps (5 total updates)"
echo " Duration: 50 steps (~10-15 minutes)"
echo "=================================================================="
echo ""

# Run training
echo "[7/7] Launching training..."
sleep 2

uv run python scripts/train.py \
  model=smollm2_135m \
  training=stage2_pretrain \
  data=mixed_pretrain \
  training.max_steps=50 \
  training.influence.update_interval=10 \
  training.logging.log_interval=5 \
  training.logging.wandb.enabled=true \
  training.logging.wandb.project=wrinklefree-influence-test \
  training.logging.wandb.name="influence-test-$(date +%s)" || {
    echo ""
    echo "❌ Training failed! Check logs above for errors."
    exit 1
}

echo ""
echo "=================================================================="
echo " ✅ Training Complete!"
echo "=================================================================="
echo " Check WandB for results: https://wandb.ai"
echo " Look for project: wrinklefree-influence-test"
echo "=================================================================="
