#!/bin/bash
# One-time setup. Run on the LANTA FRONTEND (login node): compute nodes have no internet.
#   bash lanta/setup_env.sh
# Creates the conda env, installs packages, and downloads Qwen3-ASR-0.6B-hf into the project HF cache.
set -euo pipefail
source "$(dirname "$0")/config.sh"

module load Mamba/23.11.0-0
mkdir -p "$(dirname "$ENV_PREFIX")" "$HF_HOME"

export PYTHONNOUSERSITE=1                     # never pick up ~/.local packages
PY="$ENV_PREFIX/bin/python"
if [ ! -x "$PY" ]; then                      # missing, or a half-created env from an earlier failed run
  [ -d "$ENV_PREFIX" ] && { echo "Removing incomplete env $ENV_PREFIX"; rm -rf "$ENV_PREFIX"; }
  conda create -y -p "$ENV_PREFIX" python=3.11
fi
"$PY" -V

# Always "$PY -m pip": whatever 'pip' is first on PATH can belong to another Python.
# PyTorch CUDA 12.6 wheels (bundle their own CUDA runtime; LANTA A100 drivers support CUDA 12).
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
# transformers 5.22.2 is the version the training script was tested with (Qwen3-ASR classes are in it).
"$PY" -m pip install "transformers==5.22.2" accelerate peft soundfile librosa pandas numpy jiwer tensorboard "huggingface_hub[cli]"

"$PY" - <<'EOF'
import sys, torch, transformers, pandas, numpy, peft
print("python", sys.executable)
from transformers import Qwen3ASRForConditionalGeneration  # noqa: F401  (fails here if the version is too old)
print("torch", torch.__version__, "| transformers", transformers.__version__)
EOF

# Download the base model into $HF_HOME (the sbatch jobs run with HF_HUB_OFFLINE=1).
"$ENV_PREFIX/bin/hf" download "$BASE_MODEL"
echo "Done. Env: $ENV_PREFIX   HF cache: $HF_HOME"
