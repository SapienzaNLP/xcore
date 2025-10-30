#!/usr/bin/env bash
set -euo pipefail

# --- sanity checks -----------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: 'conda' not found in PATH. Please install Miniconda/Anaconda and try again." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "${CONDA_BASE:-}" ] || [ ! -d "$CONDA_BASE" ]; then
  echo "Error: could not determine conda base (CONDA_BASE)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- create env --------------------------------------------------------------
read -rp "Enter environment name: " ENV_NAME
if [ -z "${ENV_NAME:-}" ]; then
  echo "Error: environment name is required." >&2
  exit 1
fi

read -rp "Enter python version (default 3.12): " PYTHON_VERSION
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
conda activate "$ENV_NAME"

# --- prefer OpenBLAS over MKL to avoid ITT symbol issues --------------------
# This replaces MKL with OpenBLAS in the env to prevent iJIT_NotifyEvent errors
# when importing PyTorch CPU builds.
conda install -y "blas=*=*openblas" libopenblas || true
# Clean out MKL/Intel OpenMP if they still linger (best-effort)
conda remove -y mkl mkl-service intel-openmp tbb tbb-devel --force || true

# --- install PyTorch ---------------------------------------------------------
read -rp "Enter CUDA version (e.g. '12.1' or '12.4'); leave blank for CPU-only: " CUDA_VERSION
read -rp "Enter PyTorch version (e.g. '2.5.1'; leave blank for latest): " PYTORCH_VERSION

# upgrade pip early
python -m pip install --upgrade pip

# Ensure we don't mix conda and pip torch builds
python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
conda remove -y pytorch pytorch-mutex cpuonly pytorch-cuda cudatoolkit >/dev/null 2>&1 || true

normalize_cuda_label() {
  # Map user input (12.x) to the closest available PyTorch wheel label
  case "$1" in
    "" ) echo "" ;;
    12.0|12.1|12.2|12.3|12.1.*|12.2.*|12.3.*) echo "cu121" ;;
    12.4|12.4.*) echo "cu124" ;;
    11.8|11.8.*) echo "cu118" ;;
    * ) echo "unsupported" ;;
  esac
}

CUDA_LABEL="$(normalize_cuda_label "${CUDA_VERSION:-}")"

if [ -z "${CUDA_VERSION:-}" ]; then
  # CPU-only via pip wheels to avoid MKL issues
  if [ -n "${PYTORCH_VERSION:-}" ]; then
    python -m pip install --force-reinstall --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu "torch==${PYTORCH_VERSION}"
  else
    python -m pip install --force-reinstall --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch
  fi
else
  if [ "$CUDA_LABEL" = "unsupported" ]; then
    echo "Warning: CUDA version '$CUDA_VERSION' is not directly supported by PyTorch wheels. Falling back to cu121." >&2
    CUDA_LABEL="cu121"
  fi
  WHEEL_INDEX_URL="https://download.pytorch.org/whl/${CUDA_LABEL}"
  if [ -n "${PYTORCH_VERSION:-}" ]; then
    python -m pip install --force-reinstall --no-cache-dir \
      --index-url "$WHEEL_INDEX_URL" "torch==${PYTORCH_VERSION}"
  else
    python -m pip install --force-reinstall --no-cache-dir \
      --index-url "$WHEEL_INDEX_URL" torch
  fi
fi

# --- python deps -------------------------------------------------------------
if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
else
  echo "Warning: requirements.txt not found; skipping pip install." >&2
fi

# --- verify CUDA (if requested) ---------------------------------------------
if [ -n "${CUDA_VERSION:-}" ]; then
python - <<'PY'
import os, sys, torch
print("torch", torch.__version__, "built_for_cuda:", torch.version.cuda)
ok = torch.cuda.is_available()
print("torch.cuda.is_available():", ok)
if ok:
    print("CUDA device 0:", torch.cuda.get_device_name(0))
else:
    sys.stderr.write("ERROR: CUDA was requested but is not available. Check NVIDIA driver and wheel index (cu121/cu124).\n")
    sys.exit(1)
PY
fi

# --- sanity import check -----------------------------------------------------
python - <<'PY'
import torch, pytorch_lightning as pl
print("torch", torch.__version__)
print("pytorch_lightning", pl.__version__)
PY

# --- datasets download -------------------------------------------------------
read -rp "Download datasets to ./data now? [Y/n]: " DL_DATA
DL_DATA="${DL_DATA:-Y}"
if [[ "$DL_DATA" =~ ^[Yy]$ ]]; then
  bash ./download_datasets.sh
else
  echo "Skipping dataset download."
fi

python -m spacy download en_core_web_sm

echo "Setup complete for environment '$ENV_NAME'."