# Source this ("source env.sh") before running anything on Linux / the PGX boxes.
# Keeps caches and temp files inside the project folder.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="$ROOT/.cache"
mkdir -p "$CACHE/tmp"
export PIP_CACHE_DIR="$CACHE/pip"
export TMPDIR="$CACHE/tmp"
export CUDA_CACHE_PATH="$CACHE/cuda"
export TORCH_HOME="$CACHE/torch"
export TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export PYTHONPYCACHEPREFIX="$CACHE/pycache"
export HF_HOME="${HF_HOME:-$CACHE/hf}"   # point at the data volume for the 235 GB download
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/.venv/bin/python"
