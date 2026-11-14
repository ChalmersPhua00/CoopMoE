#!/usr/bin/env bash
# Linux / PGX setup: creates .venv inside the project.
# - Inside an NGC PyTorch container, torch is already there (built for GB10), so the venv
#   reuses it (--system-site-packages) and only the small deps are installed.
# - On a bare box, torch comes from $TORCH_INDEX (default: CUDA 13.0 wheels, which ship aarch64).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

if [ ! -x "$PY" ]; then
  if python3 -c "import torch" 2>/dev/null; then
    python3 -m venv --system-site-packages .venv
  else
    python3 -m venv .venv
  fi
fi
"$PY" -m pip install --upgrade pip
if ! "$PY" -c "import torch" 2>/dev/null; then
  "$PY" -m pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
fi
"$PY" -m pip install -r requirements.txt
"$PY" - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "cuda", ok, torch.cuda.get_device_name(0) if ok else "")
if ok:
    print("compute capability", torch.cuda.get_device_capability(0))
EOF
