#!/bin/bash
# Runs inside each baseline job (fuzzball/baseline.fz); driver.py fills in every double-brace
# placeholder. One box: unpack the payload, install transformers next to the image's torch,
# get the original checkpoint onto this box's NVMe, run baseline.py.
set -euo pipefail
WORK=/tmp/baseline
HF_DIR={{HF_DIR}}        # the original checkpoint on the model's NFS volume, downloaded once
CKPT={{LOCAL}}/hf        # this box's copy, on its own NVMe
log() { echo "[bootstrap $(hostname -f) $(date +%H:%M:%S)] $*"; }

rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK"
base64 -d /opt/baseline/payload.b64 | tar xz
python3 -m venv --system-site-packages .venv   # reuses the image's torch (built for the GB10)
.venv/bin/pip install -q -r requirements.txt

# Two baseline jobs may start together: the first to create $HF_DIR.lock downloads, the other
# waits for $HF_DIR/.complete.
if [ ! -f "$HF_DIR/.complete" ]; then
  mkdir -p "$(dirname "$HF_DIR")"
  if mkdir "$HF_DIR.lock" 2>/dev/null; then
    log "downloading {{MODEL}} -> $HF_DIR (once per volume)"
    if ! .venv/bin/python - {{MODEL}} "$HF_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.jinja"])
PY
    then rmdir "$HF_DIR.lock"; exit 1; fi
    touch "$HF_DIR/.complete" && rmdir "$HF_DIR.lock"
  else
    log "another job is downloading to $HF_DIR (delete $HF_DIR.lock if it died); waiting"
    while [ ! -f "$HF_DIR/.complete" ]; do sleep 30; done
  fi
fi

log "local disk: $(findmnt -no SOURCE,FSTYPE -T {{LOCAL}} | tr -s ' ') | $(df -h --output=size,avail {{LOCAL}} | tail -1 | tr -s ' ')"
t0=$(date +%s)
rm -rf "$CKPT" && mkdir -p "$CKPT"
(cd "$HF_DIR" && find . -maxdepth 1 -type f ! -name .complete -print0 | xargs -0 -P 8 -I@ cp @ "$CKPT/")
t=$(( $(date +%s) - t0 )); bytes=$(du -sb "$CKPT" | cut -f1)
log "copied $HF_DIR -> $CKPT: $(awk "BEGIN{printf \"%.1f GB in %d s, %.2f GB/s\", $bytes/1e9, $t, $bytes/1e9/($t>0?$t:1)}")"

exec .venv/bin/python baseline.py --checkpoint "$CKPT" --offload {{LOCAL}}/offload --requests requests.json \
    --box {{BOX}} --concurrency {{CONCURRENCY}} --memory {{MEMORY}}
