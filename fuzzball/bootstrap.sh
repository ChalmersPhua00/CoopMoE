#!/bin/bash
# Runs inside the Fuzzball jobs that driver.py renders. driver.py fills in every double-brace
# placeholder. COOPMOE_FILES, COOPMOE_WORK and COOPMOE_PY (an existing Python with the
# requirements, instead of building .venv) are optional overrides.
#
#   stage   (stage.py, one node, once per model) download the checkpoint and build
#           the expert store on the team's NFS volume
#   run     (rank 0, PGX-A) copy the store from NFS to this box's local disk, start box 0's
#           engine, run the requests through the scheduler, print the results
#   engine  (rank 1, PGX-B) copy the store from NFS to this box's local disk, serve as box 1
set -euo pipefail
FILES="${COOPMOE_FILES:-/opt/coopmoe}"
WORK="${COOPMOE_WORK:-/tmp/coopmoe}"
NFS_STORE={{NFS_STORE}}   # the one persistent copy, on the team's NFS volume
STORE={{STORE}}           # what the engine reads: this box's local copy (or NFS_STORE with --read-from nfs)
rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK"
python3 - "$FILES/code.tgz.b64" <<'PY'
import base64, io, sys, tarfile
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(open(sys.argv[1]).read())), mode="r:gz") as t:
    t.extractall(**({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
PY
if [ -n "${COOPMOE_PY:-}" ]; then
  PY="$COOPMOE_PY"
else
  bash setup.sh > setup.log 2>&1 || { cat setup.log; exit 1; }
  source env.sh
fi
log() { echo "[bootstrap $1 $(hostname) $(date +%H:%M:%S)] ${*:2}"; }
log "$1" "code unpacked in $WORK"

# NFS -> this box's local disk, 8 files at a time, into $STORE.partial and renamed when
# complete. Logs where the local disk really is (unknown U2) and the copy rate (U3).
copy_store() {
  if [ ! -f "$NFS_STORE/index.json" ]; then
    log "$1" "no expert store at $NFS_STORE: build it once with stage.py"
    exit 1
  fi
  if [ "$STORE" = "$NFS_STORE" ]; then
    log "$1" "reading experts directly from NFS: $NFS_STORE"
    return
  fi
  mkdir -p "$(dirname "$STORE")"
  log "$1" "local disk: $(findmnt -no SOURCE,FSTYPE -T "$(dirname "$STORE")" | tr -s ' ') | $(df -h --output=size,avail "$(dirname "$STORE")" | tail -1 | tr -s ' ')"
  if [ -f "$STORE/index.json" ]; then
    log "$1" "local store already present: $STORE"
    return
  fi
  local t0 t bytes
  t0=$(date +%s)
  rm -rf "$STORE.partial" && mkdir -p "$STORE.partial"
  (cd "$NFS_STORE" && find . -type f -print0 | xargs -0 -P 8 -I@ cp --parents @ "$STORE.partial/")
  mv "$STORE.partial" "$STORE"
  t=$(( $(date +%s) - t0 )); bytes=$(du -sb "$STORE" | cut -f1)
  log "$1" "copied $NFS_STORE -> $STORE: $(awk "BEGIN{printf \"%.1f GB in %d s, %.2f GB/s\", $bytes/1e9, $t, $bytes/1e9/($t>0?$t:1)}")"
}

case "$1" in
  stage)
    exec "$PY" -m coopmoe stage --model {{MODEL}} --store "$NFS_STORE" --hf-dir {{HF_DIR}} ;;
  engine)
    copy_store engine
    exec "$PY" -m coopmoe engine --config {{CONFIG}} --rank 1 --host {{BIND1}} --set {{SETS}} ;;
  run)
    copy_store run
    exec "$PY" -m coopmoe run --config {{CONFIG}} --model {{MODEL}} --store "$STORE" \
        --requests "$FILES/requests.json" --out "{{RESULTS}}/${FB_WORKFLOW_ID:-$(date +%Y%m%d-%H%M%S)}" \
        --boxes {{BOXES}} --emit {{RUN_ARGS}} ;;
  *)
    echo "usage: bootstrap.sh stage|run|engine" >&2
    exit 2 ;;
esac
