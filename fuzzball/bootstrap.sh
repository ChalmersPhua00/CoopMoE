#!/bin/bash
# Runs inside the Fuzzball jobs that driver.py renders. driver.py fills in every double-brace
# placeholder. COOPMOE_FILES, COOPMOE_WORK and COOPMOE_PY (an existing Python with the
# requirements, instead of building .venv) are optional overrides.
#
#   stage   (stage.py, one node, once per model) download the checkpoint and build
#           the expert store on the team's NFS volume
#   run     (box 0, PGX-A) find box 1, copy the store from NFS to this box's local disk, start
#           box 0's engine, send each request to its box, print the results
#   engine  (box 1, PGX-B) find box 0, copy the store from NFS to this box's local disk, serve
#           as box 1 until box 0 stops it
#
# run and engine are two single-node jobs (a multinode job cannot mount a node's own NVMe), so
# they find each other through the shared NFS volume and talk over the CX-7 link (rendezvous).
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
  # Directories first: parallel cp --parents processes race to create the same parent and lose
  # with "File exists".
  (cd "$NFS_STORE" && find . -type d -print0 | (cd "$STORE.partial" && xargs -0 mkdir -p))
  (cd "$NFS_STORE" && find . -type f -print0 | xargs -0 -P 8 -I@ cp --parents @ "$STORE.partial/")
  mv "$STORE.partial" "$STORE"
  t=$(( $(date +%s) - t0 )); bytes=$(du -sb "$STORE" | cut -f1)
  log "$1" "copied $NFS_STORE -> $STORE: $(awk "BEGIN{printf \"%.1f GB in %d s, %.2f GB/s\", $bytes/1e9, $t, $bytes/1e9/($t>0?$t:1)}")"
}

# The two boxes of a linked run find each other: each writes its CX-7 address under $RDV (on
# the NFS volume both jobs mount) and waits for the other's. A single-node job has the node's
# network; its CX-7 address is the local address the kernel routes to {{CX7_NET}} (the NFS
# server, on the CX-7 network 10.10.11.x). Sets BOXES (A:port,B:port) and ADDRS (the same as
# a JSON list).
RDV={{RDV}}
PORT={{PORT}}
rendezvous() {
  local me=$1 other=$((1 - $1)) ip oip t0
  ip=$(python3 -c 'import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect((sys.argv[1], 9))
print(s.getsockname()[0])' {{CX7_NET}})
  mkdir -p "$RDV"
  echo "$ip" > "$RDV/box$me.tmp" && mv "$RDV/box$me.tmp" "$RDV/box$me"
  log link "box $me: CX-7 address $ip (the route to {{CX7_NET}}); waiting for box $other"
  t0=$(date +%s)
  while [ ! -f "$RDV/box$other" ]; do
    if [ $(( $(date +%s) - t0 )) -gt {{RDV_TIMEOUT}} ]; then
      log link "box $other's job did not start within {{RDV_TIMEOUT}} s. Fuzzball has probably put its scratch volume on this node, so it waits for this job to end."
      echo "@@coopmoe same-node"
      exit 3
    fi
    sleep 2
    # A change of ours in the directory makes this NFS client look again: otherwise it can go
    # on believing for up to a minute that the other box's file is not there.
    touch "$RDV/.poke$me" && rm -f "$RDV/.poke$me"
  done
  read -r oip < "$RDV/box$other"
  log link "box $other: CX-7 address $oip"
  if [ "$oip" = "$ip" ]; then
    log link "both jobs are on the same node"
    echo "@@coopmoe same-node"
    exit 3
  fi
  if [ "$me" = 0 ]; then
    BOXES="$ip:$PORT,$oip:$PORT"; ADDRS="[\"$ip:$PORT\",\"$oip:$PORT\"]"
  else
    BOXES="$oip:$PORT,$ip:$PORT"; ADDRS="[\"$oip:$PORT\",\"$ip:$PORT\"]"
  fi
}

case "$1" in
  stage)
    exec "$PY" -m coopmoe stage --model {{MODEL}} --store "$NFS_STORE" --hf-dir {{HF_DIR}} ;;
  engine)
    rendezvous 1
    copy_store engine
    exec "$PY" -m coopmoe engine --config {{CONFIG}} --rank 1 --set "store_dir=$STORE" "cluster.addrs=$ADDRS" ;;
  run)
    rendezvous 0
    copy_store run
    exec "$PY" -m coopmoe run --config {{CONFIG}} --model {{MODEL}} --store "$STORE" \
        --requests "$WORK/requests.json" --out "{{RESULTS}}/${FB_WORKFLOW_ID:-$(date +%Y%m%d-%H%M%S)}" \
        --boxes "$BOXES" --emit {{RUN_ARGS}} ;;
  *)
    echo "usage: bootstrap.sh stage|run|engine" >&2
    exit 2 ;;
esac
