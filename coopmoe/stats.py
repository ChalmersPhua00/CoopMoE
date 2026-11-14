"""Per-box counters reported to the benchmark: where each routed expert came from, what the
box did for its peer, admissions from the peer, bytes moved, and where time went."""
from __future__ import annotations

import threading
from collections import Counter, defaultdict

# Where a routed expert was served from, as seen by the box running the layer.
LOCAL = "local"          # own cache
PEER = "peer"            # peer ran it (activations out, partial sums back)
NVME = "nvme"            # own drive, admitted to own cache
NVME_TMP = "nvme_tmp"    # own drive, not admitted (prefill, or no slot could be freed)
PEER_MISS = "peer_miss"  # sent to the peer, which had evicted it: then read from own drive
SOURCES = (LOCAL, PEER, NVME, NVME_TMP)


class Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.experts = Counter()     # "decode.local" -> routed experts (per layer call)
            self.pairs = Counter()       # "decode.local" -> token-expert pairs
            self.served = Counter()      # work done for the peer
            self.admit = Counter()       # admissions fetched from the peer
            self.link = Counter()        # peer-link bytes from this box's side
            self.time = defaultdict(float)
            self.steps = 0               # decode steps
            self.decode_tokens = 0
            self.prefill_tokens = 0
            self.turns = 0

    def add(self, field: str, key, n=1) -> None:
        with self.lock:
            getattr(self, field)[key] += n

    def snapshot(self) -> dict:
        with self.lock:
            out = {
                "experts": dict(self.experts), "pairs": dict(self.pairs), "served": dict(self.served),
                "admit": dict(self.admit), "link": dict(self.link), "time_s": dict(self.time),
                "steps": self.steps, "decode_tokens": self.decode_tokens,
                "prefill_tokens": self.prefill_tokens, "turns": self.turns,
            }
        for phase in ("decode", "prefill"):
            tot = sum(v for k, v in out["experts"].items() if k.startswith(phase + ".") and k != f"{phase}.{PEER_MISS}")
            out[f"{phase}_share"] = {s: out["experts"].get(f"{phase}.{s}", 0) / tot for s in SOURCES} if tot else {}
        return out
