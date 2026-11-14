"""One box's expert cache: a single plain LRU over fixed-size slots, plus duplicate-first eviction.

There are no hot / warm / cold tiers. What a box holds is whatever its own traffic (and the
experts it runs for its peer) keeps recent. One rule sits on top:

  duplicate-first eviction   when a slot is needed, an expert the peer also holds (per the
                             peer's last residency bitmap) goes before any expert only this
                             box holds. With dup_protect > 0, duplicates this box used within
                             its last dup_protect decode steps are spared (an ablation: it lost
                             to the plain rule on one emulation machine and won on another).

Bookkeeping is per (layer, expert) key in numpy arrays, so picking a victim is one vectorized
pass over 12,032 keys for Qwen3-235B. Experts are read-only, so a stale view of the peer
never gives a wrong answer, only a different eviction choice.

Concurrency: the engine thread, the peer-server thread, the drive readers and the admission
thread all use the cache. A key is pinned while anyone computes on it or loads into it, and
pinned keys are never evicted. Two threads asking for the same missing expert share one load.
"""
from __future__ import annotations

import threading
from enum import Enum
from typing import Optional

import numpy as np
import torch

from coopmoe.store.layout import StoreIndex
from coopmoe.store.reader import aligned_empty

_NEVER = -(1 << 40)
_MAXI = np.iinfo(np.int64).max


class Acquire(Enum):
    HIT = "hit"            # committed, pinned for the caller (release it)
    RESERVED = "reserved"  # empty slot pinned for the caller, who must load then commit/abort
    MISS = "miss"          # not resident and no slot was taken


class ExpertCache:
    def __init__(self, index: StoreIndex, n_slots: int, device: torch.device,
                 dup_first: bool = True, dup_protect: int = 64):
        self.index = index
        self.n_slots = n_slots
        self.device = device
        self.dup_first, self.dup_protect = dup_first, dup_protect
        self.slab = aligned_empty(max(n_slots, 1) * index.stride, device).view(max(n_slots, 1), index.stride)
        nk = index.model.n_keys
        self._lock = threading.Lock()
        self.slot_of = np.full(nk, -1, dtype=np.int64)
        self.present = np.zeros(nk, dtype=bool)          # committed (usable) keys
        self.pins = np.zeros(nk, dtype=np.int32)
        self.last_tick = np.zeros(nk, dtype=np.int64)    # LRU recency, any use
        self.last_step = np.full(nk, _NEVER, dtype=np.int64)  # decode step of the last local use
        self.peer: Optional[np.ndarray] = None           # peer residency (bool per key)
        self.step = 0                                    # this box's decode step, set by the engine
        self._loading: dict[int, tuple[int, threading.Event]] = {}
        self._clear_state()

    def _clear_state(self) -> None:
        self._free = list(range(self.n_slots - 1, -1, -1))
        self._tick = 0
        self.version = 0
        self.admissions = self.evictions = self.dup_evictions = 0

    def clear(self) -> None:
        """Empty the cache (between benchmark cells, when nothing is in flight)."""
        with self._lock:
            assert not self._loading and not self.pins.any(), "clear() while experts are in use"
            self.slot_of[:] = -1
            self.present[:] = False
            self.last_tick[:] = 0
            self.last_step[:] = _NEVER
            self.peer = None
            self.step = 0
            self._clear_state()

    # -- queries ------------------------------------------------------------------
    def has(self, key: int) -> bool:
        return bool(self.present[key])

    def bitmap(self) -> tuple[int, np.ndarray]:
        with self._lock:
            return self.version, np.packbits(self.present)

    def set_peer(self, bits: Optional[np.ndarray]) -> None:
        with self._lock:
            self.peer = bits

    def occupancy(self) -> dict:
        with self._lock:
            held = int(self.present.sum())
            dup = int((self.present & self.peer).sum()) if self.peer is not None else None
            return {"slots": self.n_slots, "held": held, "free": len(self._free), "dup_held": dup,
                    "admissions": self.admissions, "evictions": self.evictions,
                    "dup_evictions": self.dup_evictions}

    # -- slot lifecycle -------------------------------------------------------------
    def _touch(self, key: int, local: bool) -> None:
        self._tick += 1
        self.last_tick[key] = self._tick
        if local:
            self.last_step[key] = self.step

    def acquire(self, key: int, reserve: bool = True, local: bool = True,
                refresh: bool = True) -> tuple[Optional[int], Acquire]:
        """Pin `key` if resident (HIT). Otherwise, with reserve=True, take a slot for the caller
        to load into (RESERVED), evicting if needed; MISS if not resident and nothing reserved.
        local=False marks a use on behalf of the peer: it refreshes LRU recency (if `refresh`)
        but does not count as this box's own use for duplicate protection."""
        while True:
            with self._lock:
                if self.present[key]:
                    self.pins[key] += 1
                    if refresh or local:
                        self._touch(key, local)
                    return int(self.slot_of[key]), Acquire.HIT
                pending = self._loading.get(key)
                if pending is None:
                    if not reserve:
                        return None, Acquire.MISS
                    slot = self._take_slot()
                    if slot is None:
                        return None, Acquire.MISS
                    self.pins[key] = 1
                    self.slot_of[key] = slot
                    self._loading[key] = (slot, threading.Event())
                    return slot, Acquire.RESERVED
            pending[1].wait()  # another thread is loading this expert; use its result

    def _victim(self) -> Optional[int]:
        cand = self.present & (self.pins == 0)
        if self.dup_first and self.peer is not None:
            dup = cand & self.peer & (self.step - self.last_step >= self.dup_protect)
            if dup.any():
                self.dup_evictions += 1
                return int(np.argmin(np.where(dup, self.last_tick, _MAXI)))
        if not cand.any():
            return None
        return int(np.argmin(np.where(cand, self.last_tick, _MAXI)))

    def _take_slot(self) -> Optional[int]:
        if self._free:
            return self._free.pop()
        key = self._victim()
        if key is None:
            return None
        slot = int(self.slot_of[key])
        self.slot_of[key] = -1
        self.present[key] = False
        self.version += 1
        self.evictions += 1
        return slot

    def commit(self, key: int, slot: int) -> None:
        """The reserved slot now holds `key`; it stays pinned for the caller."""
        with self._lock:
            _, event = self._loading.pop(key)
            self.present[key] = True
            self._touch(key, local=True)
            self.version += 1
            self.admissions += 1
        event.set()

    def abort(self, key: int, slot: int) -> None:
        with self._lock:
            _, event = self._loading.pop(key)
            self.slot_of[key] = -1
            self.pins[key] = 0
            self._free.append(slot)
        event.set()

    def release(self, key: int) -> None:
        with self._lock:
            self.pins[key] -= 1
            assert self.pins[key] >= 0
