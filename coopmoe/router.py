"""Request scheduler (runs next to PGX-A's engine): decides which box serves each request.
Never touches weights or KV.

Every request carries a category label (code, math, medicine, ...), and the run config fixes
which categories each box serves (router.box_categories). Two modes:

  category     a request goes to the box that serves its category
  round_robin  baseline: requests alternate between the boxes in arrival order, whatever
               their category

Each box serves at most engine.max_sessions (4) requests at a time. A request whose box is full
waits for a slot on that box: it never spills to the other one.

Deliberately left for later: recognizing a request's category from its text, and moving load
between the boxes when one of them is busier.
"""
from __future__ import annotations

import threading
from collections import Counter
from typing import Callable, Optional

from coopmoe.config import RouterConfig

MODES = ("category", "round_robin")


def box_of_category(box_categories: list) -> dict[str, int]:
    """[[categories of box 0], [categories of box 1]] -> {category: box}."""
    if len(box_categories) != 2:
        raise ValueError(f"router.box_categories needs two lists (box 0, box 1), got {box_categories!r}")
    out: dict[str, int] = {}
    for box, cats in enumerate(box_categories):
        for c in cats:
            if c in out:
                raise ValueError(f"category {c!r} is assigned to both boxes")
            out[c] = box
    return out


class Router:
    """call(box, header) -> (reply header, reply tensors) must be safe to use from many threads."""

    def __init__(self, rcfg: RouterConfig, cap: int, call: Callable, log=print):
        if rcfg.mode not in MODES:
            raise ValueError(f"router.mode must be one of {MODES}, got {rcfg.mode!r}")
        self.cfg, self.cap, self.call, self.log = rcfg, cap, call, log
        self.box_of = box_of_category(rcfg.box_categories)
        self.slots = [threading.Semaphore(cap), threading.Semaphore(cap)]
        self.lock = threading.Lock()
        self.sessions: dict[str, int] = {}   # open session -> its box
        self.arrivals = 0
        self.stats = Counter()
        self.placement = Counter()           # "code->0": category and where it went

    def assign(self, category: Optional[str]) -> int:
        """The scheduling decision, made once per request in arrival order."""
        with self.lock:
            if self.cfg.mode == "round_robin":
                box = self.arrivals % 2
            elif category in self.box_of:
                box = self.box_of[category]
            else:
                raise ValueError(f"no box serves category {category!r}; router.box_categories has "
                                 f"{sorted(self.box_of)}")
            self.arrivals += 1
            self.placement[f"{category}->{box}"] += 1
            return box

    def open(self, sid: str, box: int) -> None:
        """Blocks until `box` has a free slot, then holds it until close(sid)."""
        if not self.slots[box].acquire(blocking=False):
            with self.lock:
                self.stats["waits"] += 1
            self.slots[box].acquire()
        with self.lock:
            self.sessions[sid] = box
            self.stats["sessions"] += 1
            self.stats[f"sessions_box{box}"] += 1

    def run_turn(self, sid: str, ids: list, max_new: int) -> dict:
        """The request's prompt and generation, on its box. -> the box's reply header."""
        with self.lock:
            box = self.sessions[sid]
        rh, _ = self.call(box, {"op": "turn", "sid": sid, "ids": list(ids), "max_new": int(max_new)})
        with self.lock:
            self.stats["turns"] += 1
        return rh

    def close(self, sid: str) -> None:
        """Frees the session's slot (its box freed the KV when the request finished)."""
        with self.lock:
            box = self.sessions.pop(sid, None)
        if box is not None:
            self.slots[box].release()

    def summary(self) -> dict:
        with self.lock:
            s = dict(self.stats)
            s["mode"] = self.cfg.mode
            s["placement"] = dict(self.placement)
            return s
