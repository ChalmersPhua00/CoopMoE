"""Request scheduler (runs next to PGX-A's engine): decides which box serves each request.
Never touches weights or KV.

Every request carries the box make_requests.py assigned it to ("box", by its category). Two
modes:

  category     a request goes to its assigned box
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
from typing import Callable

from coopmoe.config import RouterConfig

MODES = ("category", "round_robin")


class Router:
    """call(box, header) -> (reply header, reply tensors) must be safe to use from many threads."""

    def __init__(self, rcfg: RouterConfig, cap: int, call: Callable, log=print):
        if rcfg.mode not in MODES:
            raise ValueError(f"router.mode must be one of {MODES}, got {rcfg.mode!r}")
        self.cfg, self.cap, self.call, self.log = rcfg, cap, call, log
        self.slots = [threading.Semaphore(cap), threading.Semaphore(cap)]
        self.lock = threading.Lock()
        self.sessions: dict[str, int] = {}   # open session -> its box
        self.arrivals = 0
        self.stats = Counter()
        self.placement = Counter()           # "code->0": category and where it went

    def assign(self, request: dict) -> int:
        """The scheduling decision, made once per request in arrival order."""
        category = request.get("category")
        with self.lock:
            if self.cfg.mode == "round_robin":
                box = self.arrivals % 2
            elif request.get("box") in (0, 1):
                box = request["box"]
            else:
                raise ValueError(f"request {request.get('id')!r} has no box: regenerate the requests "
                                 f"with make_requests.py")
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
