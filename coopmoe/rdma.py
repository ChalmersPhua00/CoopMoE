"""One-sided RDMA for expert copies (link.transport=rdma), through NVIDIA NIXL's UCX backend.

Each box registers two sets of pinned host buffers with the NIC, one expert record each:
    serve[i]  the peer's loader thread i reads an expert out of here
    land[i]   this box's loader thread i lands an expert read from the peer here
A copy from the peer, on this box's loader thread i:
    1. ask the peer over TCP (a few bytes) to put expert K in its serve[i]
    2. the peer copies K out of its cache into serve[i] (or replies MISS) and replies
    3. RDMA READ the peer's serve[i] into this box's land[i]; wait for it
    4. land[i] -> a cache slot or a transient buffer, as for a drive read
Expert bytes cross the link by RDMA, not through either box's TCP stack. Thread i only asks
again after its READ finished, so serve[i] is never overwritten while it is being read.

Host memory, because DGX Spark (GB10) has no GPUDirect RDMA: the NIC cannot reach cudaMalloc
memory, only pinned host memory registered with it. On a unified-memory box that is the same
LPDDR, one local copy away from a cache slot.

UCX picks the transport: RoCE on the CX-7 (UCX_NET_DEVICES selects the RDMA devices, e.g.
both rails), shared memory or TCP between two processes on one machine (the CPU test).
"""
from __future__ import annotations

import os
import threading
import time

import torch

POLL_S = 50e-6  # between completion checks: lets other threads (and HF's loop) have the GIL


class RdmaPlane:
    def __init__(self, rank: int, n_buffers: int, stride: int, nbytes: int, pin: bool):
        try:
            from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t
        except ImportError as exc:
            raise RuntimeError("link.transport=rdma needs NIXL (pip install nixl, in requirements.txt); "
                               "or set link.transport=tcp") from exc
        self.name = f"coopmoe-box{rank}-{os.getpid()}"
        # RW: the loader threads post and poll transfers concurrently.
        self.agent = nixl_agent(self.name, nixl_agent_config(
            backends=["UCX"], sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_RW))
        self.n, self.nbytes = n_buffers, nbytes
        self.serve = torch.zeros((n_buffers, stride), dtype=torch.uint8, pin_memory=pin)
        self.land = torch.zeros((n_buffers, stride), dtype=torch.uint8, pin_memory=pin)
        self.agent.register_memory(self.serve)
        self.agent.register_memory(self.land)
        self.serve_descs = self.agent.get_xfer_descs([self.serve[i, :nbytes] for i in range(n_buffers)])
        land_descs = self.agent.get_xfer_descs([self.land[i, :nbytes] for i in range(n_buffers)])
        self.local_side = self.agent.prep_xfer_dlist("NIXL_INIT_AGENT", land_descs)
        self.remote_name = None
        self.remote_side = None
        self._lock = threading.Lock()

    # -- set-up, over the TCP control connection --------------------------------------
    def info(self) -> tuple[dict, list]:
        """What the peer needs to read from this box: NIXL metadata and the serve descriptors."""
        meta = self.agent.get_agent_metadata()
        descs = self.agent.get_serialized_descs(self.serve_descs)
        return ({"name": self.name, "n": self.n, "nbytes": self.nbytes},
                [torch.frombuffer(bytearray(meta), dtype=torch.uint8),
                 torch.frombuffer(bytearray(descs), dtype=torch.uint8)])

    def connect(self, h: dict, tensors: list) -> None:
        """Adds the peer (once per peer process) from its info()."""
        if h["n"] < self.n or h["nbytes"] != self.nbytes:
            raise RuntimeError(f"peer has {h['n']} serve buffers of {h['nbytes']} B; this box needs "
                               f"{self.n} of {self.nbytes} B (same reader.threads and model on both boxes)")
        with self._lock:
            if self.remote_name == h["name"]:
                return
            name = self.agent.add_remote_agent(tensors[0].numpy().tobytes())
            remote = self.agent.deserialize_descs(tensors[1].numpy().tobytes())
            self.remote_side = self.agent.prep_xfer_dlist(name, remote)
            self.remote_name = name

    @property
    def connected(self) -> bool:
        return self.remote_side is not None

    # -- data path --------------------------------------------------------------------
    def read(self, i: int, timeout: float = 60.0) -> torch.Tensor:
        """RDMA READ of the peer's serve[i] into land[i]; returns land[i] (valid until this
        thread's next read)."""
        h = self.agent.make_prepped_xfer("READ", self.local_side, [i], self.remote_side, [i])
        try:
            st = self.agent.transfer(h)
            t0 = time.perf_counter()
            while st not in ("DONE", "ERR"):
                if time.perf_counter() - t0 > timeout:
                    raise TimeoutError(f"RDMA READ of serve[{i}] from {self.remote_name} took over {timeout} s")
                time.sleep(POLL_S)
                st = self.agent.check_xfer_state(h)
            if st == "ERR":
                raise RuntimeError(f"RDMA READ of serve[{i}] from {self.remote_name} failed")
        finally:
            self.agent.release_xfer_handle(h)
        return self.land[i, : self.nbytes]
