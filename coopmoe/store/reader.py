"""Reads expert records from this box's drive into aligned buffers.

On Linux the reader opens layer files with O_DIRECT. That matters more on a UMA box than on
a discrete one: buffered reads would fill the page cache, and the page cache lives in the
same 128 GB the expert cache and KV cache need.
"""
from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from coopmoe.store.layout import ALIGN, StoreIndex


def aligned_empty(nbytes: int, device: str | torch.device = "cpu", pin: bool = False) -> torch.Tensor:
    """uint8 tensor whose data pointer is ALIGN-aligned (needed for O_DIRECT)."""
    device = torch.device(device)
    if device.type != "cpu":
        return torch.empty(nbytes, dtype=torch.uint8, device=device)
    raw = torch.empty(nbytes + ALIGN, dtype=torch.uint8, pin_memory=pin)
    off = (-raw.data_ptr()) % ALIGN
    return raw[off : off + nbytes]


class ExpertReader:
    def __init__(self, store_dir: str, index: StoreIndex, threads: int = 8, direct_io: bool = True):
        self.store_dir, self.index = store_dir, index
        self.direct = bool(direct_io and hasattr(os, "O_DIRECT"))
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="nvme")
        self._fds: dict[int, int] = {}
        self._fd_lock = threading.Lock()
        self._stat_lock = threading.Lock()
        self.reads = 0
        self.bytes = 0

    def _fd(self, layer: int) -> int:
        fd = self._fds.get(layer)
        if fd is None:
            with self._fd_lock:
                fd = self._fds.get(layer)
                if fd is None:
                    path = self.index.layer_path(self.store_dir, layer)
                    flags = os.O_RDONLY
                    if self.direct:
                        try:
                            fd = os.open(path, flags | os.O_DIRECT)
                        except OSError:
                            self.direct = False  # filesystem without O_DIRECT (tmpfs, overlays)
                    if fd is None:
                        fd = os.open(path, flags)
                    self._fds[layer] = fd
        return fd

    def read_into(self, layer: int, expert: int, dst: torch.Tensor) -> None:
        """Blocking read of one record into dst (CPU uint8, >= stride bytes)."""
        nbytes = self.index.stride if self.direct else self.index.expert_bytes
        assert dst.device.type == "cpu" and dst.dtype == torch.uint8 and dst.numel() >= nbytes
        view = memoryview(dst.numpy())[:nbytes]
        offset = self.index.offset(expert)
        fd = self._fd(layer)
        done = 0
        while done < nbytes:
            n = os.preadv(fd, [view[done:]], offset + done)
            if n <= 0:
                raise IOError(f"short read layer {layer} expert {expert}: {done}/{nbytes}")
            done += n
        if not self.direct and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
        with self._stat_lock:
            self.reads += 1
            self.bytes += self.index.expert_bytes

    def submit(self, fn, *args) -> Future:
        return self.pool.submit(fn, *args)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


class ExpertLoader:
    """Drive -> device memory for one record.

    CPU device: read straight into the destination (aligned slab memory).
    CUDA device: read into a pinned, aligned staging buffer, then copy on a side stream. On
    GB10 that copy is LPDDR -> LPDDR (~0.07 ms for an 18.9 MB FP8 expert vs ~4.8 ms of NVMe).
    """

    def __init__(self, reader: ExpertReader, index: StoreIndex, device: torch.device, n_staging: int):
        self.reader, self.index, self.device = reader, index, device
        self.cuda = device.type == "cuda"
        self._staging: list[torch.Tensor] = []
        self._cv = threading.Condition()
        if self.cuda:
            self._staging = [aligned_empty(index.stride, "cpu", pin=True) for _ in range(n_staging)]
        self._tls = threading.local()

    def load(self, key: int, dst: torch.Tensor) -> None:
        layer, expert = self.index.unkey(key)
        if not self.cuda:
            self.reader.read_into(layer, expert, dst)
            return
        with self._cv:
            while not self._staging:
                self._cv.wait()
            buf = self._staging.pop()
        try:
            self.reader.read_into(layer, expert, buf)
            self.copy_in(dst, buf)
        finally:
            with self._cv:
                self._staging.append(buf)
                self._cv.notify()

    def copy_in(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Host bytes -> a device slot, synchronously (used for drive reads and peer fetches)."""
        n = self.index.expert_bytes
        if not self.cuda:
            dst[:n].copy_(src[:n])
            return
        stream = getattr(self._tls, "stream", None)
        if stream is None:
            stream = self._tls.stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            dst[:n].copy_(src[:n], non_blocking=True)
        stream.synchronize()


class BufferPool:
    """Fixed set of record-sized buffers for experts that are used but not admitted."""

    def __init__(self, n: int, stride: int, device: torch.device):
        self._q: "queue.Queue[torch.Tensor]" = queue.Queue()
        for _ in range(max(1, n)):
            self._q.put(aligned_empty(stride, device))

    def get(self) -> torch.Tensor:
        return self._q.get()

    def put(self, buf: torch.Tensor) -> None:
        self._q.put(buf)
