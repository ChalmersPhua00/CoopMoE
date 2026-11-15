"""Messaging between the bench and the engines and between the two engines: a JSON header plus raw
tensor payloads, over TCP. One request/response at a time per connection.

This TCP path is the portable baseline: it runs on a laptop and over the CX-7 link's IP
interface. A faster data plane (NCCL or raw verbs over RoCE) can replace PeerLink's two
connections without touching the engine, which only calls `call()`.
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
import traceback
from typing import Callable, Sequence

import torch

from coopmoe.store.layout import DTYPES

_DT = dict(DTYPES, int64=torch.int64, int32=torch.int32, uint8=torch.uint8, bool=torch.bool)
_LEN = struct.Struct("<Q")

Handler = Callable[[dict, list], tuple[dict, list]]


class RemoteError(RuntimeError):
    pass


def _dtname(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _recv_exact(sock: socket.socket, view: memoryview) -> None:
    got = 0
    while got < len(view):
        n = sock.recv_into(view[got:])
        if n == 0:
            raise ConnectionError("peer closed the connection")
        got += n


def send_msg(sock: socket.socket, header: dict, tensors: Sequence[torch.Tensor]) -> int:
    metas, bufs = [], []
    for t in tensors:
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous()
        metas.append([_dtname(t), list(t.shape)])
        bufs.append(memoryview(t.reshape(-1).view(torch.uint8).numpy()) if t.numel() else b"")
    hb = json.dumps(dict(header, _t=metas)).encode()
    total = _LEN.size + len(hb) + sum(len(b) for b in bufs)
    sock.sendall(_LEN.pack(len(hb)) + hb)
    for b in bufs:
        if len(b):
            sock.sendall(b)
    return total


def recv_msg(sock: socket.socket) -> tuple[dict, list, int]:
    lb = bytearray(_LEN.size)
    _recv_exact(sock, memoryview(lb))
    (hlen,) = _LEN.unpack(lb)
    hb = bytearray(hlen)
    _recv_exact(sock, memoryview(hb))
    header = json.loads(hb)
    tensors, total = [], _LEN.size + hlen
    for dtname, shape in header.pop("_t"):
        dt = _DT[dtname]
        numel = 1
        for s in shape:
            numel *= s
        nbytes = numel * torch.empty((), dtype=dt).element_size()
        buf = torch.empty(nbytes, dtype=torch.uint8)
        if nbytes:
            _recv_exact(sock, memoryview(buf.numpy()))
        tensors.append(buf.view(dt).reshape(shape))
        total += nbytes
    return header, tensors, total


def _tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 64 << 20)
        except OSError:
            pass


class TcpServer:
    """Accepts connections; one thread per connection runs request -> handler -> response."""

    def __init__(self, host: str, port: int, handler: Handler):
        self.handler = handler
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.bytes_in = self.bytes_out = 0
        self._thread = threading.Thread(target=self._accept_loop, name="server-accept", daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            _tune(conn)
            threading.Thread(target=self._serve, args=(conn,), name="server-conn", daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            while not self._stop.is_set():
                try:
                    header, tensors, n = recv_msg(conn)
                except (ConnectionError, OSError):
                    return
                try:
                    rh, rt = self.handler(header, tensors)
                except Exception as exc:  # report to the caller instead of killing the thread
                    traceback.print_exc()
                    rh, rt = {"error": f"{type(exc).__name__}: {exc}"}, []
                sent = send_msg(conn, rh, rt)
                with self._lock:
                    self.bytes_in += n
                    self.bytes_out += sent

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


class TcpClient:
    """One request/response at a time per connection (calls are serialized by a lock)."""

    def __init__(self, addr: str, timeout: float = 600.0):
        host, port = addr.rsplit(":", 1)
        self.addr = addr
        self.lock = threading.Lock()
        self.bytes_out = self.bytes_in = 0
        t0 = time.time()
        while True:
            try:
                self.sock = socket.create_connection((host, int(port)), timeout=10)
                break
            except OSError:
                if time.time() - t0 > timeout:
                    raise
                time.sleep(0.5)
        self.sock.settimeout(None)
        _tune(self.sock)

    def call(self, header: dict, tensors: Sequence[torch.Tensor] = ()) -> tuple[dict, list]:
        with self.lock:
            self.bytes_out += send_msg(self.sock, header, tensors)
            rh, rt, n = recv_msg(self.sock)
            self.bytes_in += n
        if "error" in rh:
            raise RemoteError(f"{self.addr}: {rh['error']}")
        return rh, rt

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
