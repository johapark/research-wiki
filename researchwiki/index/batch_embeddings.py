"""A private, persistent CPU encoder for a POSIX batch worker slot.

Each ingest still runs in a fresh process. Only text, availability probes and
NumPy arrays cross its private pipe. Each paper gets a new connection so a
killed ingest cannot strand the next one halfway through a message.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import io
import json
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import os
import subprocess
import sys
import threading

from ..log import log

_ENV = "_RESEARCHWIKI_EMBEDDING_FD"
_slot = threading.local()
_fd = os.environ.pop(_ENV, None)
_client = Connection(int(_fd)) if _fd is not None else None
if _client is not None:
    os.set_inheritable(_client.fileno(), False)
_client_lock = threading.Lock()


class EncoderSlot:
    def __init__(self):
        self.control = None
        self.process = None

    def close(self):
        if self.control is not None:
            self.control.close()
        if self.process is not None:
            self.process.wait()
        self.control = self.process = None

    @contextmanager
    def attach(self, env):
        from multiprocessing.reduction import send_handle

        if self.process is None or self.process.poll() is not None:
            self.close()
            self.control, server = Pipe()
            with server:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", __name__, str(server.fileno())],
                    pass_fds=(server.fileno(),), stdin=subprocess.DEVNULL,
                )
        client, peer = Pipe()
        with client, peer:
            send_handle(self.control, peer.fileno(), self.process.pid)
            peer.close()
            env[_ENV] = str(client.fileno())
            yield {"pass_fds": (client.fileno(),)}


def assign_slot(slots):
    _slot.encoder = slots.get_nowait()


@contextmanager
def child_options(env):
    slot = getattr(_slot, "encoder", None)
    if slot is None:
        yield {}
    else:
        with ExitStack() as stack:
            try:
                options = stack.enter_context(slot.attach(env))
            except (OSError, EOFError) as exc:
                log(f"Batch encoder unavailable, using per-paper model: {exc}", tag="semantic")
                slot.close()
                options = {}
            yield options


def active() -> bool:
    return _client is not None


def request(texts: list[str] | None) -> bytes | None:
    """None lets the caller retry locally after an encoder process failure."""
    global _client
    with _client_lock:
        if _client is None:
            return None
        try:
            _client.send_bytes(json.dumps(texts, ensure_ascii=False).encode())
            return _client.recv_bytes()
        except (EOFError, OSError) as exc:
            _client.close()
            _client = None
            log(f"Lost batch encoder, using per-paper model: {exc}", tag="semantic")
            return None


def _serve(control):
    from multiprocessing.reduction import recv_handle
    import numpy as np
    from . import embeddings

    while True:
        try:
            fd = recv_handle(control)
        except (EOFError, OSError):
            return
        with Connection(fd) as client:
            try:
                while True:
                    texts = json.loads(client.recv_bytes())
                    if texts is None:
                        client.send_bytes(b"1" if embeddings.is_available() else b"")
                    else:
                        vectors = embeddings.embed_texts(texts)
                        buffer = io.BytesIO()
                        if vectors is not None:
                            np.save(buffer, vectors, allow_pickle=False)
                        client.send_bytes(buffer.getvalue())
            except (EOFError, BrokenPipeError, ConnectionResetError):
                continue


if __name__ == "__main__":
    with Connection(int(sys.argv[1])) as control:
        _serve(control)
