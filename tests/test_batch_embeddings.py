import json
from multiprocessing.connection import Connection
import os
import subprocess
import sys

import pytest

from researchwiki.index import batch_embeddings


@pytest.mark.skipif(os.name != "posix", reason="worker descriptor passing is POSIX only")
def test_interrupted_client_cannot_leak_response_to_next_paper(monkeypatch):
    """The first PDF exits with an unread reply; the encoder stays reusable."""
    real_popen = subprocess.Popen
    server = """
import os,sys
from multiprocessing.connection import Connection
from researchwiki.index import embeddings, batch_embeddings
embeddings.is_available = lambda: True
with Connection(int(sys.argv[1])) as control:
    batch_embeddings._serve(control)
"""
    def launch(args, **kwargs):
        return real_popen([sys.executable, "-c", server, args[-1]], **kwargs)

    monkeypatch.setattr(batch_embeddings.subprocess, "Popen", launch)
    slot = batch_embeddings.EncoderSlot()
    try:
        for first in (True, False):
            env = {}
            with slot.attach(env) as options:
                fd, = options["pass_fds"]
                with Connection(os.dup(fd)) as client:
                    client.send_bytes(json.dumps(None).encode())
                    if first:
                        pid = slot.process.pid
                    else:
                        assert slot.process.pid == pid
                        assert client.recv_bytes() == b"1"
                        assert not client.poll(0.05)
    finally:
        slot.close()


def test_close_escalates_instead_of_waiting_forever(monkeypatch):
    """Closing the control pipe only reaches the encoder when it returns to
    `recv_handle`, so teardown must not block behind work in flight — a Ctrl-C
    during a first-use model download otherwise looks like a hung batch."""
    events = []

    class WedgedProcess:
        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            events.append(("wait", timeout))
            if self.waits <= 2:            # ignores the pipe close, then SIGTERM
                raise subprocess.TimeoutExpired(cmd="encoder", timeout=timeout)
            return 0

        def terminate(self):
            events.append(("terminate", None))

        def kill(self):
            events.append(("kill", None))

    class FakeControl:
        def close(self):
            events.append(("control.close", None))

    slot = batch_embeddings.EncoderSlot()
    slot.control, slot.process = FakeControl(), WedgedProcess()
    slot.close()

    assert [name for name, _ in events] == [
        "control.close", "wait", "terminate", "wait", "kill", "wait",
    ]
    assert all(t == batch_embeddings._SHUTDOWN_GRACE
               for name, t in events if name == "wait" and t is not None)
    assert slot.control is None and slot.process is None


def test_close_is_quiet_when_the_encoder_exits_on_its_own():
    """The common path: the server sees its control pipe close and exits, so
    no signal is sent."""
    events = []

    class TidyProcess:
        def wait(self, timeout=None):
            events.append("wait")
            return 0

        def terminate(self):
            events.append("terminate")

    slot = batch_embeddings.EncoderSlot()
    slot.control, slot.process = None, TidyProcess()
    slot.close()
    assert events == ["wait"]
