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
