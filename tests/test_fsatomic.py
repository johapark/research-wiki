"""Atomic + cross-process-locked file writes (researchwiki/fsatomic.py).

These back the write-safety cluster from the code review: a crash mid-write must
never truncate a wiki page, a truncated cache must read as a miss (not crash),
and two concurrent ingest subprocesses splicing into the same shared file
(index.md, back-link targets) must not clobber each other.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from researchwiki import fsatomic


def test_write_text_atomic_roundtrip_utf8(tmp_path):
    p = tmp_path / "page.md"
    fsatomic.write_text_atomic(p, "héllo — Δ world")
    assert p.read_text(encoding="utf-8") == "héllo — Δ world"


def test_write_text_atomic_leaves_no_tmp(tmp_path):
    p = tmp_path / "page.md"
    fsatomic.write_text_atomic(p, "content")
    assert not (tmp_path / "page.md.tmp").exists()


def test_write_text_atomic_overwrites(tmp_path):
    p = tmp_path / "page.md"
    fsatomic.write_text_atomic(p, "v1")
    fsatomic.write_text_atomic(p, "v2")
    assert p.read_text(encoding="utf-8") == "v2"


def test_read_json_missing_returns_default(tmp_path):
    assert fsatomic.read_json(tmp_path / "nope.json", default="D") == "D"


def test_read_json_corrupt_returns_default(tmp_path):
    c = tmp_path / "c.json"
    c.write_text("{ truncated", encoding="utf-8")
    assert fsatomic.read_json(c, default=None) is None


def test_read_json_valid(tmp_path):
    c = tmp_path / "c.json"
    fsatomic.write_json_atomic(c, {"a": 1})
    assert fsatomic.read_json(c) == {"a": 1}


def test_update_locked_no_change_returns_false(tmp_path):
    p = tmp_path / "page.md"
    fsatomic.write_text_atomic(p, "body")
    assert fsatomic.update_locked(p, lambda old: old) is False
    assert p.read_text(encoding="utf-8") == "body"


def test_update_locked_change_returns_true(tmp_path):
    p = tmp_path / "page.md"
    fsatomic.write_text_atomic(p, "body")
    assert fsatomic.update_locked(p, lambda old: old + "\nmore") is True
    assert p.read_text(encoding="utf-8").endswith("more")


def test_update_locked_missing_ok_creates(tmp_path):
    p = tmp_path / "new.md"
    assert fsatomic.update_locked(p, lambda old: old + "seed", missing_ok=True) is True
    assert p.read_text(encoding="utf-8") == "seed"


_WORKER = """
import os, sys
sys.path.insert(0, sys.argv[3])
from researchwiki.fsatomic import update_locked
target, tag = sys.argv[1], sys.argv[2]
for i in range(50):
    update_locked(target, lambda old: old + f"{tag}-{i}\\n", missing_ok=True)
"""


def test_update_locked_serializes_concurrent_processes(tmp_path):
    """4 processes each append 50 lines; the flock must prevent lost updates."""
    target = tmp_path / "shared.txt"
    target.write_text("", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER, encoding="utf-8")
    repo_root = str(Path(__file__).resolve().parents[1])
    procs = [
        subprocess.Popen([sys.executable, str(worker), str(target), f"w{n}", repo_root])
        for n in range(4)
    ]
    for p in procs:
        assert p.wait() == 0
    lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 200  # 4 workers × 50, none clobbered
