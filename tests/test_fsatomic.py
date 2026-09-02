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

import pytest

from researchwiki import fsatomic


def test_write_text_atomic_roundtrip_utf8(tmp_path):
    p = tmp_path / "page.md"
    fsatomic.write_text_atomic(p, "héllo — Δ world")
    assert p.read_text(encoding="utf-8") == "héllo — Δ world"


def test_write_text_atomic_leaves_no_tmp(tmp_path):
    p = tmp_path / "page.md"
    fsatomic.write_text_atomic(p, "content")
    assert not list(tmp_path.glob(".page.md.*.tmp"))


def test_write_text_atomic_overwrites(tmp_path):
    p = tmp_path / "page.md"
    fsatomic.write_text_atomic(p, "v1")
    fsatomic.write_text_atomic(p, "v2")
    assert p.read_text(encoding="utf-8") == "v2"


@pytest.mark.skipif(os.name == "nt", reason="Windows chmod only exposes read-only")
def test_write_text_atomic_preserves_existing_mode(tmp_path):
    target = tmp_path / "private.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o600)
    fsatomic.write_text_atomic(target, "new")
    assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX umask modes")
def test_write_text_atomic_honors_umask_for_new_file(tmp_path):
    target = tmp_path / "private.txt"
    script = (
        "import os, sys; "
        "from researchwiki.fsatomic import write_text_atomic; "
        "os.umask(0o077); write_text_atomic(sys.argv[1], 'private')"
    )
    subprocess.run([sys.executable, "-c", script, str(target)], check=True)
    assert target.stat().st_mode & 0o777 == 0o600


def test_write_text_atomic_cleans_up_after_replace_failure(tmp_path, monkeypatch):
    target = tmp_path / "page.md"

    def fail_replace(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(fsatomic.os, "replace", fail_replace)
    try:
        fsatomic.write_text_atomic(target, "content")
    except OSError as exc:
        assert "simulated" in str(exc)
    else:
        raise AssertionError("replace failure did not propagate")
    assert not list(tmp_path.glob(".page.md.*.tmp"))


def test_write_text_atomic_retries_transient_replace_permission_error(
    tmp_path, monkeypatch,
):
    """Windows sharing violations should not make a serialized write flaky."""
    target = tmp_path / "page.md"
    real_replace = fsatomic.os.replace
    calls = []
    sleeps = []

    def flaky_replace(source, destination):
        calls.append((source, destination))
        if len(calls) < 3:
            raise PermissionError("simulated transient sharing violation")
        return real_replace(source, destination)

    monkeypatch.setattr(fsatomic, "_REPLACE_PERMISSION_ATTEMPTS", 3)
    monkeypatch.setattr(fsatomic.os, "replace", flaky_replace)
    monkeypatch.setattr(fsatomic.time, "sleep", sleeps.append)

    fsatomic.write_text_atomic(target, "content")

    assert target.read_text(encoding="utf-8") == "content"
    assert len(calls) == 3
    assert sleeps == [0.01, 0.02]
    assert not list(tmp_path.glob(".page.md.*.tmp"))


_ATOMIC_WORKER = """
import sys
sys.path.insert(0, sys.argv[2])
from researchwiki.fsatomic import write_text_atomic
target = sys.argv[1]
for i in range(100):
    write_text_atomic(target, f"{sys.argv[3]}:{i}")
"""


def test_write_text_atomic_allows_concurrent_writers(tmp_path):
    """Independent cache writers may target one key; none may steal another's tmp."""
    target = tmp_path / "shared.txt"
    worker = tmp_path / "atomic_worker.py"
    worker.write_text(_ATOMIC_WORKER, encoding="utf-8")
    repo_root = str(Path(__file__).resolve().parents[1])
    procs = [
        subprocess.Popen(
            [sys.executable, str(worker), str(target), repo_root, f"w{n}"]
        )
        for n in range(8)
    ]
    assert [p.wait() for p in procs] == [0] * len(procs)
    writer, iteration = target.read_text(encoding="utf-8").split(":")
    assert writer.startswith("w")
    assert iteration.isdigit()
    assert not list(tmp_path.glob(".shared.txt.*.tmp"))


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
    """4 processes each append 50 lines; the file lock must prevent lost updates."""
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


def test_writes_pin_lf_at_the_text_io_boundary(tmp_path, monkeypatch):
    """Pages must have the same bytes whichever platform wrote them.

    Without `newline=""` Python's text mode expands each "\n" to `os.linesep`,
    so the identical page written on Windows lands as CRLF. That divergence is
    what forced `db/verify.py` to recognize a CRLF frontmatter fence, and it
    moves the `file_sha256` page identity that gates `migrate provenance`.

    A POSIX-only byte assertion passes even when the writer omits `newline=""`,
    because POSIX text mode already emits LF. Spy on the TextIOWrapper boundary
    so this test fails on every platform if the explicit no-translation contract
    is removed.
    """
    real_fdopen = fsatomic.os.fdopen
    seen = []

    def recording_fdopen(fd, *args, **kwargs):
        seen.append(kwargs.get("newline"))
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(fsatomic.os, "fdopen", recording_fdopen)
    target = tmp_path / "page.md"
    fsatomic.write_text_atomic(target, "---\ntitle: x\n---\n\nbody\n")
    assert seen == [""]
    assert b"\r" not in target.read_bytes()
    assert target.read_bytes() == b"---\ntitle: x\n---\n\nbody\n"


def test_explicit_crlf_content_is_preserved_verbatim(tmp_path):
    """Pinning LF must not rewrite content the caller deliberately made CRLF."""
    target = tmp_path / "keep.txt"
    fsatomic.write_text_atomic(target, "a\r\nb\r\n")
    assert target.read_bytes() == b"a\r\nb\r\n"
