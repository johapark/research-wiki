"""Crash-safe batch mode for `researchwiki agent ingest` and `researchwiki
ingest`, driven through the two real CLI entry points.

Both commands share `researchwiki/tasks/_ingest_batch.py` (private module,
hidden from CLI auto-discovery by its leading underscore). These tests pin
the contract that lets `--resume` work:

  - Completions and failures land in checkpoint.json atomically.
  - `--resume` skips completed inputs and retries failed by default.
  - `--no-retry` skips failed too.
  - Absolute paths are the checkpoint keys — resume from any cwd.
  - plan.json records the subcommand + passthrough flags so `--resume`
    doesn't need re-supplying and semantics don't drift.
  - Per-PDF override flags are rejected on the *command line* in batch mode
    (a --doi meant for one paper would silently apply to all N), while a
    programmatic caller can supply a different set per input — which is what
    `researchwiki import apply` needs and what that guard was never about.

The worker itself is monkeypatched to a deterministic stub — the real
`_worker` shells out to `python -m researchwiki <subcommand> <pdf>`, which
is slow, needs an API key, and hits state.db. The stub returns the same
record shape and lets us control per-input outcomes.

Hermetic: no subprocess, no network, no LLM.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from researchwiki.tasks import _ingest_batch
from researchwiki.tasks import agent as agent_cli
from researchwiki.tasks import ingest as ingest_cli


# ---------- fixtures + helpers ----------


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    """Fake wiki root; `.ingest/batch-*/` will live under here."""
    root = tmp_path / "wiki-root"
    root.mkdir()
    from researchwiki import paths

    monkeypatch.setattr(paths, "wiki_root", lambda: root)
    # Satisfy `agent ingest`'s provider preflight. `conftest._no_dotenv_leak`
    # keeps the repo's real .env out of the test environment, so without this
    # every CLI-driven test here would exit 2 on missing credentials before
    # reaching the stubbed `_worker`. These tests pin batch orchestration, not
    # provider config — the preflight itself is covered in test_preflight.py.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    return root


def _make_pdfs(root: Path, names: list[str]) -> list[str]:
    """Empty files with .pdf extension — _resolve_inputs checks only
    existence and suffix, so this is enough."""
    inbox = root / "inbox"
    inbox.mkdir(exist_ok=True)
    out = []
    for n in names:
        p = inbox / n
        p.write_bytes(b"%PDF-1.4\n")
        out.append(str(p.resolve()))
    return out


def _stub_worker(
    outcomes: dict[str, int], calls: list | None = None, arg_capture: list | None = None
):
    """Build a `_worker` replacement that returns success/failure per input.

    Optionally records every invocation into `calls` (list of pdf paths)
    and `arg_capture` (list of (subcommand, extra_args) tuples) so tests
    can assert what got dispatched.
    """

    def worker(pdf_path, batch_dir, subcommand, extra_args):
        if calls is not None:
            calls.append(pdf_path)
        if arg_capture is not None:
            arg_capture.append((list(subcommand), list(extra_args)))
        rc = outcomes.get(pdf_path, 0)
        status = "completed" if rc == 0 else "failed"
        return {"input": pdf_path, "status": status, "returncode": rc}

    return worker


def _only_batch(wiki: Path) -> Path:
    batches = list((wiki / ".ingest").glob("batch-*"))
    assert len(batches) == 1, f"expected 1 batch dir, found {len(batches)}"
    return batches[0]


# ---------- driver internals ----------


def test_atomic_write_survives_partial(tmp_path):
    """`_atomic_write_json` uses tmp+rename — target file appears only
    after the payload is fully written. Guards against a crash mid-write
    leaving a truncated checkpoint."""
    target = tmp_path / "checkpoint.json"
    _ingest_batch._atomic_write_json(target, {"completed": {}, "failed": {}})
    assert not (tmp_path / "checkpoint.json.tmp").exists()
    assert json.loads(target.read_text()) == {"completed": {}, "failed": {}}


# ---------- `agent ingest` batch mode via multi-PDF ----------


def test_agent_ingest_two_pdfs_auto_batches(wiki, monkeypatch):
    """`researchwiki agent ingest <pdf1> <pdf2>` auto-activates batch mode
    without needing an explicit --workers. Batch dir gets created; both
    PDFs run through the worker; subcommand recorded is ["agent","ingest"]."""
    pdfs = _make_pdfs(wiki, ["a.pdf", "b.pdf"])
    calls: list[str] = []
    arg_capture: list = []
    monkeypatch.setattr(
        _ingest_batch, "_worker", _stub_worker({p: 0 for p in pdfs}, calls, arg_capture)
    )
    rc = agent_cli.main(["ingest", *pdfs])
    assert rc == 0
    assert set(calls) == set(pdfs)
    for subcmd, _ in arg_capture:
        assert subcmd == ["agent", "ingest"]
    plan = json.loads((_only_batch(wiki) / "plan.json").read_text())
    assert plan["subcommand"] == ["agent", "ingest"]
    assert set(plan["inputs"]) == set(pdfs)
    assert plan["workers"] == 4
    assert plan["workers_explicit"] is False
    assert plan["relay_watch"] is False


# ---------- relay prompt mirroring ----------
#
# The scan is tested by calling it. The previous version of these tests drove the
# polling thread and drained `capsys` from the main thread while the thread wrote
# to `sys.stderr`: pytest's capture snap is getvalue -> seek -> truncate, which is
# not atomic against a concurrent write, and because a dropped notice is never
# re-emitted the loss was permanent and the assertion flaked. It also started a
# NON-daemon thread with no try/finally, so any exception before `stop.set()` hung
# the whole session at interpreter exit instead of failing a test.


@pytest.fixture
def relay_root(tmp_path, monkeypatch):
    """cwd-rooted `.llm-relay/pending/`, matching how `relay` resolves it.

    `relay` binds `wiki_root` at import, so it follows the real cwd rather than a
    monkeypatched `paths.wiki_root` — chdir is what production actually sees.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    pending = tmp_path / ".llm-relay" / "pending"
    pending.mkdir(parents=True)
    return pending


def _write_prompt(pending: Path, op_id: str, **fields) -> Path:
    payload = {"op_id": op_id, "phase": "author", "stem": f"{op_id}-stem"}
    payload.update(fields)
    path = pending / f"{op_id}.prompt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_scan_announces_each_request_once(relay_root):
    _write_prompt(relay_root, "aaa")
    seen: dict = {}
    first = _ingest_batch._scan_relay_prompts(seen)
    assert len(first) == 1
    assert "aaa-stem" in first[0]
    assert "aaa.prompt.json" in first[0]
    # Parity with the in-process notice: both file paths, and the deadline.
    assert "Awaiting response at" in first[0]
    assert "aaa.response.json" in first[0]
    assert "timeout" in first[0]
    assert _ingest_batch._scan_relay_prompts(seen) == []


def test_scan_surfaces_a_request_left_by_an_earlier_run(relay_root):
    """A prompt already on disk is the documented resume case, so it must be
    announced — the resumed worker skips its own handoff notice when the file
    already exists, leaving the mirror as the only channel."""
    _write_prompt(relay_root, "carried")
    assert len(_ingest_batch._scan_relay_prompts({})) == 1


def test_scan_re_announces_a_re_created_request(relay_root):
    """op_ids are content-addressed and `call_chat_relay` unlinks the pair on
    success, so the identical path legitimately reappears — `phases.commit`
    retries `keywords` with the same prompt, deriving the same op_id. A
    path-only `seen` set announced that once and stayed silent forever after."""
    path = _write_prompt(relay_root, "reused")
    seen: dict = {}
    assert len(_ingest_batch._scan_relay_prompts(seen)) == 1
    path.unlink()
    assert _ingest_batch._scan_relay_prompts(seen) == []
    assert seen == {}, "a vanished request is pruned, not remembered"
    recreated = _write_prompt(relay_root, "reused")
    os.utime(recreated, (1, 1))  # distinct mtime, coarse-clock-proof
    assert len(_ingest_batch._scan_relay_prompts(seen)) == 1


def test_scan_labels_an_abandoned_request_instead_of_listing_it_as_live(relay_root):
    """`prompts/chat-relay.md` tells responders not to answer abandoned requests
    speculatively, so the mirror must not print them in the same words as live
    ones — answering one writes a response nothing consumes, which then becomes
    a silent cache hit for any later byte-identical prompt."""
    old = _write_prompt(relay_root, "stale")
    ancient = time.time() - (_ingest_batch._RELAY_STALE_AFTER + 600)
    os.utime(old, (ancient, ancient))
    (message,) = _ingest_batch._scan_relay_prompts({})
    assert "STALE" in message
    assert "Do not answer it speculatively" in message
    assert "Awaiting response at" not in message


def test_scan_reports_a_retry_chain(relay_root):
    _write_prompt(relay_root, "retried", retry_of="deadbeef")
    (message,) = _ingest_batch._scan_relay_prompts({})
    assert "retry of deadbeef" in message


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff\xfe not utf-8 at all",      # UnicodeDecodeError (a ValueError)
        b"null",                            # valid JSON, not an object
        b"[1, 2, 3]",                       # ditto
        b"{truncated",                      # JSONDecodeError
    ],
)
def test_scan_survives_an_unreadable_request(relay_root, raw):
    """The narrow `(OSError, JSONDecodeError)` this replaces let a
    UnicodeDecodeError or a non-object payload's AttributeError kill the watcher
    thread, after which every later paper ran unmirrored while `relay_watch`
    still claimed otherwise."""
    (relay_root / "bad.prompt.json").write_bytes(raw)
    good = _write_prompt(relay_root, "good")
    seen: dict = {}
    messages = _ingest_batch._scan_relay_prompts(seen)
    assert any("good-stem" in m for m in messages)
    assert good in seen
    # Unrecorded, so a transient read failure is retried rather than dropped.
    assert (relay_root / "bad.prompt.json") not in seen


def test_scan_is_silent_when_there_is_no_relay_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _ingest_batch._scan_relay_prompts({}) == []


def test_mirror_context_manager_starts_and_stops_a_daemon_thread(relay_root):
    """Daemon, and joined on the way out even if the body raises — a stuck
    non-daemon watcher blocks interpreter shutdown rather than failing."""
    import threading

    before = threading.active_count()
    with _ingest_batch._mirror_relay_prompts(True):
        watcher = [t for t in threading.enumerate() if t.name == "relay-prompt-watcher"]
        assert len(watcher) == 1
        assert watcher[0].daemon is True
    assert not [t for t in threading.enumerate() if t.name == "relay-prompt-watcher"]

    with pytest.raises(RuntimeError):
        with _ingest_batch._mirror_relay_prompts(True):
            raise RuntimeError("body blew up")
    assert not [t for t in threading.enumerate() if t.name == "relay-prompt-watcher"]
    assert threading.active_count() == before


def test_mirror_starts_nothing_when_disabled(relay_root):
    import threading

    with _ingest_batch._mirror_relay_prompts(False):
        assert not [
            t for t in threading.enumerate() if t.name == "relay-prompt-watcher"
        ]


def test_watcher_loop_body_prints_to_stderr(relay_root, capsys):
    """One check that the loop body is wired to stderr at all.

    Runs it on the calling thread with `stop` pre-set: the loop scans once, then
    its first `stop.wait()` returns True and it returns. No thread, no sleep, and
    no concurrent capture — the previous shape started a thread and joined it
    before setting `stop`, which simply blocked for the join timeout.
    """
    import threading

    _write_prompt(relay_root, "live")
    stop = threading.Event()
    stop.set()
    _ingest_batch._watch_relay_prompts(stop, interval=0)
    assert "live-stem" in capsys.readouterr().err


def test_watcher_loop_exits_promptly_once_stopped(relay_root):
    """The join in `_mirror_relay_prompts` has a 1 s budget, so the loop must
    notice `stop` within one interval rather than sleeping out a long poll."""
    import threading

    stop = threading.Event()
    stop.set()
    started = time.monotonic()
    _ingest_batch._watch_relay_prompts(stop, interval=5)
    assert time.monotonic() - started < 1.0, "stop must short-circuit the wait"


def test_agent_ingest_single_pdf_does_not_batch(wiki, monkeypatch):
    """`agent ingest <one-pdf>` keeps the historical single-PDF path — no
    batch dir, no worker call. This is the backwards-compat guarantee."""
    pdfs = _make_pdfs(wiki, ["only.pdf"])
    calls: list[str] = []
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}, calls))

    # Force the single-PDF path to fail cleanly rather than hit the real
    # runner. run_ingest is imported at module top; patch it out.
    def raiser(*a, **kw):
        raise RuntimeError("single-path was reached")

    monkeypatch.setattr(agent_cli, "run_ingest", raiser)
    rc = agent_cli.main(["ingest", pdfs[0]])
    # The RuntimeError falls through to the generic handler, which returns 3
    # (internal bug) after printing the traceback — not 2, which the contract
    # reserves for environment failures. What matters here: the batch driver
    # was never entered.
    assert rc == 3
    assert calls == []
    assert not (wiki / ".ingest").exists() or not any(
        (wiki / ".ingest").glob("batch-*")
    )


def test_agent_ingest_workers_1_with_one_pdf_still_batches(wiki, monkeypatch):
    """`-w 1` is a valid opt-in to batch mode even for one PDF — you get a
    checkpoint dir on an otherwise-serial run. Useful for very long single
    ingests that you want to be able to resume."""
    pdfs = _make_pdfs(wiki, ["solo.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[0]: 0}))
    rc = agent_cli.main(["ingest", "-w", "1", *pdfs])
    assert rc == 0
    batch = _only_batch(wiki)
    assert (batch / "checkpoint.json").exists()
    assert json.loads((batch / "plan.json").read_text())["workers_explicit"] is True


def test_agent_ingest_batch_rejects_per_pdf_overrides(wiki, monkeypatch, capsys):
    """Per-PDF flags (--doi, --title, etc.) don't make sense on a batch — the
    override would silently apply to all N. Reject with a clear message."""
    pdfs = _make_pdfs(wiki, ["a.pdf", "b.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({p: 0 for p in pdfs}))
    rc = agent_cli.main(["ingest", "--doi", "10.1/x", *pdfs])
    assert rc == 1  # bad command line → 1, per the exit-code contract
    err = capsys.readouterr().err
    assert "per-PDF" in err
    assert "--doi" in err


def test_agent_ingest_passthrough_flags_recorded(wiki, monkeypatch):
    """Only non-default flags reach the per-worker subprocess. plan.json
    stores them so `--resume` replays the exact same behavior."""
    pdfs = _make_pdfs(wiki, ["a.pdf", "b.pdf"])
    arg_capture: list = []
    monkeypatch.setattr(
        _ingest_batch,
        "_worker",
        _stub_worker({p: 0 for p in pdfs}, arg_capture=arg_capture),
    )
    agent_cli.main(
        ["ingest", "--stub", "--no-semantic", "-n", "3", "--no-llm-reconcile", *pdfs]
    )
    plan = json.loads((_only_batch(wiki) / "plan.json").read_text())
    assert "--stub" in plan["extra_args"]
    assert "--no-semantic" in plan["extra_args"]
    assert "--no-llm-reconcile" in plan["extra_args"]
    assert "-n" in plan["extra_args"] and "3" in plan["extra_args"]
    for _, extra in arg_capture:
        assert extra == plan["extra_args"]


# ---------- `agent ingest --resume` ----------


def test_agent_ingest_resume_skips_completed(wiki, monkeypatch):
    """Load-bearing case: 5-PDF batch finishes 3 successes + 2 failures,
    process dies, `--resume` re-runs only the 2 failures."""
    pdfs = _make_pdfs(wiki, [f"p{i}.pdf" for i in range(5)])
    monkeypatch.setattr(
        _ingest_batch, "_worker", _stub_worker({pdfs[3]: 2, pdfs[4]: 2})
    )
    rc = agent_cli.main(["ingest", *pdfs])
    assert rc == 1
    batch_dir = _only_batch(wiki)

    touched: list[str] = []
    monkeypatch.setattr(
        _ingest_batch, "_worker", _stub_worker({p: 0 for p in pdfs}, calls=touched)
    )
    rc = agent_cli.main(["ingest", "--resume", str(batch_dir)])
    assert rc == 0
    assert set(touched) == set(pdfs[3:])
    state = json.loads((batch_dir / "checkpoint.json").read_text())
    assert set(state["completed"]) == set(pdfs)
    assert state["failed"] == {}


def test_agent_ingest_resume_no_retry_skips_failures(wiki, monkeypatch):
    pdfs = _make_pdfs(wiki, ["ok.pdf", "boom.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[1]: 2}))
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)

    touched: list[str] = []
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}, calls=touched))
    rc = agent_cli.main(["ingest", "--resume", str(batch_dir), "--no-retry"])
    assert rc == 1  # failed set non-empty
    assert touched == []


@pytest.mark.parametrize("rc", [1, 3])
def test_resume_default_does_not_retry_deterministic_failures(
    wiki, monkeypatch, rc, capsys
):
    """A worker subprocess re-runs the same argv every time, so a code-1
    (bad flag) or code-3 (internal bug) failure would fail identically on
    retry — `--resume` without `--no-retry` must leave those alone rather
    than burning a retry on something that can't change. Only code 2
    (environment error — state.db locked, index missing, provider
    unreachable) is worth re-running automatically."""
    pdfs = _make_pdfs(wiki, ["ok.pdf", "boom.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[1]: rc}))
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)

    touched: list[str] = []
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}, calls=touched))
    rc_resume = agent_cli.main(["ingest", "--resume", str(batch_dir)])
    assert rc_resume == 1  # failed set still non-empty
    assert touched == []  # not retried
    assert "boom.pdf" in capsys.readouterr().err

    state = json.loads((batch_dir / "checkpoint.json").read_text())
    assert pdfs[1] in state["failed"]  # left in place, not silently dropped


def test_resume_default_retries_environment_failures(wiki, monkeypatch):
    """The complement of the above: code 2 is exactly the class `--resume`
    should still auto-retry."""
    pdfs = _make_pdfs(wiki, ["ok.pdf", "flaky.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[1]: 2}))
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)

    touched: list[str] = []
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}, calls=touched))
    rc = agent_cli.main(["ingest", "--resume", str(batch_dir)])
    assert rc == 0
    assert touched == [pdfs[1]]


def test_resume_missing_returncode_stays_retryable(wiki, monkeypatch):
    """A checkpoint written before this feature existed (or a worker killed by
    a signal rather than returning normally) has no reliable exit code — that
    case must keep retrying, matching the pre-existing behavior, rather than
    silently stop retrying anything without a recognized code."""
    pdfs = _make_pdfs(wiki, ["ok.pdf", "legacy-failure.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[1]: 2}))
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)

    # Simulate a pre-existing checkpoint with no `returncode` field.
    state = json.loads((batch_dir / "checkpoint.json").read_text())
    del state["failed"][pdfs[1]]["returncode"]
    (batch_dir / "checkpoint.json").write_text(json.dumps(state))

    touched: list[str] = []
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}, calls=touched))
    rc = agent_cli.main(["ingest", "--resume", str(batch_dir)])
    assert rc == 0
    assert touched == [pdfs[1]]


# ---------- `researchwiki ingest` (digest path) ----------


def test_ingest_digest_workers_opts_into_batch(wiki, monkeypatch):
    """`researchwiki ingest inbox/*.pdf -w 2` runs through the batch driver
    with subcommand ["ingest"] (not ["agent","ingest"])."""
    pdfs = _make_pdfs(wiki, ["a.pdf", "b.pdf"])
    arg_capture: list = []
    monkeypatch.setattr(
        _ingest_batch,
        "_worker",
        _stub_worker({p: 0 for p in pdfs}, arg_capture=arg_capture),
    )
    rc = ingest_cli.main(["-w", "2", *pdfs])
    assert rc == 0
    for subcmd, _ in arg_capture:
        assert subcmd == ["ingest"]
    plan = json.loads((_only_batch(wiki) / "plan.json").read_text())
    assert plan["subcommand"] == ["ingest"]


def test_ingest_digest_serial_default_stays_serial(wiki, monkeypatch):
    """`researchwiki ingest a.pdf b.pdf` without --workers must keep the
    historical serial-loop behavior — no batch dir, no worker calls.

    The digest path was multi-PDF-capable before this refactor via a serial
    for-loop with per-PDF logs and a summary line. Silently switching to
    parallel would change stdout format for existing scripts.
    """
    pdfs = _make_pdfs(wiki, ["a.pdf", "b.pdf"])
    called_worker: list = []
    monkeypatch.setattr(
        _ingest_batch,
        "_worker",
        _stub_worker({p: 0 for p in pdfs}, calls=called_worker),
    )
    # Stub process_one so the serial path doesn't try to open the empty PDFs.
    monkeypatch.setattr(ingest_cli, "process_one", lambda *a, **kw: None)
    ingest_cli.main([*pdfs])
    assert called_worker == []
    assert not (wiki / ".ingest").exists() or not any(
        (wiki / ".ingest").glob("batch-*")
    )


def test_ingest_digest_workers_rejects_doi(wiki, monkeypatch):
    pdfs = _make_pdfs(wiki, ["a.pdf", "b.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({p: 0 for p in pdfs}))
    # The digest path rejects via `parser.error`, so argparse's own 2 escapes
    # this module. `__main__.main` is where that gets remapped onto the
    # contract's 1 — see test_exit_codes.py::test_argparse_usage_error_remapped.
    with pytest.raises(SystemExit) as e:
        ingest_cli.main(["-w", "2", "--doi", "10.1/x", *pdfs])
    assert e.value.code == 2


def test_ingest_digest_resume_replays_subcommand(wiki, monkeypatch):
    """A batch started via `researchwiki ingest -w 2` resumes back to
    `researchwiki ingest ...`, not `researchwiki agent ingest`. plan.json's
    subcommand field is the source of truth."""
    pdfs = _make_pdfs(wiki, ["a.pdf", "b.pdf", "c.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[2]: 2}))
    ingest_cli.main(["-w", "2", *pdfs])
    batch_dir = _only_batch(wiki)

    arg_capture: list = []
    monkeypatch.setattr(
        _ingest_batch,
        "_worker",
        _stub_worker({p: 0 for p in pdfs}, arg_capture=arg_capture),
    )
    rc = ingest_cli.main(["--resume", str(batch_dir)])
    assert rc == 0
    for subcmd, _ in arg_capture:
        assert subcmd == ["ingest"]


# ---------- shared: input validation, CLI misuse ----------


def test_missing_pdf_rejected(wiki):
    # A path that doesn't exist is a bad command line (1), not a broken
    # environment (2) — `_resolve_inputs` is validating argv, not the disk.
    with pytest.raises(SystemExit) as e:
        agent_cli.main(["ingest", "/does/not/exist.pdf", "/also/nope.pdf"])
    assert e.value.code == 1


def test_non_pdf_rejected(wiki, tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    pdfs = _make_pdfs(wiki, ["a.pdf"])
    with pytest.raises(SystemExit) as e:
        agent_cli.main(["ingest", pdfs[0], str(txt)])
    assert e.value.code == 1


def test_resume_bad_dir_returns_1(wiki, tmp_path):
    # `--resume <not a batch dir>` is a bad command line, so code 1 per the
    # exit-code contract — not 2, which means the environment is broken.
    rc = agent_cli.main(["ingest", "--resume", str(tmp_path / "nope")])
    assert rc == 1


def test_agent_ingest_no_args_returns_1(wiki, capsys):
    rc = agent_cli.main(["ingest"])
    assert rc == 1
    assert "need PDF" in capsys.readouterr().err


# ---------- post-run epilogue (evolve + concept-attach visibility) ----------


def _seed_batch_dir(
    root: Path, worker_stem_to_log: dict[str, str]
) -> tuple[Path, dict]:
    """Build a fake `.ingest/batch-<ts>/` complete with `checkpoint.json` +
    per-worker log files. Returns (batch_dir, state) suitable for feeding
    into `_print_batch_epilogue`.
    """
    batch_dir = root / ".ingest" / "batch-test"
    batch_dir.mkdir(parents=True)
    completed = {}
    for stem, log_body in worker_stem_to_log.items():
        pdf = str((root / "inbox" / f"{stem}.pdf").resolve())
        completed[pdf] = {"input": pdf, "status": "completed", "returncode": 0}
        _ingest_batch._worker_log_path(batch_dir, pdf).write_text(log_body)
    return batch_dir, {"completed": completed, "failed": {}}


def test_epilogue_aggregates_evolve_proposals_across_workers(wiki, capsys):
    """Each worker's `[agent] evolve → N proposal(s) … actionable=N` line
    lives inside `worker-{stem}.log`. Without the epilogue, users only see
    `[i/N] ok:` per completion and miss the proposal counts entirely — the
    Ahmad/Santos ingest-batch that motivated this fix."""
    log_a = (
        "[agent] evolve   → 2 proposal(s) at /path/to/ahmad-evolution-proposals/  "
        "(knn=8 above_thr=4 judged=4 actionable=2)\n"
    )
    log_b = (
        "[agent] evolve   → 1 proposal(s) at /path/to/santos-evolution-proposals/  "
        "(knn=8 above_thr=4 judged=4 actionable=1)\n"
    )
    log_c = "[agent] evolve   → no actionable proposals (knn=8 above_thr=4 judged=4 actionable=0)\n"
    batch_dir, state = _seed_batch_dir(
        wiki, {"ahmad": log_a, "santos": log_b, "nitu": log_c}
    )

    _ingest_batch._print_batch_epilogue(batch_dir, state)

    out = capsys.readouterr().out
    assert "post-run summary" in out
    assert "3 actionable proposal(s) across 2 paper(s)" in out
    assert "ahmad: 2 actionable" in out
    assert "santos: 1 actionable" in out
    # nitu has zero actionable → not listed per-paper.
    assert "nitu:" not in out


def test_epilogue_surfaces_concept_attach_near_miss(wiki, capsys):
    """The FH-shaped miss: a concept-hub whose vocabulary the ingested paper
    only uses in body prose logs a near-miss inside the worker log. The
    epilogue surfaces every one so reviewers can decide to attach manually."""
    log = (
        "[concepts] concept-attach: skipped ahmad-2026-fh→familial-hypercholesterolemia, "
        "term only in body prose (not in kc/results/methodology)\n"
    )
    batch_dir, state = _seed_batch_dir(wiki, {"ahmad": log})

    _ingest_batch._print_batch_epilogue(batch_dir, state)

    out = capsys.readouterr().out
    assert "concept-attach near-miss: 1" in out
    assert "ahmad → familial-hypercholesterolemia" in out


def test_epilogue_reports_concept_attach_joined(wiki, capsys):
    log = "[concepts] concept-attach paper-xyz: joined hub-one, hub-two\n"
    batch_dir, state = _seed_batch_dir(wiki, {"paper-xyz": log})

    _ingest_batch._print_batch_epilogue(batch_dir, state)

    out = capsys.readouterr().out
    assert "concept-attach: 2 hub attachment(s)" in out
    assert "paper-xyz → hub-one" in out
    assert "paper-xyz → hub-two" in out


def test_epilogue_silent_when_nothing_to_report(wiki, capsys):
    """A batch that produced no actionable evolve proposals and no
    concept-attach signal shouldn't add noise — silence is a feature."""
    log = "[agent] evolve   → no actionable proposals (knn=8 above_thr=4 judged=4 actionable=0)\n"
    batch_dir, state = _seed_batch_dir(wiki, {"paper-abc": log})

    _ingest_batch._print_batch_epilogue(batch_dir, state)

    assert capsys.readouterr().out == ""


def test_epilogue_missing_log_does_not_crash(wiki, capsys):
    """One worker log went missing (renamed, moved) — epilogue must degrade
    gracefully rather than kill the batch's exit code path."""
    batch_dir = wiki / ".ingest" / "batch-test"
    batch_dir.mkdir(parents=True)
    pdf = str((wiki / "inbox" / "ghost.pdf").resolve())
    state = {"completed": {pdf: {"input": pdf, "status": "completed"}}, "failed": {}}

    # No log file exists → parser returns empty summary → epilogue prints nothing.
    _ingest_batch._print_batch_epilogue(batch_dir, state)
    assert capsys.readouterr().out == ""


def test_run_batch_calls_epilogue_on_clean_completion(wiki, monkeypatch, capsys):
    """End-to-end: a normal (non-interrupted) run must invoke the epilogue.
    Stub `_worker` to emit an evolve-actionable log line and assert the
    epilogue's aggregate reaches stdout."""
    pdfs = _make_pdfs(wiki, ["p1.pdf"])

    def worker(pdf_path, batch_dir, subcommand, extra_args):
        _ingest_batch._worker_log_path(batch_dir, pdf_path).write_text(
            "[agent] evolve   → 3 proposal(s) at /path/to/p1-evolution-proposals/  "
            "(knn=8 above_thr=4 judged=4 actionable=3)\n"
        )
        return {"input": pdf_path, "status": "completed", "returncode": 0}

    monkeypatch.setattr(_ingest_batch, "_worker", worker)

    # `-w 1` opts a single-PDF invocation into batch mode.
    rc = agent_cli.main(["ingest", "-w", "1", *pdfs])
    assert rc == 0
    out = capsys.readouterr().out
    assert "post-run summary" in out
    assert "3 actionable" in out
    assert "p1: 3 actionable" in out


def test_cli_no_longer_lists_ingest_batch():
    """The refactor removed `researchwiki ingest-batch` — verify the
    auto-discovery no longer surfaces it (the module is now
    `_ingest_batch.py`, hidden by leading underscore)."""
    from researchwiki.__main__ import _discover_tasks

    tasks = _discover_tasks()
    assert "ingest-batch" not in tasks
    # But the two hosts remain.
    assert "agent" in tasks
    assert "ingest" in tasks


# ---------- per-input args ----------
#
# The CLI still refuses `--doi`/`--title`/`--authors`/`--year` in batch mode
# (`agent._BATCH_INCOMPATIBLE_FLAGS`), and rightly so: one `--doi` has no
# meaning across N PDFs. A programmatic caller — `researchwiki import apply` —
# holds a *different* DOI for every input, which is the case that guard was
# never about.


def test_per_input_args_reach_only_their_own_worker(tmp_path, monkeypatch):
    from researchwiki.tasks import _ingest_batch

    a = tmp_path / "a.pdf"
    a.write_bytes(b"%PDF-1.4\n")
    b = tmp_path / "b.pdf"
    b.write_bytes(b"%PDF-1.4\n")
    seen = {}

    def fake_worker(pdf_path, batch_dir, subcommand, extra_args):
        seen[Path(pdf_path).name] = list(extra_args)
        return {"input": pdf_path, "status": "completed", "returncode": 0}

    monkeypatch.setattr(_ingest_batch, "_worker", fake_worker)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()

    _ingest_batch.new_batch(
        [str(a), str(b)],
        ["agent", "ingest"],
        ["-n", "1"],
        workers=1,
        per_input_args={str(a.resolve()): ["--doi", "10.1234/a"]},
    )
    assert seen["a.pdf"] == ["-n", "1", "--doi", "10.1234/a"]
    assert seen["b.pdf"] == ["-n", "1"]  # no leakage between inputs


def test_per_input_args_are_persisted_for_resume(tmp_path, monkeypatch):
    import json as _json

    from researchwiki.tasks import _ingest_batch

    a = tmp_path / "a.pdf"
    a.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        _ingest_batch,
        "_worker",
        lambda p, d, s, e: {"input": p, "status": "completed", "returncode": 0},
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    _ingest_batch.new_batch(
        [str(a)],
        ["agent", "ingest"],
        [],
        workers=1,
        per_input_args={str(a.resolve()): ["--year", "2024"]},
    )
    batch = sorted((tmp_path / ".ingest").glob("batch-*"))[-1]
    plan = _json.loads((batch / "plan.json").read_text())
    assert plan["per_input_args"][str(a.resolve())] == ["--year", "2024"]


def test_a_plan_without_per_input_args_still_resumes(tmp_path, monkeypatch):
    """Additive key, not a schema bump: batch dirs written before this existed
    must keep resuming."""
    import json as _json

    from researchwiki.tasks import _ingest_batch

    a = tmp_path / "a.pdf"
    a.write_bytes(b"%PDF-1.4\n")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    batch = tmp_path / ".ingest" / "batch-legacy"
    batch.mkdir(parents=True)
    (batch / "plan.json").write_text(
        _json.dumps(
            {
                "started_at": "2026-01-01T00:00:00",
                "subcommand": ["agent", "ingest"],
                "workers": 1,
                "inputs": [str(a.resolve())],
                "extra_args": [],
            }
        )
    )
    seen = []
    monkeypatch.setattr(
        _ingest_batch,
        "_worker",
        lambda p, d, s, e: (
            seen.append(list(e)) or {"input": p, "status": "completed", "returncode": 0}
        ),
    )
    _ingest_batch.resume_batch(batch, no_retry=False)
    assert seen == [[]]


# ---------- unresumable inputs (WI-1 fix c) ----------
#
# A worker that dies *after* `_move_pdf` shutil.move'd the PDF out of inbox/
# records nothing in the checkpoint (that write only happens once the
# subprocess returns). The old resume re-queued the vanished path; these pin
# that it is now classified terminal and explained instead.


def test_resume_skips_input_whose_pdf_vanished(wiki, monkeypatch, capsys):
    """The load-bearing case: promote moved the PDF, then the worker died."""
    pdfs = _make_pdfs(wiki, ["ok.pdf", "moved.pdf"])

    # Only `ok.pdf` reports back; `moved.pdf` leaves no checkpoint record,
    # exactly as a mid-promote crash would.
    def worker(pdf_path, batch_dir, subcommand, extra_args):
        if pdf_path == pdfs[1]:
            raise KeyboardInterrupt
        return {"input": pdf_path, "status": "completed", "returncode": 0}

    monkeypatch.setattr(_ingest_batch, "_worker", worker)
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)

    # Simulate promote having moved it into papers/.
    Path(pdfs[1]).unlink()

    touched: list[str] = []
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}, calls=touched))
    rc = agent_cli.main(["ingest", "--resume", str(batch_dir)])

    assert rc == 1
    assert touched == [], "a vanished input must not be re-queued"
    state = json.loads((batch_dir / "checkpoint.json").read_text())
    assert pdfs[1] in state["unresumable"]
    assert state["unresumable"][pdfs[1]]["reason"] == "input PDF no longer on disk"
    err = capsys.readouterr().err
    assert "no longer on disk" in err
    assert "moved.pdf" in err
    assert "recovery.md" in err


def test_unresumable_is_terminal_across_two_resumes(wiki, monkeypatch):
    """Second resume must not re-report or re-queue it — the bucket is
    terminal, like `completed`, not a retry queue."""
    pdfs = _make_pdfs(wiki, ["gone.pdf", "keep.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[0]: 2}))
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)
    Path(pdfs[0]).unlink()

    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}))
    assert agent_cli.main(["ingest", "--resume", str(batch_dir)]) == 1

    touched: list[str] = []
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}, calls=touched))
    assert agent_cli.main(["ingest", "--resume", str(batch_dir)]) == 1
    assert touched == []


def test_retryable_failure_with_vanished_pdf_is_unresumable(wiki, monkeypatch):
    """Exit 2 is retryable *by exit code*, but not when the input is gone —
    a promote that failed after the PDF move is exactly this shape."""
    pdfs = _make_pdfs(wiki, ["boom.pdf", "fine.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[0]: 2}))
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)
    Path(pdfs[0]).unlink()

    touched: list[str] = []
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}, calls=touched))
    assert agent_cli.main(["ingest", "--resume", str(batch_dir)]) == 1
    assert touched == []


def test_worker_log_presence_distinguishes_crash_from_deletion(
    wiki, monkeypatch, capsys
):
    """`_worker` opens its log before `subprocess.run`, so the log's
    existence already records that the subprocess launched — no extra
    bookkeeping needed. Absent log ⇒ the user moved the file."""
    pdfs = _make_pdfs(wiki, ["never-started.pdf", "other.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[0]: 2}))
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)
    # Stub worker wrote no log; remove the input as if the user moved it.
    Path(pdfs[0]).unlink()

    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}))
    agent_cli.main(["ingest", "--resume", str(batch_dir)])

    state = json.loads((batch_dir / "checkpoint.json").read_text())
    assert state["unresumable"][pdfs[0]]["worker_started"] is False
    assert "moved or deleted outside this batch" in capsys.readouterr().err


def test_denominator_counts_unresumable(wiki, monkeypatch, capsys):
    """`[i/N]` must keep counting the batch the user submitted; an
    unresumable input is terminal, not absent."""
    pdfs = _make_pdfs(wiki, ["a.pdf", "b.pdf", "c.pdf"])
    monkeypatch.setattr(
        _ingest_batch, "_worker", _stub_worker({pdfs[1]: 2, pdfs[2]: 2})
    )
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)
    Path(pdfs[1]).unlink()

    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}))
    capsys.readouterr()
    agent_cli.main(["ingest", "--resume", str(batch_dir)])
    out = capsys.readouterr().out
    assert "/3]" in out, f"denominator lost an input: {out!r}"


def test_all_inputs_present_is_unaffected(wiki, monkeypatch):
    """Guard against the check firing on a healthy resume."""
    pdfs = _make_pdfs(wiki, ["a.pdf", "b.pdf"])
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({pdfs[1]: 2}))
    agent_cli.main(["ingest", *pdfs])
    batch_dir = _only_batch(wiki)

    touched: list[str] = []
    monkeypatch.setattr(_ingest_batch, "_worker", _stub_worker({}, calls=touched))
    assert agent_cli.main(["ingest", "--resume", str(batch_dir)]) == 0
    assert touched == [pdfs[1]]
    state = json.loads((batch_dir / "checkpoint.json").read_text())
    assert "unresumable" not in state or state["unresumable"] == {}
