"""Crash-safe batch driver shared by `researchwiki agent ingest` and
`researchwiki ingest`.

Both commands take `nargs="+"` on their PDF positional. When invoked with
multiple PDFs (or an explicit `--workers`/`--resume`), they delegate here
so a mid-batch crash is recoverable: a `checkpoint.json` records every
completion and failure atomically, and `--resume <batch-dir>` picks up
where the previous run left off.

This module is *not* a CLI subcommand (leading-underscore filename hides
it from `__main__.py`'s auto-discovery). Callers import `new_batch()` /
`resume_batch()` and pass the subcommand they want each worker to run —
`["agent", "ingest"]` for the agent path, `["ingest"]` for the digest
path. That subcommand is recorded in `plan.json` so `--resume` doesn't
need the user to remember which command started the batch.

Design notes:
  - Subprocess-per-worker (`python -m researchwiki <subcommand> <pdf>`) —
    full isolation across lru_cache and DB connections, matches CLAUDE.md's
    bash-fan-out mental model. One crash can't corrupt siblings.
  - `concurrent.futures.ThreadPoolExecutor` over subprocesses (threads are
    I/O-bound on `subprocess.wait`).
  - Absolute paths as checkpoint keys — `--resume` works from any cwd.
  - Atomic checkpoint via tmp+rename. Every worker completion re-writes.
  - SIGINT cancels unstarted futures, lets in-flight subprocesses finish.
    Killing mid-ingest could strand state.db mid-transaction; the 5s
    busy_timeout doesn't help if the writer never comes back. Second
    Ctrl-C SIGKILLs the whole process group.

Pattern lifted from hermes-agent `batch_runner.py::BatchRunner`.
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


def _now_iso() -> str:
    """Local-time ISO stamp, second precision — matches `ingested_at`."""
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def _batch_dir_for_new_run() -> Path:
    """Timestamped batch dir under `.ingest/`. `.ingest/` is gitignored."""
    from ..paths import wiki_root
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    d = wiki_root() / ".ingest" / f"batch-{stamp}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON via tmp+rename — rename is atomic on POSIX, so a crash
    mid-write can't leave a truncated checkpoint that fails json.loads on
    resume."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _read_checkpoint(batch_dir: Path) -> dict:
    """Read checkpoint.json. Missing file → empty state (fresh batch)."""
    p = batch_dir / "checkpoint.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_checkpoint(batch_dir: Path, state: dict) -> None:
    _atomic_write_json(batch_dir / "checkpoint.json", state)


def _resolve_inputs(paths: list[str]) -> list[str]:
    """Absolute-path every input. Reject non-existent or non-PDF entries so
    a typo doesn't silently produce a smaller batch than expected. Dedup
    while preserving order (shell glob can occasionally yield dupes)."""
    out: list[str] = []
    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            print(f"ingest-batch: not found: {p}", file=sys.stderr)
            sys.exit(1)
        if path.suffix.lower() != ".pdf":
            print(f"ingest-batch: not a PDF: {p}", file=sys.stderr)
            sys.exit(1)
        out.append(str(path))
    if not out:
        print("ingest-batch: no PDFs given", file=sys.stderr)
        sys.exit(1)
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def _worker_log_path(batch_dir: Path, pdf_path: str) -> Path:
    """Log path for one worker, keyed by basename + a short hash of the full
    input path. The hash disambiguates two same-named PDFs from different
    directories (inputs are absolute paths, so that's legal) — without it the
    second worker clobbered the first's log and the epilogue misattributed
    its evolve/concept counts. Deterministic, so `_print_batch_epilogue` and
    `resume_batch` recompute the same name from the checkpoint's input path."""
    digest = hashlib.sha1(pdf_path.encode("utf-8")).hexdigest()[:8]
    return batch_dir / f"worker-{Path(pdf_path).stem}-{digest}.log"


def _worker(pdf_path: str, batch_dir: Path, subcommand: list[str],
            extra_args: list[str]) -> dict:
    """Run one ingest subprocess. Module-level so tests can monkeypatch it.

    Full per-worker output lands in `worker-{stem}-{hash8}.log` beside the
    checkpoint; the returned record is deliberately minimal — resume only
    needs `input` + `status`.
    """
    log_path = _worker_log_path(batch_dir, pdf_path)
    cmd = [sys.executable, "-m", "researchwiki", *subcommand, pdf_path, *extra_args]
    with log_path.open("w", encoding="utf-8") as log_fp:
        log_fp.write(f"# cmd: {' '.join(cmd)}\n")
        log_fp.flush()
        proc = subprocess.run(cmd, stdout=log_fp, stderr=subprocess.STDOUT, check=False)
    status = "completed" if proc.returncode == 0 else "failed"
    return {"input": pdf_path, "status": status, "returncode": proc.returncode}


def _split_unresumable(batch_dir: Path, pending: list[str]) -> tuple[list[str], list[str]]:
    """Partition `pending` into (still-runnable, unresumable).

    Unresumable = the input path no longer exists. Applied to everything about
    to be queued, including retryable `failed` entries: a promote that failed
    with exit 2 *after* the PDF move is retryable by exit code but not by path.
    """
    runnable: list[str] = []
    gone: list[str] = []
    for pdf in pending:
        (runnable if Path(pdf).exists() else gone).append(pdf)
    return runnable, gone


def _report_unresumable(batch_dir: Path, unresumable: list[str], state: dict) -> None:
    """Explain the unresumable bucket and what to check, on stderr.

    Shaped like the `skipped_non_retryable` block below it — same layout,
    different cause — so the two read as one family in a resume's output.
    """
    n = len(unresumable)
    print(
        f"ingest-batch: {n} input(s) no longer on disk — not re-runnable as-is.",
        file=sys.stderr,
    )
    for pdf in unresumable:
        rec = (state.get("unresumable") or {}).get(pdf) or {}
        if rec.get("worker_started"):
            why = ("worker started, never recorded a result — promote most likely "
                   "got as far as moving the PDF into papers/ and then died")
        else:
            why = "no worker log — the input was moved or deleted outside this batch"
        print(f"    · {Path(pdf).name}", file=sys.stderr)
        print(f"      {why}", file=sys.stderr)
        if rec.get("worker_started"):
            print(f"      log: {_worker_log_path(batch_dir, pdf).name}", file=sys.stderr)
    print(
        "  These are recorded as terminal and won't be re-queued. Check each by "
        "hand — wiki page present? index.md bullet? back-links? log.md entry? — "
        "then finish or delete the partial page and re-ingest the PDF from "
        "papers/. See prompts/recovery.md § Half-landed promote.",
        file=sys.stderr,
    )


def _should_retry(record: dict) -> bool:
    """Decide whether a failed PDF is worth re-running on `--resume`.

    Each worker is a fresh `researchwiki <subcommand> <pdf>` subprocess with
    the same argv every time, so its exit code (the contract in CLAUDE.md's
    Exit-code contract section) tells us whether a retry can plausibly change
    anything:
      1  bad argv — same flags, same failure, every time. Not retryable.
      2  environment error (state.db locked, index missing, provider
         unreachable) — exactly the transient class a retry might clear.
      3  internal bug — deterministic given the same input; a retry hits the
         same code path. Not retryable (unlike 2, nothing external caused it).
    A missing/unrecognized code (older checkpoint written before this existed,
    or the worker was killed by a signal rather than returning normally) stays
    retryable — that's the pre-existing behavior, and a signal kill (OOM,
    Ctrl-C forwarded to the child) is itself the kind of transient condition
    a retry is meant to survive.
    """
    rc = record.get("returncode")
    return rc not in (1, 3)


# Regexes matching lines the per-worker subprocess emits during ingest.
# `[agent] evolve   → N proposal(s) at /path (…actionable=N)`, or
# `[agent] evolve   → no actionable proposals (…actionable=0)`.
_EVOLVE_ACTIONABLE_RE = re.compile(
    r"\[agent\] evolve\s+→\s+(\d+)\s+proposal.*?actionable=(\d+)"
)
# `[concepts] concept-attach: skipped <paper>→<hub>, term only in body prose …`
_CONCEPT_NEARMISS_RE = re.compile(
    r"\[concepts\] concept-attach:\s+skipped\s+(\S+)→(\S+),\s+term only in body prose"
)
# `[concepts] concept-attach <paper>: joined <hub>[, <hub>…]`
_CONCEPT_JOINED_RE = re.compile(
    r"\[concepts\] concept-attach\s+(\S+):\s+joined\s+(.+?)$", re.MULTILINE
)


def _summarize_worker_log(log_path: Path) -> dict:
    """Parse one worker log for the actions a downstream reviewer needs to
    know about: evolve proposals waiting for review, concept-hub near-misses,
    and hub attachments that already fired.

    Returns `{evolve_actionable: int, concept_joined: [hub, …], concept_near_missed: [hub, …]}`.
    Silent on any I/O error — this runs post-batch as an informational
    summary, not a gate.
    """
    out = {"evolve_actionable": 0,
           "concept_joined": [], "concept_near_missed": []}
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return out
    for m in _EVOLVE_ACTIONABLE_RE.finditer(text):
        out["evolve_actionable"] += int(m.group(2))
    for m in _CONCEPT_NEARMISS_RE.finditer(text):
        out["concept_near_missed"].append(m.group(2))
    for m in _CONCEPT_JOINED_RE.finditer(text):
        for hub in [h.strip() for h in m.group(2).split(",") if h.strip()]:
            out["concept_joined"].append(hub)
    return out


def _print_batch_epilogue(batch_dir: Path, state: dict) -> None:
    """After the pool drains, aggregate per-worker log summaries and surface
    anything the reviewer might otherwise miss inside `.ingest/batch-*/worker-*.log`.

    Two motivating misses this catches:
      - Evolve emits `[agent] evolve → N proposal(s) … actionable=N` inside
        each worker log; without this epilogue the batch driver only prints
        `[N/M] ok: …` and the proposal counts stay hidden.
      - Concept-attach's near-miss + join lines also live inside worker logs.
        Attach-hook signals (e.g. FH-shaped abbreviation misses) belong in
        the batch summary so they surface at the same time as ingest results.

    Silent when nothing to report. Non-fatal (no exceptions escape).
    """
    try:
        completed = list((state.get("completed") or {}).values())
        if not completed:
            return
        totals = {"evolve_actionable": 0}
        per_paper_evolve: list[tuple[str, int]] = []
        joined: list[tuple[str, str]] = []
        near_missed: list[tuple[str, str]] = []
        for rec in completed:
            pdf = rec.get("input")
            if not pdf:
                continue
            stem = Path(pdf).stem
            summary = _summarize_worker_log(_worker_log_path(batch_dir, pdf))
            if summary["evolve_actionable"]:
                per_paper_evolve.append((stem, summary["evolve_actionable"]))
                totals["evolve_actionable"] += summary["evolve_actionable"]
            joined.extend((stem, h) for h in summary["concept_joined"])
            near_missed.extend((stem, h) for h in summary["concept_near_missed"])

        if not (per_paper_evolve or joined or near_missed):
            return

        print()
        print("ingest-batch: post-run summary")
        if per_paper_evolve:
            print(f"  evolve: {totals['evolve_actionable']} actionable "
                  f"proposal(s) across {len(per_paper_evolve)} paper(s) — "
                  f"review under .ingest/*-evolution-proposals/")
            for stem, n in per_paper_evolve:
                print(f"    · {stem}: {n} actionable")
        if joined:
            print(f"  concept-attach: {len(joined)} hub attachment(s)")
            for stem, hub in joined:
                print(f"    · {stem} → {hub}")
        if near_missed:
            print(f"  concept-attach near-miss: {len(near_missed)} "
                  "(body prose only; run `researchwiki concepts` to review)")
            for stem, hub in near_missed:
                print(f"    · {stem} → {hub}")
    except Exception:
        # Epilogue is best-effort; never let a formatting bug affect the
        # ingest exit code.
        pass


def _run_batch(batch_dir: Path, pending: list[str], workers: int,
               subcommand: list[str], extra_args: list[str],
               per_input_args: dict[str, list[str]] | None = None) -> int:
    """Drive the thread pool. Returns 0 iff every pending item completed.

    SIGINT: cancel unstarted futures, let in-flight subprocesses finish
    their current PDF, persist the checkpoint on the way out.
    """
    state = _read_checkpoint(batch_dir)
    state.setdefault("completed", {})
    state.setdefault("failed", {})

    # `failed` is in the denominator because `resume_batch` can now hold back
    # non-retryable failures (exit 1 / 3) instead of queueing them. Without it
    # the `[i/N]` line would silently start counting a smaller batch than the
    # user submitted, which reads as "some inputs vanished". `unresumable` is
    # in it for the same reason — those inputs are terminal, not absent.
    total = (len(state["completed"]) + len(state["failed"])
             + len(state.get("unresumable") or {}) + len(pending))
    done_before = len(state["completed"])
    if not pending:
        print(f"nothing pending ({done_before}/{total} already complete)")
        return 0 if not (state.get("failed") or state.get("unresumable")) else 1

    print(f"ingest-batch: {len(pending)} pending, {done_before} done, "
          f"{workers} worker{'s' if workers != 1 else ''} → {batch_dir}")

    interrupted = False
    # Capture the caller's SIGINT handler so we can restore it in the outer
    # finally. Without this, a batch that installs `SIG_DFL` for second-Ctrl-C
    # leaks that policy into post-batch work (evolve, epilogue, tests).
    try:
        prev_sigint = signal.getsignal(signal.SIGINT)
    except (ValueError, OSError):
        prev_sigint = None
    installed_dfl = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # Per-input flags are composed at submit time rather than passed
        # down, so `_worker` keeps its signature and every existing caller and
        # test is unaffected.
        per_input_args = per_input_args or {}
        futures = {
            pool.submit(_worker, p, batch_dir, subcommand,
                        [*extra_args, *per_input_args.get(p, [])]): p
            for p in pending
        }
        try:
            try:
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    idx = len(state["completed"]) + len(state["failed"]) + 1
                    if result["status"] == "completed":
                        state["completed"][result["input"]] = result
                        tag = "ok"
                    else:
                        state["failed"][result["input"]] = result
                        tag = f"FAIL rc={result['returncode']}"
                    _write_checkpoint(batch_dir, state)
                    print(f"[{idx}/{total}] {tag}: {Path(result['input']).name}")
            except KeyboardInterrupt:
                interrupted = True
                # Second Ctrl-C: hand the signal back to the default handler so
                # the whole process group aborts immediately instead of the
                # doc-comment's promise of "SIGKILL" that no code enforced.
                # Restored in the outer `finally` so post-batch work still runs
                # under the caller's original handler.
                try:
                    signal.signal(signal.SIGINT, signal.SIG_DFL)
                    installed_dfl = True
                except (ValueError, OSError):
                    pass  # signal only settable from main thread; no-op otherwise
                print("\ningest-batch: interrupt received — waiting for in-flight "
                      "workers to finish (Ctrl-C again aborts immediately)",
                      file=sys.stderr)
                # In-flight workers finish and their state.db writes complete,
                # but their results don't make it into checkpoint.json (the
                # as_completed loop is dead). --resume will re-run them. For
                # personal-wiki scale that's a few wasted minutes at most.
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            # Belt-and-braces flush: whatever happened above (normal exit,
            # KeyboardInterrupt, or unexpected exception), the last observed
            # `state` must be on disk before we leave. Every worker completion
            # already writes the checkpoint inside the loop; this extra flush
            # is a no-op on the happy path and a lifeline on the sad one.
            try:
                _write_checkpoint(batch_dir, state)
            except Exception as e:
                print(f"ingest-batch: final checkpoint flush failed: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
            # Restore only if we actually installed SIG_DFL. When the prior
            # handler wasn't Python-settable (getsignal returned None), fall
            # back to the default KeyboardInterrupt handler rather than leaking
            # SIG_DFL into post-batch work (evolve, epilogue, tests).
            if installed_dfl:
                try:
                    signal.signal(
                        signal.SIGINT,
                        prev_sigint if prev_sigint is not None else signal.default_int_handler,
                    )
                except (ValueError, OSError):
                    pass

    if interrupted:
        resume_cmd = f"researchwiki {' '.join(subcommand)} --resume {batch_dir}"
        print(f"ingest-batch: interrupted. "
              f"{len(state['completed'])}/{total} done, "
              f"{len(state['failed'])} failed, "
              f"{total - len(state['completed']) - len(state['failed'])} unstarted. "
              f"Resume: {resume_cmd}",
              file=sys.stderr)
        return 1
    _print_batch_epilogue(batch_dir, state)
    return 0 if not (state["failed"] or state.get("unresumable")) else 1


# ---------- public entry points ----------

def new_batch(pdfs: list[str], subcommand: list[str],
              extra_args: list[str], workers: int,
              per_input_args: dict[str, list[str]] | None = None) -> int:
    """Create a fresh batch dir, drive the ingest pool. Returns exit code.

    `subcommand` is the CLI verb chain each worker's subprocess runs — e.g.
    `["agent", "ingest"]` or `["ingest"]`. Recorded in plan.json so
    `resume_batch` doesn't need it re-supplied.

    `per_input_args` maps an absolute input path to flags for that PDF alone —
    `--doi`, `--title`, `--authors`, `--year`, `--supplementary`. The CLI still
    **refuses** those flags in batch mode (`agent._BATCH_INCOMPATIBLE_FLAGS`),
    and rightly so: one `--doi` has no meaning across N PDFs. But a programmatic
    caller like `researchwiki import apply` holds a *different* DOI for every
    input, which is the whole point of importing from a reference manager, and
    that case the guard was never about. Keys are the same absolute paths
    `_resolve_inputs` produces, so `--resume` still works from any cwd.
    """
    resolved = _resolve_inputs(pdfs)
    batch_dir = _batch_dir_for_new_run()
    plan = {
        "started_at": _now_iso(),
        "subcommand": subcommand,
        "workers": workers,
        "inputs": resolved,
        "extra_args": extra_args,
        "per_input_args": per_input_args or {},
    }
    _atomic_write_json(batch_dir / "plan.json", plan)
    return _run_batch(batch_dir, resolved, workers, subcommand, extra_args,
                      per_input_args)


def resume_batch(batch_dir: Path, no_retry: bool,
                 workers_override: int | None = None) -> int:
    """Continue an interrupted batch. Reads plan.json for its subcommand +
    passthrough flags so semantics don't drift across a restart.

    Retried failures re-enter the queue (dropped from `failed` so the retry
    either promotes to `completed` or overwrites with a fresh failure) —
    but only when `_should_retry` judges the recorded exit code worth
    re-running (code 2, environment error; not 1 or 3, both deterministic
    given the same argv). `--no-retry` overrides this and leaves every
    failure where it is, retryable or not.
    """
    # Both of these mean the `--resume` argument points somewhere that isn't a
    # batch directory — a bad command line, not a broken environment.
    if not batch_dir.is_dir():
        print(f"ingest-batch: not a directory: {batch_dir}", file=sys.stderr)
        return 1
    plan_path = batch_dir / "plan.json"
    if not plan_path.exists():
        print(f"ingest-batch: no plan.json in {batch_dir}", file=sys.stderr)
        return 1
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    state = _read_checkpoint(batch_dir)
    completed = set((state.get("completed") or {}).keys())
    failed = set((state.get("failed") or {}).keys())
    unresumable_before = set((state.get("unresumable") or {}).keys())

    pending: list[str] = []
    skipped_non_retryable: list[str] = []
    for pdf in plan["inputs"]:
        if pdf in completed or pdf in unresumable_before:
            continue
        if pdf in failed:
            if no_retry:
                continue
            if not _should_retry(state["failed"][pdf]):
                skipped_non_retryable.append(pdf)
                continue
        pending.append(pdf)

    # An input whose PDF is no longer on disk cannot be re-run. The common
    # cause is a worker that died *after* `_move_pdf` shutil.move'd the file
    # out of inbox/ into papers/ — the checkpoint records nothing (it is only
    # written once the subprocess returns), so the old code re-queued a path
    # that no longer existed and the half-landed paper was never repaired.
    #
    # Whether the worker ever started is recoverable without new bookkeeping:
    # `_worker` opens its log file *before* `subprocess.run`, so the log's
    # existence already proves the subprocess was launched. That distinguishes
    # "died mid-promote" (log present — inspect the wiki page) from "the user
    # moved or deleted the input" (no log).
    pending, unresumable = _split_unresumable(batch_dir, pending)
    if unresumable:
        state.setdefault("unresumable", {})
        state.setdefault("failed", {})
        for pdf in unresumable:
            state["unresumable"][pdf] = {
                "input": pdf,
                "status": "unresumable",
                "reason": "input PDF no longer on disk",
                "worker_started": _worker_log_path(batch_dir, pdf).exists(),
            }
            # The buckets must stay disjoint: `_run_batch` sums them for the
            # `[i/N]` denominator, so an input left in `failed` *and* moved to
            # `unresumable` inflates N. A retryable-by-exit-code failure whose
            # PDF has since vanished belongs only in the latter.
            state["failed"].pop(pdf, None)
        _write_checkpoint(batch_dir, state)
        _report_unresumable(batch_dir, unresumable, state)

    if skipped_non_retryable:
        print(f"ingest-batch: {len(skipped_non_retryable)} failure(s) not retried "
              f"— exit code 1 (bad input) or 3 (internal bug), so the same argv "
              f"would fail the same way. Fix the cause, then re-run the PDF "
              f"directly to see the error:", file=sys.stderr)
        for pdf in skipped_non_retryable:
            print(f"    · {Path(pdf).name}  "
                  f"(see {_worker_log_path(batch_dir, pdf).name})",
                  file=sys.stderr)

    if not no_retry and failed:
        state.setdefault("failed", {})
        for pdf in list(state["failed"]):
            if pdf in pending:
                del state["failed"][pdf]
        _write_checkpoint(batch_dir, state)

    workers = workers_override if workers_override is not None else plan.get("workers", 4)
    subcommand = plan.get("subcommand", ["agent", "ingest"])  # legacy default
    # `.get` with a default, so a batch dir written before per-input args
    # existed still resumes — the key is additive, not a schema bump.
    return _run_batch(batch_dir, pending, workers, subcommand,
                      plan.get("extra_args", []),
                      plan.get("per_input_args", {}))
