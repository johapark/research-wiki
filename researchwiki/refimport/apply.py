"""Stage `ready` records into `inbox/` and hand them to `agent ingest`.

The whole phase is a copy plus a dispatch. There is no journal, no staging
directory and no backup, because there is nothing here that a crash could leave
half-done: the only mutation is copying a PDF, and everything after that belongs
to `_ingest_batch`, which already keeps a crash-safe `checkpoint.json`.
Recovery is `researchwiki agent ingest --resume <batch-dir>` — the path users
already know.

**Liveness is re-checked here, and only liveness.** Pairing, triage verdicts,
derived stems and argv are frozen in the manifest, so `apply` cannot reach a
different conclusion than the `inspect` the user read. But *whether a paper is
already in the wiki* is a fact about the world now, not a decision made at
inspect time — and without re-checking it, `apply --limit 30` run twice would
ingest the same 30 papers twice. Re-checking makes `--limit N` mean "the next N
still-pending records", which is what a wave-by-wave import needs.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..fsatomic import file_sha256
from ..paths import inbox_dir
from ..wiki import read_wiki_dois, read_wiki_stems


@dataclass
class Plan:
    """What `apply` would do, resolved against the wiki as it is right now."""

    staged: list[dict] = field(default_factory=list)
    already_present: list[dict] = field(default_factory=list)
    already_staged: list[dict] = field(default_factory=list)
    missing_pdf: list[dict] = field(default_factory=list)

    @property
    def total_ready(self) -> int:
        return (
            len(self.staged)
            + len(self.already_present)
            + len(self.already_staged)
            + len(self.missing_pdf)
        )


def read_inbox_index() -> dict[int, list[Path]]:
    """`inbox/`'s PDFs grouped by byte size — a cheap pre-filter for identity.

    Named and used like `read_wiki_dois`/`read_wiki_stems`: read once per
    `plan_wave`, then looked up N times. Size first so the common case (an empty
    or unrelated `inbox/`) hashes nothing at all.

    `inbox_dir()` is globbed unresolved, which still works when `inbox/` is a
    directory symlink into a synced folder — a layout CLAUDE.md supports — and
    keeps the spelling the user recognizes for the report.
    """
    index: dict[int, list[Path]] = {}
    try:
        for p in sorted(inbox_dir().glob("*.pdf")):
            try:
                index.setdefault(p.stat().st_size, []).append(p)
            except OSError:
                continue
    except OSError:
        return {}
    return index


def _already_in_inbox(src: Path, index: dict[int, list[Path]]) -> Path | None:
    """The copy of `src` already staged in `inbox/`, if there is one.

    Compares **bytes**, not names. `stage` deliberately uniquifies same-named
    sources into `X--1.pdf` because two different papers legitimately share an
    exporter-generated name, so a name match is not an identity match — and the
    leftover this looks for may itself be under the `--N` spelling. Content is
    the one test immune to both, and to path spelling.
    """
    try:
        candidates = index.get(src.stat().st_size, [])
    except OSError:
        return None
    if not candidates:
        return None
    try:
        want = file_sha256(src)
        for p in candidates:
            if file_sha256(p) == want:
                return p
    except OSError:
        return None
    return None


def plan_wave(records: list[dict], limit: int = 0) -> Plan:
    """Choose the next wave from the manifest's `ready` records.

    Four ways a record that `inspect` called ready can drop out here, all of
    them checked against the live filesystem rather than the manifest:

      - its DOI or stem is now in the wiki (a previous wave landed it)
      - its PDF has since been moved or deleted
      - a byte-identical copy is already sitting in `inbox/`
      - the limit is reached

    All but the limit are reported, never silently skipped: a record vanishing
    between phases is something the user should see, not something to absorb.

    The `inbox/` check is what makes re-running a failed wave safe. `stage`
    uniquifies on collision, so without it a leftover `inbox/X.pdf` produced
    `X--1.pdf` and stranded `X.pdf` as permanent phantom backlog — which the next
    `agent ingest inbox/*.pdf` would ingest as a separate paper. Reported rather
    than silently reused, because the user may have partly processed it.
    """
    # All read once, not per record: `find_stem_collision` re-walks `wiki/`
    # on every call, which turns this loop into O(records x pages).
    known_dois = {k.lower(): v for k, v in read_wiki_dois().items()}
    known_stems = read_wiki_stems()
    inbox_index = read_inbox_index()
    plan = Plan()

    for rec in records:
        if rec.get("verdict") != "ready":
            continue

        doi = (rec.get("doi") or "").lower()
        stem = rec.get("derived_stem")
        if doi and doi in known_dois:
            plan.already_present.append({**rec, "landed_as": known_dois[doi]})
            continue
        if stem and stem in known_stems:
            plan.already_present.append({**rec, "landed_as": stem})
            continue

        src = rec.get("primary_pdf")
        if not src or not Path(src).is_file():
            plan.missing_pdf.append(rec)
            continue

        # After the wiki and existence checks: a record already in the wiki is
        # more definitively done, and `src` has to exist to be hashed.
        leftover = _already_in_inbox(Path(src), inbox_index)
        if leftover is not None:
            plan.already_staged.append({**rec, "staged_as": str(leftover)})
            continue

        if limit and len(plan.staged) >= limit:
            continue
        plan.staged.append(rec)

    return plan


def stage(
    records: list[dict], *, dry_run: bool = False
) -> list[tuple[Path, list[str]]]:
    """Copy each record's PDF into `inbox/`; return `(path, argv)` pairs.

    **Copy, never move or symlink.** The source is the user's own library and
    must survive the import untouched, and `CLAUDE.md`'s PDF rule is explicit
    that files enter `inbox/` as copies. `agent ingest` then *moves* from
    `inbox/` to `papers/{stem}.pdf`, so the copy is consumed rather than
    accumulating.

    Filenames are made unique on collision rather than overwritten. Reference
    managers export names like `Nature-2026.3.pdf` that carry no identity, so a
    collision in `inbox/` is plausible and silently clobbering one paper with
    another would be unrecoverable.
    """
    inbox = inbox_dir()
    out: list[tuple[Path, list[str]]] = []
    # Names claimed earlier in *this* wave. Testing `dest.exists()` alone is
    # not enough: under `--dry-run` nothing is written, so every same-named
    # source resolved to the same destination and the preview disagreed with
    # what `apply` would actually do — in the one mode whose entire job is to
    # predict that accurately.
    claimed: set[Path] = set()
    for rec in records:
        src = Path(rec["primary_pdf"])
        dest = inbox / src.name
        n = 1
        while dest.exists() or dest in claimed:
            dest = inbox / f"{src.stem}--{n}{src.suffix}"
            n += 1
        claimed.add(dest)
        if not dry_run:
            inbox.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        out.append((dest, list(rec.get("ingest_args") or [])))
    return out


def dispatch(
    staged: list[tuple[Path, list[str]]],
    *,
    workers: int,
    extra_args: list[str] | None = None,
    workers_explicit: bool = True,
) -> int:
    """Hand the wave to `_ingest_batch` with per-record overrides.

    Paths are validated first because `_ingest_batch._resolve_inputs` calls
    `sys.exit(1)` on a missing file — fine for a CLI entry point, fatal for a
    caller that would rather report which record broke and keep going.
    """
    from ..tasks._ingest_batch import new_batch

    usable = [(p, args) for p, args in staged if p.is_file()]
    if not usable:
        return 1
    per_input = {str(p.resolve()): args for p, args in usable}
    return new_batch(
        [str(p) for p, _ in usable],
        ["agent", "ingest"],
        list(extra_args or []),
        workers,
        per_input_args=per_input,
        workers_explicit=workers_explicit,
    )
