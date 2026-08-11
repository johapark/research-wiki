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

from ..paths import inbox_dir
from ..wiki import find_stem_collision, read_wiki_dois


@dataclass
class Plan:
    """What `apply` would do, resolved against the wiki as it is right now."""

    staged: list[dict] = field(default_factory=list)
    already_present: list[dict] = field(default_factory=list)
    missing_pdf: list[dict] = field(default_factory=list)

    @property
    def total_ready(self) -> int:
        return len(self.staged) + len(self.already_present) + len(self.missing_pdf)


def plan_wave(records: list[dict], limit: int = 0) -> Plan:
    """Choose the next wave from the manifest's `ready` records.

    Three ways a record that `inspect` called ready can drop out here, all of
    them checked against the live filesystem rather than the manifest:

      - its DOI or stem is now in the wiki (a previous wave landed it)
      - its PDF has since been moved or deleted
      - the limit is reached

    The first two are reported, never silently skipped: a record vanishing
    between phases is something the user should see, not something to absorb.
    """
    known_dois = {k.lower(): v for k, v in read_wiki_dois().items()}
    plan = Plan()

    for rec in records:
        if rec.get("verdict") != "ready":
            continue

        doi = (rec.get("doi") or "").lower()
        stem = rec.get("derived_stem")
        if doi and doi in known_dois:
            plan.already_present.append({**rec, "landed_as": known_dois[doi]})
            continue
        if stem and find_stem_collision(stem) is not None:
            plan.already_present.append({**rec, "landed_as": stem})
            continue

        src = rec.get("primary_pdf")
        if not src or not Path(src).is_file():
            plan.missing_pdf.append(rec)
            continue

        if limit and len(plan.staged) >= limit:
            continue
        plan.staged.append(rec)

    return plan


def stage(records: list[dict], *, dry_run: bool = False) -> list[tuple[Path, list[str]]]:
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
    for rec in records:
        src = Path(rec["primary_pdf"])
        dest = inbox / src.name
        n = 1
        while dest.exists():
            dest = inbox / f"{src.stem}--{n}{src.suffix}"
            n += 1
        if not dry_run:
            inbox.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        out.append((dest, list(rec.get("ingest_args") or [])))
    return out


def dispatch(staged: list[tuple[Path, list[str]]], *, workers: int,
             extra_args: list[str] | None = None) -> int:
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
    return new_batch([str(p) for p, _ in usable], ["agent", "ingest"],
                     list(extra_args or []), workers, per_input_args=per_input)
