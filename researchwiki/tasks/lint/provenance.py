"""Recover `ingested_at` / `author_model` from the ingest telemetry log.

The repair behind `lint --fix` for `missing_author_model`. Deliberately *not* a
command of its own: this is a one-shot cleanup for pages that predate the field,
not an operation anybody runs regularly, and it belongs where the gap is already
reported rather than as a fourth `backfill` target nobody would find.

Both fields are recoverable *facts* rather than derivations — every ingest writes
`ingest_iterations` rows carrying the model it used and when it committed.
Zero tokens, no network: one SQLite read and one atomic write per page.

**The recovery rule.** For each stem, take the last committed attempt *that used
a real model*. Read `ingested_at` from that commit row's `created_at`. Read
`author_model` from the draft the commit row links to as its winner, falling
back to the attempt's `author` rows on legacy logs that lack the link. When that
attempt's author has no exact id, `author_model` is left blank. It is never
taken from an older attempt, because an older attempt did not write the page.

"Real model" is load-bearing, not a formality. `agents.llm` records
`model=f"stub:{model}"` for a deterministic placeholder whose text begins "STUB
DRAFT … in lieu of real LLM generation", so a stub attempt authored nothing.
`asai-2023` is the worked case: its last two committed attempts are both
`stub:gemini-3.5-flash`, while the prose actually on disk came from the
`solar-pro3` attempt eleven days earlier. Anchoring on the newest commit would
credit a placeholder run for a page it did not write, and would date the page to
a test run rather than its ingest.

Validated against the pages that already carry `ingested_at`: the rule reproduces
the recorded stamp within seconds, the residue being structural — `promote`
stamps the frontmatter when it builds it, while the commit row lands once the run
finishes. Accurate to the day, near-exact to the second, and finer than any
consumer of the field renders.

**It never overwrites.** A stem can have many committed attempts (`cui-2024-scgpt`
has 14). "Last real committed attempt" means *the last time the pipeline
committed this paper*, which for a re-ingested page need not be the run that
produced the page now on disk — a fine basis for filling a blank, a bad one for
correcting a value a page already asserts. Pages with several committed attempts
are flagged in the report so the imprecision is visible rather than implied.
A placeholder (`TODO`) or family alias (`gpt-5.6`) is not an asserted value,
because `lint` reports it as missing. It is replaced only by an exact id that
refines it (`gpt-5.6` → `gpt-5.6-terra`); a contradiction is reported and left.

**Recovered values are marked.** Both fields carry a trailing
`# recovered from ingest_iterations` comment: YAML and Dataview ignore it, and a
reader can tell a stamped value from a reconstructed one. No new frontmatter key,
so the page contract is unchanged.

**A migrated wiki gets nothing, by design.** `researchwiki migrate` never writes
`ingest_iterations` — a migrated page was not ingested by this pipeline, so
nothing recorded when it landed or what wrote it. There is no honest fallback: a
file timestamp is not an ingest date (back-link splicing resets both mtime and
birthtime) and no artefact on disk names a model. With no telemetry this repair
fills nothing and says so; per page, a stem the log has never seen is skipped and
*counted*, so a part-migrated wiki shows how many pages are beyond reach instead
of quietly ignoring them.

The source is durable: `db rebuild` reconciles `papers` and `claims` from
markdown and never touches `ingest_iterations`, so recovery survives a rebuild.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ...fsatomic import write_text_atomic
from ...provenance import normalized_author_model, specific_author_model
from ...wiki import commit_page, read_pages

#: `model_used` values that name a non-call rather than a model. `promote`
#: writes `author_model` only for a real one, so recovering these would assert
#: that "(skipped)" wrote the prose.
_NOT_A_MODEL = frozenset({"", "stub", "(skipped)", "(no calls)", "(local)"})

#: Marks a value this target reconstructed. A trailing comment rather than a new
#: key: `category_suggestion_strength: weak  # …` and the ingest template's
#: `author_model: "TODO"  # …` already use the idiom, every YAML reader drops
#: it, and adding a key would change the page contract for a provenance note.
RECOVERED_MARKER = "# recovered from ingest_iterations"


@dataclass
class Recovery:
    """One page's recoverable fields. `fields` lists only what is *missing*."""

    path: Path
    key: str
    stem: str
    ingested_at: str | None = None
    author_model: str | None = None
    attempts: int = 0
    fields: list[str] = field(default_factory=list)


# ---------- the telemetry read ----------


def _is_real_model(name: str) -> bool:
    """False for values that name a non-call rather than a model.

    The `stub:` prefix is the one that matters and the reason this is a function
    rather than a set membership test. `agents.llm` returns
    `model=f"stub:{model}"` for a deterministic placeholder whose text begins
    "STUB DRAFT … in lieu of real LLM generation", so a stub attempt authored
    nothing. `asai-2023` is the worked case: its last two committed attempts are
    both `stub:gemini-3.5-flash`, while the prose actually on disk came from the
    `solar-pro3` attempt eleven days earlier. Recovering the newest attempt there
    would credit a placeholder run for a page it did not write.
    """
    n = (name or "").strip().lower()
    return bool(n) and n not in _NOT_A_MODEL and not n.startswith("stub:")


def _attempt_author(
    parent_model: str | None, attempt_models: set[str],
) -> tuple[bool, str | None]:
    """`(authored, model)` for one committed attempt.

    `authored` is False only when the attempt wrote nothing, i.e. a stub. Such
    an attempt is skipped, so an earlier real run still stands. `model` is None
    when a real model authored the page but no exact id can be named for it.
    That case must *not* be skipped the same way: the older run it would fall
    back to did not write the page now on disk.

    The commit row's `parent_iteration_id` names the draft that won, whether
    that was an author, evolve, or debug output, so its model is the author.
    Legacy rows without that link fall back to the attempt's author rows, and
    are attributable only when they carry exactly one real model. A tournament
    across several models without the link has no single author.
    """
    if parent_model is not None:
        if not _is_real_model(parent_model):
            return False, None
        return True, specific_author_model(parent_model) or None
    real = {m for m in attempt_models if _is_real_model(m)}
    if not real:
        return False, None
    if len(real) > 1:
        return True, None
    return True, specific_author_model(next(iter(real))) or None


def _read_log() -> dict[str, tuple[int, str | None, int]]:
    """`{stem: (commit_epoch, author_model | None, n_committed_attempts)}`.

    The chosen attempt is the **last committed attempt that a real model
    authored**, not simply the last committed one — see `_is_real_model`. A stem
    whose every committed attempt was a stub is omitted entirely: nothing about
    it can be recovered honestly.

    `author_model` is None when that attempt's author cannot be recorded
    exactly, because it was an ambiguous tournament or used a family alias.
    The commit date is still known, so `ingested_at` stays recoverable, but no
    model is credited. That includes the model of an older run.

    Degrades to `{}` when the DB is missing or unreadable, which `survey` turns
    into the migration refusal.
    """
    try:
        from ...db.connection import get_connection
        conn = get_connection()
    except Exception as exc:                      # pragma: no cover - env-specific
        print(f"  ! telemetry unavailable ({exc})", file=sys.stderr)
        return {}
    try:
        commits = conn.execute(
            "SELECT c.paper_stem, c.attempt_id, c.created_at, w.model_used "
            "  FROM ingest_iterations c "
            "  LEFT JOIN ingest_iterations w ON w.id = c.parent_iteration_id "
            " WHERE c.role = 'commit' AND c.decision LIKE 'committed%' "
            "   AND c.paper_stem IS NOT NULL "
            " ORDER BY c.created_at"
        ).fetchall()
        authors = conn.execute(
            "SELECT DISTINCT attempt_id, model_used FROM ingest_iterations "
            " WHERE role = 'author' AND model_used IS NOT NULL"
        ).fetchall()
    except Exception:
        return {}
    finally:
        conn.close()

    by_attempt: dict[str, set[str]] = {}
    for attempt, model in authors:
        by_attempt.setdefault(attempt, set()).add(model.strip())

    out: dict[str, tuple[int, str | None]] = {}
    totals: dict[str, int] = {}
    for stem, attempt, ts, parent_model in commits:   # ascending, so later wins
        totals[stem] = totals.get(stem, 0) + 1
        authored, model = _attempt_author(
            parent_model, by_attempt.get(attempt, set()),
        )
        if authored:
            out[stem] = (int(ts), model)
    return {s: (ts, m, totals[s]) for s, (ts, m) in out.items()}


def telemetry_author_models() -> dict[str, str]:
    """Exact recoverable author model by paper stem.

    Kept as a small public view so the reviewed provenance migration uses the
    same stub filtering and last-real-committed-attempt rule as ``lint --fix``.
    Stems whose latest real attempt cannot be attributed exactly are absent.
    """
    return {stem: row[1] for stem, row in _read_log().items() if row[1]}


# ---------- candidate selection ----------


@dataclass
class Survey:
    """What the log can and cannot reach.

    `no_telemetry` is the part that matters for a part-migrated wiki: those pages
    are missing the fields *and* have no rows, so they are unreachable rather
    than merely unprocessed. Counting them separately is what keeps the report
    from implying the job is finishable.
    """

    recoverable: list[Recovery] = field(default_factory=list)
    no_telemetry: list[str] = field(default_factory=list)
    has_telemetry: bool = True
    #: Pages whose recorded generic `author_model` the log contradicts rather
    #: than refines. Left alone, because the log's last run need not be the
    #: one that wrote the page.
    conflicts: list[str] = field(default_factory=list)


def _replaceable(recorded: str, recovered: str) -> bool:
    """Whether a recovered exact id may replace what the page records.

    A placeholder (`TODO`, blank) asserts nothing, so it is always replaceable.
    A generic value such as `gpt-5.6` does assert a family. It may give way only
    to an id that refines it, like `gpt-5.6-terra`. An unrelated id would be a
    correction, which this repair never makes.
    """
    recorded = normalized_author_model(recorded)
    return not recorded or recorded.lower() in recovered.lower()


def survey() -> Survey:
    """Paper pages missing either field, split by whether the log can supply it.

    Scoped to telemetry-backed `paper` and `commentary` pages. Reference,
    synthesis, concept, and idea pages are authored outside the ingest telemetry
    path and must be completed manually.

    `has_telemetry` is False when the log holds no committed ingest at all, which
    is the migrated-wiki case: every candidate then lands in `no_telemetry` and
    nothing is written. Reported rather than raised, so `lint --fix` on a
    migrated wiki completes its other repairs instead of aborting.
    """
    log = _read_log()
    out = Survey(has_telemetry=bool(log))

    for page in read_pages():
        if page.page_type not in ("paper", "commentary"):
            continue
        # Same test `lint` reports with: a placeholder or family alias is not
        # provenance, so `--fix` must be able to repair what the report lists.
        recorded_model = page.str_field("author_model")
        missing_model = not specific_author_model(recorded_model)
        missing_date = not page.str_field("ingested_at").strip()
        if not (missing_model or missing_date):
            continue
        entry = log.get(page.stem)
        if entry is None:
            out.no_telemetry.append(page.key)     # migrated / hand-written
            continue
        ts, model, n_attempts = entry
        rec = Recovery(path=page.path, key=page.key, stem=page.stem, attempts=n_attempts)

        if missing_date:
            rec.ingested_at = datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")
            rec.fields.append("ingested_at")
        if missing_model and model:
            if _replaceable(recorded_model, model):
                rec.author_model = model
                rec.fields.append("author_model")
            else:
                out.conflicts.append(page.key)
        if rec.fields:
            out.recoverable.append(rec)
    out.recoverable.sort(key=lambda r: r.key)
    out.no_telemetry.sort()
    out.conflicts.sort()
    return out


# ---------- the write ----------


def _apply(rec: Recovery) -> bool:
    """Insert the recovered fields in one atomic write. True if the file changed.

    Both fields land together rather than through two passes, so a page is never
    left carrying half a recovery. Placement mirrors `promote._build_frontmatter`
    — `author_model` then `ingested_at`, after `keywords` where it exists — so a
    recovered page reads like an ingested one.
    """
    text = rec.path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        print(f"    ! {rec.key}: no frontmatter; skipped", file=sys.stderr)
        return False
    end = text.find("\n---\n", 4)
    if end < 0:
        print(f"    ! {rec.key}: unterminated frontmatter; skipped", file=sys.stderr)
        return False

    lines = text[4:end].split("\n")

    def _anchor(*keys: str) -> int:
        """Index to insert after: the last of `keys` present, else end of block."""
        for want in keys:
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].startswith(f"{want}:"):
                    return i + 1
        return len(lines)

    # `ingested_at` is written UNQUOTED on purpose: CLAUDE.md requires a real
    # YAML timestamp there, because Dataview's date column will not parse a
    # string. The trailing comment does not change the parsed type.
    if rec.author_model:
        line = f'author_model: "{rec.author_model}"  {RECOVERED_MARKER}'
        # A placeholder or family alias is replaced in place: inserting beside
        # it would leave two `author_model:` keys, and YAML keeps the last.
        existing = [i for i, text in enumerate(lines)
                    if text.startswith("author_model:")]
        if existing:
            lines[existing[0]] = line
            for i in reversed(existing[1:]):
                del lines[i]
        else:
            lines.insert(_anchor("keywords", "tags", "year"), line)
    if rec.ingested_at:
        lines.insert(_anchor("author_model", "keywords", "tags", "year"),
                     f"ingested_at: {rec.ingested_at}  {RECOVERED_MARKER}")

    write_text_atomic(rec.path, "---\n" + "\n".join(lines) + "\n---\n" + text[end + 5:])
    commit_page(rec.path)
    return True


# ---------- CLI ----------


def apply_provenance_fixes() -> dict[str, int]:
    """Fill what the log can supply. Returns counts for `lint`'s fix summary.

    Called only under `lint --fix`. Prints its own detail — the same shape
    `apply_backlink_fixes` uses — so the repair explains itself where it happens
    rather than threading another payload through the report emitters.
    """
    found = survey()
    stats: dict = {"pages": 0, "ingested_at": 0, "author_model": 0,
                   "no_telemetry": len(found.no_telemetry), "multi_attempt": 0,
                   # Page keys whose `author_model` this run filled, so `lint`
                   # can drop them from the finding it just repaired instead of
                   # re-walking every page to notice.
                   "author_model_keys": []}

    if not found.has_telemetry:
        if found.no_telemetry:
            print(f"  provenance: no ingest telemetry in this wiki, so the "
                  f"{len(found.no_telemetry)} page(s) missing these fields cannot be "
                  f"recovered — they were not ingested by this pipeline (migrated or "
                  f"hand-written), and nothing on disk records an ingest date or a model.")
        return stats

    for rec in found.recoverable:
        if not _apply(rec):
            continue
        stats["pages"] += 1
        for f in rec.fields:
            stats[f] += 1
        if rec.author_model:
            stats["author_model_keys"].append(rec.key)
        if rec.attempts > 1:
            stats["multi_attempt"] += 1
        note = f"  [{rec.attempts} committed attempts]" if rec.attempts > 1 else ""
        detail = ", ".join(
            f"{f}={getattr(rec, f)}" for f in ("ingested_at", "author_model")
            if getattr(rec, f)
        )
        print(f"  provenance: {rec.key} → {detail}{note}")

    if stats["pages"]:
        print(f"  provenance: recovered {stats['ingested_at']} ingested_at and "
              f"{stats['author_model']} author_model value(s) across "
              f"{stats['pages']} page(s), marked `{RECOVERED_MARKER}`.")
    if stats["multi_attempt"]:
        print(f"  provenance: {stats['multi_attempt']} page(s) have several committed "
              f"attempts — the recovered stamp is the last time the pipeline committed "
              f"that paper, which for a re-ingested page may postdate the page on disk.")
    if found.no_telemetry:
        print(f"  provenance: {len(found.no_telemetry)} page(s) left alone — no telemetry "
              f"for them (migrated or hand-written), e.g. {found.no_telemetry[0]}")
    if found.conflicts:
        print(f"  provenance: {len(found.conflicts)} page(s) left alone — the recorded "
              f"generic author_model disagrees with the log's model, so only a "
              f"reviewer can say which is right, e.g. {found.conflicts[0]}")
    return stats
