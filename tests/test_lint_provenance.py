"""Recovering `ingested_at` / `author_model` from the ingest telemetry log.

The repair behind `lint --fix` for `missing_author_model`. What these tests pin
is the difference between *recovering a recorded fact* and *inventing a
plausible value*, because everything worth getting wrong here lives on that line:

  - a stub attempt authored nothing, so it must never be credited — even when it
    is the most recent commit for that stem;
  - a migrated wiki has no telemetry, so the repair fills nothing rather than
    reaching for a file timestamp;
  - a value already on the page is never overwritten, because "last committed
    attempt" is not necessarily the run that produced the page on disk.

Hermetic: tmp wiki, tmp DB (the autouse `conftest` fixture points
`RESEARCHWIKI_DB_PATH` at tmp_path), no LLM, no network.
"""

from __future__ import annotations

import sqlite3

import pytest

from researchwiki.db.connection import db_path
from researchwiki.tasks.lint.provenance import (
    RECOVERED_MARKER,
    apply_provenance_fixes,
    survey,
)

REAL = 1_760_000_000        # a real ingest
LATER_STUB = 1_770_000_000  # a stub run, eleven days "later"


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    root = tmp_path / "wiki"
    (root / "ai").mkdir(parents=True)
    fake = lambda: root  # noqa: E731
    monkeypatch.setattr("researchwiki.paths.wiki_dir", fake)
    monkeypatch.setattr("researchwiki.wiki.wiki_dir", fake)
    return root


def _page(wiki, stem, extra: str = "", ptype: str = "paper") -> None:
    (wiki / "ai" / f"{stem}.md").write_text(
        f"---\ntitle: T\ntype: {ptype}\nyear: 2024\nkeywords: [a, b]\n{extra}---\n\n"
        "## Summary\n\nBody.\n", encoding="utf-8")


def _log(rows: list[tuple]) -> None:
    """Seed `ingest_iterations` with (attempt, stem, role, decision, model, ts).

    A row may carry a seventh element, the 1-based position of an earlier row
    in the same call, which is written as `parent_iteration_id`. The commit
    row's parent is the winning draft, which is how the real runner links them.
    """
    conn = sqlite3.connect(db_path())
    conn.execute("""CREATE TABLE IF NOT EXISTS ingest_iterations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id TEXT, paper_stem TEXT,
        pdf_filename TEXT, iteration INTEGER, role TEXT, decision TEXT,
        model_used TEXT, created_at INTEGER, parent_iteration_id INTEGER)""")
    ids: list[int] = []
    for row in rows:
        attempt, stem, role, decision, model, ts, *parent = row
        parent_id = ids[parent[0] - 1] if parent and parent[0] else None
        cur = conn.execute(
            "INSERT INTO ingest_iterations (attempt_id, paper_stem, role, decision, "
            "model_used, created_at, pdf_filename, iteration, parent_iteration_id) "
            "VALUES (?,?,?,?,?,?,'x.pdf',1,?)",
            (attempt, stem, role, decision, model, ts, parent_id))
        ids.append(cur.lastrowid)
    conn.commit()
    conn.close()


def _committed(attempt, stem, model, ts):
    return [(attempt, stem, "author", "kept", model, ts),
            (attempt, stem, "commit", "committed-to-wiki", None, ts + 10)]


# ---------- the happy path ----------

def test_recovers_both_fields_from_the_log(wiki):
    _page(wiki, "a-2024-x")
    _log(_committed("A1", "a-2024-x", "solar-pro3", REAL))

    found = survey()
    assert [r.key for r in found.recoverable] == ["ai/a-2024-x"]
    rec = found.recoverable[0]
    assert rec.author_model == "solar-pro3"
    assert rec.ingested_at.startswith("20")          # a real ISO stamp

    apply_provenance_fixes()
    text = (wiki / "ai" / "a-2024-x.md").read_text(encoding="utf-8")
    assert f'author_model: "solar-pro3"  {RECOVERED_MARKER}' in text
    assert f"ingested_at: {rec.ingested_at}  {RECOVERED_MARKER}" in text


def test_the_recovered_timestamp_stays_unquoted(wiki):
    """CLAUDE.md requires a real YAML timestamp — Dataview's date column cannot
    parse a quoted string, and the whole point of recovering the field is to make
    `views.md` sortable."""
    _page(wiki, "a-2024-x")
    _log(_committed("A1", "a-2024-x", "solar-pro3", REAL))
    apply_provenance_fixes()

    import yaml
    text = (wiki / "ai" / "a-2024-x.md").read_text(encoding="utf-8")
    fm = yaml.safe_load(text.split("---")[1])
    assert not isinstance(fm["ingested_at"], str), "must parse as a timestamp"


# ---------- the line between fact and invention ----------

def test_a_stub_attempt_is_never_credited(wiki):
    """`agents.llm` records `stub:{model}` for a placeholder whose text says so.
    It authored nothing, so a later stub commit must not displace the real run —
    the `asai-2023` case, whose two newest commits are both stubs."""
    _page(wiki, "a-2024-x")
    _log(_committed("A1", "a-2024-x", "gemini-3.5-flash", REAL)
         + _committed("A2", "a-2024-x", "stub:gemini-3.5-flash", LATER_STUB))

    rec = survey().recoverable[0]
    assert rec.author_model == "gemini-3.5-flash"
    # and the date follows the real attempt, not the stub run
    assert rec.ingested_at < "2026"


def test_a_stem_whose_every_attempt_was_a_stub_yields_nothing(wiki):
    _page(wiki, "a-2024-x")
    _log(_committed("A1", "a-2024-x", "stub:solar-pro3", REAL))

    found = survey()
    assert found.recoverable == []
    assert found.no_telemetry == ["ai/a-2024-x"]


@pytest.mark.parametrize("sentinel", ["stub", "(skipped)", "(no calls)", "(local)", ""])
def test_non_model_sentinels_are_not_recovered(wiki, sentinel):
    _page(wiki, "a-2024-x")
    _log(_committed("A1", "a-2024-x", sentinel, REAL))
    assert survey().recoverable == []


@pytest.mark.parametrize("model", ["gpt-5.6", "codex/gpt-5.6", "claude-4"])
def test_generic_model_aliases_are_not_credited(wiki, model):
    """The date is still a recorded fact; only the model is unattributable."""
    _page(wiki, "a-2024-x")
    _log(_committed("A1", "a-2024-x", model, REAL))
    rec = survey().recoverable[0]
    assert rec.fields == ["ingested_at"] and rec.author_model is None


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5.5", "gpt-4.1", "qwen3:32b"])
def test_exact_bare_version_ids_are_recovered(wiki, model):
    """Priced bare-version ids and Ollama tags name one model, not a family."""
    _page(wiki, "a-2024-x")
    _log(_committed("A1", "a-2024-x", model, REAL))
    assert survey().recoverable[0].author_model == model


def test_an_unattributable_later_run_does_not_fall_back_to_an_older_one(wiki):
    """The later run wrote the page, so the earlier run's model is not the
    author even though the later one cannot be named exactly."""
    _page(wiki, "a-2024-x")
    _log(_committed("A1", "a-2024-x", "claude-sonnet-4-6", REAL)
         + _committed("A2", "a-2024-x", "gpt-5.6", REAL + 5000))
    rec = survey().recoverable[0]
    assert rec.author_model is None
    assert "author_model" not in rec.fields


def test_the_committed_winner_is_credited_in_a_mixed_tournament(wiki):
    """The commit row links to the winning draft; the losing author's model
    must not be credited, and the mix must not make the attempt unattributable."""
    _page(wiki, "a-2024-x")
    _log([
        ("A1", "a-2024-x", "author", "kept", "gpt-5", REAL),
        ("A1", "a-2024-x", "author", "kept", "gpt-5.6-terra", REAL + 1),
        ("A1", "a-2024-x", "commit", "committed-to-wiki", None, REAL + 10, 1),
    ])
    assert survey().recoverable[0].author_model == "gpt-5"


def test_a_debug_winner_is_credited_over_the_original_author(wiki):
    _page(wiki, "a-2024-x")
    _log([
        ("A1", "a-2024-x", "author", "kept", "gpt-5.6-luna", REAL),
        ("A1", "a-2024-x", "debug", "kept", "gpt-5.6-terra", REAL + 1, 1),
        ("A1", "a-2024-x", "commit", "committed-to-wiki", None, REAL + 10, 2),
    ])
    assert survey().recoverable[0].author_model == "gpt-5.6-terra"


def test_a_legacy_tournament_without_a_winner_link_is_unattributable(wiki):
    _page(wiki, "a-2024-x")
    _log([
        ("A1", "a-2024-x", "author", "kept", "gpt-5", REAL),
        ("A1", "a-2024-x", "author", "kept", "gpt-5.6-terra", REAL + 1),
        ("A1", "a-2024-x", "commit", "committed-to-wiki", None, REAL + 10),
    ])
    assert survey().recoverable[0].author_model is None


@pytest.mark.parametrize("recorded", ["TODO", "gpt-5.6"])
def test_a_placeholder_or_alias_is_replaced_in_place(wiki, recorded):
    """`lint` reports these as missing, so `--fix` must repair them — by
    replacing the line, since a second `author_model:` key would leave YAML
    reading whichever comes last."""
    _page(wiki, "a-2024-x",
          f'author_model: "{recorded}"\ningested_at: 2026-01-01T00:00:00\n')
    _log(_committed("A1", "a-2024-x", "gpt-5.6-terra", REAL))

    stats = apply_provenance_fixes()
    assert stats["author_model_keys"] == ["ai/a-2024-x"]
    text = (wiki / "ai" / "a-2024-x.md").read_text(encoding="utf-8")
    assert text.count("author_model:") == 1
    assert f'author_model: "gpt-5.6-terra"  {RECOVERED_MARKER}' in text


def test_a_contradicted_alias_is_reported_not_replaced(wiki, capsys):
    """`gpt-5.6` → `claude-sonnet-4-6` would be a correction, not a refinement,
    and the log's newest run need not be the page's author."""
    _page(wiki, "a-2024-x",
          'author_model: "gpt-5.6"\ningested_at: 2026-01-01T00:00:00\n')
    _log(_committed("A1", "a-2024-x", "claude-sonnet-4-6", REAL))

    found = survey()
    assert found.recoverable == [] and found.conflicts == ["ai/a-2024-x"]
    apply_provenance_fixes()
    assert "disagrees with the log" in capsys.readouterr().out
    text = (wiki / "ai" / "a-2024-x.md").read_text(encoding="utf-8")
    assert 'author_model: "gpt-5.6"\n' in text


def test_an_uncommitted_attempt_is_ignored(wiki):
    """A run that authored but never committed did not produce the page."""
    _page(wiki, "a-2024-x")
    _log([("A1", "a-2024-x", "author", "kept", "solar-pro3", REAL)])
    assert survey().recoverable == []


# ---------- migration: fill nothing, say so ----------

def test_a_wiki_with_no_telemetry_recovers_nothing(wiki, capsys):
    """`migrate` never writes `ingest_iterations`, so a migrated page has no
    recorded ingest date or model. There is no fallback worth having."""
    _page(wiki, "migrated-2019-x")

    found = survey()
    assert found.has_telemetry is False
    assert found.no_telemetry == ["ai/migrated-2019-x"]

    stats = apply_provenance_fixes()
    assert stats["pages"] == 0 and stats["no_telemetry"] == 1
    assert "not ingested by this pipeline" in capsys.readouterr().out
    assert "ingested_at" not in (wiki / "ai" / "migrated-2019-x.md").read_text(encoding="utf-8")


def test_a_part_migrated_wiki_counts_what_it_cannot_reach(wiki):
    """The mixed case: some pages ingested here, some imported. The imported ones
    are reported rather than silently skipped, so the job doesn't look finishable."""
    _page(wiki, "ingested-2024-x")
    _page(wiki, "migrated-2019-y")
    _log(_committed("A1", "ingested-2024-x", "solar-pro3", REAL))

    found = survey()
    assert [r.key for r in found.recoverable] == ["ai/ingested-2024-x"]
    assert found.no_telemetry == ["ai/migrated-2019-y"]


# ---------- never overwrite ----------

def test_a_recorded_value_is_left_alone(wiki):
    """"Last committed attempt" is not always the run that produced the page —
    `ghareeb-2026` has seven, its stamp 3.6 days before the newest. Fine for
    filling a blank, wrong for correcting an assertion."""
    _page(wiki, "a-2024-x",
          'author_model: "claude-opus-4-7"\ningested_at: 2026-01-01T00:00:00\n')
    _log(_committed("A1", "a-2024-x", "solar-pro3", REAL))

    assert survey().recoverable == []
    text = (wiki / "ai" / "a-2024-x.md").read_text(encoding="utf-8")
    assert 'author_model: "claude-opus-4-7"' in text
    assert RECOVERED_MARKER not in text


def test_only_the_missing_half_is_filled(wiki):
    _page(wiki, "a-2024-x", "ingested_at: 2026-01-01T00:00:00\n")
    _log(_committed("A1", "a-2024-x", "solar-pro3", REAL))

    rec = survey().recoverable[0]
    assert rec.fields == ["author_model"] and rec.ingested_at is None
    apply_provenance_fixes()
    text = (wiki / "ai" / "a-2024-x.md").read_text(encoding="utf-8")
    assert "ingested_at: 2026-01-01T00:00:00\n" in text     # untouched, unmarked
    assert f'author_model: "solar-pro3"  {RECOVERED_MARKER}' in text


# ---------- scope ----------

@pytest.mark.parametrize("ptype", ["synthesis", "idea", "concept", "whitepaper"])
def test_only_paper_pages_are_in_scope(wiki, ptype):
    _page(wiki, f"a-2024-{ptype}", ptype=ptype)
    _log(_committed("A1", f"a-2024-{ptype}", "solar-pro3", REAL))
    assert survey().recoverable == []


def test_reports_pages_with_several_committed_attempts(wiki):
    _page(wiki, "a-2024-x")
    _log(_committed("A1", "a-2024-x", "solar-pro3", REAL)
         + _committed("A2", "a-2024-x", "gemini-3.5-flash", REAL + 5000))

    rec = survey().recoverable[0]
    assert rec.attempts == 2
    assert rec.author_model == "gemini-3.5-flash"     # the later real attempt
