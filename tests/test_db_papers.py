"""`researchwiki db papers` — structured lookups over the papers mirror.

Seeds a temp DB (via RESEARCHWIKI_DB_PATH, which wins over per-repo pathing),
then drives the CLI to pin the composable filters + exit codes.
"""

import time

import pytest

from researchwiki.db.connection import get_connection
from researchwiki.tasks import db


@pytest.fixture
def seeded_db(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    conn = get_connection()
    now = int(time.time())
    rows = [
        # stem, category, page_type, title, year, doi, venue, authors, senior
        ("a-2024-x", "cgt", "paper", "A", 2024, "10.1/a", "Nature", "Alice Smith", "Zed Boss"),
        ("b-2026-y", "cgt", "paper", "B", 2026, None, "Cell", "Bob Jones", "Alice Smith"),
        ("c-2020-z", "ai", "paper", "C", 2020, "TODO", "NeurIPS", "Carol Lee", "Carol Lee"),
    ]
    for stem, cat, pt, title, year, doi, venue, authors, senior in rows:
        conn.execute(
            "INSERT INTO papers (stem, category, page_type, title, year, doi, venue, "
            "authors, senior_authors, page_path, page_mtime, raw_frontmatter, indexed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (stem, cat, pt, title, year, doi, venue, authors, senior,
             f"/wiki/{cat}/{stem}.md", now, "{}", now),
        )
    conn.commit()
    conn.close()
    # capsys is consumed per-test; clear the fixture's own output noise
    capsys.readouterr()
    return tmp_path


def _run(argv):
    return db.main(argv)


def test_count_all(seeded_db, capsys):
    assert _run(["papers", "--count"]) == 0
    assert capsys.readouterr().out.strip() == "3"


def test_year_range(seeded_db, capsys):
    assert _run(["papers", "--year", "2024-2026", "--count"]) == 0
    assert capsys.readouterr().out.strip() == "2"


def test_year_exact(seeded_db, capsys):
    assert _run(["papers", "--year", "2020", "--count"]) == 0
    assert capsys.readouterr().out.strip() == "1"


def test_no_doi_matches_null_and_todo(seeded_db, capsys):
    # b has NULL doi, c has 'TODO' → both count as missing.
    assert _run(["papers", "--no-doi", "--count"]) == 0
    assert capsys.readouterr().out.strip() == "2"


def test_category_filter(seeded_db, capsys):
    assert _run(["papers", "--category", "ai", "--count"]) == 0
    assert capsys.readouterr().out.strip() == "1"


def test_author_matches_authors_and_senior(seeded_db, capsys):
    # "Alice Smith" is first author of a-2024 and senior author of b-2026 → 2 hits.
    assert _run(["papers", "--author", "Alice", "--count"]) == 0
    assert capsys.readouterr().out.strip() == "2"


def test_venue_substring(seeded_db, capsys):
    assert _run(["papers", "--venue", "Nature", "--count"]) == 0
    assert capsys.readouterr().out.strip() == "1"


def test_json_output(seeded_db, capsys):
    import json
    assert _run(["papers", "--category", "cgt", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {r["stem"] for r in data} == {"a-2024-x", "b-2026-y"}


def test_bad_year_exits_2(seeded_db, capsys):
    assert _run(["papers", "--year", "abc"]) == 2


def test_no_match_tsv_exits_1(seeded_db, capsys):
    # TSV mode with zero rows → exit 1 (no-result-where-expected).
    assert _run(["papers", "--category", "nonexistent"]) == 1
