"""Coverage tracking for `claim-overlap`, and the nudge that surfaces it.

Claim-overlap moved from auto-on-ingest to opt-in (`--claim-overlap`) because it
spends an LLM judge call per candidate pair to confirm a link on roughly one
paper in ten. Batching is cheaper; the cost is that coverage decays with nothing
to notice it. `claim_overlap_runs` plus a size-gated `status` nudge is what keeps
that from being silent, so both need pinning:

  - a stem with no run row is pending, and one with a matching fingerprint is not
  - changing a stem's claims re-opens it (a regrade invalidates the comparison)
  - a dry run must NOT record coverage
  - the nudge fires only at/above the threshold, and decays once shown
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from researchwiki.tasks import claim_overlap as co


@pytest.fixture
def conn(tmp_path):
    """Minimal papers+claims+runs DB, so the backlog query has real tables."""
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE papers (stem TEXT PRIMARY KEY, category TEXT, page_type TEXT,
                             year INTEGER, tags TEXT, page_path TEXT);
        CREATE TABLE claims (id INTEGER PRIMARY KEY AUTOINCREMENT, paper_stem TEXT,
                             section TEXT, text TEXT, is_cross_ref INTEGER DEFAULT 0);
        CREATE TABLE claim_overlap_runs (
            paper_stem TEXT PRIMARY KEY, ran_at INTEGER NOT NULL,
            claims_fingerprint TEXT NOT NULL, n_claims INTEGER NOT NULL,
            n_candidates INTEGER NOT NULL, n_judged INTEGER NOT NULL,
            n_confirmed INTEGER NOT NULL, sim_threshold REAL NOT NULL,
            source TEXT NOT NULL DEFAULT 'run');
    """)
    return c


def _paper(conn, stem, *, claims=("a claim",), page_type="paper", tags="ingested-via-agent"):
    conn.execute(
        "INSERT INTO papers (stem, category, page_type, year, tags, page_path) "
        "VALUES (?,?,?,?,?,?)",
        (stem, "compbio", page_type, 2024, tags, f"wiki/compbio/{stem}.md"),
    )
    for t in claims:
        conn.execute(
            "INSERT INTO claims (paper_stem, section, text, is_cross_ref) VALUES (?,?,?,0)",
            (stem, "results", t),
        )
    conn.commit()


# ---------- fingerprint ----------

def test_fingerprint_is_order_independent():
    """Claim row order isn't meaningful — `db rebuild` reassigns ids."""
    a = [{"section": "results", "text": "x"}, {"section": "limitations", "text": "y"}]
    assert co.claims_fingerprint(a) == co.claims_fingerprint(list(reversed(a)))


def test_fingerprint_distinguishes_section():
    """The same sentence under a different H2 is a different claim."""
    assert co.claims_fingerprint([{"section": "results", "text": "x"}]) != \
           co.claims_fingerprint([{"section": "limitations", "text": "x"}])


def test_fingerprint_changes_with_text():
    assert co.claims_fingerprint([{"section": "results", "text": "x"}]) != \
           co.claims_fingerprint([{"section": "results", "text": "x!"}])


# ---------- backlog ----------

def test_unrecorded_stem_is_pending(conn):
    _paper(conn, "a-2024-x")
    assert co.find_backlog(conn) == ["a-2024-x"]


def test_recorded_stem_is_not_pending(conn):
    _paper(conn, "a-2024-x")
    co.record_run(conn, "a-2024-x",
                  fingerprint=co.claims_fingerprint(co._claims_for_stem(conn, "a-2024-x")),
                  n_claims=1, n_candidates=0, n_judged=0, n_confirmed=0,
                  sim_threshold=0.83)
    assert co.find_backlog(conn) == []


def test_changed_claims_reopen_a_recorded_stem(conn):
    """A regrade or re-ingest invalidates the earlier comparison."""
    _paper(conn, "a-2024-x")
    co.record_run(conn, "a-2024-x",
                  fingerprint=co.claims_fingerprint(co._claims_for_stem(conn, "a-2024-x")),
                  n_claims=1, n_candidates=0, n_judged=0, n_confirmed=0,
                  sim_threshold=0.83)
    conn.execute("UPDATE claims SET text='a revised claim' WHERE paper_stem='a-2024-x'")
    conn.commit()
    assert co.find_backlog(conn) == ["a-2024-x"]


def test_non_paper_pages_are_never_pending(conn):
    """Synthesis / idea / concept pages carry no graded paper claims."""
    _paper(conn, "syn-thing", page_type="synthesis")
    _paper(conn, "idea-thing", page_type="idea")
    assert co.find_backlog(conn) == []


def test_paper_without_claims_is_not_pending(conn):
    """Nothing for the overlap finder to match — surfaced by zero_claim_papers."""
    _paper(conn, "a-2024-x", claims=())
    assert co.find_backlog(conn) == []


def test_cross_ref_claims_do_not_count(conn):
    _paper(conn, "a-2024-x", claims=())
    conn.execute("INSERT INTO claims (paper_stem, section, text, is_cross_ref) "
                 "VALUES ('a-2024-x','results','xref',1)")
    conn.commit()
    assert co.find_backlog(conn) == []


def test_record_run_upserts_rather_than_duplicating(conn):
    """Re-running a stem supersedes its row — what makes draining idempotent."""
    _paper(conn, "a-2024-x")
    fp = co.claims_fingerprint(co._claims_for_stem(conn, "a-2024-x"))
    for confirmed in (0, 2):
        co.record_run(conn, "a-2024-x", fingerprint=fp, n_claims=1, n_candidates=3,
                      n_judged=3, n_confirmed=confirmed, sim_threshold=0.83)
    rows = conn.execute("SELECT n_confirmed FROM claim_overlap_runs").fetchall()
    assert rows == [(2,)]


# ---------- mark-covered migration ----------

def test_mark_covered_only_claims_agent_ingested_papers(conn):
    """Digest-path papers were never covered — the hook is agent-only."""
    _paper(conn, "agent-2024-x", tags="ingested-via-agent")
    _paper(conn, "digest-2024-y", tags="")
    res = co.mark_covered(conn=conn)
    assert res["marked"] == 1
    assert co.find_backlog(conn) == ["digest-2024-y"]


def test_mark_covered_dry_run_writes_nothing(conn):
    _paper(conn, "agent-2024-x")
    res = co.mark_covered(conn=conn, dry_run=True)
    assert res["marked"] == 1 and res["dry_run"] is True
    assert co.find_backlog(conn) == ["agent-2024-x"]


def test_mark_covered_rows_are_tagged_as_back_records(conn):
    """`source` keeps their zero counts from being read as measurements."""
    _paper(conn, "agent-2024-x")
    co.mark_covered(conn=conn)
    (src,) = conn.execute("SELECT source FROM claim_overlap_runs").fetchone()
    assert src == "marked"


# ---------- the status nudge ----------

@pytest.fixture
def stamp(tmp_path, monkeypatch):
    monkeypatch.setattr("researchwiki.paths.wiki_root", lambda: tmp_path)
    return tmp_path / co.BACKLOG_STAMP


def _backlog_of(monkeypatch, n):
    monkeypatch.setattr(co, "find_backlog", lambda conn=None: [f"s{i}" for i in range(n)])


def test_nudge_silent_below_threshold(monkeypatch, stamp):
    _backlog_of(monkeypatch, co.BACKLOG_THRESHOLD - 1)
    assert co.backlog_warning() is None
    assert not stamp.exists(), "a warning that never fired must not consume the window"


def test_nudge_fires_at_threshold_and_names_the_command(monkeypatch, stamp):
    _backlog_of(monkeypatch, co.BACKLOG_THRESHOLD)
    msg = co.backlog_warning()
    assert msg is not None
    assert str(co.BACKLOG_THRESHOLD) in msg
    assert "claim-overlap --backlog" in msg


def test_nudge_decays_once_shown(monkeypatch, stamp):
    _backlog_of(monkeypatch, 50)
    assert co.backlog_warning() is not None
    assert stamp.exists()
    assert co.backlog_warning() is None, "should stay quiet for the decay window"


def test_nudge_returns_after_the_decay_window(monkeypatch, stamp):
    _backlog_of(monkeypatch, 50)
    stale = int(time.time() - (co.BACKLOG_DECAY_DAYS + 1) * 86400)
    stamp.write_text(str(stale))
    assert co.backlog_warning() is not None


def test_peeking_does_not_consume_the_window(monkeypatch, stamp):
    _backlog_of(monkeypatch, 50)
    assert co.backlog_warning(touch=False) is not None
    assert not stamp.exists()


def test_nudge_survives_a_missing_db(monkeypatch, stamp):
    """Cold install: no DB is not something to warn about."""
    def boom(conn=None): raise RuntimeError("no such table")
    monkeypatch.setattr(co, "find_backlog", boom)
    assert co.backlog_warning() is None


# ---------- which verdicts earn a bullet ----------
#
# `measures_same` is a real relation but the weakest one the judge accepts, and
# in practice it fires on shared methodology ("both quantify indel frequency by
# deep sequencing, on different CRISPR systems"). That is not the source citing,
# building on, or contrasting the other paper, so it fails CLAUDE.md's cross-link
# corollary and gets a typed edge without a Related Papers bullet. Over a
# 56-stem drain it was 6 of 9 confirmed links, so the boundary carries real
# weight and would be easy to erase by accident.

def test_measures_same_does_not_earn_a_bullet():
    assert "measures_same" not in co._CROSS_LINK_VERDICTS
    assert "measures_same" in co._EDGE_ONLY_VERDICTS


@pytest.mark.parametrize("verdict", ["corroborates", "refines", "builds_on", "cross_link"])
def test_engaging_verdicts_do_earn_a_bullet(verdict):
    assert verdict in co._CROSS_LINK_VERDICTS


def test_the_two_sets_are_disjoint():
    """A verdict in both would make the bullet decision order-dependent."""
    assert not (co._CROSS_LINK_VERDICTS & co._EDGE_ONLY_VERDICTS)


def test_every_edge_only_verdict_still_maps_to_a_relation():
    """An edge-only verdict whose relation is None would write nothing at all —
    silently dropping the pair instead of recording it."""
    for v in co._EDGE_ONLY_VERDICTS:
        assert co._relation_from_verdict(v) is not None


def test_relation_verdicts_is_the_union():
    """The `none`/unparseable boundary must not drift from the two sets."""
    assert co._RELATION_VERDICTS == co._CROSS_LINK_VERDICTS | co._EDGE_ONLY_VERDICTS
    assert "none" not in co._RELATION_VERDICTS


def test_judge_prompt_still_offers_every_verdict_the_code_handles():
    """If the prompt stops emitting a verdict the sets accept, the split is dead
    code; if it emits one they don't, the pair is dropped as a coincidence."""
    emitted = set(co._JUDGE_SCHEMA["properties"]["verdict"]["enum"]) - {"none"}
    handled = co._RELATION_VERDICTS - {"cross_link"}   # legacy, not emitted
    assert emitted == handled


# ---------- end to end: what actually lands on disk ----------

class _Cand:
    def __init__(self, existing_stem):
        self.existing_stem = existing_stem
        self.cosine = 0.9
        # `position` is required — `_format_prompt` cites claims as
        # `section#position` so the judge can see where each one sits.
        self.new_claim = {"section": "results", "position": 0, "text": "new claim"}
        self.existing_claim = {"section": "results", "position": 1, "text": "old claim"}


class _Page:
    def __init__(self, path, key, stem):
        self.path, self.key, self.stem = path, key, stem


def _two_pages(tmp_path, monkeypatch, verdict):
    """Wire `run()` against two real files and a judge stubbed to `verdict`."""
    new_p = tmp_path / "new-2026-a.md"
    old_p = tmp_path / "old-2020-b.md"
    for p in (new_p, old_p):
        p.write_text("---\ntitle: t\n---\n\n## Related Papers\n\n")

    pages = [_Page(new_p, "compbio/new-2026-a", "new-2026-a"),
             _Page(old_p, "compbio/old-2020-b", "old-2020-b")]
    monkeypatch.setattr(co, "read_pages", lambda: pages)
    monkeypatch.setattr("researchwiki.grade.claim_overlap.find_claim_overlaps",
                        lambda *a, **k: [_Cand("old-2020-b")])
    edges = []
    monkeypatch.setattr(co, "_persist_typed_edge",
                        lambda *a, **k: edges.append(a[4]))   # relation
    monkeypatch.setattr(co, "record_run", lambda *a, **k: None)
    res = co.run("new-2026-a", new_claims=[{"section": "results", "text": "x"}],
                 judge_fn=lambda _p: {"verdict": verdict, "rationale": "r"})
    return res, edges, new_p.read_text(), old_p.read_text()


def test_measures_same_writes_an_edge_but_no_bullet(tmp_path, monkeypatch):
    res, edges, new_body, old_body = _two_pages(tmp_path, monkeypatch, "measures_same")
    assert edges == ["measures_same"], "the typed edge must still be recorded"
    assert res["applied"] == []
    assert len(res["edge_only"]) == 1
    assert "[[" not in new_body and "[[" not in old_body, \
        "no Related Papers bullet on either page"


def test_builds_on_writes_reciprocal_bullets_and_an_edge(tmp_path, monkeypatch):
    res, edges, new_body, old_body = _two_pages(tmp_path, monkeypatch, "builds_on")
    assert edges == ["builds_on"]
    assert len(res["applied"]) == 1 and res["edge_only"] == []
    assert "[[compbio/old-2020-b]] — claim-grounded match" in new_body
    assert "[[compbio/new-2026-a]] — claim-grounded match" in old_body


def test_none_writes_neither(tmp_path, monkeypatch):
    res, edges, new_body, old_body = _two_pages(tmp_path, monkeypatch, "none")
    assert edges == [] and res["applied"] == [] and res["edge_only"] == []
    assert len(res["coincidence"]) == 1
    assert "[[" not in new_body and "[[" not in old_body


# ---------- the ingest flag flip ----------

def _ingest_args(argv):
    from researchwiki.tasks.agent import build_parser
    return build_parser().parse_args(["ingest", *argv])


def test_claim_overlap_is_off_by_default_at_ingest():
    """The whole point of the change — no judge calls unless asked for."""
    assert _ingest_args(["x.pdf"]).claim_overlap is False


def test_claim_overlap_opts_in():
    assert _ingest_args(["x.pdf", "--claim-overlap"]).claim_overlap is True


def test_no_cross_link_no_longer_governs_claim_overlap():
    """`--no-cross-link` now covers only concept attach + contradiction alerts.

    It used to gate claim-overlap too, so switching claim-overlap off meant
    losing the other two hooks as well.
    """
    a = _ingest_args(["x.pdf", "--no-cross-link"])
    assert a.no_cross_link is True and a.claim_overlap is False
    b = _ingest_args(["x.pdf", "--no-cross-link", "--claim-overlap"])
    assert b.no_cross_link is True and b.claim_overlap is True


def test_batch_workers_inherit_the_opt_in():
    """Batch mode re-invokes per PDF; a dropped flag would silently disagree."""
    from researchwiki.tasks.agent import _batch_passthrough_args
    assert "--claim-overlap" in _batch_passthrough_args(_ingest_args(["x.pdf", "--claim-overlap"]))
    assert "--claim-overlap" not in _batch_passthrough_args(_ingest_args(["x.pdf"]))
