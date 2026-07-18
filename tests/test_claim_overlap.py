"""Proactive claim-overlap cross-linker — compute core, cache, and auto-apply.

No network: embeddings are monkeypatched to deterministic vectors and the LLM
judge is stubbed. A temp repo (chdir) + temp DB (RESEARCHWIKI_DB_PATH) give the
command a real wiki to read pages from and write links into.
"""

import numpy as np
import pytest

from researchwiki.db.connection import get_connection
from researchwiki.grade import claim_overlap as core
from researchwiki.index import claim_embeddings as cache
from researchwiki.tasks import claim_overlap as cmd


# --- deterministic fake embedder: text → axis by keyword -------------------

_AXES = {"SHARED": [1.0, 0.0, 0.0], "ALPHA": [0.0, 1.0, 0.0], "BETA": [0.0, 0.0, 1.0]}


def _fake_embed(texts):
    out = []
    for t in texts:
        vec = [0.0, 0.0, 0.0]
        for kw, ax in _AXES.items():
            if kw in t:
                vec = [a + b for a, b in zip(vec, ax)]
        if vec == [0.0, 0.0, 0.0]:
            vec = [0.0, 0.0, 0.0001]
        arr = np.array(vec, dtype=np.float32)
        arr /= np.linalg.norm(arr)
        out.append(arr)
    return np.vstack(out).astype(np.float32)


@pytest.fixture
def embed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # semantic_cache_dir() is cwd-relative
    monkeypatch.setattr(core, "_NUMPY", True)
    monkeypatch.setattr("researchwiki.index.embeddings.is_available", lambda: True, raising=True)
    calls = {"n": 0}

    def _counted(texts):
        calls["n"] += len(texts)
        return _fake_embed(texts)

    monkeypatch.setattr("researchwiki.index.embeddings.embed_texts", _counted, raising=True)
    return calls


# --- claim-embedding cache -------------------------------------------------

def _rows(*specs):
    return [{"paper_stem": s, "section": "results", "position": p, "text": t}
            for s, p, t in specs]


def test_cache_embeds_then_reuses(embed):
    rows = _rows(("a", 0, "SHARED one"), ("b", 0, "ALPHA two"))
    v1 = cache.get_claim_embeddings(rows)
    assert v1 is not None and embed["n"] == 2
    v2 = cache.get_claim_embeddings(rows)          # identical rows → all reused
    assert embed["n"] == 2                          # no new embeds
    assert np.allclose(v1, v2)


def test_cache_invalidates_on_text_change(embed):
    cache.get_claim_embeddings(_rows(("a", 0, "SHARED one")))
    assert embed["n"] == 1
    cache.get_claim_embeddings(_rows(("a", 0, "SHARED one EDITED")))  # same id, new text
    assert embed["n"] == 2                          # re-embedded the changed row


# --- compute core: find_claim_overlaps -------------------------------------

def _seed_claims(conn, rows):
    for stem, pos, text in rows:
        conn.execute("INSERT OR IGNORE INTO papers "
                     "(stem, category, page_type, title, page_path, page_mtime, raw_frontmatter, indexed_at) "
                     "VALUES (?,?,?,?,?,?,?,?)",
                     (stem, "ai", "paper", stem, f"/x/{stem}.md", 0, "{}", 0))
        conn.execute("INSERT INTO claims (paper_stem, section, position, text, is_cross_ref) "
                     "VALUES (?,?,?,?,0)", (stem, "results", pos, text))
    conn.commit()


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    return get_connection()


def test_finds_cross_paper_match_only(embed, db):
    _seed_claims(db, [
        ("new", 0, "SHARED result"),      # matches old's SHARED claim
        ("new", 1, "ALPHA aside"),
        ("old", 0, "SHARED result"),
        ("other", 0, "BETA unrelated"),
    ])
    cands = core.find_claim_overlaps("new", sim_threshold=0.9, conn=db)
    stems = {c.existing_stem for c in cands}
    assert stems == {"old"}               # BETA paper below threshold, excluded
    assert cands[0].cosine == pytest.approx(1.0, abs=1e-4)


def test_collapses_to_one_pair_per_existing_paper(embed, db):
    _seed_claims(db, [
        ("new", 0, "SHARED a"),
        ("old", 0, "SHARED b"),
        ("old", 1, "SHARED c"),           # two matching claims, same paper
    ])
    cands = core.find_claim_overlaps("new", sim_threshold=0.9, conn=db)
    assert len(cands) == 1 and cands[0].existing_stem == "old"


# --- command: judge + auto-apply -------------------------------------------

def _write_page(tmp_path, cat, stem, body="## Summary\n\nx\n"):
    d = tmp_path / "wiki" / cat
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.md").write_text(f"---\ntype: paper\ncategory: [{cat}]\n---\n\n{body}")


@pytest.fixture
def wiki_and_db(embed, tmp_path, monkeypatch):
    # embed already chdir'd to tmp_path; add DB + pages + claims.
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    conn = get_connection()
    _seed_claims(conn, [("new-2026-x", 0, "SHARED finding"), ("old-2020-y", 0, "SHARED finding")])
    _write_page(tmp_path, "ai", "new-2026-x")
    _write_page(tmp_path, "ai", "old-2020-y")
    return tmp_path, conn


def test_cross_link_applies_reciprocal(wiki_and_db):
    tmp_path, conn = wiki_and_db
    res = cmd.run("new-2026-x", sim_threshold=0.9, judge_fn=lambda _p: {"verdict": "cross_link", "rationale": "same finding"}, conn=conn)
    assert len(res["applied"]) == 1
    new_body = (tmp_path / "wiki/ai/new-2026-x.md").read_text()
    old_body = (tmp_path / "wiki/ai/old-2020-y.md").read_text()
    assert "[[ai/old-2020-y]]" in new_body          # forward
    assert "[[ai/new-2026-x]]" in old_body          # reciprocal
    assert "auto-added; claim-overlap" in new_body


def test_coincidence_writes_nothing(wiki_and_db):
    tmp_path, conn = wiki_and_db
    res = cmd.run("new-2026-x", sim_threshold=0.9, judge_fn=lambda _p: {"verdict": "coincidence", "rationale": "different domain"}, conn=conn)
    assert res["applied"] == [] and len(res["coincidence"]) == 1
    assert "[[ai/old-2020-y]]" not in (tmp_path / "wiki/ai/new-2026-x.md").read_text()


def test_dry_run_judges_but_does_not_write(wiki_and_db):
    tmp_path, conn = wiki_and_db
    res = cmd.run("new-2026-x", sim_threshold=0.9, dry_run=True, judge_fn=lambda _p: {"verdict": "cross_link", "rationale": "x"}, conn=conn)
    assert len(res["applied"]) == 1
    assert "[[ai/old-2020-y]]" not in (tmp_path / "wiki/ai/new-2026-x.md").read_text()


def test_already_linked_skips_without_judging(wiki_and_db):
    tmp_path, conn = wiki_and_db
    # Pre-link the new page to old.
    p = tmp_path / "wiki/ai/new-2026-x.md"
    p.write_text(p.read_text() + "\n## Related Papers\n\n- [[ai/old-2020-y]] — prior\n")
    judged = {"n": 0}

    def _judge(_p):
        judged["n"] += 1
        return {"verdict": "cross_link", "rationale": "should not run"}

    res = cmd.run("new-2026-x", sim_threshold=0.9, judge_fn=_judge, conn=conn)
    assert judged["n"] == 0                          # skipped before the LLM
    assert len(res["skipped"]) == 1 and res["applied"] == []


def test_unknown_stem_raises(wiki_and_db):
    _, conn = wiki_and_db
    with pytest.raises(LookupError):
        cmd.run("nonexistent-stem", conn=conn)


def test_supplied_new_claims_bypass_db(embed, tmp_path, monkeypatch):
    # The ingest hook supplies new claims from the committed page BEFORE they're
    # in the DB. Seed only the EXISTING paper's claims; supply the new ones.
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    conn = get_connection()
    _seed_claims(conn, [("old-2020-y", 0, "SHARED finding")])  # new-2026-x NOT in DB
    _write_page(tmp_path, "ai", "new-2026-x")
    _write_page(tmp_path, "ai", "old-2020-y")

    supplied = [{"paper_stem": "new-2026-x", "section": "results", "position": 0, "text": "SHARED finding"}]
    res = cmd.run("new-2026-x", new_claims=supplied, sim_threshold=0.9,
                  judge_fn=lambda _p: {"verdict": "cross_link", "rationale": "match"}, conn=conn)
    assert len(res["applied"]) == 1
    assert "[[ai/new-2026-x]]" in (tmp_path / "wiki/ai/old-2020-y.md").read_text()


def test_claims_from_page_parses_committed_markdown(embed, tmp_path):
    _write_page(tmp_path, "ai", "p-2026-z",
                body="## Key Contributions\n\n- First contribution claim.\n- Second one.\n")
    rows = cmd.claims_from_page("p-2026-z", tmp_path / "wiki/ai/p-2026-z.md")
    assert [r["text"] for r in rows] == ["First contribution claim.", "Second one."]
    assert all(r["paper_stem"] == "p-2026-z" for r in rows)


def test_judge_failure_is_distinct_from_coincidence(wiki_and_db):
    # A None verdict (LLM unreachable/unparseable) must NOT be silently counted
    # as coincidence and must NOT write a link.
    tmp_path, conn = wiki_and_db
    res = cmd.run("new-2026-x", sim_threshold=0.9, judge_fn=lambda _p: None, conn=conn)
    assert res["applied"] == []
    assert res["coincidence"] == []
    assert len(res["judge_failed"]) == 1
    assert "[[ai/old-2020-y]]" not in (tmp_path / "wiki/ai/new-2026-x.md").read_text()


# --- typed verdicts + claim-graph edge emission ---------------------------------


def test_typed_corroborates_verdict_applies_link(wiki_and_db):
    tmp_path, conn = wiki_and_db
    res = cmd.run("new-2026-x", sim_threshold=0.9,
                  judge_fn=lambda _p: {"verdict": "corroborates",
                                        "rationale": "same finding on same benchmark"},
                  conn=conn)
    assert len(res["applied"]) == 1
    assert res["applied"][0]["relation"] == "corroborates"


def test_typed_none_verdict_does_not_apply(wiki_and_db):
    tmp_path, conn = wiki_and_db
    res = cmd.run("new-2026-x", sim_threshold=0.9,
                  judge_fn=lambda _p: {"verdict": "none",
                                        "rationale": "different domains"},
                  conn=conn)
    assert res["applied"] == []
    # `none` is the new coincidence — surfaces there for backwards compat.
    assert len(res["coincidence"]) == 1


def test_directed_verdicts_persist_directed_edge(tmp_path, monkeypatch, embed):
    """`refines` and `builds_on` are directed relations; the persisted edge
    must carry directed=True. `corroborates` and `measures_same` are symmetric."""
    from researchwiki.tasks.claim_overlap import _persist_typed_edge
    from researchwiki.claim_graph import open_edges_db, query

    # Isolate the edges cache to tmp_path/.claim-graph/edges.db.
    monkeypatch.chdir(tmp_path)

    new_claim = {"section": "key_contributions", "position": 0, "text": "NEW claim"}
    old_claim = {"section": "results", "position": 1, "text": "EXISTING claim"}

    for rel in ("corroborates", "measures_same", "refines", "builds_on"):
        _persist_typed_edge(
            "new-x", new_claim, "old-y", old_claim,
            relation=rel, rationale=f"{rel} rationale", cosine=0.9,
        )

    conn = open_edges_db()
    try:
        edges = query(conn, relation=None)
        by_rel = {e.relation: e for e in edges}
    finally:
        conn.close()

    assert set(by_rel.keys()) == {"corroborates", "measures_same", "refines", "builds_on"}
    assert by_rel["refines"].directed is True
    assert by_rel["builds_on"].directed is True
    assert by_rel["corroborates"].directed is False
    assert by_rel["measures_same"].directed is False


def test_legacy_cross_link_verdict_maps_to_builds_on(wiki_and_db):
    """Older callers/tests use the pre-typed `cross_link` verdict. Still
    applies the reciprocal link AND emits a `builds_on` edge in the graph."""
    tmp_path, conn = wiki_and_db
    res = cmd.run("new-2026-x", sim_threshold=0.9,
                  judge_fn=lambda _p: {"verdict": "cross_link", "rationale": "..."},
                  conn=conn)
    assert len(res["applied"]) == 1
    assert res["applied"][0]["relation"] == "builds_on"
