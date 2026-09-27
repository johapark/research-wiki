"""`scout recent`: new papers near a seed set, ranked locally.

Pins what the feature promises: seeds resolve deterministically inside S2's
100-id budget, what the wiki already holds or the user declined never
resurfaces, update triggers come from the pages that state them, a trigger
match needs both a standardized score and neighbourhood agreement, and the
abstract never reaches the snapshot.

Hermetic: the provider, page index and embedder are faked; no network or model.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from researchwiki.providers import ScholarlyArticle
from researchwiki.providers.semantic_scholar import SemanticScholarProvider
from researchwiki.scouting import recent as R
from researchwiki.tasks import scout

TODAY = dt.date(2026, 9, 27)


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("wiki/single-cell", "wiki/genomics", "wiki/synthesis",
                 "wiki/ideas", "wiki/proposals", "wiki/other"):
        (tmp_path / name).mkdir(parents=True)
    return tmp_path


def _write(root, rel, fm: dict, body: str = "## Summary\n\nx\n"):
    lines = ["---"] + [f"{k}: {json.dumps(v)}" for k, v in fm.items()] + ["---", "", body]
    path = root / "wiki" / f"{rel}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _paper(root, cat, stem, year, doi=None, title=None):
    fm = {"title": title or stem, "type": "paper", "year": year}
    if doi:
        fm["doi"] = doi
    return _write(root, f"{cat}/{stem}", fm)


def _article(doi=None, pid=None, title="t", date="2026-09-01", year=2026, abstract="a"):
    return ScholarlyArticle(title=title, year=year, doi=doi, abstract=abstract,
                            publication_date=date, raw={"paperId": pid} if pid else {})


# ---------- seeds ----------

def test_seeds_from_a_category_prefer_the_newest_papers(wiki):
    _paper(wiki, "single-cell", "a-2020-x", 2020, "10.1/a")
    _paper(wiki, "single-cell", "b-2026-x", 2026, "10.1/b")
    _paper(wiki, "single-cell", "c-2024-x", 2024, "10.1/c")
    _paper(wiki, "single-cell", "d-2025-x", 2025)            # no DOI: unusable
    _paper(wiki, "genomics", "e-2026-x", 2026, "10.1/e")
    seeds = R.resolve_seeds(R.read_pages(), categories=["single-cell"],
                            page_keys=[], stems=[], budget=2)
    assert seeds.dois == ["10.1/b", "10.1/c"]
    assert seeds.available == 3


def test_page_seeds_are_the_papers_it_cites(wiki):
    _paper(wiki, "single-cell", "a-2020-x", 2020, "10.1/a")
    _paper(wiki, "genomics", "e-2026-x", 2026, "10.1/e")
    _paper(wiki, "genomics", "f-2026-x", 2026, "10.1/f")
    _write(wiki, "synthesis/s", {"title": "S", "type": "synthesis"},
           "## Short answer\n\nx[^a]\n\n[^a]: [[single-cell/a-2020-x]]\n\n"
           "See [[genomics/e-2026-x#kc-1234abcd]].\n")
    seeds = R.resolve_seeds(R.read_pages(), categories=[], stems=[],
                            page_keys=["synthesis/s"], budget=100)
    assert sorted(seeds.dois) == ["10.1/a", "10.1/e"]


def test_an_unknown_seed_is_an_input_error(wiki):
    with pytest.raises(R.SeedError, match="unknown category"):
        R.resolve_seeds(R.read_pages(), categories=["nope"], page_keys=[],
                        stems=[], budget=100)
    with pytest.raises(R.SeedError, match="no wiki page"):
        R.resolve_seeds(R.read_pages(), categories=[], page_keys=["synthesis/x"],
                        stems=[], budget=100)


# ---------- filtering ----------

def test_filter_drops_wiki_declined_and_old_papers():
    arts = [
        _article(doi="10.1/in-wiki"),
        _article(doi="10.9/preprint", title="A Draft Human Pangenome Reference"),
        _article(doi="10.1/declined"),
        _article(doi="10.1/old", date="2025-01-01"),
        _article(pid="abc123", date=None, year=2026),         # undated, year in window
        _article(doi="10.1/undated-old", date=None, year=2024),
        _article(doi="10.1/keep"),
        _article(doi="10.1/keep"),                            # duplicate
        _article(),                                           # no identifier
    ]
    kept, counts = R.filter_candidates(
        arts, cutoff=dt.date(2026, 7, 29), wiki_dois={"10.1/in-wiki"},
        wiki_titles={R._norm_title("A draft human pangenome reference.")},
        declined={"10.1/declined"},
    )
    assert [c.key for c in kept] == ["s2:abc123", "10.1/keep"]
    assert counts == {"returned": 9, "in_wiki": 2, "declined": 1,
                      "outside_window": 2, "no_identifier": 1}


# ---------- triggers ----------

def test_triggers_come_from_update_sections_and_open_proposals(wiki):
    _write(wiki, "synthesis/s", {"title": "S", "type": "synthesis"},
           "## Short answer\n\nNot a trigger at all, just an ordinary claim here.\n\n"
           "## What would update this page\n\n"
           "- A head-to-head benchmark of causal and non-causal perturbation models.\n"
           "  - with a nested detail that belongs to the parent bullet\n"
           "- Too short.\n\n"
           "## References\n\n[^a]: [[single-cell/a-2020-x]]\n")
    _write(wiki, "ideas/i", {"title": "I", "type": "idea"},
           "## Caveats\n\n### What would change the conclusion\n\n"
           "- A paper relaxing the independence requirement without losing power.\n\n"
           "### Other\n\n- Something unrelated that is long enough to count as one.\n")
    open_body = ("## Decisive uncertainty\n\nWhether the advantage comes from graph "
                 "structure or from control alignment remains unresolved.\n\n## Feedback\n")
    _write(wiki, "proposals/p1", {"title": "P1", "type": "proposal", "status": "deferred"},
           open_body)
    _write(wiki, "proposals/p2", {"title": "P2", "type": "proposal", "status": "published"},
           open_body)
    got = {(t.page, t.text) for t in R.collect_triggers(R.read_pages())}
    assert got == {
        ("synthesis/s", "A head-to-head benchmark of causal and non-causal perturbation "
                        "models. with a nested detail that belongs to the parent bullet"),
        ("ideas/i", "A paper relaxing the independence requirement without losing power."),
        ("proposals/p1", "Whether the advantage comes from graph structure or from "
                         "control alignment remains unresolved."),
    }


def test_trigger_text_keeps_the_words_of_a_linked_stem():
    assert R._clean("fixed by [[genomics/siren-2021-pangenomics-enables]] and "
                    "[[x|Giraffe]][^a]") == "fixed by siren 2021 pangenomics enables and Giraffe"


# ---------- scoring ----------

def _fake_index(monkeypatch, paper_vecs, stems):
    arr = np.array(paper_vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    rows = [{"key": f"cat/{s}", "stem": s, "page_type": "paper"} for s in stems]
    monkeypatch.setattr(R, "_load_page_index", lambda: (arr, rows))


def _fake_embed(monkeypatch, table):
    def embed(texts):
        out = np.array([table[t] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)
    monkeypatch.setattr(R, "_embed", embed)


def _cand(text):
    return R.Candidate(paper_id=None, doi=f"10.1/{text}", title=text,
                       publication_date="2026-09-01", year=2026, venue="",
                       citation_count=0, has_abstract=True, score_text=text)


def test_trigger_match_needs_neighbourhood_agreement(monkeypatch):
    """Two candidates equally close to a trigger; only the one whose nearest
    papers the trigger's page cites is nominated."""
    stems = [f"p{i}" for i in range(8)]
    vecs = [[1, 0, 0.05 * i] for i in range(4)] + [[0, 1, 0.05 * i] for i in range(4)]
    _fake_index(monkeypatch, vecs, stems)
    monkeypatch.setattr(R, "NEIGHBOR_K", 2)
    monkeypatch.setattr(R, "TRIGGER_Z_MIN", 0.5)
    monkeypatch.setattr(R, "PAGE_RELEVANCE_MIN", -1.0)
    _fake_embed(monkeypatch, {"near-cited": [1, 0.1, 1], "near-other": [0.1, 1, 1],
                              "trigger": [0.5, 0.5, 1]})
    trig = R.Trigger(page="synthesis/s", page_type="synthesis", text="trigger",
                     cited=frozenset({"p0", "p1", "p2", "p3"}))
    a, b = _cand("near-cited"), _cand("near-other")
    assert R.score_candidates([a, b], [trig])
    assert [t["page"] for t in a.triggers] == ["synthesis/s"]
    assert b.triggers == []
    assert a.nearest[0]["key"].startswith("cat/p")


def test_each_trigger_keeps_only_its_best_candidates(monkeypatch):
    stems = [f"p{i}" for i in range(6)]
    _fake_index(monkeypatch, [[1, 0.1 * i, 0] for i in range(6)], stems)
    monkeypatch.setattr(R, "TRIGGER_Z_MIN", -10.0)
    monkeypatch.setattr(R, "PAGE_RELEVANCE_MIN", -1.0)
    monkeypatch.setattr(R, "PER_TRIGGER_CAP", 2)
    table = {f"c{i}": [1, 0.1 * i, 0.2 * i] for i in range(5)}
    table["trigger"] = [1, 0, 1]
    _fake_embed(monkeypatch, table)
    trig = R.Trigger(page="synthesis/s", page_type="synthesis", text="trigger",
                     cited=frozenset(stems))
    cands = [_cand(f"c{i}") for i in range(5)]
    R.score_candidates(cands, [trig])
    assert sum(bool(c.triggers) for c in cands) == 2


def test_page_relevance_floor_drops_papers_far_from_the_pages_citations(monkeypatch):
    """Both candidates clear the trigger and neighbourhood gates; only the one
    close to the page's cited papers survives the floor. The floor runs before
    the per-trigger cap, so the dropped paper does not take a slot."""
    stems = ["p0", "p1", "p2", "q0", "q1", "q2"]
    vecs = [[1, 0, 0], [1, 0.05, 0], [1, 0.1, 0], [0.6, 1, 0], [0.6, 1.05, 0], [0.6, 1.1, 0]]
    _fake_index(monkeypatch, vecs, stems)
    monkeypatch.setattr(R, "TRIGGER_Z_MIN", -10.0)
    monkeypatch.setattr(R, "NEIGHBOR_K", 6)
    monkeypatch.setattr(R, "PER_TRIGGER_CAP", 1)
    monkeypatch.setattr(R, "PAGE_RELEVANCE_MIN", 0.95)
    _fake_embed(monkeypatch, {"close": [1, 0.02, 0], "far": [0.6, 1, 0.9],
                              "trigger": [0.7, 0.7, 0.5]})
    trig = R.Trigger(page="synthesis/s", page_type="synthesis", text="trigger",
                     cited=frozenset(stems[:3]))
    close, far = _cand("close"), _cand("far")
    R.score_candidates([far, close], [trig])
    assert far.triggers == []
    assert [t["page"] for t in close.triggers] == ["synthesis/s"]
    assert close.triggers[0]["page_relevance"] >= 0.95


def test_page_with_too_few_cited_papers_cannot_match(monkeypatch):
    _fake_index(monkeypatch, [[1, 0, 0], [1, 0.1, 0]], ["p0", "p1"])
    monkeypatch.setattr(R, "TRIGGER_Z_MIN", -10.0)
    monkeypatch.setattr(R, "PAGE_RELEVANCE_MIN", -1.0)
    _fake_embed(monkeypatch, {"x": [1, 0, 0], "trigger": [1, 0, 0]})
    trig = R.Trigger(page="synthesis/s", page_type="synthesis", text="trigger",
                     cited=frozenset({"p0", "p1"}))
    cand = _cand("x")
    R.score_candidates([cand], [trig])
    assert cand.triggers == []


def test_scoring_degrades_without_an_index(monkeypatch):
    monkeypatch.setattr(R, "_load_page_index", lambda: None)
    cand = _cand("x")
    assert R.score_candidates([cand], []) is False
    assert cand.fit is None


# ---------- run / snapshot ----------

class _Provider:
    def __init__(self, articles):
        self.articles = articles
        self.calls = []

    def get_recommendations_for_seeds(self, pos, neg, *, limit):
        self.calls.append((list(pos), list(neg), limit))
        return self.articles


def test_snapshot_never_carries_the_abstract(wiki, monkeypatch):
    _paper(wiki, "single-cell", "a-2026-x", 2026, "10.1/a")
    monkeypatch.setattr(R, "score_candidates", lambda c, t: False)
    secret = "VERBATIM ABSTRACT TEXT"
    provider = _Provider([_article(doi="10.1/new", abstract=secret)])
    snap = R.run(categories=["single-cell"], today=TODAY, provider=provider)
    on_disk = (wiki / ".s2-cache" / "recent").glob("2026-09-27__*.json")
    assert secret not in json.dumps(snap)
    assert all(secret not in p.read_text() for p in on_disk)
    assert snap["candidates"][0]["has_abstract"] is True


def test_declines_are_filtered_and_sent_as_negative_seeds(wiki, monkeypatch):
    _paper(wiki, "single-cell", "a-2026-x", 2026, "10.1/a")
    monkeypatch.setattr(R, "score_candidates", lambda c, t: True)
    assert R.add_decline("https://doi.org/10.1/NO", "off-topic") == "10.1/no"
    R.add_decline("s2:Abc", "no doi")
    provider = _Provider([_article(doi="10.1/no"), _article(pid="Abc"),
                          _article(doi="10.1/yes")])
    snap = R.run(categories=["single-cell"], today=TODAY, provider=provider)
    assert provider.calls == [(["10.1/a"], ["10.1/no"], R.POOL)]
    assert [c["key"] for c in snap["candidates"]] == ["10.1/yes"]
    assert snap["counts"]["declined"] == 2
    assert R.remove_decline("10.1/no") is True
    assert R.remove_decline("10.1/no") is False


def test_first_seen_survives_a_rerun(wiki, monkeypatch):
    _paper(wiki, "single-cell", "a-2026-x", 2026, "10.1/a")
    monkeypatch.setattr(R, "score_candidates", lambda c, t: True)
    provider = _Provider([_article(doi="10.1/new")])
    R.run(categories=["single-cell"], today=TODAY, provider=provider)
    later = R.run(categories=["single-cell"], today=TODAY + dt.timedelta(days=3),
                  provider=provider)
    assert later["candidates"][0]["first_seen"] == TODAY.isoformat()


def test_no_model_is_ever_called(wiki, monkeypatch):
    from researchwiki.agents import llm
    monkeypatch.setattr(llm, "call", lambda **kw: pytest.fail("scout recent called a model"))
    _paper(wiki, "single-cell", "a-2026-x", 2026, "10.1/a")
    monkeypatch.setattr(R, "score_candidates", lambda c, t: True)
    R.run(categories=["single-cell"], today=TODAY,
          provider=_Provider([_article(doi="10.1/new")]))


# ---------- provider ----------

def test_multi_seed_request_is_sorted_capped_and_cached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = SemanticScholarProvider()
    seen = []

    def fake_post(url, payload, cache_path):
        seen.append((url, payload, cache_path.name))
        return {"recommendedPapers": [{"paperId": "p1", "title": "T",
                                       "publicationDate": "2026-09-01",
                                       "externalIds": {"DOI": "10.1/X"}}]}

    monkeypatch.setattr(provider, "_post_fetch", fake_post)
    got = provider.get_recommendations_for_seeds(["10.1/B", "10.1/a"], ["10.1/a", "10.1/n"])
    first = provider.get_recommendations_for_seeds(["10.1/a", "10.1/b"], ["10.1/n"])
    assert got[0].doi == "10.1/x" and got[0].publication_date == "2026-09-01"
    url, payload, name = seen[0]
    assert "publicationDate" in url and "abstract" in url
    assert payload == {"positivePaperIds": ["DOI:10.1/a", "DOI:10.1/b"],
                       "negativePaperIds": ["DOI:10.1/n"]}
    assert name.startswith("s2_recs_multi_v1__") and seen[1][2] == name
    assert first[0].title == "T"
    with pytest.raises(ValueError, match="at most 100"):
        provider.get_recommendations_for_seeds([f"10.1/{i}" for i in range(101)])


# ---------- CLI ----------

def test_scout_dispatches_recent_mode(monkeypatch):
    seen = []
    monkeypatch.setattr(scout.recent, "main", lambda argv: seen.append(argv) or 0)
    assert scout.main(["recent", "--category", "x"]) == 0
    assert seen == [["--category", "x"]]


def test_cli_requires_a_seed_and_a_decline_reason(wiki, capsys):
    assert R.main([]) == 1
    assert "--category" in capsys.readouterr().err
    assert R.main(["--decline", "10.1/x"]) == 1
    assert R.main(["--decline", "10.1/x", "--reason", "off-topic"]) == 0
    assert R.main(["--undecline", "10.1/x"]) == 0
    assert R.main(["--undecline", "10.1/x"]) == 1


def test_cli_renders_page_matches_first(wiki, monkeypatch, capsys):
    _paper(wiki, "single-cell", "a-2026-x", 2026, "10.1/a")
    snapshot = {
        "generated_at": "2026-09-27T09:00:00", "since": "2026-07-29",
        "seeds": {"label": "single-cell", "used": 1, "available": 1, "negatives": 0},
        "counts": {"returned": 2, "in_wiki": 0, "declined": 0, "outside_window": 0},
        "page_matches": [{"key": "10.1/u", "doi": "10.1/u", "paper_id": None,
                              "title": "Update me", "publication_date": "2026-09-01",
                              "venue": "Cell", "fit": 0.9, "first_seen": "2026-09-27",
                              "nearest": [{"key": "single-cell/a-2026-x", "score": 0.9}],
                              "triggers": [{"page": "synthesis/s", "z": 3.1,
                                            "text": "A head-to-head benchmark."}]}],
        "candidates": [{"key": "s2:p", "doi": None, "paper_id": "p", "title": "Other",
                        "publication_date": None, "year": 2026, "venue": "",
                        "fit": 0.8, "first_seen": "2026-09-20", "nearest": [],
                        "triggers": []}],
    }
    monkeypatch.setattr(R, "run", lambda **kw: snapshot)
    assert R.main(["--category", "single-cell"]) == 0
    out = capsys.readouterr().out
    assert out.index("Update me") < out.index("Other")
    assert "on-topic for [[synthesis/s]]; closest trigger (z 3.1)" in out
    assert "NEW 2026-09-01" in out
    assert "semanticscholar.org/paper/p" in out and "2026 (undated)" in out


def test_seed_error_exits_1(wiki, capsys):
    assert R.main(["--category", "nope"]) == 1
    assert "unknown category" in capsys.readouterr().err
