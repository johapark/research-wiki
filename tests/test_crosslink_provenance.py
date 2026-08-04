"""Cross-link bullets must state only the relationship that was established.

Observed 2026-08-04 ingesting a Nature Reviews Genetics pangenome review:
S2 reported referenceCount 139 but `/references` came back empty (citation
graph unresolved for a new DOI), so the run had zero citation evidence. The
gleaning pass then reopened LLM-rejected semantic neighbours, and each
resulting link got a reciprocal bullet reading "cites this paper" — a
citation the pipeline never checked. The PDF mentioned none of the linked
papers.

Two defects, tested here:
  1. the back-link note ignored `CrosslinkCandidate.kind`
  2. gleaning could not tell "no citations found" from "citation graph empty"
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from researchwiki.agents import promote
from researchwiki.agents.phases import crosslinks


@dataclass
class _Cand:
    wikilink: str
    kind: str


@pytest.fixture
def captured(monkeypatch):
    """Capture (target_key, note) instead of writing to disk."""
    seen: list[tuple[str, str]] = []

    def fake_append(target_path, source_key, note="DEFAULT"):
        seen.append((target_path.stem, note))
        return True

    monkeypatch.setattr(
        "researchwiki.backlinks.append_related_paper", fake_append
    )
    monkeypatch.setattr(promote, "wiki_dir", lambda: __import__("pathlib").Path("/w"))
    return seen


def test_topical_backlink_does_not_claim_a_citation(captured):
    """The bug: a semantic-only link asserted 'cites this paper'."""
    promote._append_backlinks([_Cand("compbio/x", "topical")], "compbio", "src")
    (_, note), = captured
    assert "cites this paper" not in note
    assert "cited by this paper" not in note
    assert "topically related" in note


def test_cited_by_source_keeps_citation_phrasing(captured):
    """Source cites the target → on the target, 'cites this paper' is right."""
    promote._append_backlinks(
        [_Cand("compbio/x", "cited_by_source")], "compbio", "src"
    )
    (_, note), = captured
    assert note.startswith("cites this paper")


def test_cites_source_reverses_the_direction(captured):
    """Target cites the source, so the same wording would be backwards.

    This direction was also wrong before the fix — every kind got the one
    hardcoded default.
    """
    promote._append_backlinks(
        [_Cand("compbio/x", "cites_source")], "compbio", "src"
    )
    (_, note), = captured
    assert note.startswith("cited by this paper")


def test_unknown_kind_falls_back_to_the_weakest_claim(captured):
    promote._append_backlinks([_Cand("compbio/x", "")], "compbio", "src")
    (_, note), = captured
    assert "topically related" in note


def test_every_kind_is_marked_for_refinement():
    """The `(auto-added; refine)` marker is what tells a later pass to rewrite."""
    for note in promote._BACKLINK_NOTES.values():
        assert "(auto-added; refine)" in note


# ---------- gleaning suppression ----------

class _Hit:
    def __init__(self, key):
        self.key, self.title, self.score = key, key, 0.9


def _judge(monkeypatch, *, allow_gleaning, gleaned=("compbio/g",)):
    """Run _judge_candidates with pass 1 rejecting every candidate."""
    hits = [_Hit(f"compbio/h{i}") for i in range(4)]
    # The real prompt builder reads each candidate's page off disk.
    monkeypatch.setattr(crosslinks, "_build_judge_prompt", lambda *a, **k: "P")
    monkeypatch.setattr(
        crosslinks, "_parse_judge_response",
        lambda _text: [{"wikilink": h.key, "verdict": "none"} for h in hits],
    )
    monkeypatch.setattr(
        crosslinks.llm, "call",
        lambda *a, **k: type("R", (), {"text": "{}"})(),
    )
    monkeypatch.setattr(
        crosslinks, "_gleaning_pass",
        lambda *a, **k: [
            crosslinks.CrosslinkCandidate(
                doi="", wikilink=g, kind="topical", title=g,
                year=None, verified=True, relationship="(gleaning)",
            ) for g in gleaned
        ],
    )
    return crosslinks._judge_candidates(
        {}, {}, hits, allow_gleaning=allow_gleaning
    )


def test_gleaning_suppressed_yields_no_candidates(monkeypatch):
    """With the citation graph unresolved, rejected neighbours stay rejected."""
    assert _judge(monkeypatch, allow_gleaning=False) == []


def test_gleaning_still_runs_when_citations_were_available(monkeypatch):
    """Suppression must be conditional — the recall pass is otherwise wanted."""
    out = _judge(monkeypatch, allow_gleaning=True)
    assert [c.wikilink for c in out] == ["compbio/g"]


def test_crosslink_candidates_reports_unresolved_citation_graph(monkeypatch):
    """referenceCount > 0 but /references empty is the signal to suppress."""
    class _Article:
        reference_count = 139

    class _Provider:
        def get_by_doi(self, doi): return _Article()
        def get_references(self, a): return []
        def get_citations(self, a): return []

    monkeypatch.setattr(crosslinks, "read_wiki_dois", lambda: {"10.1/x": "c/x"})
    monkeypatch.setattr(crosslinks, "SemanticScholarProvider", _Provider)
    monkeypatch.setattr(crosslinks, "extract_ref_dois", lambda *a, **k: [])
    monkeypatch.setattr(
        "researchwiki.providers.crossref.fetch_crossref_refs", lambda d: []
    )
    stats: dict = {}
    crosslinks.crosslink_candidates(
        __import__("pathlib").Path("/nonexistent.pdf"),
        {"doi": "10.1038/whatever"},
        stats=stats,
    )
    assert stats["citation_graph_unresolved"] is True


def test_populated_reference_list_is_not_flagged(monkeypatch):
    """A resolved graph that simply shares nothing with the wiki is normal."""
    class _Article:
        reference_count = 40

    class _Ref:
        doi, title, year = "10.9/other", "Other", 2020

    class _Provider:
        def get_by_doi(self, doi): return _Article()
        def get_references(self, a): return [_Ref()]
        def get_citations(self, a): return []

    monkeypatch.setattr(crosslinks, "read_wiki_dois", lambda: {"10.1/x": "c/x"})
    monkeypatch.setattr(crosslinks, "SemanticScholarProvider", _Provider)
    monkeypatch.setattr(crosslinks, "extract_ref_dois", lambda *a, **k: [])
    monkeypatch.setattr(
        "researchwiki.providers.crossref.fetch_crossref_refs", lambda d: []
    )
    stats: dict = {}
    crosslinks.crosslink_candidates(
        __import__("pathlib").Path("/nonexistent.pdf"),
        {"doi": "10.1038/whatever"},
        stats=stats,
    )
    assert stats["citation_graph_unresolved"] is False
