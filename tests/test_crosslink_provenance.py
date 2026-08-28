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


# ---------- `lint --fix` must mirror the direction too ----------
#
# Observed 2026-08-04 on a 2026 review page: 13 of its 14 auto-added bullets
# read "cites this paper" against targets from 2019-2025, which cannot cite a
# 2026 paper. `promote` had written the correct note on each target page, but
# the reciprocal bullet came from `lint --fix`, which called
# `append_related_paper` with no `note=` and so took the one hardcoded default
# regardless of direction.

from researchwiki.backlinks import (          # noqa: E402
    CITED_BY_NOTE, CITES_NOTE, EARLIER_NOTE, MORE_RECENT_NOTE, TOPICAL_NOTE,
    invert_relationship_note,
)
from researchwiki.tasks.lint import link_checks   # noqa: E402


def test_inverting_cites_yields_cited_by():
    """`[[T]] — cites this paper` on S means T cites S; on T, S is cited by T."""
    assert invert_relationship_note(CITES_NOTE) == CITED_BY_NOTE


def test_inverting_cited_by_yields_cites():
    assert invert_relationship_note(CITED_BY_NOTE) == CITES_NOTE


def test_inversion_is_an_involution():
    for note in (CITES_NOTE, CITED_BY_NOTE):
        assert invert_relationship_note(invert_relationship_note(note)) == note


@pytest.mark.parametrize("note", [
    "topically related (auto-added; refine)",
    "near-duplicate claim (auto-added; claim-overlap)",
    "instantiates this concept (auto-added; concept-link)",
    "foundational Enformer model cited as a landmark architecture in Fig. 2",
    "",
])
def test_uninvertible_notes_degrade_to_the_weakest_claim(note):
    """Understating is safe; asserting an unchecked citation is fabrication."""
    assert invert_relationship_note(note) == TOPICAL_NOTE


@pytest.mark.parametrize("note", [MORE_RECENT_NOTE, EARLIER_NOTE])
def test_recency_notes_are_not_directional_claims(note):
    """The recency phrasings must never read as citations.

    They exist to say "newer/older work on this topic" without asserting a
    citation, which only holds while their text stays free of citation
    language — reword either constant to contain "cites this paper" and
    `invert_relationship_note` would start flipping it.
    """
    assert invert_relationship_note(note) == TOPICAL_NOTE
    assert "(auto-added; refine)" in note


def test_cited_by_is_not_swallowed_by_the_cites_probe():
    """"cited by this paper" must not match on the "cites this paper" branch."""
    assert invert_relationship_note("CITED BY THIS PAPER (auto-added)") == CITES_NOTE


def test_mirrored_note_reads_direction_off_the_source_bullet():
    src = (
        "## Related Papers\n\n"
        "- [[compbio/other-2020-thing]] — cites this paper (auto-added; refine)\n"
        "- [[compbio/jaganathan-2019-predicting-splicing]] — cites this paper\n"
    )
    note = link_checks._mirrored_note(src, "compbio/jaganathan-2019-predicting-splicing")
    assert note == CITED_BY_NOTE


def test_mirrored_note_matches_a_bare_stem_wikilink():
    """CLAUDE.md mandates bare `[[stem]]` in tables, so both forms must resolve."""
    src = "- [[jaganathan-2019-predicting-splicing]] — cited by this paper\n"
    note = link_checks._mirrored_note(src, "compbio/jaganathan-2019-predicting-splicing")
    assert note == CITES_NOTE


@pytest.mark.parametrize("src_prose", ["", "- [[compbio/unrelated-2020-x]] — cites this paper"])
def test_mirrored_note_without_a_matching_bullet_is_weakest(src_prose):
    assert link_checks._mirrored_note(src_prose, "compbio/absent-2019-y") == TOPICAL_NOTE


def test_apply_backlink_fixes_passes_the_mirrored_note(monkeypatch, tmp_path):
    """End to end: the bug's exact shape — a 2026 review citing a 2019 paper."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "researchwiki.backlinks.append_related_paper",
        lambda target_path, source_key, note=TOPICAL_NOTE: (
            seen.append((source_key, note)) or True
        ),
    )
    monkeypatch.setattr(link_checks, "wiki_dir", lambda: tmp_path)
    review = tmp_path / "compbio" / "nagai-2026-review.md"
    # Stubbed so the prose map can be keyed by a tmp_path Path; `page_key`'s
    # own wiki-relative derivation is covered in the walk tests.
    monkeypatch.setattr(
        link_checks, "page_key",
        lambda p: f"{p.parent.name}/{p.stem}",
    )
    prose = {review: "- [[compbio/jaganathan-2019-splicing]] — cites this paper\n"}

    written = link_checks.apply_backlink_fixes(
        [("compbio/nagai-2026-review", "compbio/jaganathan-2019-splicing")], prose
    )

    assert written == {"compbio/jaganathan-2019-splicing": 1}
    (source_key, note), = seen
    assert source_key == "compbio/nagai-2026-review"
    assert note == CITED_BY_NOTE, "the 2019 paper does not cite the 2026 review"


def test_apply_backlink_fixes_without_prose_understates(monkeypatch, tmp_path):
    """The optional argument is a degradation, never a fabrication."""
    seen: list[str] = []
    monkeypatch.setattr(
        "researchwiki.backlinks.append_related_paper",
        lambda target_path, source_key, note=TOPICAL_NOTE: (
            seen.append(note) or True
        ),
    )
    monkeypatch.setattr(link_checks, "wiki_dir", lambda: tmp_path)
    link_checks.apply_backlink_fixes([("compbio/a-2024-x", "compbio/b-2020-y")])
    assert seen == [TOPICAL_NOTE]


def test_append_related_paper_default_note_asserts_nothing(tmp_path):
    """A caller that forgets `note=` must not fabricate a citation."""
    from researchwiki.backlinks import append_related_paper

    p = tmp_path / "t.md"
    p.write_text("---\ntitle: t\n---\n\n## Related Papers\n\n")
    append_related_paper(p, "cgt/smith-2024-x")
    body = p.read_text()
    assert "cites this paper" not in body
    assert "topically related" in body


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
    """Suppression is conditional; a second strict evidence pass is allowed."""
    out = _judge(monkeypatch, allow_gleaning=True)
    assert [c.wikilink for c in out] == ["compbio/g"]


def test_first_pass_borderline_verdict_does_not_become_a_link(monkeypatch):
    """Semantic adjacency is a proposal signal, never durable link evidence."""
    hit = _Hit("compbio/h0")
    monkeypatch.setattr(crosslinks, "_build_judge_prompt", lambda *a, **k: "P")
    monkeypatch.setattr(
        crosslinks, "_parse_judge_response",
        lambda _text: [{
            "wikilink": hit.key,
            "verdict": "borderline",
            "rationale": "same problem, different method",
        }],
    )
    monkeypatch.setattr(
        crosslinks.llm, "call",
        lambda *a, **k: type("R", (), {"text": "{}"})(),
    )

    assert crosslinks._judge_candidates({}, {}, [hit]) == []


def test_gleaning_borderline_verdict_does_not_become_a_link(monkeypatch):
    """The recall pass must use the same source-supported threshold as pass 1."""
    hit = _Hit("compbio/h0")
    monkeypatch.setattr(crosslinks, "_build_gleaning_prompt", lambda *a, **k: "P")
    monkeypatch.setattr(
        crosslinks, "_parse_judge_response",
        lambda _text: [{
            "wikilink": hit.key,
            "verdict": "borderline",
            "rationale": "plausible but uncertain",
        }],
    )
    monkeypatch.setattr(
        crosslinks.llm, "call",
        lambda *a, **k: type("R", (), {"text": "{}"})(),
    )

    assert crosslinks._gleaning_pass(
        {}, {}, [hit], [hit.key], {hit.key: hit},
    ) == []


def test_judge_schema_cannot_request_a_speculative_link():
    verdict = (
        crosslinks._JUDGE_SCHEMA["properties"]["verdicts"]["items"]
        ["properties"]["verdict"]
    )
    assert verdict["enum"] == ["topical", "none"]


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


def test_topical_candidates_skip_stale_semantic_rows(monkeypatch, tmp_path):
    """A removed page can remain in the derived index until the next upsert."""
    present = tmp_path / "compbio" / "present.md"
    present.parent.mkdir(parents=True)
    present.write_text("---\ntitle: Present\ntype: paper\n---\n\n## Summary\nPresent.\n")
    hits = [_Hit("compbio/missing"), _Hit("compbio/present")]

    monkeypatch.setattr(crosslinks, "wiki_dir", lambda: tmp_path)
    monkeypatch.setattr(crosslinks.semantic_pages, "index_exists", lambda: True)
    monkeypatch.setattr(
        crosslinks.semantic_pages, "query_text", lambda *args, **kwargs: hits,
    )

    out = crosslinks.propose_crosslinks(
        {"title": "New paper"}, {}, use_stub=True,
    )

    assert out == [], "stub similarity cannot perform source-engagement verification"


def test_judge_prompt_tolerates_candidate_removed_during_prompt_build(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(crosslinks, "wiki_dir", lambda: tmp_path)

    prompt = crosslinks._build_judge_prompt({}, {}, [_Hit("compbio/gone")])

    assert "# Candidate wiki pages" in prompt
    assert "compbio/gone" not in prompt
