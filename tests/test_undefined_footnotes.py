"""Undefined `[^id]` footnote references.

Why this needs its own check: a footnote with no definition cites nothing, and
the two page gates both under-report it. `check-grounding` fails a unit only
when the undefined footnote is its *sole* citation, so a paragraph citing
`[^a][^b]` with `b` undefined passes — and `grade synthesis` then grades that
unit against `a` alone, reporting an empty `unresolved_citations`. Both gates
go green while a paper's worth of claims was never checked. Found by an eval
run whose page referenced `[^kinney-2025]` six times and defined it zero.
"""
from __future__ import annotations

from pathlib import Path

from researchwiki.tasks.lint.claim_anchors import find_undefined_footnote_refs
from researchwiki.tasks.lint.contracts import LINT_JSON_KEYS

P = Path("wiki/cgt/a-2024-paper.md")


def _run(body: str, page: Path = P):
    return find_undefined_footnote_refs({page: body})


def test_lint_json_exposes_the_key():
    assert "undefined_footnote_refs" in LINT_JSON_KEYS


def test_defined_footnote_passes():
    assert _run("Claim.[^a]\n\n[^a]: [[cgt/x-2020-y]]\n") == []


def test_undefined_footnote_is_flagged():
    got = _run("Claim.[^a]\n\n[^b]: [[cgt/x-2020-y]]\n")
    assert len(got) == 1
    assert "`[^a]`" in got[0]["detail"]
    assert got[0]["kind"] == "undefined_footnote_ref"


def test_the_mixed_case_the_gates_miss_is_flagged():
    """The motivating case: one defined, one not. Both gates pass this."""
    got = _run("Both tools enumerate candidates.[^a][^b]\n\n[^a]: [[cgt/x-2020-y]]\n")
    assert len(got) == 1
    assert "`[^b]`" in got[0]["detail"] and "`[^a]`" not in got[0]["detail"]


def test_detail_names_the_consequence_not_just_the_id():
    got = _run("Claim.[^a]\n")
    assert "under-grade" in got[0]["detail"] or "skip" in got[0]["detail"]


def test_many_missing_ids_are_truncated_with_a_count():
    body = "Claim." + "".join(f"[^m{i}]" for i in range(8)) + "\n"
    (v,) = _run(body)
    assert "8 footnote ref(s)" in v["detail"] and "+3 more" in v["detail"]


def test_root_meta_pages_are_skipped():
    """log.md quotes footnote ids in prose about citations; it carries none of
    its own. Same exclusion broken_wikilinks makes."""
    assert _run("entry mentioning [^brixi-2026]\n", page=Path("wiki/log.md")) == []


def test_footnotes_inside_fenced_code_do_not_count():
    body = "Claim.[^a]\n\n```\n[^example]: how to write one\n```\n\n[^a]: [[cgt/x-2020-y]]\n"
    assert _run(body) == []


def test_a_page_with_no_footnotes_is_not_reported():
    assert _run("Plain prose with [[cgt/x-2020-y]] and no footnotes.\n") == []
