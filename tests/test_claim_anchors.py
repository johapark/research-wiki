"""Tests for [[stem#slug]] claim anchors — grounding + dangling lint.

Covers the assessment gates for page-level anchor support:
  - extract_claim_anchors pulls (stem, slug) pairs from wikilinks
  - grounding treats a resolving anchor as a citation
  - grounding treats a dangling anchor as ungrounded (with a specific reason)
  - the plain `[[stem]]` form (no anchor) is unchanged by the new code path
  - the lint dangling-anchor scan surfaces the right pages
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.grade.grounding import (
    check,
    extract_claim_anchors,
)
from researchwiki.tasks.lint.claim_anchors import find_dangling_claim_anchors


# ---------- anchor extraction ----------


def test_extract_single_anchor():
    text = "GraphRAG introduces a pipeline. [[edge-2024#kc-15edb372]]"
    assert extract_claim_anchors(text) == [("edge-2024", "kc-15edb372")]


def test_extract_multiple_anchors_dedups():
    text = ("First fact [[stem-a#res-abc12345]] and second [[stem-b#res-def67890]] "
            "and a repeat of the first [[stem-a#res-abc12345]].")
    got = extract_claim_anchors(text)
    assert got == [
        ("stem-a", "res-abc12345"),
        ("stem-b", "res-def67890"),
    ]


def test_extract_ignores_plain_wikilinks():
    text = "See [[stem-a]] and [[compbio/stem-b]] for context."
    assert extract_claim_anchors(text) == []


def test_extract_supports_category_prefix():
    text = "A finding [[compbio/stem-a#kc-abcd1234]] holds."
    assert extract_claim_anchors(text) == [("compbio/stem-a", "kc-abcd1234")]


def test_extract_strips_alias_suffix():
    text = "See [[stem-a#kc-abcd1234|GraphRAG's contribution]]."
    got = extract_claim_anchors(text)
    assert got == [("stem-a", "kc-abcd1234")]


# ---------- grounding with anchor validation ----------


LONG_CLAIM = (
    "This is a sufficiently long claim about a specific finding "
    "in the paper that must be grounded against a wikilink citation "
    "or it will be reported as ungrounded by the grader"
)


def test_resolving_anchor_grounds_the_unit():
    text = f"{LONG_CLAIM}. [[stem-a#kc-15edb372]]"
    report = check(text, valid_anchors={("stem-a", "kc-15edb372")})
    assert report.total_claims == 1
    assert report.grounded_claims == 1


def test_dangling_anchor_does_not_ground():
    text = f"{LONG_CLAIM}. [[stem-a#kc-deadbeef]]"
    report = check(text, valid_anchors=set())
    assert report.total_claims == 1
    assert report.grounded_claims == 0
    ungrounded = report.ungrounded_units
    assert len(ungrounded) == 1
    # Specific flag reason so authors know it's a slug mistake, not missing cite.
    assert "dangling" in (ungrounded[0].flag_reason or "").lower()


def test_plain_wikilink_still_grounds_when_no_anchor():
    text = f"{LONG_CLAIM}. [[stem-a]]"
    report = check(text, valid_anchors=set())
    assert report.total_claims == 1
    assert report.grounded_claims == 1


def test_mixed_dangling_and_valid_anchors():
    text = (
        f"{LONG_CLAIM}. [[stem-a#kc-15edb372]]\n\n"
        f"{LONG_CLAIM}. [[stem-a#kc-deadbeef]]"
    )
    report = check(text, valid_anchors={("stem-a", "kc-15edb372")})
    assert report.total_claims == 2
    assert report.grounded_claims == 1
    assert len(report.ungrounded_units) == 1


def test_resolve_anchors_disabled_treats_all_as_grounded():
    """Back-compat: pre-Phase-3 callers relied on plain-wikilink grounding
    accepting any `[[stem#anything]]`. `resolve_anchors=False` restores that."""
    text = f"{LONG_CLAIM}. [[stem-a#kc-anything]]"
    report = check(text, resolve_anchors=False)
    assert report.total_claims == 1
    assert report.grounded_claims == 1


# ---------- dangling-anchor lint check ----------


def test_find_dangling_returns_empty_when_no_anchors():
    pages_body: dict[Path, str] = {
        Path("wiki/compbio/foo.md"): "A plain page. [[stem-a]] link only.",
    }
    # No anchors → no DB access needed → empty result regardless of DB state.
    assert find_dangling_claim_anchors(pages_body) == []


def test_find_dangling_surfaces_unresolved_pair(monkeypatch):
    """Stub the resolver so we can test the lint in isolation."""
    from researchwiki.tasks.lint import claim_anchors as ca

    def fake_resolve(pairs):
        return {("stem-a", "res-real1234")}  # only this one exists

    monkeypatch.setattr(ca, "_resolve_claim_anchors", fake_resolve)

    pages_body: dict[Path, str] = {
        Path("wiki/compbio/foo.md"): (
            "Real claim. [[stem-a#res-real1234]]\n\n"
            "Bogus claim. [[stem-a#kc-deadbeef]]"
        ),
        Path("wiki/synthesis/bar.md"): (
            "Another bogus. [[stem-b#res-nope0000]]"
        ),
    }
    dangling = find_dangling_claim_anchors(pages_body)
    # Two dangling anchors: kc-deadbeef and res-nope0000.
    slugs = sorted(d["slug"] for d in dangling)
    assert slugs == ["kc-deadbeef", "res-nope0000"]
    # Each carries the source page.
    pages_with_dangling = {d["page"].name for d in dangling}
    assert pages_with_dangling == {"foo.md", "bar.md"}
