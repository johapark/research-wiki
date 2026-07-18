"""Auto-apply gate for memory-evolution proposals.

The narrowest gate that ships: refine proposals in the bullet-append
flavor, at confidence >= 0.9, with all structural checks passing, get
materialized in place. Everything else — refine's line-replace flavor,
enhance (paragraph rewrites), and contrast — keeps going to disk for
human review. Important to pin precisely: a regression here either
silently rewrites synthesis pages we didn't intend to touch (corrupted
wiki) or refuses to apply correctly-shaped proposals (no value over the
existing manual flow).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.agents.phases.evolution import (
    EvolutionProposal,
    _insert_bullet_under,
    _insert_in_referenced_papers,
    _split_frontmatter,
    _update_generated_at,
    auto_apply_proposal,
)


# ---------- frontmatter / body helpers (pure) ----------

def test_split_frontmatter_basic():
    body = "---\nfoo: bar\n---\n# Body\nstuff"
    fm, rest = _split_frontmatter(body)
    assert fm == "---\nfoo: bar\n---\n"
    assert rest == "# Body\nstuff"


def test_split_frontmatter_missing_returns_empty():
    body = "# No frontmatter\nstuff"
    fm, rest = _split_frontmatter(body)
    assert fm == ""
    assert rest == body


def test_insert_in_referenced_papers_appends_to_existing_list():
    fm = (
        "---\n"
        "title: Foo\n"
        "referenced_papers:\n"
        "  - [[a/x]]\n"
        "  - [[b/y]]\n"
        "tags: []\n"
        "---\n"
    )
    new_fm, changed = _insert_in_referenced_papers(fm, "[[c/z]]")
    assert changed
    assert "  - [[c/z]]" in new_fm
    # Inserted after [[b/y]] (last entry), before tags
    lines = new_fm.split("\n")
    z_idx = next(i for i, l in enumerate(lines) if "[[c/z]]" in l)
    tags_idx = next(i for i, l in enumerate(lines) if l.startswith("tags:"))
    assert z_idx == tags_idx - 1


def test_insert_in_referenced_papers_idempotent():
    fm = "---\nreferenced_papers:\n  - [[a/x]]\n---\n"
    new_fm, changed = _insert_in_referenced_papers(fm, "[[a/x]]")
    assert not changed
    assert new_fm == fm


def test_insert_in_referenced_papers_refuses_when_list_missing():
    fm = "---\ntitle: Foo\ntags: []\n---\n"
    new_fm, changed = _insert_in_referenced_papers(fm, "[[a/x]]")
    assert not changed
    assert new_fm == fm


def test_insert_bullet_under_appends_at_section_end():
    body = (
        "# Title\n"
        "## Evidence\n"
        "- existing bullet\n"
        "\n"
        "## Next section\n"
        "more text\n"
    )
    new_body, changed = _insert_bullet_under(body, "## Evidence", "[[a/b]] — new")
    assert changed
    lines = new_body.split("\n")
    # New bullet inserted just before `## Next section` (after walking past blanks)
    next_section_idx = next(i for i, l in enumerate(lines) if l == "## Next section")
    assert lines[next_section_idx - 1] == "- [[a/b]] — new" or \
           lines[next_section_idx - 2] == "- [[a/b]] — new"


def test_insert_bullet_under_missing_section():
    body = "# Title\n## Evidence\nstuff\n"
    new_body, changed = _insert_bullet_under(body, "## Nonexistent", "x")
    assert not changed
    assert new_body == body


def test_insert_bullet_under_respects_h3_boundaries():
    """A `## Section` ends at the next `## ` or higher-level heading,
    NOT at sub-headings (`### Foo`)."""
    body = (
        "## Section\n"
        "### Subsection\n"
        "- old bullet\n"
        "\n"
        "## Next\n"
    )
    new_body, changed = _insert_bullet_under(body, "## Section", "[[x/y]] new")
    assert changed
    # New bullet should land BEFORE `## Next`, after the subsection content
    new_idx = new_body.index("[[x/y]] new")
    next_idx = new_body.index("## Next")
    assert new_idx < next_idx


def test_update_generated_at_replaces_existing():
    fm = "---\ntitle: Foo\ngenerated_at: 2026-01-01\ntags: []\n---\n"
    new_fm = _update_generated_at(fm, "2026-06-16")
    assert "generated_at: 2026-06-16" in new_fm
    assert "generated_at: 2026-01-01" not in new_fm


def test_update_generated_at_inserts_when_missing():
    fm = "---\ntitle: Foo\n---\n"
    new_fm = _update_generated_at(fm, "2026-06-16")
    assert "generated_at: 2026-06-16" in new_fm


# ---------- auto_apply_proposal (end-to-end on a temp wiki) ----------

def _proposal(*, verdict="refine", confidence=0.95,
              source_key="ai/source-paper", target_key="synthesis/target-page",
              add_bullet_under="## Evidence",
              bullet_text=None, patch=None) -> EvolutionProposal:
    """Builder for auto-apply gate tests.

    Default shape is `verdict="refine"` with the bullet-append patch — the
    only combination the gate accepts. Callers pass `patch=` explicitly to
    build refine-with-line-replace, enhance, and contrast fixtures.
    """
    if patch is None:
        if bullet_text is None:
            bullet_text = f"[[{source_key}]] — new contribution"
        patch = {
            "add_bullet_under": add_bullet_under,
            "bullet_text": bullet_text,
        }
    return EvolutionProposal(
        source_key=source_key, target_key=target_key,
        verdict=verdict, confidence=confidence,
        rationale="test", patch=patch,
    )


def _make_target(tmp_path: Path, key: str = "synthesis/target-page") -> Path:
    """Create a synthesis page in the temp wiki dir."""
    page_path = tmp_path / f"{key}.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_text(
        "---\n"
        "title: Target Page\n"
        "type: synthesis\n"
        "referenced_papers:\n"
        "  - [[other/existing-paper]]\n"
        "generated_at: 2026-01-01\n"
        "tags: [synthesis]\n"
        "---\n"
        "\n"
        "## Evidence\n"
        "- existing bullet\n"
        "\n"
        "## Next section\n"
        "other stuff\n"
    )
    return page_path


# ---- Gate: verdict + confidence ----

def test_gate_blocks_refine_line_replace(tmp_path):
    """Refine with the line-replace patch shape modifies existing prose —
    higher blast radius than appending a bullet, so it's blocked from
    auto-apply. The gate identifies the shape by the absence of
    `add_bullet_under`."""
    _make_target(tmp_path)
    p = _proposal(verdict="refine", confidence=0.99, patch={
        "target_line_match": "- existing bullet",
        "new_line": "- existing bullet, now with [[ai/source-paper]] context",
    })
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path)
    assert not ok
    assert "bullet-append" in reason


def test_gate_blocks_enhance(tmp_path):
    """Enhance rewrites a whole paragraph — the highest-blast-radius edit
    the proposer emits. It never auto-applies regardless of confidence.
    Reviewer must run the grounding + fidelity gates by hand after."""
    _make_target(tmp_path)
    p = _proposal(verdict="enhance", confidence=1.0, patch={
        "target_section": "## Evidence",
        "target_paragraph_match": "existing paragraph anchor",
        "new_paragraph": "Rewritten paragraph citing [[ai/source-paper]].",
    })
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path)
    assert not ok
    assert "verdict=enhance" in reason


def test_gate_blocks_contrast(tmp_path):
    _make_target(tmp_path)
    p = _proposal(verdict="contrast", confidence=0.99)
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path)
    assert not ok
    assert "verdict=contrast" in reason


def test_gate_blocks_low_confidence(tmp_path):
    _make_target(tmp_path)
    p = _proposal(confidence=0.85)
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path)
    assert not ok
    assert "confidence" in reason and "0.85" in reason


def test_gate_blocks_at_threshold_boundary(tmp_path):
    """Exactly 0.9 → applies; 0.89 → blocked. The conjunction is `>= 0.9`."""
    _make_target(tmp_path)
    ok_at, _ = auto_apply_proposal(_proposal(confidence=0.90), wiki_root_dir=tmp_path)
    assert ok_at
    # Re-create the page since the previous call modified it
    _make_target(tmp_path)
    ok_under, _ = auto_apply_proposal(_proposal(confidence=0.89), wiki_root_dir=tmp_path)
    assert not ok_under


# ---- is_actionable: which verdicts reach the proposal writer ----
#
# Regression guard: `is_actionable()` is the filter both the agent phase
# (memory_evolve) and the CLI (tasks/evolve) apply *before* a proposal is
# written to disk. If a live verdict is missing from this set, its proposals
# are silently discarded and never surface for review. The list must track
# VALID_VERDICTS minus "none".

@pytest.mark.parametrize("verdict", ["refine", "enhance", "contrast"])
def test_is_actionable_true_for_live_verdicts(verdict):
    assert _proposal(verdict=verdict).is_actionable()


def test_is_actionable_false_for_none():
    assert not _proposal(verdict="none", patch={}).is_actionable()


def test_is_actionable_covers_every_non_none_verdict():
    """The actionable set must equal VALID_VERDICTS minus "none" — otherwise
    a newly-added verdict (as `enhance` once was) gets judged but dropped
    before it can be written."""
    from researchwiki.agents.phases.evolution import VALID_VERDICTS

    actionable = {
        v for v in VALID_VERDICTS
        if _proposal(verdict=v, patch={}).is_actionable()
    }
    assert actionable == VALID_VERDICTS - {"none"}


# ---- Gate: structural checks ----

def test_gate_blocks_when_target_missing(tmp_path):
    p = _proposal(target_key="synthesis/does-not-exist")
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path)
    assert not ok
    assert "target page missing" in reason


def test_gate_blocks_when_section_missing(tmp_path):
    _make_target(tmp_path)
    p = _proposal(add_bullet_under="## Nonexistent Section")
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path)
    assert not ok
    assert "not found" in reason


def test_gate_blocks_when_source_link_already_present(tmp_path):
    """Defensive — `_select_neighbors` already excludes targets that
    contain the source, so this case shouldn't arise from the normal
    flow. But direct callers might pass us a stale proposal."""
    page_path = _make_target(tmp_path)
    # Inject the source link into the body
    body = page_path.read_text().replace(
        "- existing bullet",
        "- existing bullet referencing [[ai/source-paper]] already",
    )
    page_path.write_text(body)
    p = _proposal()
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path)
    assert not ok
    assert "already linked" in reason


def test_gate_blocks_when_bullet_text_lacks_source_link(tmp_path):
    _make_target(tmp_path)
    p = _proposal(bullet_text="bullet without source wikilink")
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path)
    assert not ok
    assert "doesn't contain" in reason


def test_gate_blocks_when_patch_fields_missing(tmp_path):
    """An empty patch on a `refine`-verdict proposal fails the shape check
    first — `add_bullet_under` is absent, so the gate can't tell whether
    this is a malformed bullet-append refine or a line-replace refine
    stripped of its fields. Either way, no auto-apply."""
    _make_target(tmp_path)
    p = _proposal()
    p.patch = {}
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path)
    assert not ok
    assert "bullet-append" in reason


# ---- render_proposal_md format ----

def test_render_enhance_shows_old_and_new_paragraph():
    """Enhance proposal files must show BOTH the paragraph anchor and the
    new prose so the reviewer can diff before applying. Also surfaces the
    checklist for footnote preservation."""
    from researchwiki.agents.phases.evolution import render_proposal_md
    p = EvolutionProposal(
        source_key="genetics/ahmad-2026",
        target_key="synthesis/fh-review",
        verdict="enhance", confidence=0.8, rationale="prevalence framing shifted",
        patch={
            "target_section": "## Evidence",
            "target_paragraph_match": "FH prevalence has historically been quoted as 1 in 500…",
            "new_paragraph": "FH prevalence is now ~1 in 311 per two meta-analyses [[genetics/ahmad-2026]], superseding the historical 1-in-500 anchor.[^earlier]",
        },
    )
    out = render_proposal_md(p)
    assert "# ENHANCE — [[synthesis/fh-review]]" in out
    assert "## Patch (enhance — rewrite paragraph)" in out
    assert "target_paragraph_match" not in out          # not the raw key
    assert "FH prevalence has historically been quoted as 1 in 500" in out
    assert "FH prevalence is now ~1 in 311" in out
    assert "Every `[^footnote-id]`" in out


def test_render_refine_bullet_append_labeled_distinctly():
    """The renderer labels the two refine flavors differently so the
    reviewer knows which shape they're looking at."""
    from researchwiki.agents.phases.evolution import render_proposal_md
    append = EvolutionProposal(
        source_key="ai/x", target_key="synthesis/y",
        verdict="refine", confidence=0.95, rationale="new fact",
        patch={"add_bullet_under": "## Evidence",
               "bullet_text": "[[ai/x]] — new fact"},
    )
    line = EvolutionProposal(
        source_key="ai/x", target_key="synthesis/y",
        verdict="refine", confidence=0.7, rationale="old line stale",
        patch={"target_line_match": "Old line.",
               "new_line": "New line citing [[ai/x]]."},
    )
    assert "append bullet" in render_proposal_md(append)
    assert "replace line" in render_proposal_md(line)


# ---- Successful apply ----

def test_apply_writes_bullet_into_section(tmp_path):
    page_path = _make_target(tmp_path)
    p = _proposal(bullet_text="[[ai/source-paper]] — adds machine X result of 50%")
    ok, reason = auto_apply_proposal(p, wiki_root_dir=tmp_path, today="2026-06-16")
    assert ok, reason
    body = page_path.read_text()
    bullet = "- [[ai/source-paper]] — adds machine X result of 50%"
    assert bullet in body
    # Bullet should be under ## Evidence, before ## Next section
    evidence_idx = body.index("## Evidence")
    next_idx = body.index("## Next section")
    bullet_idx = body.index(bullet)
    assert evidence_idx < bullet_idx < next_idx


def test_apply_adds_to_referenced_papers(tmp_path):
    page_path = _make_target(tmp_path)
    p = _proposal()
    ok, _ = auto_apply_proposal(p, wiki_root_dir=tmp_path, today="2026-06-16")
    assert ok
    body = page_path.read_text()
    assert "  - [[ai/source-paper]]" in body
    # Existing reference still there
    assert "  - [[other/existing-paper]]" in body


def test_apply_updates_generated_at(tmp_path):
    page_path = _make_target(tmp_path)
    p = _proposal()
    ok, _ = auto_apply_proposal(p, wiki_root_dir=tmp_path, today="2026-06-16")
    assert ok
    body = page_path.read_text()
    assert "generated_at: 2026-06-16" in body
    assert "generated_at: 2026-01-01" not in body


def test_apply_preserves_other_sections(tmp_path):
    page_path = _make_target(tmp_path)
    p = _proposal()
    ok, _ = auto_apply_proposal(p, wiki_root_dir=tmp_path, today="2026-06-16")
    assert ok
    body = page_path.read_text()
    # Existing content intact
    assert "## Next section" in body
    assert "other stuff" in body
    assert "- existing bullet" in body
    # Frontmatter delimiters intact
    assert body.count("---\n") >= 2
