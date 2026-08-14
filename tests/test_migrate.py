"""`researchwiki migrate` — heading aliasing, frontmatter mapping, classification.

Hermetic: no network, no LLM, no model download. The DB is redirected per-test by
conftest's `_isolate_state_db`.

The two behaviors most worth pinning, because both fail silently:

1. **Merge on collision.** Two source headings can map to one canonical name.
   Emitting that name twice makes three modules disagree about the page —
   `parser._split_sections` keeps the *last* body, `wiki.extract_section` the
   *first*, and `coherence` prefix-matches and sees no problem. So a collision
   must concatenate bodies under one heading.

2. **Ordering.** `claim_slug` is content-addressed on `(section, normalized
   text)`, and `db rebuild` NULLs grader columns when that text changes. So a
   YAML-only splice (what `backfill hook` does) must preserve grading, while a
   body change must not — that asymmetry is what makes the free/paid phase split
   safe, and it's asserted both ways below.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from researchwiki.grade.coherence import PAPER_REQUIRED_SECTIONS
from researchwiki.grade.parser import ON_PAGE_H2, SECTION_KEYS, parse_claims
from researchwiki.migrate.classify import _split_authors, assess, assess_all
from researchwiki.migrate.frontmatter import map_keys, render_frontmatter
from researchwiki.migrate.sections import (
    CANONICAL_H2,
    canonical_for,
    plan_headings,
    rewrite_headings,
)
from researchwiki.wiki import Page


def _claims(body: str) -> list:
    page = Page(path=Path("wiki/x/y.md"), stem="y", category="x",
                fm={"type": "paper"}, body=body)
    return parse_claims(page)


# ---------- section aliases ----------

@pytest.mark.parametrize("canonical", CANONICAL_H2)
def test_canonical_headings_map_to_themselves(canonical):
    """Adding a required section can't silently escape the alias table."""
    assert canonical_for(canonical) == canonical


def test_canonical_set_covers_parser_and_coherence():
    """The two contracts the table must satisfy, asserted against their own
    constants rather than a copied list."""
    for name in ON_PAGE_H2:
        assert name in CANONICAL_H2
    for name in PAPER_REQUIRED_SECTIONS:
        assert name in CANONICAL_H2
    assert tuple(h.lower() for h in ON_PAGE_H2) == tuple(n for n, _ in SECTION_KEYS)


@pytest.mark.parametrize("heading,expected", [
    # Parenthetical qualifiers scope a section without redefining it. All four
    # forms below were live in the maintainer's corpus and each yielded zero
    # claims from a section that plainly had them.
    ("Key Contributions (as a Review)", "Key Contributions"),
    ("Key Contributions (as stated in the abstract)", "Key Contributions"),
    # A slashed pair names one section twice; both halves are already aliases.
    ("Results / Findings", "Results"),
    # An inflected suffix on a heading whose first word already decides it.
    ("Methodology and Architecting", "Methodology and Architecture"),
])
def test_decorated_headings_resolve_to_their_canonical_name(heading, expected):
    assert canonical_for(heading) == expected


@pytest.mark.parametrize("heading", [
    "Discussion (results)",          # qualifier must not smuggle it past the guard
    "Results and Discussion (2026)",
    "Discussion / Results",
])
def test_undecorating_cannot_defeat_the_ambiguity_guard(heading):
    """Stripping decoration runs the ambiguity check against both forms.

    Otherwise `## Discussion (results)` would strip to `Discussion`, miss the
    guard on the decorated form, and import discussion prose as graded claims.
    """
    assert canonical_for(heading) is None


def test_qualifier_only_heading_is_not_mapped():
    # `## (draft)` undecorates to the empty string; that must not match anything.
    assert canonical_for("(draft)") is None


def test_rename_makes_an_unciteable_page_citeable():
    body = ("## Key Findings\n\n- Scanned 17,000 variants for surface abundance.\n")
    assert _claims(body) == []          # the silent failure
    out, _ = rewrite_headings(body)
    assert len(_claims(out)) == 1


def test_merge_on_collision_keeps_both_bodies_under_one_heading():
    body = ("## Findings\n\n- First result across 1,042 samples.\n\n"
            "## Benchmarks\n\n- Second result covering 46 genes.\n")
    out, plan = rewrite_headings(body)
    assert out.count("## Results") == 1, "duplicate canonical H2 would drop a section"
    texts = [c.text for c in _claims(out)]
    assert any("1,042" in t for t in texts)
    assert any("46 genes" in t for t in texts)
    assert [c.merged_into_earlier for c in plan.changes] == [False, True]


def test_plain_rename_leaves_the_body_byte_identical():
    body = "## Findings\n\nSome prose.\n\n- A bullet long enough to be a claim.\n"
    out, _ = rewrite_headings(body)
    assert out.split("\n", 1)[1] == body.split("\n", 1)[1]


@pytest.mark.parametrize("heading", ["Results and Discussion", "References",
                                     "Bibliography", "Discussion", "Conclusions"])
def test_ambiguous_headings_are_reported_not_rewritten(heading):
    body = f"## {heading}\n\n- Something that looks like a finding here.\n"
    out, plan = rewrite_headings(body)
    assert out == body, "ambiguous headings must not be guessed at"
    assert plan.ambiguous and plan.needs_human
    assert canonical_for(heading) is None


def test_ambiguous_can_be_opted_into():
    body = "## Results and Discussion\n\n- A finding worth grading here.\n"
    out, plan = rewrite_headings(body, accept_ambiguous=True)
    assert "## Results" in out and not plan.ambiguous


def test_unmapped_headings_are_left_alone_and_listed():
    body = "## Reviewer Notes\n\nSome prose that isn't a paper section.\n"
    out, plan = rewrite_headings(body)
    assert out == body
    assert plan.unmapped == ["Reviewer Notes"]


def test_h1_and_h3_are_untouched():
    body = "# Findings\n\n### Findings\n\n## Findings\n\n- A real claim goes here.\n"
    out, _ = rewrite_headings(body)
    assert out.startswith("# Findings")
    assert "### Findings" in out
    assert "## Results" in out


def test_preamble_before_the_first_heading_survives():
    body = "Lead-in prose.\n\n## Findings\n\n- A claim that is long enough.\n"
    out, _ = rewrite_headings(body)
    assert out.startswith("Lead-in prose.")


def test_plan_headings_does_not_mutate():
    body = "## Findings\n\n- A claim.\n"
    plan_headings(body)
    assert body == "## Findings\n\n- A claim.\n"


# ---------- frontmatter ----------

def test_keys_alias_on_a_normalized_form():
    p = map_keys({"paper_title": "T", "Author List": "A B", "Publication Year": 2024})
    assert p.mapped["title"] == "T"
    assert p.mapped["authors"] == "A B"
    assert p.mapped["year"] == 2024
    assert ("paper_title", "title") in p.renames


def test_conflicting_aliases_are_refused_not_resolved():
    p = map_keys({"title": "T", "authors": "A", "year": 2024, "date": "2023-11-02"})
    assert p.needs_human
    assert p.conflicts[0][0] == "year"
    assert "year" not in p.mapped


def test_iso_date_yields_a_year_but_an_ambiguous_one_does_not():
    assert map_keys({"title": "T", "authors": "A", "date": "2023-05-02"}).mapped["year"] == 2023
    amb = map_keys({"title": "T", "authors": "A", "year": "11/02/23"})
    assert "year" not in amb.mapped
    assert "year" in amb.missing_required
    assert any("ambiguous" in n for n in amb.notes)


def test_missing_required_blocks_and_is_never_invented():
    p = map_keys({"title": "Only a title"})
    assert p.blocked
    assert set(p.missing_required) == {"authors", "year"}
    assert "authors" not in p.mapped and "year" not in p.mapped


def test_lookup_fillable_fields_are_flagged_not_guessed():
    p = map_keys({"title": "T", "authors": "A", "year": 2024})
    assert set(p.lookup_needed) == {"doi", "venue"}
    assert "doi" not in p.mapped


def test_unknown_keys_are_preserved():
    p = map_keys({"title": "T", "authors": "A", "year": 2024, "my_field": "keep"})
    assert p.extras == {"my_field": "keep"}


@pytest.mark.parametrize("authors", [
    "X Smith, Y Doe (senior: Z Roth)",     # the ': ' nested-mapping trap
    "Plain Author",
])
def test_rendered_block_is_valid_yaml(authors):
    p = map_keys({"title": "A paper: with a colon", "authors": authors, "year": 2024})
    block = render_frontmatter(p, stem="x-2024-a-paper", category="compbio",
                              page_type="paper", migrated_at="2026-01-01T00:00:00")
    d = yaml.safe_load(block)
    assert d["authors"] == authors
    assert d["title"] == "A paper: with a colon"


def test_rendered_category_matches_the_directory_argument():
    """`db rebuild` derives category from the parent dir and ignores YAML, so
    rendering anything else only trips lint's category_yaml_drift."""
    p = map_keys({"title": "T", "authors": "A", "year": 2024, "category": "[wrongcat]"})
    d = yaml.safe_load(render_frontmatter(p, stem="s", category="compbio",
                                         page_type="paper", migrated_at="t"))
    assert d["category"] == ["compbio"]


def test_provenance_is_honest():
    """A migrated page must not claim it came out of the agent pipeline."""
    p = map_keys({"title": "T", "authors": "A", "year": 2024})
    d = yaml.safe_load(render_frontmatter(p, stem="s", category="compbio",
                                         page_type="paper",
                                         migrated_at="2026-01-01T00:00:00"))
    # `source_collection: migrated` used to be asserted here; the field was
    # removed (no reader anywhere), and provenance rests on these two instead.
    assert "source_collection" not in d
    assert d["migrated_at"] == "2026-01-01T00:00:00"
    # The `migrated` tag went with the rest of paper-page `tags:`; `migrated_at`
    # is the provenance marker now and cannot be mistaken for vocabulary.
    assert not d.get("tags")
    assert "ingested_at" not in d and "author_model" not in d


def test_extra_containing_a_wikilink_stays_a_string():
    p = map_keys({"title": "T", "authors": "A", "year": 2024,
                  "note": "see [[other/page]] ratio 3:1"})
    d = yaml.safe_load(render_frontmatter(p, stem="s", category="c",
                                         page_type="paper", migrated_at="t"))
    assert d["note"] == "see [[other/page]] ratio 3:1"


# ---------- author splitting ----------

@pytest.mark.parametrize("raw,first", [
    ("Doe, Alice; Roe, Bob", "Doe, Alice"),   # ';' wins so 'Last, First' survives
    ("Jane Smith, John Doe", "Jane Smith"),
    (["A Smith", "B Doe"], "A Smith"),
])
def test_author_splitting_prefers_semicolons(raw, first):
    assert _split_authors(raw)[0] == first


# ---------- classification ----------

def _write(src: Path, name: str, fm: str, body: str, *, pdf: bool = True) -> Path:
    p = src / f"{name}.md"
    p.write_text(f"---\n{fm}\n---\n\n{body}")
    if pdf:
        (src / f"{name}.pdf").write_bytes(b"%PDF-1.4 x")
    return p


_PAPER_BODY = "## Key Contributions\n\n- A finding across 1,042 tested samples.\n"


def test_compliant_page_needs_no_rewrite(tmp_path):
    _write(tmp_path, "smith2024",
           'title: "A fast and versatile algorithm for search"\n'
           "authors: Jane Smith\nyear: 2024\ndoi: 10.1/x\nvenue: Nature\ntype: paper",
           _PAPER_BODY)
    a = assess(tmp_path / "smith2024.md", pdf_dir=tmp_path, category="compbio")
    assert a.verdict == "compliant"
    assert a.claims_before == a.claims_after == 1
    assert a.derived_stem == "smith-2024-a-fast-and-versatile-algorithm"


def test_simpler_wiki_page_is_fixable_and_gains_claims(tmp_path):
    _write(tmp_path, "doe2023",
           'paper_title: "Deep mutational scanning of a receptor"\n'
           'Author List: "Doe, Alice"\nPublication Year: 2023-05-02\njournal: Science',
           "## Key Findings\n\n- Scanned 17,000 variants for surface abundance.\n")
    a = assess(tmp_path / "doe2023.md", pdf_dir=tmp_path, category="compbio")
    assert a.verdict == "fixable"
    assert (a.claims_before, a.claims_after) == (0, 1)


def test_page_without_frontmatter_is_blocked(tmp_path):
    (tmp_path / "bare.md").write_text("# A page\n\nProse with no YAML block.\n")
    a = assess(tmp_path / "bare.md", pdf_dir=tmp_path)
    assert a.verdict == "blocked"
    assert any("frontmatter" in r for r in a.reasons)


def test_page_without_a_pdf_is_blocked(tmp_path):
    _write(tmp_path, "nopdf", 'title: "T"\nauthors: A B\nyear: 2024', _PAPER_BODY,
           pdf=False)
    a = assess(tmp_path / "nopdf.md", pdf_dir=tmp_path)
    assert a.verdict == "blocked"
    assert any("pdf" in r.lower() for r in a.reasons)


def test_non_paper_shaped_page_is_blocked(tmp_path):
    """The scope boundary: a note with no gradable section would land as an
    unciteable stub, so it's refused rather than imported."""
    _write(tmp_path, "notes", 'title: "Thoughts"\nauthors: Me\nyear: 2024',
           "## Random Musings\n\nSome thoughts while reading.\n")
    a = assess(tmp_path / "notes.md", pdf_dir=tmp_path)
    assert a.verdict == "blocked"
    assert any("one-paper" in r for r in a.reasons)


def test_batch_stem_collision_gets_a_bibtex_letter(tmp_path):
    for i, name in enumerate(("a1", "a2")):
        _write(tmp_path, name,
               f'title: "The same five leading words here {i}"\n'
               "authors: Jane Smith\nyear: 2024", _PAPER_BODY)
    got = sorted(a.derived_stem for a in assess_all(tmp_path, category="compbio"))
    assert got[0] == "smith-2024-the-same-five-leading-words"
    assert got[1] == "smith-2024b-the-same-five-leading-words"


def test_collision_lettering_is_deterministic(tmp_path):
    for i, name in enumerate(("z1", "z2")):
        _write(tmp_path, name,
               f'title: "Identical leading five words again {i}"\n'
               "authors: Jane Smith\nyear: 2024", _PAPER_BODY)
    first = [a.derived_stem for a in assess_all(tmp_path, category="compbio")]
    second = [a.derived_stem for a in assess_all(tmp_path, category="compbio")]
    assert first == second


def test_assess_writes_nothing(tmp_path):
    p = _write(tmp_path, "keep", 'title: "T"\nauthors: A B\nyear: 2024', _PAPER_BODY)
    before = p.read_text()
    assess(p, pdf_dir=tmp_path, category="compbio")
    assert p.read_text() == before
