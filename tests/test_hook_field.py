"""The `hook:` catalog gloss: author trailer, backfill proposer, lint checks.

Three routes produce or police the field and they must agree on its shape:

  - `phases.draft.split_gloss_trailer` — the ingest path. The author emits
    HANDLE/HOOK after the six sections; this parses them off so the critic and
    graders never see them.
  - `phases.commit._parse_gloss_response` — the backfill / migration path, a
    standalone lightweight call over an existing page body.
  - `tasks.lint.yaml_checks.find_missing_hook` / `find_hook_too_long` — the
    review queue and the advisory ceilings.

The regression these pin down: the field must be *omitted* when the model gives
us nothing usable, never backfilled from a Summary slice. The retired index-line
generator did exactly that and produced 244 entries stating each paper's
question instead of its finding, 97 of them cut mid-word by a 200-char slice.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from researchwiki.agents import runner
from researchwiki.agents.phases.commit import (
    _extract_gloss_context,
    _parse_gloss_response,
)
from researchwiki.agents.phases.draft import Draft, split_gloss_trailer
from researchwiki.agents.promote import _build_frontmatter, _yaml_dq
from researchwiki.tasks.lint.walk import all_pages
from researchwiki.tasks.lint.yaml_checks import (
    HOOK_MAX_CHARS,
    find_hook_too_long,
    find_missing_author_model,
    find_missing_hook,
)
from researchwiki.wiki import read_page


BODY = """## Summary

Prose about the paper.

## Key Contributions

- A finding across 1,042 samples.

## Related Papers

(none)
"""

TRAILER = (
    "\n---\n\nHANDLE: NanoSeq Population\n"
    "HOOK: Updated NanoSeq compatible with whole-exome capture (<5 errors per "
    "10^9 bp); 1,042 oral epithelium samples reveal 46 genes under positive "
    "selection.\n"
)


# ---------- author trailer (ingest path) ----------

def test_trailer_is_parsed_and_stripped_from_body():
    body, handle, hook = split_gloss_trailer(BODY + TRAILER)
    assert handle == "NanoSeq Population"
    assert hook.startswith("Updated NanoSeq compatible")
    # The stripping is the safety property: a trailer that survived into the
    # body would be graded as page prose and rendered on the page.
    assert "HANDLE" not in body and "HOOK" not in body
    assert body.rstrip().endswith("(none)")


def test_missing_trailer_leaves_body_untouched():
    body, handle, hook = split_gloss_trailer(BODY)
    assert (handle, hook) == ("", "")
    assert body == BODY


def test_trailer_tolerates_quotes_bold_and_whitespace():
    messy = BODY + '\n---\nHANDLE: **"Evo 2"**\nHOOK:   "A   result-first   gloss."  \n'
    _, handle, hook = split_gloss_trailer(messy)
    assert handle == "Evo 2"
    assert hook == "A result-first gloss."


def test_runaway_hook_is_rejected_not_truncated():
    """A hook far past budget means the format was ignored, not that the model
    was verbose. Rejecting yields '' (→ omitted field → lint queue); truncating
    would commit a sentence fragment, which is the old bug."""
    _, handle, hook = split_gloss_trailer(
        BODY + "\n---\nHANDLE: X\nHOOK: " + ("word " * 400) + "\n"
    )
    assert handle == "X"
    assert hook == ""


def test_overlong_handle_is_rejected():
    _, handle, _ = split_gloss_trailer(
        BODY + "\n---\nHANDLE: " + ("a" * 60) + "\nHOOK: Fine gloss.\n"
    )
    assert handle == ""


def test_hook_is_flattened_to_one_line():
    """`hook:` is a single-line YAML scalar; a wrapped hook must not break it."""
    _, _, hook = split_gloss_trailer(
        BODY + "\n---\nHANDLE: X\nHOOK: first part\nof the same hook\n"
    )
    assert "\n" not in hook
    assert hook == "first part of the same hook"


# ---------- backfill proposer (migration path) ----------

def test_proposer_accepts_the_same_shape_as_the_trailer():
    handle, hook = _parse_gloss_response(
        "HANDLE: DOCKS\nHOOK: Universal hitting sets lower minimizer density."
    )
    assert handle == "DOCKS"
    assert hook == "Universal hitting sets lower minimizer density."


def test_proposer_degrades_each_field_independently():
    handle, hook = _parse_gloss_response("HANDLE: DOCKS")
    assert (handle, hook) == ("DOCKS", "")
    handle, hook = _parse_gloss_response("HOOK: A gloss with no handle.")
    assert (handle, hook) == ("", "A gloss with no handle.")
    assert _parse_gloss_response("") == ("", "")


def test_gloss_context_includes_contributions_not_just_summary():
    """Summary alone reliably yields the paper's question; the findings and the
    numbers a result-first hook needs live in Key Contributions."""
    ctx = _extract_gloss_context(BODY)
    assert "Prose about the paper." in ctx
    assert "1,042 samples" in ctx


# ---------- YAML serialisation ----------

@pytest.mark.parametrize("hook", [
    'Sharpens [[genomics/roberts-2004-reducing]] — density 1.5× lower.',
    'Ratio 3:1, "quoted" text, a # hash, and a \\ backslash.',
    "Plain gloss with no metacharacters.",
])
def test_hook_survives_a_yaml_round_trip(hook):
    page = _build_frontmatter(
        {"title": "T: with colon", "year": 2026, "venue": "Nature"},
        "x-2026-t", "genomics", BODY, short_name="X", hook=hook,
    )
    parsed = yaml.safe_load(page.split("---", 2)[1])
    assert parsed["hook"] == " ".join(hook.split())


def test_wikilink_hook_is_quoted_so_yaml_reads_it_as_a_string():
    """Unquoted `[[x/y]]` parses as a nested flow sequence — the failure lint
    reports as `unquoted_wikilink_lists`."""
    assert _yaml_dq("see [[a/b]]").startswith('"')


def test_empty_hook_is_omitted_entirely():
    page = _build_frontmatter(
        {"title": "T", "year": 2026}, "x-2026-t", "genomics", BODY,
        short_name="X", hook="",
    )
    assert "hook:" not in page.split("---", 2)[1]


# ---------- lint checks ----------

@pytest.fixture
def tmp_wiki(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    fake = lambda: wiki  # noqa: E731
    monkeypatch.setattr("researchwiki.paths.wiki_dir", fake)
    monkeypatch.setattr("researchwiki.tasks.lint.walk.wiki_dir", fake)
    return wiki


def _mkpage(wiki: Path, key: str, fm: str) -> Path:
    p = wiki / f"{key}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\n{fm}\n---\n\n## Summary\n\nBody.\n")
    return p


def _walk(wiki: Path):
    pages = all_pages()
    fm = {}
    for p in pages:
        pg = read_page(p)
        fm[p] = pg.fm if pg else {}
    return pages, fm


def test_missing_hook_lists_only_pages_without_one(tmp_wiki):
    _mkpage(tmp_wiki, "genomics/a-2026-x", 'type: paper\ntitle: A\nhook: "A gloss."')
    _mkpage(tmp_wiki, "genomics/b-2026-y", "type: paper\ntitle: B")
    assert find_missing_hook(*_walk(tmp_wiki)) == ["genomics/b-2026-y"]


def test_root_bookkeeping_is_not_expected_to_carry_a_hook(tmp_wiki):
    """index.md / log.md / views.md are not catalogued in themselves."""
    (tmp_wiki / "index.md").write_text("---\ntype: meta\n---\n\n# Index\n")
    (tmp_wiki / "log.md").write_text("---\ntype: meta\n---\n\n# Log\n")
    assert find_missing_hook(*_walk(tmp_wiki)) == []


def test_explicitly_exempt_types_are_skipped(tmp_wiki):
    _mkpage(tmp_wiki, "genomics/dash", "type: dashboard\ntitle: D")
    assert find_missing_hook(*_walk(tmp_wiki)) == []


def test_page_without_a_type_still_needs_a_hook(tmp_wiki):
    """23 pages predate the `type:` requirement; inferring `paper` keeps them in
    scope rather than silently excusing them."""
    _mkpage(tmp_wiki, "genomics/c-2026-z", "title: C")
    assert find_missing_hook(*_walk(tmp_wiki)) == ["genomics/c-2026-z"]


def test_hook_too_long_uses_the_per_type_ceiling(tmp_wiki):
    long = "x" * 500
    # 500 chars is over the paper ceiling (400) but under synthesis's (1000).
    _mkpage(tmp_wiki, "genomics/p-2026-a", f'type: paper\ntitle: P\nhook: "{long}"')
    _mkpage(tmp_wiki, "synthesis/s-topic", f'type: synthesis\ntitle: S\nhook: "{long}"')
    hits = find_hook_too_long(*_walk(tmp_wiki))
    assert [(k, cap) for k, _, cap in hits] == [
        ("genomics/p-2026-a", HOOK_MAX_CHARS["paper"])
    ]


def test_hook_too_long_sorts_longest_first(tmp_wiki):
    _mkpage(tmp_wiki, "genomics/a-2026-a", f'type: paper\ntitle: A\nhook: "{"x" * 600}"')
    _mkpage(tmp_wiki, "genomics/b-2026-b", f'type: paper\ntitle: B\nhook: "{"x" * 450}"')
    assert [k for k, _, _ in find_hook_too_long(*_walk(tmp_wiki))] == [
        "genomics/a-2026-a", "genomics/b-2026-b",
    ]


def test_a_hook_within_its_ceiling_is_not_reported(tmp_wiki):
    _mkpage(tmp_wiki, "genomics/a-2026-a", f'type: paper\ntitle: A\nhook: "{"x" * 399}"')
    assert find_hook_too_long(*_walk(tmp_wiki)) == []


# ---------- trailer survives a revision (runner handoff) ----------
#
# The author parses the trailer once, then `split_gloss_trailer` strips it from
# `Draft.text` so no downstream consumer sees it. That makes the revision phases
# the field's blind spot: evolve and DEBUG are prompted with the *stripped* body
# and have no trailer of their own to parse, so a freshly-constructed Draft
# defaults both fields to '' — and the runner's `ctx.winner = revised` swap then
# discards the author's values. Symptom in the wild: 3 of 4 papers in one batch
# committed with no `hook:` and `short_name: TODO`, all three of them the ones
# whose critic flagged something. The one paper the critic passed kept its hook.

def _prior_draft():
    return Draft(
        text=BODY, model="m", temperature=0.2, input_tokens=1, output_tokens=1,
        iteration_id=100, handle="NanoSeq Population",
        hook="Updated NanoSeq compatible with whole-exome capture.",
        scores={"semantic_score": 0.60, "n_drift": 0, "mean_bm25": 4.0},
    )


def _revision_ctx(prior):
    """Minimal Context stand-in for the two revision wrappers."""
    return SimpleNamespace(
        metadata={"paper_type": "research"}, sections={}, use_stub=False,
        attempt_id="a", paper_stem="s", pdf_filename="s.pdf", iteration=3,
        drafts=[prior], winner=prior, crosslink_candidates=[],
        next_iter=lambda: None,
    )


def test_evolve_revision_inherits_the_authors_handle_and_hook(monkeypatch):
    prior = _prior_draft()
    monkeypatch.setattr(runner.phases, "evolve", lambda **kw: SimpleNamespace(
        text="## Summary\n\nRevised prose.\n", model="m", temperature=0.2,
        input_tokens=10, output_tokens=20,
    ))
    monkeypatch.setattr(runner, "write_iteration", lambda **kw: 101)

    revised = runner._phase_evolve(_revision_ctx(prior), None, prior, object())

    assert revised.text != prior.text, "sanity: the revision replaced the body"
    assert revised.handle == prior.handle
    assert revised.hook == prior.hook


def test_debug_repair_inherits_the_authors_handle_and_hook(monkeypatch):
    prior = _prior_draft()
    ctx = _revision_ctx(prior)
    rows = []
    monkeypatch.setattr(runner.phases, "debug", lambda **kw: SimpleNamespace(
        text="## Summary\n\nRepaired prose.\n", model="m", temperature=0.0,
        input_tokens=10, output_tokens=20, issues_addressed=["drift"],
    ))
    monkeypatch.setattr(runner, "write_iteration", lambda **kw: rows.append(kw) or 102)
    monkeypatch.setattr(runner, "_phase_grade", lambda *a, **kw: None)
    # Revert path: the Draft is built (and appended) before the improvement
    # gate, so the inheritance is observable without mocking promote's gates.
    monkeypatch.setattr(runner, "_is_strict_improvement", lambda *a: False)

    runner._phase_debug(
        ctx, None, BODY, 3, SimpleNamespace(reasons=[], promoted=False), None,
        issues=["drift"],
    )

    repaired = ctx.drafts[-1]
    assert repaired.text != prior.text, "sanity: the repair replaced the body"
    assert repaired.handle == prior.handle
    assert repaired.hook == prior.hook
    assert rows[0]["duration_ms"] >= 0


# ---------- venue / title quoting (regression, 2026-08-11) ----------
#
# `hook:` was quoted from the start; `venue:` and `title:` were not, and a
# colon-bearing journal name is not exotic. *Molecular Therapy: Nucleic Acids*
# landed a page whose ENTIRE frontmatter failed to parse — which takes the page
# out of `db rebuild`, out of the index, and out of every query, while the file
# still looks perfectly correct to a human reading it. Found on the first real
# reference-manager import.

@pytest.mark.parametrize("venue", [
    "Molecular Therapy: Nucleic Acids",
    "OMICS: A Journal of Integrative Biology",
    "Cell Reports: Methods",
    "Nature",
    'A venue with "quotes" in it',
    "Genome Biology #2",
])
def test_venue_survives_a_yaml_round_trip(venue):
    page = _build_frontmatter(
        {"title": "T", "year": 2026, "venue": venue},
        "x-2026-t", "genomics", BODY, short_name="X",
    )
    parsed = yaml.safe_load(page.split("---", 2)[1])
    assert parsed["venue"] == venue


@pytest.mark.parametrize("title", [
    "ATAC-seq: A Method for Assaying Chromatin Accessibility",
    'A title with "quoted" words',
    "Ratio 3:1 in a title",
])
def test_title_survives_a_yaml_round_trip(title):
    page = _build_frontmatter(
        {"title": title, "year": 2026, "venue": "Nature"},
        "x-2026-t", "genomics", BODY, short_name="X",
    )
    parsed = yaml.safe_load(page.split("---", 2)[1])
    assert parsed["title"] == title


def test_the_whole_frontmatter_block_parses_with_a_colon_venue():
    """The actual failure: not that `venue` was wrong, but that one bad value
    made every other field unreadable too."""
    page = _build_frontmatter(
        {"title": "T", "year": 2026, "venue": "Molecular Therapy: Nucleic Acids",
         "doi": "10.1234/test.2022.0001"},
        "ada-2022-a-paper", "other", BODY, short_name="X",
    )
    parsed = yaml.safe_load(page.split("---", 2)[1])
    assert parsed["doi"] == "10.1234/test.2022.0001"
    assert parsed["type"] == "paper"


# --- author_model: provenance for the page's own prose -----------------------
#
# Scoped deliberately to pages the pipeline wrote (`ingested_at:` present).
# The field is optional by contract and 31 of 120 corpus pages predate it, so an
# unscoped check would bury 11 real gaps under legacy noise. These tests pin the
# scope, because widening it is the tempting and wrong change.

def test_missing_author_model_flags_an_ingested_page_without_one(tmp_wiki):
    _mkpage(tmp_wiki, "genomics/a-2026-x",
            'type: paper\ntitle: A\ningested_at: 2026-06-16T23:32:46')
    assert find_missing_author_model(*_walk(tmp_wiki)) == ["genomics/a-2026-x"]


def test_an_ingested_page_carrying_the_field_is_clean(tmp_wiki):
    _mkpage(tmp_wiki, "genomics/a-2026-x",
            'type: paper\ntitle: A\ningested_at: 2026-06-16T23:32:46\n'
            'author_model: "claude-opus-4-7"')
    assert find_missing_author_model(*_walk(tmp_wiki)) == []


def test_a_page_predating_the_pipeline_is_out_of_scope(tmp_wiki):
    """No `ingested_at:` means no ingest run wrote this page, so nothing was
    ever in a position to record the model. `park-2023` in the real corpus is
    exactly this shape — flagging it would be noise, not a finding."""
    _mkpage(tmp_wiki, "genomics/legacy-2019-z", "type: paper\ntitle: Legacy")
    assert find_missing_author_model(*_walk(tmp_wiki)) == []


@pytest.mark.parametrize("ptype", ["synthesis", "idea", "concept", "whitepaper", "guidance"])
def test_only_paper_pages_are_in_scope(tmp_wiki, ptype):
    """Reference docs take the field on the manual path where it is genuinely
    optional, and the synthesis/idea scaffolds emit `author_model: "TODO"` for a
    human to fill rather than guaranteeing a value."""
    _mkpage(tmp_wiki, f"genomics/p-2026-{ptype}",
            f"type: {ptype}\ntitle: P\ningested_at: 2026-06-16T23:32:46")
    assert find_missing_author_model(*_walk(tmp_wiki)) == []


def test_an_empty_author_model_counts_as_missing(tmp_wiki):
    """An empty or whitespace value carries no more provenance than no key —
    and `promote` writes the key unconditionally when it has a model, so a blank
    one means the value was lost, not that the field was never wanted."""
    _mkpage(tmp_wiki, "genomics/a-2026-x",
            'type: paper\ntitle: A\ningested_at: 2026-06-16T23:32:46\nauthor_model: "  "')
    assert find_missing_author_model(*_walk(tmp_wiki)) == ["genomics/a-2026-x"]


def test_root_bookkeeping_is_skipped(tmp_wiki):
    (tmp_wiki / "index.md").write_text(
        "---\ntype: meta\ningested_at: 2026-06-16T23:32:46\n---\n\n# Index\n")
    assert find_missing_author_model(*_walk(tmp_wiki)) == []
