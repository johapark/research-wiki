"""Trigger-eval harness.

Tests the *harness*, not the triggers — whether a given trigger line is any good
is what running the command tells you, and that costs tokens. What is pinned here
is the machinery that makes those numbers trustworthy:

  - an errored grading is excluded from the denominators, never scored as a
    failure (without this one timeout silently depresses a pass rate);
  - routing to the *wrong* prompt counts as a miss, not a pass;
  - code-loaded `*-system` prompts are not reported as orphans;
  - `--dry-run` spends nothing.

Hermetic: the LLM is a stub, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchwiki.eval import triggers as tr
from researchwiki.tasks import eval_triggers as cli


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    for slug, body in [
        ("recovery", "# Recovery\n\nRe-ingest after broken metadata.\n"),
        ("share-page", "# Share\n\nProduce a standalone markdown.\n"),
        ("ask-system", "You are the ask agent.\n"),
        ("author-system-research", "You are the author.\n"),
        ("unlinked", "# Unlinked\n\nNobody points here.\n"),
    ]:
        (tmp_path / "prompts" / f"{slug}.md").write_text(body, encoding="utf-8")

    (tmp_path / "CLAUDE.md").write_text(
        "# Wiki\n\n"
        "When lint flags missing_doi, re-ingest with overrides. "
        "Full workflow in [`prompts/recovery.md`](./prompts/recovery.md).\n\n"
        "When the user asks to share a page, produce a standalone doc. "
        "Procedure in [`prompts/share-page.md`](./prompts/share-page.md).\n",
        encoding="utf-8")

    from researchwiki import paths
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(tr, "wiki_root", lambda: tmp_path)
    return tmp_path


# ---------- pointer extraction ----------

def test_collects_pointers_with_their_trigger_line(repo):
    pointers = tr.collect_pointers()
    assert [p.slug for p in pointers] == ["recovery", "share-page"]
    assert "missing_doi" in pointers[0].line
    assert "Re-ingest after broken metadata" in pointers[0].body


def test_anchored_links_resolve_to_the_same_prompt(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "recovery.md").write_text("x", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(
        "See [`recovery.md` § Half-landed](./prompts/recovery.md#half-landed-promote).\n",
        encoding="utf-8")
    monkeypatch.setattr(tr, "wiki_root", lambda: tmp_path)
    assert [p.slug for p in tr.collect_pointers()] == ["recovery"]


def test_a_pointer_to_a_missing_file_is_skipped(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "CLAUDE.md").write_text(
        "See [`gone.md`](./prompts/gone.md).\n", encoding="utf-8")
    monkeypatch.setattr(tr, "wiki_root", lambda: tmp_path)
    assert tr.collect_pointers() == []


def test_system_prompts_are_not_orphans(repo):
    """`ask-system` and `author-system-research` are loaded by code through
    `prompt_lib`, so their absence from CLAUDE.md is correct. Counting them
    would be seven permanent false positives in the real repo."""
    assert tr.orphan_prompts() == ["unlinked"]


@pytest.mark.parametrize("slug,gated", [
    ("recovery", True),
    ("ask-system", False),
    ("author-system-research", False),   # marker is not a suffix here
    ("suggest-splits-system", False),
])
def test_is_gated_prompt(slug, gated):
    assert tr.is_gated_prompt(slug) is gated


# ---------- scoring ----------

def _graded(slug, should_fire, chose, error=None):
    return tr.Graded(case=tr.Case(slug, "req", should_fire), chose=chose, error=error)


def test_correct_when_it_fires_for_its_own_case():
    assert _graded("recovery", True, "recovery").correct is True
    assert _graded("recovery", False, None).correct is True


def test_routing_to_another_prompt_is_a_miss():
    """The all-triggers catalogue is what makes this observable — a grader that
    picks `share-page` for a recovery request has not merely 'not fired'."""
    g = _graded("recovery", True, "share-page")
    assert g.correct is False


def test_firing_when_it_should_not_is_a_miss():
    assert _graded("recovery", False, "recovery").correct is False


def test_errors_are_excluded_from_the_denominators():
    """The detail that makes the rates trustworthy: one provider hiccup must
    not depress a pass rate."""
    graded = [
        _graded("recovery", True, "recovery"),
        _graded("recovery", True, "recovery"),
        _graded("recovery", True, None, error="timeout"),
    ]
    rep = tr.summarize(graded)["recovery"]
    assert rep.should_fire_total == 2, "the errored case is not in the denominator"
    assert rep.should_fire_hit == 2
    assert rep.recall == 1.0
    assert rep.errors == 1
    assert rep.misses == []


def test_summarize_separates_the_two_directions():
    graded = [
        _graded("recovery", True, "recovery"),
        _graded("recovery", True, None),
        _graded("recovery", False, None),
        _graded("recovery", False, "recovery"),
    ]
    rep = tr.summarize(graded)["recovery"]
    assert (rep.should_fire_hit, rep.should_fire_total) == (1, 2)
    assert (rep.should_not_hit, rep.should_not_total) == (1, 2)
    assert len(rep.misses) == 2


def test_rates_are_none_when_a_direction_has_no_cases():
    rep = tr.summarize([_graded("recovery", True, "recovery")])["recovery"]
    assert rep.recall == 1.0
    assert rep.precision is None


# ---------- reply parsing ----------

@pytest.mark.parametrize("text,expected", [
    ('{"slug": "recovery"}', {"slug": "recovery"}),
    ('```json\n{"slug": "recovery"}\n```', {"slug": "recovery"}),
    ('Sure! {"slug": null} hope that helps', {"slug": None}),
    ("not json at all", None),
    ("", None),
    ('["a", "b"]', None),
])
def test_parse_json_tolerates_model_habits(text, expected):
    assert tr._parse_json(text) == expected


def test_grade_case_survives_a_provider_exception(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("provider down")
    monkeypatch.setattr(tr.llm, "call", boom)
    g = tr.grade_case(tr.Case("recovery", "req", True), "catalogue")
    assert g.error and "provider down" in g.error
    assert g.chose is None


def test_grade_all_returns_one_result_per_case(monkeypatch):
    monkeypatch.setattr(tr, "grade_case",
                        lambda c, cat, use_stub=False: _graded(c.slug, c.should_fire, c.slug))
    cases = [tr.Case("recovery", f"r{i}", True) for i in range(7)]
    assert len(tr.grade_all(cases, [])) == 7


def test_grade_all_on_no_cases(monkeypatch):
    assert tr.grade_all([], []) == []


# ---------- CLI ----------

def test_dry_run_spends_nothing(repo, monkeypatch, capsys):
    def boom(**kwargs):
        raise AssertionError("dry run must not call the provider")
    monkeypatch.setattr(tr.llm, "call", boom)

    assert cli.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 trigger-gated prompt(s)" in out
    assert "would spend 2 generator call(s) + 20 grader call(s)" in out


def test_dry_run_names_orphans(repo, capsys):
    cli.main(["--dry-run"])
    assert "unlinked" in capsys.readouterr().out


def test_slug_filter_narrows_the_run(repo, capsys):
    cli.main(["--dry-run", "--slug", "recovery"])
    out = capsys.readouterr().out
    assert "recovery" in out and "share-page" not in out


def test_no_pointers_exits_1(tmp_path, monkeypatch, capsys):
    (tmp_path / "CLAUDE.md").write_text("# nothing here\n", encoding="utf-8")
    (tmp_path / "prompts").mkdir()
    monkeypatch.setattr(tr, "wiki_root", lambda: tmp_path)
    assert cli.main([]) == 1
    assert "no prompt pointers" in capsys.readouterr().err


def test_provider_failure_exits_2(repo, monkeypatch, capsys):
    monkeypatch.setattr(tr, "generate_cases",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")))
    assert cli.main([]) == 2
    assert "provider error" in capsys.readouterr().err


def test_json_report_shape(repo, monkeypatch, capsys):
    monkeypatch.setattr(tr, "generate_cases", lambda p, n, use_stub=False: [
        tr.Case(p.slug, "fires", True), tr.Case(p.slug, "holds", False),
    ])
    monkeypatch.setattr(tr, "grade_all", lambda cases, ptrs, use_stub=False: [
        tr.Graded(case=c, chose=(c.slug if c.should_fire else None)) for c in cases
    ])

    assert cli.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["orphan_prompts"] == ["unlinked"]
    assert payload["prompts"]["recovery"]["recall"] == 1.0
    assert payload["prompts"]["recovery"]["precision"] == 1.0
    assert payload["prompts"]["recovery"]["misses"] == []


def test_text_report_names_the_miss(repo, monkeypatch, capsys):
    monkeypatch.setattr(tr, "generate_cases", lambda p, n, use_stub=False: [
        tr.Case(p.slug, "a request it should have caught", True),
    ])
    monkeypatch.setattr(tr, "grade_all", lambda cases, ptrs, use_stub=False: [
        tr.Graded(case=c, chose=None) for c in cases
    ])

    cli.main([])
    out = capsys.readouterr().out
    assert "misses" in out
    assert "a request it should have caught" in out
    assert "should have fired, routed to no prompt" in out


# ---------- gating text is the section, not the link's line ----------

def test_gating_text_is_the_enclosing_section(tmp_path, monkeypatch):
    """The real case this exists for: `export-bibliography` states its trigger
    one paragraph above the paragraph carrying the link. Taking the link's line
    alone fed the grader a passage about citekeys and then scored the trigger as
    having missed — blaming the prompt for the extractor's choice."""
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "export-bibliography.md").write_text("x", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(
        "### Export — emit the corpus as a bibliography\n\n"
        'Trigger: *"can I get a bib file"*, *"export my library"*.\n\n'
        "**The citekey is the page stem.** Read "
        "[`prompts/export-bibliography.md`](./prompts/export-bibliography.md).\n\n"
        "### Something Else\n\nUnrelated.\n",
        encoding="utf-8")
    monkeypatch.setattr(tr, "wiki_root", lambda: tmp_path)

    p = tr.collect_pointers()[0]
    assert "can I get a bib file" in p.line, "the trigger sentence must be included"
    assert "Something Else" not in p.line, "the next section must not bleed in"
    assert p.line.startswith("### Export")


def test_section_extraction_stops_at_the_next_heading(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(
        "## First\n\nSee [`a`](./prompts/a.md).\n\n## Second\n\nother content\n",
        encoding="utf-8")
    monkeypatch.setattr(tr, "wiki_root", lambda: tmp_path)
    assert "other content" not in tr.collect_pointers()[0].line


def test_section_is_capped(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(
        "## Big\n\nSee [`a`](./prompts/a.md).\n\n" + ("filler line\n" * 500),
        encoding="utf-8")
    monkeypatch.setattr(tr, "wiki_root", lambda: tmp_path)
    assert len(tr.collect_pointers()[0].line) <= tr.MAX_SECTION_CHARS


def test_a_link_before_any_heading_still_resolves(tmp_path, monkeypatch):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(
        "Preamble with [`a`](./prompts/a.md) before any heading.\n", encoding="utf-8")
    monkeypatch.setattr(tr, "wiki_root", lambda: tmp_path)
    assert tr.collect_pointers()[0].line.startswith("Preamble")
