"""Contracts for safe, bounded bottom-up synthesis proposals."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

from researchwiki.synthesis_candidates.detect import Candidate, _check_synthesis_coverage
from researchwiki.synthesis_candidates.judge import MAX_MEMBERS_PER_JUDGE, judge_candidate
from researchwiki.synthesis_candidates.render import render_proposal
from researchwiki.wiki import Page


def _candidate(*, members: list[str], judged: bool = True) -> Candidate:
    return Candidate(
        members=members,
        titles={k: k for k in members},
        density=1.0,
        edges=[],
        edge_signal_counts={},
        common_keywords=["exact topic"],
        nearest_synthesis=None,
        nearest_synthesis_overlap=0.0,
        verdict="new",
        members_missing_from_nearest=[],
        judged=judged,
        judge_topic='Exact "topic"',
    )


def _page(key: str) -> Page:
    category, stem = key.split("/", 1)
    return Page(
        path=Path("wiki") / category / f"{stem}.md",
        stem=stem,
        category=category,
        fm={"type": "paper", "title": stem, "keywords": ["exact topic"]},
        body="## Summary\n\nA short summary.",
    )


def _command_from_proposal(text: str) -> list[str]:
    block = text.split("```\n", 1)[1].split("\n   ```", 1)[0]
    # Shell continuation removes the newline. The final member must not carry
    # one, otherwise the fence / following shell line becomes part of the call.
    assert not block.rstrip().endswith("\\")
    return shlex.split(block.replace("\\\n", " "))


def test_new_proposal_renders_a_valid_synthesize_invocation(tmp_path, monkeypatch):
    members = ["ai/a-2026-x", "ai/b-2025-y"]
    candidate = _candidate(members=members)
    candidate.member_verdicts = [
        SimpleNamespace(key=k, verdict="in_scope", rationale="fits") for k in members
    ]
    rendered = render_proposal(candidate)
    argv = _command_from_proposal(rendered)

    assert argv[:2] == ["researchwiki", "synthesize"]
    assert "--type" not in argv and "--category" not in argv
    assert argv[argv.index("--papers") + 1:] == members
    assert argv[argv.index("--topic-seed") + 1] == 'Exact "topic"'

    # Execute the rendered command in a disposable wiki. This catches parser
    # drift as well as shell-continuation errors, not merely string shape.
    for key in members:
        category, stem = key.split("/", 1)
        page = tmp_path / "wiki" / category / f"{stem}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(f"---\ntype: paper\ncategory: [{category}]\n---\n\n## Summary\n\ntext\n")
    monkeypatch.chdir(tmp_path)
    from researchwiki.tasks import synthesize
    assert synthesize.main(argv[2:]) == 0
    written = (tmp_path / "wiki/synthesis/exact-topic.md").read_text()
    assert "category: [ai]" in written
    assert 'author_model: "TODO"' in written
    assert 'topic_seed: "Exact \'topic\'"' in written


def test_extend_proposal_never_instructs_synthesis_referenced_papers_edits():
    candidate = _candidate(members=["ai/a-2026-x"])
    candidate.verdict = "extend"
    candidate.nearest_synthesis = "synthesis/existing"
    candidate.nearest_synthesis_overlap = 0.5
    candidate.members_missing_from_nearest = ["ai/a-2026-x"]
    rendered = render_proposal(candidate)
    assert "YAML `referenced_papers:`" not in rendered
    assert "`## References`" in rendered


def test_claim_anchor_citations_count_toward_existing_synthesis_coverage():
    synthesis = Page(
        path=Path("wiki/synthesis/existing.md"),
        stem="existing",
        category="synthesis",
        fm={"type": "synthesis"},
        body="## Evidence\n\nSupported claim. [[a-2026-x#kc-1234abcd]]\n",
    )
    nearest, coverage, refs = _check_synthesis_coverage(
        ["ai/a-2026-x", "ai/b-2025-y"], [synthesis]
    )
    assert nearest == "synthesis/existing"
    assert coverage == 0.5
    assert refs == {"ai/a-2026-x"}


def test_judge_batches_every_member_and_requires_complete_responses(monkeypatch):
    members = [f"ai/p{i}" for i in range(MAX_MEMBERS_PER_JUDGE + 2)]
    candidate = _candidate(members=members, judged=False)
    pages = [_page(k) for k in members]
    from researchwiki.agents import llm

    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs["prompt"])
        if kwargs["schema"].get("required") == ["topic"]:
            return SimpleNamespace(
                text=json.dumps({"topic": "Bounded topic"}),
                input_tokens=10, output_tokens=20,
            )
        keys = [line[6:-2] for line in kwargs["prompt"].splitlines()
                if line.startswith("### [[") and line.endswith("]]")]
        payload = {"topic": "Bounded topic", "verdicts": [
            {"key": k, "verdict": "in_scope", "rationale": "fits"} for k in keys
        ]}
        return SimpleNamespace(text=json.dumps(payload), input_tokens=10, output_tokens=20)

    monkeypatch.setattr(llm, "call", fake_call)
    judge_candidate(candidate, [], pages)

    assert len(calls) == 3
    assert "# Whole cluster" in calls[0]
    assert all("# Fixed synthesis topic\nBounded topic" in prompt for prompt in calls[1:])
    assert candidate.judged is True
    assert candidate.judge_batches == 3
    assert candidate.judge_topic == "Bounded topic"
    assert {v.key for v in candidate.member_verdicts} == set(members)
    assert candidate.judge_input_tokens == 30
    assert candidate.judge_output_tokens == 60


def test_judge_rejects_a_truncated_batch_instead_of_filtering_members(monkeypatch):
    members = ["ai/a", "ai/b"]
    candidate = _candidate(members=members, judged=False)
    from researchwiki.agents import llm

    monkeypatch.setattr(
        llm, "call",
        lambda **_: SimpleNamespace(
            text=json.dumps({"topic": "partial", "verdicts": [
                {"key": "ai/a", "verdict": "in_scope", "rationale": "only one"},
            ]}),
            input_tokens=10, output_tokens=20,
        ),
    )
    judge_candidate(candidate, [], [_page(k) for k in members])

    assert candidate.judged is False
    assert candidate.member_verdicts == []
    assert candidate.judge_batches == 1


def test_synthesis_cli_defaults_to_local_preview(monkeypatch, capsys):
    from researchwiki.tasks import _synthesis_candidates as cli

    seen = {}

    def fake_find(**kwargs):
        seen.update(kwargs)
        return [], {"n_papers": 0, "n_edges_above_threshold": 0,
                    "n_clusters_found": 0, "n_already_covered": 0,
                    "n_extend": 0, "n_new": 0, "n_judged": 0,
                    "judge_input_tokens": 0, "judge_output_tokens": 0}

    monkeypatch.setattr(cli, "find_candidates", fake_find)
    assert cli.main([]) == 0
    assert seen["judge"] is False
    assert "No actionable synthesis candidates" in capsys.readouterr().out


def test_synthesis_cli_returns_two_when_semantic_index_is_missing(monkeypatch, capsys):
    from researchwiki.tasks import _synthesis_candidates as cli

    monkeypatch.setattr(
        cli,
        "find_candidates",
        lambda **_kwargs: ([], {"error": "semantic index is unavailable"}),
    )

    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: semantic index is unavailable\n"
