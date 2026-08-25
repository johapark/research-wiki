"""Typed model-config failures must survive graceful judge fallbacks.

The judge layers intentionally turn ordinary provider/runtime failures into a
safe no-verdict result. `ModelConfigUnavailable`, however, is an
`EnvironmentFailure`: swallowing it makes a deliberately selected broken
models file look like a successful no-op. These tests keep both contracts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki import __main__ as cli
from researchwiki.agents.model_config import ModelConfigUnavailable
from researchwiki.synthesis_candidates.detect import Candidate
from researchwiki.wiki import Page


def _raise_config_failure(**_kwargs):
    raise ModelConfigUnavailable("selected models config is unavailable")


def _concept_candidate() -> dict:
    return {
        "term": "protein dynamics",
        "slug": "protein-dynamics",
        "pages": 4,
        "categories": 2,
        "label": "concept-ready (bridge)",
    }


def test_shared_llm_judge_propagates_model_config_failure(monkeypatch):
    from researchwiki.agents import judge, llm

    monkeypatch.setattr(llm, "call", _raise_config_failure)
    with pytest.raises(ModelConfigUnavailable):
        judge.run_llm_judge(phase="critic", system="s", prompt="p")


def test_concept_chunk_judge_propagates_model_config_failure(
    monkeypatch, tmp_path,
):
    from researchwiki import paths
    from researchwiki.agents import llm
    from researchwiki.concepts import triage

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / triage.TRIAGE_SYSTEM_FILENAME).write_text("judge concepts")
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.setattr(llm, "call", _raise_config_failure)

    with pytest.raises(ModelConfigUnavailable):
        triage._judge_chunk([_concept_candidate()], use_stub=False)


def test_concept_triage_public_api_propagates_model_config_failure():
    from researchwiki.concepts import triage

    def judge_fn(_chunk):
        raise ModelConfigUnavailable("selected models config is unavailable")

    with pytest.raises(ModelConfigUnavailable):
        triage.triage_candidates([_concept_candidate()], judge_fn=judge_fn)


def test_synthesis_candidate_judge_propagates_model_config_failure(monkeypatch):
    from researchwiki.agents import llm
    from researchwiki.synthesis_candidates import judge

    candidate = Candidate(
        members=["cat/paper"],
        titles={"cat/paper": "Paper"},
        density=1.0,
        edges=[],
        edge_signal_counts={},
        common_keywords=["protein dynamics"],
        nearest_synthesis=None,
        nearest_synthesis_overlap=0.0,
        verdict="new",
        members_missing_from_nearest=[],
    )
    page = Page(
        path=Path("wiki/cat/paper.md"),
        stem="paper",
        category="cat",
        fm={"title": "Paper", "keywords": ["protein dynamics"]},
        body="## Summary\nA paper about protein dynamics.\n",
    )
    monkeypatch.setattr(llm, "call", _raise_config_failure)

    with pytest.raises(ModelConfigUnavailable):
        judge.judge_candidate(candidate, syntheses=[], paper_pages=[page])
    assert candidate.judged is False


def test_synthesis_topic_judge_propagates_model_config_failure(monkeypatch):
    """The whole-cluster topic call must obey the same fail-closed contract."""
    from researchwiki.agents import llm
    from researchwiki.synthesis_candidates import judge

    members = [f"cat/paper-{i:02d}" for i in range(17)]
    candidate = Candidate(
        members=members,
        titles={key: f"Paper {i}" for i, key in enumerate(members)},
        density=1.0,
        edges=[],
        edge_signal_counts={},
        common_keywords=["protein dynamics"],
        nearest_synthesis=None,
        nearest_synthesis_overlap=0.0,
        verdict="new",
        members_missing_from_nearest=[],
    )
    pages = [
        Page(
            path=Path(f"wiki/{key}.md"),
            stem=key.rsplit("/", 1)[-1],
            category="cat",
            fm={"title": f"Paper {i}", "keywords": ["protein dynamics"]},
            body="## Summary\nA paper about protein dynamics.\n",
        )
        for i, key in enumerate(members)
    ]
    monkeypatch.setattr(llm, "call", _raise_config_failure)

    with pytest.raises(ModelConfigUnavailable):
        judge.judge_candidate(candidate, syntheses=[], paper_pages=pages)
    assert candidate.judged is False
    assert candidate.judge_batches == 1


def test_cli_maps_concept_triage_model_config_failure_to_exit_2(
    monkeypatch, tmp_path, capsys,
):
    import researchwiki.concepts as concepts
    from researchwiki.agents import llm

    (tmp_path / "wiki").mkdir()
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "concept-triage-system.md").write_text("judge concepts")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_discover_tasks", lambda: {"candidates": "candidates"})
    monkeypatch.setattr(
        concepts, "collect_candidates", lambda **_kwargs: [_concept_candidate()],
    )
    monkeypatch.setattr(llm, "call", _raise_config_failure)

    assert cli.main(["candidates", "concepts", "--triage", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "selected models config is unavailable" in err
    assert "Traceback" not in err
