"""Regression checks for the default, model-free post-ingest review queue."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from researchwiki import impact_review as impact
from researchwiki.index import pages_semantic
from researchwiki.wiki import read_page


SOURCE = "single-cell/pullin-2024-a-comparison-of-marker-gene"
TARGET = "synthesis/cell-typing-benchmark-comparability"


def _fixture(tmp_path: Path, monkeypatch, *, index_ok=True):
    wiki = tmp_path / "wiki"
    source_path = wiki / f"{SOURCE}.md"
    target_path = wiki / f"{TARGET}.md"
    source_path.parent.mkdir(parents=True)
    target_path.parent.mkdir(parents=True)
    source_path.write_text(
        "---\n"
        "title: A comparison of marker gene selection methods\n"
        "type: paper\n"
        "keywords: [marker gene selection, Wilcoxon rank-sum]\n"
        "---\n"
        "## Summary\nMarker-gene selection for cell-type annotation.\n"
        "## Key Contributions\nCompared many marker-selection methods.\n",
        encoding="utf-8",
    )
    target_path.write_text(
        "---\n"
        "title: What makes cell-typing benchmarks comparable?\n"
        "type: synthesis\n"
        "topic_seed: scRNA-seq cell type annotation benchmark open-set rejection\n"
        "---\n"
        "## Question\nWhat makes cell-typing benchmarks comparable?\n"
        "## Short answer\nCompare marker selection and input preparation.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(impact, "wiki_dir", lambda: wiki)
    monkeypatch.setattr(impact, "ingest_dir", lambda: tmp_path / ".ingest")
    monkeypatch.setattr(impact, "read_pages", lambda: [read_page(source_path), read_page(target_path)])
    monkeypatch.setattr(pages_semantic, "index_exists", lambda: index_ok)
    monkeypatch.setattr(pages_semantic, "page_index_text", lambda page: page.body)
    monkeypatch.setattr(impact, "_passage_scores", lambda *args: {})
    monkeypatch.setattr(
        pages_semantic, "query_text",
        lambda *args, **kwargs: [SimpleNamespace(key=TARGET, score=0.8537)]
        if index_ok else [],
    )
    return source_path, target_path


def test_pullin_pre_edit_candidate_and_decision_reopen(tmp_path, monkeypatch):
    _, target_path = _fixture(tmp_path, monkeypatch)
    first = impact.scan(SOURCE)
    assert first["state"] == "complete"
    assert [row["target"] for row in first["candidates"]] == [TARGET]
    assert len(first["audit"]) == 1
    assert len(impact.pending_reviews()) == 1

    path = impact.review_path(SOURCE.split("/")[-1])
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    saved["candidates"][0]["decision"] = "not_relevant"
    saved["candidates"][0]["reason"] = "Reviewed source claims and page scope."
    path.write_text(yaml.safe_dump(saved), encoding="utf-8")
    assert impact.pending_reviews() == []
    assert impact.scan(SOURCE)["candidates"][0]["decision"] == "not_relevant"

    target_path.write_text(target_path.read_text(encoding="utf-8") +
                           "\n## New analysis\nA substantive change.\n", encoding="utf-8")
    assert impact.pending_reviews()[0][1]["state"] == "stale"
    rescanned = impact.scan(SOURCE)
    assert rescanned["candidates"][0]["decision"] == "pending"
    assert rescanned["candidates"][0]["reason"] == ""


def test_missing_index_is_unscanned_not_clean(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, index_ok=False)
    result = impact.scan(SOURCE)
    assert result["state"] == "unscanned"
    assert result["candidates"] == []
    assert impact.pending_reviews()[0][1]["state"] == "unscanned"


def test_failed_rescan_preserves_prior_decisions(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    impact.scan(SOURCE)
    path = impact.review_path(SOURCE.split("/")[-1])
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    saved["candidates"][0].update(
        decision="not_relevant", reason="Reviewed against the page question.")
    path.write_text(yaml.safe_dump(saved), encoding="utf-8")

    monkeypatch.setattr(pages_semantic, "index_exists", lambda: False)
    failed = impact.scan(SOURCE)
    assert failed["state"] == "unscanned"
    assert failed["candidates"][0]["decision"] == "not_relevant"
    assert impact.pending_reviews()[0][1]["state"] == "unscanned"


def test_change_to_non_candidate_target_marks_receipt_stale(tmp_path, monkeypatch):
    _, target_path = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pages_semantic, "query_text",
        lambda *args, **kwargs: [SimpleNamespace(key=TARGET, score=0.2)],
    )
    first = impact.scan(SOURCE)
    assert first["candidates"] == []
    assert impact.pending_reviews() == []

    target_path.write_text(target_path.read_text(encoding="utf-8") +
                           "\n## New evidence\nThe comparison now includes marker genes.\n",
                           encoding="utf-8")
    assert impact.pending_reviews()[0][1]["state"] == "stale"


def test_moderate_match_with_shared_terms_in_same_category_is_candidate(
    tmp_path, monkeypatch,
):
    source_path, target_path = _fixture(tmp_path, monkeypatch)
    source_path.write_text(source_path.read_text(encoding="utf-8").replace(
        "Wilcoxon rank-sum]", "Wilcoxon rank-sum, orthologous cluster groups]"
    ), encoding="utf-8")
    target_path.write_text(target_path.read_text(encoding="utf-8").replace(
        "type: synthesis\n", "type: synthesis\ncategory: [single-cell]\n"
    ) + "\nMarker selection using orthologous cluster groups and Wilcoxon ranks.\n",
                           encoding="utf-8")
    monkeypatch.setattr(impact, "_pairs", lambda _text: set())
    monkeypatch.setattr(pages_semantic, "query_text", lambda *args, **kwargs: [
        SimpleNamespace(key=TARGET, score=0.8059)
    ])
    result = impact.scan(SOURCE)
    assert len(result["audit"][0]["shared_terms"]) >= impact.MIN_SHARED_TERMS
    assert [row["target"] for row in result["candidates"]] == [TARGET]


def test_cited_target_is_audited_without_pending_candidate(tmp_path, monkeypatch):
    _, target_path = _fixture(tmp_path, monkeypatch)
    target_path.write_text(target_path.read_text(encoding="utf-8") +
                           f"\n[^pullin]: [[{SOURCE}]]\n", encoding="utf-8")
    result = impact.scan(SOURCE)
    assert result["candidates"] == []
    assert result["audit"][0]["already_cited"] is True


def test_cited_target_reopens_after_source_change(tmp_path, monkeypatch):
    source_path, target_path = _fixture(tmp_path, monkeypatch)
    first = impact.scan(SOURCE)
    assert first["candidates"][0]["decision"] == "pending"
    target_path.write_text(target_path.read_text(encoding="utf-8") +
                           f"\n[^pullin]: [[{SOURCE}]]\n", encoding="utf-8")
    source_path.write_text(source_path.read_text(encoding="utf-8") +
                           "\n## Results\nNew substantive result.\n", encoding="utf-8")
    second = impact.scan(SOURCE)
    assert second["candidates"][0]["target"] == TARGET
    assert second["candidates"][0]["decision"] == "pending"
