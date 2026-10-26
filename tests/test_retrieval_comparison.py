"""Suite comparisons skip absent anchors and refuse corpus drift."""

import pytest

from researchwiki.benchmark import retrieval_comparison as comparison
from researchwiki.benchmark.fixture import ExpectedPage, RetrievalFixture
from researchwiki.benchmark.retrieval import RetrievedPage, score_pages_fixture
from researchwiki.benchmark.retrieval_preflight import FixtureAvailability
from researchwiki.errors import EnvironmentFailure
from researchwiki.tasks import benchmark_retrieval


@pytest.fixture
def suite(monkeypatch, tmp_path):
    path = tmp_path / "fixture.yaml"
    path.write_text("fixture_type: pages\n", encoding="utf-8")
    fixture = RetrievalFixture("available", "pages", "query", 5,
                               expected_claims=[], expected_pages=[ExpectedPage("a", "critical")],
                               must_not_appear=[], source_path=path)
    missing = RetrievalFixture("missing", "pages", "query", 5,
                               expected_claims=[], expected_pages=[ExpectedPage("absent", "critical")],
                               must_not_appear=[], source_path=path)
    monkeypatch.setattr(comparison, "find_fixtures", lambda: ["retrieval/yes", "retrieval/no", "proposals/other"])
    monkeypatch.setattr(comparison, "load_fixture", lambda name: fixture if name.endswith("yes") else missing)
    monkeypatch.setattr(comparison, "validate_fixture_anchors", lambda f:
                        FixtureAvailability([] if f.fixture_id == "available" else ["absent"], []))
    monkeypatch.setattr(comparison.encoders, "corpus_fingerprint", lambda: {"sha256": "same", "pages": 1, "claims": 1})
    monkeypatch.setattr(comparison.encoders, "model_details", lambda model, **kw: {"model": model})
    monkeypatch.setattr(comparison, "_performance", lambda *a, **kw:
                        {"query_samples": 3, "query_encode_median_ms": 1, "query_encode_p95_ms": 2})
    calls = []

    def score(f, model, backend, **options):
        calls.append((f.fixture_id, model, backend))
        return score_pages_fixture(f, [RetrievedPage("a", 1)], model, backend)

    monkeypatch.setattr(comparison, "score_retrieval", score)
    return calls


def test_suite_skips_missing_anchors_and_preserves_per_fixture_diffs(suite):
    report = comparison.compare_retrieval("bge-small", "harrier-270m")
    assert report["fixtures_scored"] == 1
    assert report["fixtures_skipped"][0]["missing_expected"] == ["absent"]
    assert len(suite) == 6
    assert len(report["comparisons"]) == 3
    assert all(item["diff"]["delta_ndcg"] == 0 for item in report["comparisons"])
    assert report["candidate"]["summary"]["hybrid"]["mean_recall_critical"] == 1
    assert "1 unavailable" in comparison.format_comparison(report)


def test_corpus_drift_invalidates_the_entire_comparison(suite, monkeypatch):
    snapshots = iter([{"sha256": "before"}, {"sha256": "after"}])
    monkeypatch.setattr(comparison.encoders, "corpus_fingerprint", lambda: next(snapshots))
    with pytest.raises(EnvironmentFailure, match="corpus changed"):
        comparison.compare_retrieval("bge-small", "harrier-270m")


def test_all_missing_fixtures_are_an_environment_error(suite, monkeypatch):
    monkeypatch.setattr(comparison, "validate_fixture_anchors", lambda f: FixtureAvailability(["absent"], []))
    with pytest.raises(EnvironmentFailure, match="no retrieval fixtures"):
        comparison.compare_retrieval("bge-small", "harrier-270m")


def test_cli_writes_a_reviewable_json_report(suite, tmp_path, capsys):
    out = tmp_path / "report.json"
    assert benchmark_retrieval.main(["--out", str(out), "--json"]) == 0
    import json
    report = json.loads(capsys.readouterr().out)
    assert json.loads(out.read_text()) == report
    assert report["schema_version"] == 1


def test_invalid_batch_size_returns_user_input_error(suite):
    assert benchmark_retrieval.main(["--batch-size", "0"]) == 1
