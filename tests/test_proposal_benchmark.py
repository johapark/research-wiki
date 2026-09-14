"""Guard metric denominators, evaluator separation, and source-group splits."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from researchwiki.benchmark.proposals import (
    AXES, _hash, check_pack, load_suite, main, prepare, score_reviews, validate_suite,
)

FIXTURES = Path(__file__).resolve().parents[1] / "benchmark-fixtures/proposals"


def proposal(decision="shortlist", scores=None, **overrides):
    return {"scores": scores or dict.fromkeys(AXES, 2),
            "reasons": dict.fromkeys(AXES, "Checked the proposed mechanism and its cited evidence."),
            "critical_factual_failure": False, "constraint_failure": False,
            "decision": decision, "decision_reason": "A bounded next test is worthwhile.", **overrides}


def block(case_id="P", proposals=None, ok=True, **extras):
    return {"case_id": case_id, "operational_success": ok,
            "proposals": [] if proposals is None else proposals, **extras}


def expectations(**labels):
    return {"cases": {key: {"expectation": label} for key, label in labels.items()}}


def test_curated_suite_has_expected_counts_and_disjoint_sources():
    inputs, expected = load_suite(FIXTURES)
    assert len(inputs["cases"]) == 12
    assert sum(c["split"] == "development" for c in inputs["cases"]) == 8
    assert sum(c["split"] == "heldout" for c in inputs["cases"]) == 4
    assert sum(c["expectation"] == "opportunity" for c in expected["cases"].values()) == 9
    assert {c["family"] for c in inputs["cases"]} == {"synthesis", "idea", "transfer", "restraint", "feedback"}
    assert "shortlist_conditions" not in json.dumps(inputs)
    assert "acceptable_approaches" not in json.dumps(inputs)
    assert expected["review_status"] == "agent_calibrated_not_human_validated"
    assert expected["rubric_version"] == 2
    by_id = {c["id"]: c for c in inputs["cases"]}
    assert by_id["X01"]["topic"] == by_id["X02"]["topic"]
    assert len(expected["cases"]["F01"]["feedback_requirements"]) == 5


def test_source_leakage_and_missing_evidence_fail_before_preparation():
    inputs, expected = load_suite(FIXTURES)
    broken = deepcopy(inputs)
    broken["source_groups"]["chromatin"]["change"] = inputs["source_groups"]["agent-evidence"]["amem"]
    with pytest.raises(ValueError, match="leakage"):
        validate_suite(broken, expected)
    broken_expected = deepcopy(expected)
    broken_expected["cases"]["S01"]["essential_refs"].append("hipporag#missing")
    with pytest.raises(ValueError, match="essential refs absent"):
        validate_suite(inputs, broken_expected)


def test_all_abstain_cannot_win_positive_cases():
    result = score_reviews([block("P"), block("N")], expectations(P="opportunity", N="abstain"), {"P", "N"})
    assert result["useful_proposal_yield"] == 0
    assert result["unnecessary_abstention"] == 1
    assert result["appropriate_abstention"] == 1
    assert result["proposal_precision"] is None


def test_operational_failure_never_counts_as_successful_abstention_or_yield():
    result = score_reviews([block("N", ok=False), block("P", [proposal()], ok=False)],
                           expectations(P="opportunity", N="abstain"), {"P", "N"})
    assert result["appropriate_abstention"] == 0
    assert result["operational_success"] == 0
    assert result["useful_proposal_yield"] == 0
    assert result["proposal_precision"] == 0
    assert result["proposal_scores"][0]["total"] == 10  # raw quality remains diagnostic


def test_extra_weak_proposals_lower_precision_without_increasing_yield():
    result = score_reviews([block(proposals=[proposal(), proposal("reject"), proposal("defer")])],
                           expectations(P="opportunity"), {"P"})
    assert result["useful_proposal_yield"] == 1
    assert result["proposal_precision"] == pytest.approx(1 / 3)


def test_human_utility_is_distinct_from_uncalibrated_threshold():
    scores = dict(zip(AXES, [2, 2, 1, 1, 1]))
    result = score_reviews([block(proposals=[proposal(scores=scores)])], expectations(P="opportunity"), {"P"})
    assert result["useful_proposal_yield"] == 1
    assert result["proposal_scores"][0]["threshold_shortlist"] is False
    assert result["proposal_scores"][0]["threshold_disagrees"] is True


def test_high_total_does_not_override_a_deferred_editorial_decision():
    p = proposal("defer", scores=dict(zip(AXES, [2, 2, 1, 2, 1])))
    result = score_reviews([block(proposals=[p])], expectations(P="opportunity"), {"P"})
    assert result["proposal_scores"][0]["total"] == 8
    assert result["proposal_scores"][0]["threshold_shortlist"] is True
    assert result["proposal_scores"][0]["threshold_disagrees"] is True
    assert result["useful_proposal_yield"] == 0


def test_other_axes_cannot_compensate_for_a_critical_factual_failure():
    p = proposal("reject", scores=dict(zip(AXES, [0, 2, 2, 2, 2])), critical_factual_failure=True)
    result = score_reviews([block(proposals=[p])], expectations(P="opportunity"), {"P"})
    assert result["proposal_scores"][0]["total"] == 8
    assert result["proposal_scores"][0]["threshold_shortlist"] is False
    assert result["critical_factual_failure_rate"] == 1
    assert result["useful_proposal_yield"] == 0


def test_incomplete_or_contradictory_reviews_require_adjudication():
    expected = expectations(P="opportunity", N="abstain")
    with pytest.raises(ValueError, match="exactly once"):
        score_reviews([block("P")], expected, {"P", "N"})
    with pytest.raises(ValueError, match="adjudicate"):
        score_reviews([block("P", [proposal(critical_factual_failure=True)]), block("N")], expected, {"P", "N"})
    with pytest.raises(ValueError, match="abstention label"):
        score_reviews([block("P"), block("N", [proposal()])], expected, {"P", "N"})


def test_feedback_needs_substantive_checks_and_parent():
    expected = expectations(F="opportunity")
    expected["cases"]["F"].update(feedback_requirements=["Reduce scope"], expected_parent="prop-parent")
    review = block("F", [proposal(parent_proposal=None)], feedback_checks=[{"passed": True, "reason": "Scope narrowed."}])
    assert score_reviews([review], expected, {"F"})["feedback_compliance"] == 0
    review["proposals"][0]["parent_proposal"] = "prop-parent"
    assert score_reviews([review], expected, {"F"})["feedback_compliance"] == 1
    review["feedback_checks"][0]["passed"] = False
    assert score_reviews([review], expected, {"F"})["feedback_compliance"] == 0


@pytest.mark.parametrize("invalid", [None, True, 3, -1, "2"])
def test_missing_or_invalid_scores_are_not_silently_zero(invalid):
    p = proposal()
    p["scores"]["evidence"] = invalid
    with pytest.raises(ValueError, match="integers"):
        score_reviews([block(proposals=[p])], expectations(P="opportunity"), {"P"})


def test_pack_integrity_detects_file_changes(tmp_path):
    packet = {"evidence": []}
    content = json.dumps(packet).encode()
    (tmp_path / "packet.json").write_bytes(content)
    manifest = {"cases": [{"input_path": "packet.json"}], "files": {"packet.json": _hash(content)}}
    manifest["pack_id"] = _hash(json.dumps(manifest, sort_keys=True).encode())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert check_pack(tmp_path)["pack_id"] == manifest["pack_id"]
    (tmp_path / "packet.json").write_text("{}")
    with pytest.raises(ValueError, match="file hash mismatch"):
        check_pack(tmp_path)


def test_score_cli_preserves_execution_mode_and_rejects_mixed_mode(tmp_path, monkeypatch, capsys):
    manifest = {"pack_id": "unit", "cases": [{"id": "P", "split": "development"}]}
    monkeypatch.setattr("researchwiki.benchmark.proposals.check_pack", lambda path: manifest)
    (tmp_path / "evaluator").mkdir()
    (tmp_path / "evaluator/expectations.yaml").write_text(yaml.safe_dump(expectations(P="opportunity")))
    path = tmp_path / "reviews.json"
    reviews = {"pack_id": "unit", "split": "development", "candidate": "blind-A",
               "reviewer_kind": "agent", "reviewer": "test-reviewer",
               "repeat": 1, "execution_mode": "fixed-packet", "reviews": [block("P")]}
    path.write_text(json.dumps(reviews))
    monkeypatch.setattr("sys.argv", ["proposals", "score", "--pack", str(tmp_path), "--reviews", str(path)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["execution_mode"] == "fixed-packet"
    assert result["useful_proposal_yield"] == 0
    assert result["reviewer_kind"] == "agent"
    assert result["reviewer"] == "test-reviewer"
    reviews["execution_mode"] = "mixed"
    path.write_text(json.dumps(reviews))
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "execution_mode" in capsys.readouterr().err
    reviews["execution_mode"] = "fixed-packet"
    for kind in (None, "mixed"):
        reviews["reviewer_kind"] = kind
        path.write_text(json.dumps(reviews))
        with pytest.raises(SystemExit):
            main()
        assert "reviewer_kind" in capsys.readouterr().err


@pytest.mark.parametrize("claim_state", ["current", "missing", "stale"])
def test_prepare_freezes_evidence_and_refuses_missing_or_stale_claims(tmp_path, monkeypatch, claim_state):
    from researchwiki.wiki import Page

    inputs = {"version": 1, "suite_id": "unit", "max_proposals": 3,
              "source_groups": {"group": {"source": "test-paper"}},
              "evidence_sets": {"selected": {"source": ["kc-example"]}},
              "histories": {}, "cases": [{"id": "P", "split": "development", "group": "group",
                  "family": "idea", "topic": "A bounded test?", "target_category": "test",
                  "evidence_set": "selected"}]}
    expected = {"version": 1, "suite_id": "unit", "rubric_version": 1,
                "review_status": "pending", "cases": {"P": {"expectation": "opportunity"}}}
    spec = tmp_path / "spec"
    spec.mkdir()
    (spec / "inputs.yaml").write_text(yaml.safe_dump(inputs))
    (spec / "expectations.yaml").write_text(yaml.safe_dump(expected))
    (spec / "SCORING.md").write_text("Frozen rubric.")
    (spec / "CALIBRATION.md").write_text("Agent-only calibration.")
    path = tmp_path / "test-paper.md"
    body = "## Key Contributions\n- A test method preserves the selected input constraints.\n"
    path.write_text(body)
    page = Page(path, "test-paper", "test", {"type": "paper"}, body)
    hit = {"paper_stem": page.stem, "claim_slug": "kc-example", "section": "key_contributions",
           "text": "A test method preserves the selected input constraints."}
    if claim_state == "stale":
        hit["text"] = "An old claim absent from the current wiki."
    monkeypatch.setattr("researchwiki.wiki.read_pages", lambda: [page])
    monkeypatch.setattr("researchwiki.search.claims_by_stem",
                        lambda *args, **kwargs: [] if claim_state == "missing" else [hit])
    # Bundled PDF exercises real extraction without accessing a personal corpus.
    pdf = FIXTURES.parent / "pdfs/chuai-2018-deepcrispr-optimized-crispr-guide-rna.pdf"
    monkeypatch.setattr("researchwiki.paths.resolve_pdf", lambda stem: pdf)
    out = tmp_path / "pack"
    if claim_state != "current":
        with pytest.raises(ValueError, match="claim missing|stale DB claim"):
            prepare(out, spec)
        assert not out.exists()
        return
    manifest = prepare(out, spec)
    assert check_pack(out) == manifest
    assert manifest["rubric_version"] == 1
    assert (out / "evaluator/SCORING.md").read_text() == "Frozen rubric."
    assert (out / "evaluator/CALIBRATION.md").read_text() == "Agent-only calibration."
    packet = json.loads((out / "inputs/P.json").read_text())
    assert packet["evidence"][0]["text"] == hit["text"]
    assert packet["evidence"][0]["citation"] == "[[test-paper#kc-example]]"
    assert "expectation" not in packet
    assert "PDF page 1" in (out / "sources/test-paper.txt").read_text()
    assert manifest["sources"][page.stem]["pdf_sha256"] == _hash(pdf.read_bytes())
    with pytest.raises(ValueError, match="already exists"):
        prepare(out, spec)
