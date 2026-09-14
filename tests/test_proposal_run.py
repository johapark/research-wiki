"""Offline runner tests: no provider, retrieval, personal corpus, or credentials."""
from copy import deepcopy
import json

import pytest

from researchwiki.agents.llm import LLMResponse
from researchwiki.benchmark import proposal_run as runner
from researchwiki.benchmark.proposals import _hash
from researchwiki import proposal_generation as generation


@pytest.fixture
def setup(tmp_path, monkeypatch):
    packet = {"topic": "Compare methods", "target_category": "ai", "cross_category": False,
              "search_plan": None, "prior_proposals": [], "existing_pages": [],
              "evidence": [{"id": "e01", "category": "ai", "paper_stem": "paper-one",
                  "section": "key_contributions", "claim_slug": "kc-one",
                  "text": "A source-grounded test claim.", "citation": "[[paper-one#kc-one]]"}]}
    pack = tmp_path / "pack"
    pack.mkdir()
    cases, files = [], {}
    for cid, split in (("S01", "development"), ("F01", "development"), ("S02", "heldout")):
        content = deepcopy(packet)
        if split == "heldout":
            content["topic"] = "HOLDOUT_SECRET"
        (pack / f"{cid}.json").write_text(json.dumps(content))
        files[f"{cid}.json"] = _hash((pack / f"{cid}.json").read_bytes())
        cases.append({"id": cid, "split": split, "input_path": f"{cid}.json"})
    (pack / "expectations.yaml").write_text("EVALUATOR_SECRET")
    files["expectations.yaml"] = _hash((pack / "expectations.yaml").read_bytes())
    manifest = {"suite_id": "unit", "max_proposals": 3, "rubric_version": 2,
                "cases": cases, "files": files}
    manifest["pack_id"] = runner._identity(manifest, "pack_id")
    (pack / "manifest.json").write_text(json.dumps(manifest))
    route = {"provider": "openai-compatible", "model": "test-model", "temperature": 0.6,
             "max_tokens": 4000, "reasoning_effort": None, "rpm": None,
             "endpoint": "https://example.invalid/v1", "config_path": "test.yaml", "config_sha256": "test"}
    monkeypatch.setattr(runner, "current_route", lambda: dict(route))
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected call or live retrieval")
    monkeypatch.setattr(runner.llm, "call", forbidden)
    monkeypatch.setattr(generation, "build_evidence_packet", forbidden)
    monkeypatch.setattr(generation, "expand_cross_category", forbidden)
    monkeypatch.setattr(generation, "read_pages", forbidden)
    return pack, tmp_path / "plan", packet, route


def response(text='{"proposals": []}'):
    return LLMResponse(text=text, model="test-model", temperature=0.6, input_tokens=100, output_tokens=20)


def test_plan_is_offline_disjoint_and_equal_except_for_system_policy(setup):
    pack, out, packet, _ = setup
    plan = runner.plan_comparison(pack, out)
    assert len(plan["entries"]) == 4
    assert runner.check_plan(out) == plan
    requests = {}
    for entry in plan["entries"]:
        data = (out / entry["request_path"]).read_text()
        assert "HOLDOUT_SECRET" not in data and "EVALUATOR_SECRET" not in data
        request = json.loads(data)
        assert request["max_tokens"] == 4000
        assert request["model"] == "test-model"
        if entry["case_id"] == "S01":
            requests[plan["candidate_identity"][entry["candidate"]]] = request
    production = generation.proposal_request(packet)
    assert requests["production"]["system"] == production["system"]
    assert requests["production"]["prompt"] == production["prompt"]
    for key in requests["production"]:
        if key != "system":
            assert requests["baseline"][key] == requests["production"][key]
    assert requests["baseline"]["schema"] == production["schema"]
    with pytest.raises(ValueError, match="already exists"):
        runner.plan_comparison(pack, out)


def test_approval_route_and_file_integrity_block_calls(setup):
    pack, out, _, route = setup
    plan = runner.plan_comparison(pack, out)
    with pytest.raises(ValueError, match="approve-plan"):
        runner.run_comparison(out, "wrong")
    route["endpoint"] = "https://changed.invalid/v1"
    with pytest.raises(ValueError, match="provider/config changed"):
        runner.run_comparison(out, plan["plan_id"])
    assert not (out / "run").exists()
    route["endpoint"] = plan["route"]["endpoint"]
    request = out / plan["entries"][0]["request_path"]
    request.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run_comparison(out, plan["plan_id"])
    assert not (out / "run").exists()


def test_run_preserves_failures_usage_and_identity_hidden_review(setup, monkeypatch):
    pack, out, _, _ = setup
    plan = runner.plan_comparison(pack, out)
    replies = iter([response(), response("not JSON"), RuntimeError("provider failed"), response()])
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        result = next(replies)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(runner.llm, "call", call)
    summary = runner.run_comparison(out, plan["plan_id"])
    assert len(calls) == 4  # No generation retries or hidden extra calls.
    assert summary["operational_success"] == 2
    assert summary["validation_failures"] == 1
    assert summary["provider_failures"] == 1
    assert summary["reported_input_tokens"] == 300
    assert summary["all_final_responses_have_usage"] is False
    assert sum(c["attempted"] for c in summary["candidates"].values()) == 4
    bad = json.loads((out / "run" / f"{plan['entries'][1]['request_id']}.json").read_text())
    assert bad["raw_response"] == "not JSON"
    assert bad["usage"]["output_tokens"] == 20
    assert bad["proposals"] is None  # Invalid output is not an empty abstention.
    for alias in ("A", "B"):
        review = json.loads((out / f"run/review/reviews-{alias}.json").read_text())
        assert len(review["reviews"]) == 2
        assert review["reviewer_kind"] is None
        outputs = (out / f"run/review/outputs-{alias}.json").read_text()
        assert "test-model" not in outputs and "candidate_identity" not in outputs
        assert "system" not in outputs and "input_tokens" not in outputs
    with pytest.raises(ValueError, match="never replayed"):
        runner.run_comparison(out, plan["plan_id"])


def test_raw_response_is_saved_before_validation_and_bugs_propagate(setup, monkeypatch):
    pack, out, _, _ = setup
    plan = runner.plan_comparison(pack, out)
    monkeypatch.setattr(runner.llm, "call", lambda **kwargs: response())
    def broken_parser(text, packet):
        path = out / "run" / f"{plan['entries'][0]['request_id']}.json"
        saved = json.loads(path.read_text())
        assert saved["raw_response"] == text and saved["usage"]["input_tokens"] == 100
        raise KeyError("internal parser defect")
    monkeypatch.setattr(generation, "parse_proposal_response", broken_parser)
    with pytest.raises(KeyError, match="internal parser defect"):
        runner.run_comparison(out, plan["plan_id"])
    assert not (out / "run/summary.json").exists()


def test_midrun_changes_stop_before_next_dispatch(setup, monkeypatch):
    pack, out, _, _ = setup
    plan = runner.plan_comparison(pack, out)
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        (out / plan["entries"][1]["request_path"]).write_text("tampered")
        return response()
    monkeypatch.setattr(runner.llm, "call", call)
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run_comparison(out, plan["plan_id"])
    assert len(calls) == 1


def test_ordinary_generation_uses_shared_request_and_parser(setup, monkeypatch):
    _, _, packet, _ = setup
    seen = []
    monkeypatch.setattr(runner.llm, "call", lambda **kw: seen.append(kw) or response())
    result, usage = generation.generate_proposals(packet)
    assert result == [] and usage["input_tokens"] == 100
    assert seen == [generation.proposal_request(packet)]


def test_path_outside_hashed_pack_is_rejected_before_planning(setup):
    pack, out, _, _ = setup
    manifest = json.loads((pack / "manifest.json").read_text())
    manifest["cases"][0]["input_path"] = "../private.json"
    manifest["pack_id"] = runner._identity(manifest, "pack_id")
    (pack / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unhashed pack file"):
        runner.plan_comparison(pack, out)
