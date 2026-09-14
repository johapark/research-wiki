"""Markdown proposal ledger, derived DB rows, and bounded generation."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from researchwiki.db.connection import init_schema
from researchwiki.db.rebuild import _upsert_one
from researchwiki.proposals import (
    append_feedback,
    create_proposal,
    parse_proposal,
)
from researchwiki.tasks.lint.proposal_contract import (
    find_proposal_contract_violations,
)
from researchwiki.wiki import read_page


def _proposal() -> dict:
    return {
        "title": "Transfer selective retrieval across domains",
        "page_type": "idea",
        "direction": "cross-category-application",
        "question": "Can selective retrieval reduce unnecessary evidence reads?",
        "thesis": "A bounded router could reserve expensive reads for uncertain cases.",
        "why_it_matters": "It could reduce cost while preserving difficult cases.",
        "evidence_connections": [
            {"insight": "Both systems make evidence access conditional.",
             "evidence_ids": ["e01", "e02"]},
        ],
        "distinct_from_existing": "Existing pages compare retrieval accuracy, not transfer.",
        "decisive_uncertainty": "The router may miss evidence before uncertainty is visible.",
        "outline": ["Target bottleneck", "Transfer", "Discriminating experiment"],
        "source_method": "Selective retrieval",
        "target_problem": "Expensive source-PDF reading",
        "transfer_mapping": "Retrieval actions map to evidence-surface escalation.",
        "mechanism": "An uncertainty signal gates access to the expensive evidence tier.",
        "assumptions_to_test": "Uncertainty is observable before the expensive read.",
        "necessary_adaptations": "Route among claims, pages, and PDFs.",
        "baseline": "Always read the top PDF passages.",
        "first_experiment": "Compare grounded recall and reads per query.",
    }


def _evidence() -> list[dict]:
    return [
        {"id": "e01", "paper_stem": "a-2026-method", "claim_slug": "kc-aaaa",
         "text": "A method claim."},
        {"id": "e02", "paper_stem": "b-2026-target", "claim_slug": "lim-bbbb",
         "text": "A target limitation."},
    ]


def test_proposal_and_feedback_round_trip_through_markdown(tmp_path, monkeypatch):
    import researchwiki.proposals as proposals

    wiki = tmp_path / "wiki"
    monkeypatch.setattr(proposals, "wiki_dir", lambda: wiki)
    monkeypatch.setattr(proposals, "commit_page", lambda _path: None)

    record = create_proposal(
        _proposal(), evidence_items=_evidence(), topic_seed="selective retrieval",
        target_category="ai", author_model="test-model",
    )
    assert record.path.is_file()
    assert record.direction == "cross-category-application"
    assert "[[a-2026-method#kc-aaaa]]" in record.path.read_text()

    monkeypatch.setattr(
        proposals, "read_pages", lambda: [read_page(record.path)],
    )
    feedback = append_feedback(
        record.proposal_id, decision="deferred",
        reason="Needs a stronger target-domain baseline.", actor="user",
        resulting_page="[[ideas/selective-retrieval]]",
    )
    reparsed = parse_proposal(read_page(record.path))
    assert reparsed is not None
    assert reparsed.status == "deferred"
    assert reparsed.resulting_page == "[[ideas/selective-retrieval]]"
    assert reparsed.feedback == [feedback]


def test_db_rows_are_rebuilt_entirely_from_proposal_markdown(tmp_path, monkeypatch):
    import researchwiki.proposals as proposals

    wiki = tmp_path / "wiki"
    monkeypatch.setattr(proposals, "wiki_dir", lambda: wiki)
    monkeypatch.setattr(proposals, "commit_page", lambda _path: None)
    record = create_proposal(
        _proposal(), evidence_items=_evidence(), topic_seed="selective retrieval",
        target_category="ai", author_model="test-model",
    )
    monkeypatch.setattr(proposals, "read_pages", lambda: [read_page(record.path)])
    append_feedback(
        record.proposal_id, decision="shortlisted", reason="Worth drafting.",
    )

    conn = sqlite3.connect(tmp_path / "state.db")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    with conn:
        page, n_claims = _upsert_one(conn, record.path, 123)
    assert page is not None and n_claims == 0
    row = conn.execute("SELECT * FROM proposals").fetchone()
    assert row["proposal_id"] == record.proposal_id
    assert row["status"] == "shortlisted"
    assert row["question"].startswith("Can selective retrieval")
    fb = conn.execute("SELECT * FROM proposal_feedback").fetchone()
    assert fb["decision"] == "shortlisted"
    assert fb["reason"] == "Worth drafting."

    # If the canonical record becomes invalid, rebuilding must not preserve a
    # stale proposal row from its last valid revision.
    text = record.path.read_text()
    record.path.write_text(text.replace(f'proposal_id: "{record.proposal_id}"\n', ""))
    with conn:
        page, _ = _upsert_one(conn, record.path, 124)
    assert page is not None
    assert conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0
    conn.close()


def test_generation_rejects_invented_evidence_ids(monkeypatch):
    from researchwiki import proposal_generation as generation
    from researchwiki.agents import llm

    packet = {
        "topic": "retrieval",
        "target_category": "ai",
        "search_plan": None,
        "prior_proposals": [],
        "evidence": [{
            "id": "e01", "category": "ai", "paper_stem": "a",
            "section": "results", "text": "Supported text.",
            "supporting_text": "", "supporting_provenance": "",
        }],
    }
    proposal = _proposal()
    proposal["evidence_connections"] = [
        {"insight": "Invented source", "evidence_ids": ["e99"]},
    ]
    monkeypatch.setattr(
        llm, "call", lambda **_kwargs: SimpleNamespace(
            text=json.dumps({"proposals": [proposal]}), model="test-model",
            input_tokens=10, output_tokens=20,
        ),
    )
    try:
        generation.generate_proposals(packet)
    except ValueError as exc:
        assert "unknown evidence ids" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invented evidence id was accepted")


def test_generation_accepts_bounded_supported_proposal(monkeypatch):
    from researchwiki import proposal_generation as generation
    from researchwiki.agents import llm

    packet = {
        "topic": "retrieval",
        "target_category": "ai",
        "search_plan": None,
        "prior_proposals": [],
        "evidence": [
            {"id": "e01", "category": "ai", "paper_stem": "a",
             "section": "results", "text": "A supported result.",
             "supporting_text": "", "supporting_provenance": ""},
            {"id": "e02", "category": "systems", "paper_stem": "b",
             "section": "limitations", "text": "A supported limitation.",
             "supporting_text": "", "supporting_provenance": ""},
        ],
    }
    monkeypatch.setattr(
        llm, "call", lambda **_kwargs: SimpleNamespace(
            text=json.dumps({"proposals": [_proposal()]}), model="test-model",
            input_tokens=10, output_tokens=20,
        ),
    )
    proposals, usage = generation.generate_proposals(packet)
    assert proposals == [_proposal()]
    assert usage == {"model": "test-model", "input_tokens": 10, "output_tokens": 20}


def test_cross_category_packet_searches_outside_target(monkeypatch):
    from researchwiki import proposal_generation as generation
    from researchwiki.agents import llm

    target_hit = {"paper_stem": "target-paper", "claim_slug": "lim-11111111",
                  "section": "limitations", "text": "Target bottleneck."}
    source_hit = {"paper_stem": "source-paper", "claim_slug": "met-22222222",
                  "section": "methodology", "text": "Reusable method."}
    monkeypatch.setattr(generation, "_page_categories", lambda: {
        "target-paper": "biology", "source-paper": "systems",
    })
    monkeypatch.setattr(generation, "_claims_for_papers", lambda _stems: [])
    calls = []

    def fake_query(query, **_kwargs):
        calls.append(query)
        return [target_hit] if query == "target bottleneck" else [source_hit]

    monkeypatch.setattr(generation, "claim_query", fake_query)
    monkeypatch.setattr(generation, "_relevant_history", lambda _topic: [])
    monkeypatch.setattr(
        llm, "call", lambda **_kwargs: SimpleNamespace(
            text=json.dumps({"target_problem": "Target bottleneck",
                             "required_capabilities": ["selective routing"],
                             "queries": ["conditional evidence access"]}),
            model="planner", input_tokens=5, output_tokens=6,
        ),
    )
    packet = generation.build_evidence_packet(
        "target bottleneck", target_category="biology", cross_category=True,
    )
    assert calls == ["target bottleneck"]
    assert packet["planning_pending"]
    packet = generation.expand_cross_category(packet)
    assert calls == ["target bottleneck", "conditional evidence access"]
    assert [item["category"] for item in packet["evidence"]] == ["biology", "systems"]
    assert packet["search_plan"]["queries"] == ["conditional evidence access"]


def test_proposal_contract_accepts_renderer_and_flags_malformed_feedback(
    tmp_path, monkeypatch,
):
    import researchwiki.proposals as proposals

    wiki = tmp_path / "wiki"
    monkeypatch.setattr(proposals, "wiki_dir", lambda: wiki)
    monkeypatch.setattr(proposals, "commit_page", lambda _path: None)
    record = create_proposal(
        _proposal(), evidence_items=_evidence(), topic_seed="selective retrieval",
        target_category="ai", author_model="test-model",
    )
    monkeypatch.setattr(proposals, "read_pages", lambda: [read_page(record.path)])
    append_feedback(record.proposal_id, decision="shortlisted", reason="Draft this.")
    page = read_page(record.path)
    assert find_proposal_contract_violations(
        [record.path], {record.path: page.body}, {record.path: page.fm},
    ) == []

    broken = record.path.read_text().replace("- Actor: user", "- By: user")
    record.path.write_text(broken)
    page = read_page(record.path)
    findings = find_proposal_contract_violations(
        [record.path], {record.path: page.body}, {record.path: page.fm},
    )
    assert any(item["kind"] == "proposal_malformed_feedback" for item in findings)


def test_preview_accepts_exact_selected_content_without_calls(tmp_path, monkeypatch):
    import researchwiki.proposals as ledger
    from researchwiki.proposal_preview import save_preview, accept_preview
    from researchwiki.agents import llm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ledger, "commit_page", lambda _path: None)
    monkeypatch.setattr(llm, "call", lambda **kw: (_ for _ in ()).throw(AssertionError("model call")))
    evidence = _evidence()
    for item in evidence:
        item["citation"] = f"[[{item['paper_stem']}#{item['claim_slug']}]]"
    packet = {"topic": "routing", "target_category": "ai", "evidence": evidence}
    second = {**_proposal(), "title": "Second reviewed choice", "thesis": "Exact reviewed hypothesis."}
    path = save_preview(packet, [_proposal(), second], {"model": "test-author"})
    records = accept_preview(path, [2])
    assert len(records) == 1
    assert records[0].title == second["title"]
    assert records[0].thesis == second["thesis"]
    assert records[0].author_model == "test-author"
    assert evidence[0]["citation"] in records[0].path.read_text()
    append_feedback(records[0].proposal_id, decision="shortlisted", reason="Keep exactly this.")
    before = records[0].path.read_text()
    assert accept_preview(path, [2])[0].proposal_id == records[0].proposal_id
    assert records[0].path.read_text() == before
    assert len(list((tmp_path / "wiki/proposals").glob("*.md"))) == 1


def test_explicit_papers_fail_before_search(monkeypatch):
    import pytest
    from researchwiki import proposal_generation as generation
    monkeypatch.setattr(generation, "_page_categories", lambda: {"empty": "ai"})
    monkeypatch.setattr(generation, "claims_by_stem", lambda *a, **kw: [])
    monkeypatch.setattr(generation, "claim_query", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("search")))
    for stems, message in [(["unknown"], "unknown explicit"), (["empty"], "no claim evidence"),
                           ([str(i) for i in range(9)], "too many")]:
        with pytest.raises(ValueError, match=message):
            generation.build_evidence_packet("question", papers=stems)


def test_existing_pages_are_bounded_context(monkeypatch):
    from researchwiki import proposal_generation as generation
    pages = [SimpleNamespace(page_type="synthesis", stem=f"routing-{i}", category="synthesis",
                             fm={"title": "Routing decision rules"},
                             body="## Question\n\nWhen should routing change?\n\n## Short answer\n\nUse evidence.")
             for i in range(5)]
    monkeypatch.setattr(generation, "read_pages", lambda: pages)
    selected = generation._existing_pages("routing decision")
    assert len(selected) == 3
    assert "When should routing change?" in selected[0]["summary"]
    assert generation._existing_pages("unrelated") == []


def test_cross_category_prepare_cli_never_calls_model(tmp_path, monkeypatch, capsys):
    from researchwiki import proposal_generation as generation
    from researchwiki.tasks import proposals as cli
    from researchwiki.agents import llm
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "content_categories", lambda: ["ai"])
    monkeypatch.setattr(generation, "_page_categories", lambda: {"a": "ai"})
    monkeypatch.setattr(generation, "claim_query", lambda *a, **kw: [
        {"paper_stem": "a", "claim_slug": "kc-12345678", "text": "A claim"}])
    monkeypatch.setattr(llm, "call", lambda **kw: (_ for _ in ()).throw(AssertionError("model call")))
    assert cli.main(["generate", "routing", "--target-category", "ai", "--cross-category", "--prepare-only"]) == 0
    assert json.loads(capsys.readouterr().out)["planning_pending"] is True


def test_generate_then_accept_cli_uses_receipt(tmp_path, monkeypatch, capsys):
    import researchwiki.proposals as ledger
    from researchwiki.tasks import proposals as cli
    from researchwiki.agents import llm
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ledger, "commit_page", lambda _path: None)
    evidence = _evidence()
    for item in evidence:
        item.update(category="ai", section="methodology",
                    citation=f"[[{item['paper_stem']}#{item['claim_slug']}]]")
    packet = {"topic": "routing", "target_category": "ai", "evidence": evidence,
              "existing_pages": [{"title": "Prior routing synthesis", "summary": "Covered comparison"}],
              "prior_proposals": [{"proposal_id": "prop-123456789012", "status": "rejected",
                                    "latest_feedback": "Needs a baseline"}]}
    monkeypatch.setattr(cli, "build_evidence_packet", lambda *a, **kw: packet)
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({"proposals": [_proposal()]}), model="test-model",
                               input_tokens=1, output_tokens=2)
    monkeypatch.setattr(llm, "call", generate)
    assert cli.main(["generate", "routing", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "Prior routing synthesis" in calls[0]["prompt"]
    assert "Needs a baseline" in calls[0]["prompt"]
    assert cli.main(["accept", result["preview"], "--select", "1"]) == 0
    assert len(calls) == 1
    assert len(list((tmp_path / "wiki/proposals").glob("*.md"))) == 1


def test_invalid_preview_selection_and_mappings_do_not_write(tmp_path, monkeypatch):
    import pytest
    from researchwiki.proposal_preview import save_preview, accept_preview, validate_receipt
    monkeypatch.chdir(tmp_path)
    evidence = _evidence()
    for item in evidence:
        item["citation"] = f"[[{item['paper_stem']}#{item['claim_slug']}]]"
    packet = {"topic": "routing", "target_category": "ai", "evidence": evidence}
    path = save_preview(packet, [_proposal()], {"model": "test"})
    with pytest.raises(ValueError, match="select proposal numbers"):
        accept_preview(path, [1, 2])
    assert not (tmp_path / "wiki").exists()
    receipt = json.loads(path.read_text())
    receipt["packet"]["evidence"][0]["citation"] = "[[invented#kc-aaaa]]"
    with pytest.raises(ValueError, match="citation mappings"):
        validate_receipt(receipt)


def test_explicit_claims_use_query_ranking_before_quota(monkeypatch, tmp_path):
    from researchwiki import proposal_generation as generation
    monkeypatch.chdir(tmp_path)
    contributions = [{"paper_stem": "graph", "claim_slug": f"kc-{i}",
                      "text": f"Construction result {i}", "section": "key_contributions"}
                     for i in range(5)]
    limitation = {"paper_stem": "graph", "claim_slug": "lim-loss", "section": "limitations",
                  "text": "Projection into VCF can lose nested variant relationships."}
    monkeypatch.setattr(generation, "_page_categories", lambda: {"graph": "bio", "quiet": "bio"})
    quiet = {"paper_stem": "quiet", "claim_slug": "kc-quiet", "text": "Context claim"}
    monkeypatch.setattr(generation, "claims_by_stem", lambda stem, **kw:
                        contributions + [limitation] if stem == "graph" else [quiet])
    queries = []
    def query(topic, **kwargs):
        queries.append(topic)
        return [limitation] + contributions
    monkeypatch.setattr(generation, "claim_query", query)
    packet = generation.build_evidence_packet("nested VCF projection", papers=["graph", "quiet"],
                                               target_category="bio", cross_category=True)
    assert packet["evidence"][0]["claim_slug"] == "lim-loss"
    assert sum(item["paper_stem"] == "graph" for item in packet["evidence"]) == 3
    assert {item["paper_stem"] for item in packet["evidence"]} == {"graph", "quiet"}
    assert queries == ["nested VCF projection"]


def test_explicit_ranking_lexical_fallback_and_stable_ties():
    from researchwiki.proposal_generation import _rank_explicit_claims
    claims = [{"paper_stem": "p", "claim_slug": str(i), "text": text}
              for i, text in enumerate(["Irrelevant performance", "PageRank graph retrieval", "Other result"])]
    ranked = _rank_explicit_claims("graph retrieval", claims, [])
    assert [item["claim_slug"] for item in ranked] == ["1", "0", "2"]
    assert claims[0]["claim_slug"] == "0"  # caller's evidence is unchanged


def test_context_matching_requires_primary_and_substantive_overlap():
    from researchwiki.proposal_generation import _context_score
    assert _context_score("the and which", "The question", "And the answer") == 0
    assert _context_score("qubit fabrication conditions", "Research conditions", "A general approach") == 0
    assert _context_score("qubit fabrication", "Agent memory", "Qubit fabrication in a passing example") == 0
    assert _context_score("memory", "Agent memory", "") > 0
    assert _context_score("memory retrieval", "Agent memory", "Retrieval tradeoffs") > 0
    assert (_context_score("memory retrieval", "Memory retrieval", "")
            > _context_score("memory retrieval", "Memory", "Retrieval"))


def test_history_filters_noise_but_preserves_relevant_feedback(monkeypatch):
    from researchwiki import proposal_generation as generation
    record = SimpleNamespace(title="Memory retrieval", question="When should memory retrieval change?",
                             topic_seed="memory retrieval", thesis="A bounded hypothesis",
                             proposal_id="prop-123456789012", status="rejected",
                             feedback=[SimpleNamespace(reason="Needs a controlled baseline.")])
    monkeypatch.setattr(generation, "load_proposals", lambda: [record])
    assert generation._relevant_history("Which fabrication conditions yield reproducible superconducting qubits?") == []
    found = generation._relevant_history("memory retrieval")
    assert found[0]["status"] == "rejected"
    assert found[0]["latest_feedback"] == "Needs a controlled baseline."


def test_api_prompt_contains_contract_and_transfer_mode(monkeypatch):
    from researchwiki import proposal_generation as generation
    from researchwiki.agents import llm
    # Simulate a transport which ignores the schema argument entirely.
    def call(**kwargs):
        schema = json.loads(kwargs["system"].split("REQUIRED OUTPUT JSON SCHEMA:\n")[1])
        assert schema == generation._PROPOSAL_SCHEMA
        assert 'return {"proposals": []}' in kwargs["system"]
        assert "parent_proposal MUST" in kwargs["system"]
        assert "MODE: CROSS-CATEGORY APPLICATION ONLY" in kwargs["prompt"]
        assert "NON-target category" in kwargs["prompt"]
        return SimpleNamespace(text='{"proposals": []}', model="test", input_tokens=1, output_tokens=1)
    monkeypatch.setattr(llm, "call", call)
    found, _ = generation.generate_proposals({"topic": "transfer", "target_category": "ai",
                                             "cross_category": True, "evidence": [
                                                 {**item, "category": "ai", "section": "methodology"}
                                                 for item in _evidence()]})
    assert found == []


def test_planner_contract_rejects_missing_fields_before_search(monkeypatch):
    import pytest
    from researchwiki import proposal_generation as generation
    from researchwiki.agents import llm
    monkeypatch.setattr(generation, "_page_categories", lambda: {})
    monkeypatch.setattr(generation, "claim_query", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("search")))
    for response, error in [({"queries": ["graph preservation"]}, "target_problem"),
                            ({"target_problem": "loss", "queries": ["graph preservation"]}, "required_capabilities"),
                            ({"target_problem": "loss", "required_capabilities": [""], "queries": ["graph"]}, "required_capabilities")]:
        def call(response=response, **kwargs):
            schema = json.loads(kwargs["system"].split("REQUIRED OUTPUT JSON SCHEMA:\n")[1])
            assert schema == generation._SEARCH_PLAN_SCHEMA
            assert "domain-independent" in kwargs["system"]
            return SimpleNamespace(text=json.dumps(response))
        monkeypatch.setattr(llm, "call", call)
        with pytest.raises(ValueError, match=error):
            generation.expand_cross_category({"topic": "loss", "target_category": "bio", "evidence": []})


def test_json_envelope_is_not_extracted_from_wrong_shape():
    import pytest
    from researchwiki.proposal_generation import _parse_json
    for text in ('[]', '[{"proposals": []}]'):
        with pytest.raises(ValueError, match="JSON object"):
            _parse_json(text)
    assert _parse_json('```json\n{"proposals": []}\n```') == {"proposals": []}


def test_transfer_requires_both_categories_even_with_valid_schema():
    import pytest
    from researchwiki.proposal_generation import validate_candidates
    evidence = [{**item, "category": "ai"} for item in _evidence()]
    with pytest.raises(ValueError, match="target and source-category"):
        validate_candidates([_proposal()], {"evidence": evidence, "cross_category": True,
                                           "target_category": "ai"})
