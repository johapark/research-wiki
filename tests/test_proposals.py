"""Markdown proposal ledger, derived DB rows, and bounded generation."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

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
         "text": "A method claim.", "category": "systems"},
        {"id": "e02", "paper_stem": "b-2026-target", "claim_slug": "lim-bbbb",
         "text": "A target limitation.", "category": "ai"},
    ]


def _fake_corpus(monkeypatch, evidence: list[dict]) -> None:
    """Make the wiki's paper→category lookup agree with `evidence`."""
    from researchwiki import proposal_preview
    categories = {item["paper_stem"]: item["category"] for item in evidence}
    monkeypatch.setattr(proposal_preview, "paper_categories", lambda _pages=None: categories)


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


def test_feedback_headings_preserve_reasons_and_later_decisions(tmp_path, monkeypatch):
    import researchwiki.proposals as ledger
    from researchwiki.proposal_generation import _relevant_history

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ledger, "commit_page", lambda _path: None)
    record = create_proposal(
        _proposal(), evidence_items=_evidence(), topic_seed="selective retrieval",
        target_category="ai", author_model="test-model",
    )
    first = append_feedback(
        record.proposal_id, decision="deferred",
        reason="Needs revision.\n\n## Required controls\n\nCompare equal evidence budgets.",
    )
    second = append_feedback(
        record.proposal_id, decision="shortlisted",
        reason="# Decision\n\nControls added.\n\n### Next step\n\nProceed to scoping.",
    )
    page = read_page(record.path)
    reparsed = parse_proposal(page)
    assert reparsed.feedback == [first, second]
    assert reparsed.status == "shortlisted"
    assert find_proposal_contract_violations(
        [record.path], {record.path: page.body}, {record.path: page.fm},
    ) == []
    assert _relevant_history("selective retrieval")[0]["latest_feedback"] == second.reason

    conn = sqlite3.connect(tmp_path / "rebuilt.db")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    with conn:
        _upsert_one(conn, record.path, 123)
    rows = conn.execute("SELECT feedback_id, decision, reason FROM proposal_feedback").fetchall()
    assert {row["feedback_id"]: (row["decision"], row["reason"]) for row in rows} == {
        event.feedback_id: (event.decision, event.reason) for event in (first, second)
    }
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
    monkeypatch.setattr(generation, "paper_categories", lambda _pages=None: {
        "target-paper": "biology", "source-paper": "systems",
    })
    monkeypatch.setattr(generation, "_claims_for_papers", lambda _stems: [])
    calls = []

    def fake_query(query, **_kwargs):
        calls.append(query)
        return [target_hit] if query == "target bottleneck" else [source_hit]

    monkeypatch.setattr(generation, "claim_query", fake_query)
    monkeypatch.setattr(generation, "_relevant_history", lambda _topic, **_kw: [])
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
    _fake_corpus(monkeypatch, evidence)
    second = {**_proposal(), "title": "Second reviewed choice", "thesis": "Exact reviewed hypothesis."}
    path = save_preview(packet, [_proposal(), second], {"model": "gpt-5.6-terra"})
    records = accept_preview(path, [2])
    assert len(records) == 1
    assert records[0].title == second["title"]
    assert records[0].thesis == second["thesis"]
    assert records[0].author_model == "gpt-5.6-terra"
    assert evidence[0]["citation"] in records[0].path.read_text()
    append_feedback(records[0].proposal_id, decision="shortlisted", reason="Keep exactly this.")
    before = records[0].path.read_text()
    assert accept_preview(path, [2])[0].proposal_id == records[0].proposal_id
    assert records[0].path.read_text() == before
    assert len(list((tmp_path / "wiki/proposals").glob("*.md"))) == 1


def test_explicit_papers_fail_before_search(monkeypatch):
    import pytest
    from researchwiki import proposal_generation as generation
    monkeypatch.setattr(generation, "paper_categories", lambda _pages=None: {"empty": "ai"})
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
    monkeypatch.setattr(generation, "paper_categories", lambda _pages=None: {"a": "ai"})
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
        item.update(section="methodology",
                    citation=f"[[{item['paper_stem']}#{item['claim_slug']}]]")
    packet = {"topic": "routing", "target_category": "ai", "evidence": evidence,
              "existing_pages": [{"title": "Prior routing synthesis", "summary": "Covered comparison"}],
              "prior_proposals": [{"proposal_id": "prop-123456789012", "status": "rejected",
                                    "latest_feedback": "Needs a baseline"}]}
    monkeypatch.setattr(cli, "build_evidence_packet", lambda *a, **kw: packet)
    monkeypatch.setattr(llm, "preflight_providers", lambda: None)
    _fake_corpus(monkeypatch, evidence)
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({"proposals": [_proposal()]}), model="gpt-5.6-terra",
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
    _fake_corpus(monkeypatch, evidence)
    path = save_preview(packet, [_proposal()], {"model": "gpt-5.6-terra"})
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
    monkeypatch.setattr(generation, "paper_categories", lambda _pages=None: {"graph": "bio", "quiet": "bio"})
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
    monkeypatch.setattr(generation, "load_proposals", lambda _pages=None: [record])
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
    monkeypatch.setattr(generation, "paper_categories", lambda _pages=None: {})
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


@pytest.mark.parametrize("cross_category", [True, False, None])
@pytest.mark.parametrize("categories", [("ai", "ai"), ("systems", "methods"), ("ai", "")])
def test_transfer_requires_both_categories_even_with_valid_schema(cross_category, categories):
    from researchwiki.proposal_generation import validate_candidates
    evidence = [{**item, "category": category} for item, category in zip(_evidence(), categories)]
    packet = {"evidence": evidence, "target_category": "ai"}
    if cross_category is not None:
        packet["cross_category"] = cross_category
    with pytest.raises(ValueError, match="target and source-category"):
        validate_candidates([_proposal()], packet)


def test_accept_rejects_same_category_transfer_without_writing(tmp_path, monkeypatch):
    from researchwiki.proposal_preview import accept_preview

    monkeypatch.chdir(tmp_path)
    evidence = [{**item, "category": "ai",
                 "citation": f"[[{item['paper_stem']}#{item['claim_slug']}]]"}
                for item in _evidence()]
    _fake_corpus(monkeypatch, evidence)
    receipt = {"version": 1, "packet": {"topic": "retrieval", "target_category": "ai",
                                         "evidence": evidence},
               "proposals": [_proposal()], "usage": {"model": "gpt-5.6-terra"}}
    path = tmp_path / "preview.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="target and source-category"):
        accept_preview(path, [1])
    assert not (tmp_path / "wiki").exists()


# ---------- ledger integrity (review fixes) ----------


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """A proposals directory with one saved record; no DB, no index."""
    import researchwiki.proposals as proposals

    wiki = tmp_path / "wiki"
    monkeypatch.setattr(proposals, "wiki_dir", lambda: wiki)
    monkeypatch.setattr(proposals, "commit_page", lambda _path: None)
    record = create_proposal(
        _proposal(), evidence_items=_evidence(), topic_seed="selective retrieval",
        target_category="ai", author_model="gpt-5.6-terra",
    )
    return proposals, record


def test_backslash_resulting_page_keeps_the_page_parseable(ledger):
    """A Windows-style path was spliced into a regex replacement template, which
    turned JSON's `\\\\` escapes back into `\\` — invalid YAML — and the record
    vanished from every view."""
    proposals, record = ledger
    proposals.append_feedback(record.proposal_id, decision="published", reason="Done.",
                              resulting_page=r"wiki\ideas\x.md")
    page = read_page(record.path)
    assert page.fm.get("resulting_page") == r"wiki\ideas\x.md"
    assert proposals.find_proposal(record.proposal_id) is not None


@pytest.mark.parametrize("key", ["resulting_page", "updated_at"])
def test_an_emptied_field_does_not_swallow_the_next_line(ledger, key):
    """`resulting_page:` with no value (what Obsidian writes) used to absorb the
    following `author_model:` line, because `\\s*` also matches the newline."""
    proposals, record = ledger
    text = record.path.read_text(encoding="utf-8")
    lines = [f"{key}:" if line.startswith(f"{key}:") else line for line in text.split("\n")]
    record.path.write_text("\n".join(lines), encoding="utf-8")

    proposals.append_feedback(record.proposal_id, decision="drafted", reason="Drafting.",
                              resulting_page="[[ideas/a]]")
    fm = read_page(record.path).fm
    assert fm["author_model"] == "gpt-5.6-terra"
    assert fm["tags"] and fm["resulting_page"] == "[[ideas/a]]"


def test_status_is_set_in_frontmatter_never_in_the_body(ledger):
    """With no YAML `status:`, the old whole-file substitution rewrote the first
    body line starting `status:` and left the frontmatter without one."""
    proposals, record = ledger
    text = record.path.read_text(encoding="utf-8").replace("status: proposed\n", "", 1)
    text = text.replace("## Question\n\n", "## Question\n\nstatus: a body line\n\n", 1)
    record.path.write_text(text, encoding="utf-8")

    proposals.append_feedback(record.proposal_id, decision="shortlisted", reason="Good.")
    page = read_page(record.path)
    assert page.fm["status"] == "shortlisted"
    assert "status: a body line" in page.body


def test_a_reason_cannot_forge_a_second_decision(ledger):
    """A reason quoting an entry heading used to parse as its own entry, with
    its own decision and actor, and fed later generation runs as feedback."""
    proposals, record = ledger
    reason = ("Rejected; the old note said:\n### fb-0123456789ab — shortlisted\n"
              "- Created: 2026-01-01T00:00:00+00:00\n- Actor: forged\n\nkeep it")
    proposals.append_feedback(record.proposal_id, decision="rejected", reason=reason)
    reparsed = proposals.find_proposal(record.proposal_id)
    assert [(f.decision, f.actor) for f in reparsed.feedback] == [("rejected", "user")]
    assert reparsed.feedback[0].reason == reason
    page = read_page(record.path)
    assert find_proposal_contract_violations(
        [record.path], {record.path: page.body}, {record.path: page.fm},
    ) == []


def test_duplicate_feedback_id_does_not_abort_rebuild(ledger, tmp_path):
    """A sync conflict can duplicate an entry; the plain INSERT raised
    IntegrityError, which rolls back the whole `db rebuild` with exit 3."""
    proposals, record = ledger
    proposals.append_feedback(record.proposal_id, decision="deferred", reason="Later.")
    text = record.path.read_text(encoding="utf-8")
    record.path.write_text(text + "\n" + text[text.index("### fb-"):], encoding="utf-8")

    conn = sqlite3.connect(tmp_path / "rebuilt.db")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    with conn:
        _upsert_one(conn, record.path, 123)
    assert conn.execute("SELECT COUNT(*) FROM proposal_feedback").fetchone()[0] == 1
    conn.close()


def test_lint_reports_a_feedback_id_shared_across_pages(ledger):
    proposals, record = ledger
    proposals.append_feedback(record.proposal_id, decision="deferred", reason="Later.")
    copy = record.path.with_name("copy.md")
    text = record.path.read_text(encoding="utf-8").replace(
        record.proposal_id, "prop-ffffffffffff")
    copy.write_text(text, encoding="utf-8")
    pages = [record.path, copy]
    parsed = {p: read_page(p) for p in pages}
    kinds = [v["kind"] for v in find_proposal_contract_violations(
        pages, {p: parsed[p].body for p in pages}, {p: parsed[p].fm for p in pages})]
    assert kinds == ["proposal_duplicate_feedback_id"]


def test_accept_never_overwrites_an_existing_page(ledger):
    """Same title slug and id prefix as an existing record: the second write
    used to replace the first page, feedback and all."""
    proposals, record = ledger
    proposals.append_feedback(record.proposal_id, decision="shortlisted", reason="Keep.")
    before = record.path.read_text(encoding="utf-8")
    short = record.proposal_id.removeprefix("prop-")
    clash = short[:8] + "0000" if not short.endswith("0000") else short[:8] + "1111"
    other = create_proposal(
        _proposal(), evidence_items=_evidence(), topic_seed="t", target_category="ai",
        author_model="gpt-5.6-terra", acceptance_id=clash,
    )
    assert other.path != record.path
    assert record.path.read_text(encoding="utf-8") == before


def test_proposals_sort_by_instant_not_by_string():
    from researchwiki.proposals import _timestamp_key
    tokyo = "2026-09-24T09:00:00+09:00"      # 00:00 UTC on the 24th
    pacific = "2026-09-23T20:00:00-07:00"    # 03:00 UTC on the 24th — later
    assert _timestamp_key(pacific) > _timestamp_key(tokyo)
    assert tokyo > pacific                   # what the string sort did


@pytest.mark.parametrize("model", ["TODO", "gpt-5.6", "", "test"])
def test_receipt_requires_an_exact_author_model(model, monkeypatch):
    from researchwiki.proposal_preview import validate_receipt
    evidence = [{**item, "citation": f"[[{item['paper_stem']}#{item['claim_slug']}]]"}
                for item in _evidence()]
    receipt = {"version": 1, "packet": {"topic": "t", "target_category": "ai",
                                         "evidence": evidence},
               "proposals": [], "usage": {"model": model}}
    if model == "test":
        # A bare word is not a family alias; it is the caller's claim to make.
        validate_receipt(receipt)
        return
    with pytest.raises(ValueError, match="exact author model"):
        validate_receipt(receipt)


def test_receipt_categories_are_checked_against_the_wiki(tmp_path, monkeypatch):
    """A hand-edited receipt could relabel same-category evidence as a transfer."""
    from researchwiki.proposal_preview import accept_preview
    monkeypatch.chdir(tmp_path)
    evidence = [{**item, "citation": f"[[{item['paper_stem']}#{item['claim_slug']}]]"}
                for item in _evidence()]
    _fake_corpus(monkeypatch, [{**item, "category": "ai"} for item in evidence])
    path = tmp_path / "preview.json"
    path.write_text(json.dumps({
        "version": 1, "packet": {"topic": "t", "target_category": "ai", "evidence": evidence},
        "proposals": [_proposal()], "usage": {"model": "gpt-5.6-terra"}}))
    with pytest.raises(ValueError, match="files it under 'ai'"):
        accept_preview(path, [1])
    assert not (tmp_path / "wiki").exists()


def test_unplanned_cross_category_receipt_is_rejected():
    from researchwiki.proposal_preview import validate_receipt
    evidence = [{**item, "citation": f"[[{item['paper_stem']}#{item['claim_slug']}]]"}
                for item in _evidence()]
    receipt = {"version": 1, "packet": {"topic": "t", "target_category": "ai",
                                         "cross_category": True, "planning_pending": True,
                                         "evidence": evidence},
               "proposals": [], "usage": {"model": "gpt-5.6-terra"}}
    with pytest.raises(ValueError, match="unplanned cross-category"):
        validate_receipt(receipt)


def test_chat_search_plan_adds_source_evidence_without_a_model_call(
    tmp_path, monkeypatch, capsys,
):
    """The documented chat path could only produce a target-only packet, which
    no cross-category proposal can pass."""
    from researchwiki import proposal_generation as generation
    from researchwiki.tasks import proposals as cli
    from researchwiki.agents import llm
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "content_categories", lambda: ["ai", "systems"])
    monkeypatch.setattr(generation, "read_pages", lambda: [])
    monkeypatch.setattr(generation, "paper_categories",
                        lambda _pages=None: {"t": "ai", "s": "systems"})
    monkeypatch.setattr(generation, "claim_query", lambda query, **kw: (
        [{"paper_stem": "t", "claim_slug": "lim-11111111", "text": "Target limit."}]
        if query == "routing" else
        [{"paper_stem": "s", "claim_slug": "met-22222222", "text": "Source method."}]))
    monkeypatch.setattr(llm, "call", lambda **kw: (_ for _ in ()).throw(AssertionError("model call")))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"target_problem": "Expensive reads",
                                "required_capabilities": ["selective access"],
                                "queries": ["conditional access"]}))
    assert cli.main(["generate", "routing", "--target-category", "ai", "--cross-category",
                     "--search-plan", str(plan), "--prepare-only"]) == 0
    packet = json.loads(capsys.readouterr().out)
    assert packet["planning_pending"] is False
    assert {item["category"] for item in packet["evidence"]} == {"ai", "systems"}


def test_slugless_claims_never_enter_the_packet():
    from researchwiki.proposal_generation import _select_diverse
    hits = [{"paper_stem": "a", "claim_slug": None, "text": "No slug."},
            {"paper_stem": "a", "claim_slug": "kc-12345678", "text": "Slugged."}]
    chosen = _select_diverse(hits, {"a": "ai"})
    assert [item["claim_slug"] for item in chosen] == ["kc-12345678"]


def test_generate_preflights_before_any_retrieval(monkeypatch):
    from researchwiki.tasks import proposals as cli
    from researchwiki.agents import llm
    monkeypatch.setattr(llm, "preflight_providers", lambda: (_ for _ in ()).throw(
        llm.ProviderUnavailable("the active model config cannot attribute authored pages.")))
    monkeypatch.setattr(cli, "build_evidence_packet",
                        lambda *a, **kw: pytest.fail("retrieval ran before preflight"))
    with pytest.raises(llm.ProviderUnavailable):
        cli.main(["generate", "routing"])


def test_partial_write_is_logged_and_exits_1(tmp_path, monkeypatch, capsys):
    """Proposal 2 failing after proposal 1 landed used to escape as exit 3 with
    no log.md entry for the page already on disk."""
    import researchwiki.proposals as ledger_mod
    from researchwiki import proposal_preview
    from researchwiki.tasks import proposals as cli
    from researchwiki.agents import llm
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ledger_mod, "wiki_dir", lambda: tmp_path / "wiki")
    monkeypatch.setattr(ledger_mod, "commit_page", lambda _path: None)
    monkeypatch.setattr(llm, "preflight_providers", lambda: None)
    evidence = [{**item, "section": "methodology",
                 "citation": f"[[{item['paper_stem']}#{item['claim_slug']}]]"}
                for item in _evidence()]
    _fake_corpus(monkeypatch, evidence)
    packet = {"topic": "routing", "target_category": "ai", "evidence": evidence}
    monkeypatch.setattr(cli, "build_evidence_packet", lambda *a, **kw: packet)
    second = {**_proposal(), "title": "Second"}
    monkeypatch.setattr(llm, "call", lambda **kw: SimpleNamespace(
        text=json.dumps({"proposals": [_proposal(), second]}), model="gpt-5.6-terra",
        input_tokens=1, output_tokens=1))
    real_create = proposal_preview.create_proposal
    calls = []

    def flaky(proposal, **kw):
        calls.append(proposal["title"])
        if len(calls) == 2:
            raise OSError("disk full")
        return real_create(proposal, **kw)
    monkeypatch.setattr(proposal_preview, "create_proposal", flaky)
    logged = []
    monkeypatch.setattr(cli, "append_log_md", lambda *a: logged.append(a))

    assert cli.main(["generate", "routing", "--write"]) == 1
    assert "disk full" in capsys.readouterr().err
    assert len(logged) == 1 and "generated 1 proposal(s)" in logged[0][1]


def test_feedback_log_entry_is_one_line(ledger, monkeypatch):
    """A reason with `## ` headings wrote fake H2 entries into log.md."""
    from researchwiki.tasks import proposals as cli
    _proposals, record = ledger
    logged = []
    monkeypatch.setattr(cli, "append_log_md", lambda *a: logged.append(a))
    assert cli.main(["feedback", record.proposal_id, "--decision", "deferred",
                     "--reason", "Needs work.\n\n## Required controls\n\nA baseline."]) == 0
    _kind, headline, details = logged[0]
    assert "\n" not in headline and "\n" not in details
    assert "## Required controls" in details    # kept, but inline
