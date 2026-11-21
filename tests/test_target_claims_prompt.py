"""Target-claims prompt assembly — the extractor is fed the whole substantive
paper (references excluded) up to a char budget, priority-trimmed so
Results/Discussion/captions survive when a long paper overflows.

Regression guard for the bug where a blind `full_text[:18000]` prefix + tiny
per-section caps meant the extractor never saw the deep Results table where
benchmark ratios live, and emitted only qualitative claims.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from researchwiki.agents.phases.target_claims import (
    _allocate,
    _build_prompt,
    _rich_sections,
    render_for_author_prompt,
    TargetClaim,
    TargetClaimsOutput,
    extract_target_claims,
)
from researchwiki.agents.phases.draft import _build_author_prompt

BIG = 120_000


def _long_paper() -> str:
    """Synthetic PDF text: a huge Methods section pushes Results deep, with a
    benchmark number ~10K chars into Results and a limitation in Discussion.
    A References section carries a decoy number that must NOT be extracted."""
    methods_filler = "methods detail sentence. " * 2000          # ~50K chars
    results_pad = "result narrative sentence. " * 400            # ~10K chars
    return (
        "Abstract\nWe present a caller.\n\n"
        "Introduction\nBackground prose.\n\n"
        "Methods\n" + methods_filler + "\n\n"
        "Results\n" + results_pad +
        "On the benchmark VariantMedium reached AUPRC 0.918 here.\n\n"
        "Discussion\nWe note LIMITMARK indel calling is limited.\n\n"
        "References\n1. Author et al. Journal DECOY 0.404 (2020).\n"
    )


def test_deep_results_number_reaches_the_prompt():
    full = _long_paper()
    assert full.find("0.918") > 50000           # deep past the old windows
    prompt = _build_prompt({"title": "X"}, {}, full, BIG)
    assert "0.918" in prompt and "AUPRC" in prompt


def test_discussion_limitation_reaches_the_prompt():
    prompt = _build_prompt({"title": "X"}, {}, _long_paper(), BIG)
    assert "LIMITMARK" in prompt


def test_references_are_excluded():
    prompt = _build_prompt({"title": "X"}, {}, _long_paper(), BIG)
    assert "DECOY" not in prompt and "0.404" not in prompt
    assert "References" not in prompt


def test_budget_trims_low_priority_first():
    """Under a tight budget, Results/Discussion survive in full and low-density
    Methods is trimmed to whatever budget is left over — never the reverse."""
    prompt = _build_prompt({"title": "X"}, {}, _long_paper(), 12000)
    assert "0.918" in prompt          # Results number kept
    assert "LIMITMARK" in prompt      # Discussion kept
    # Methods gets only leftover budget, not its full ~2000 sentences.
    assert prompt.count("methods detail sentence") < 100


def test_allocate_orders_by_priority_and_respects_budget():
    rich = {"results": "R" * 5000, "discussion": "D" * 5000,
            "methods": "M" * 5000, "references": "X" * 5000}
    got = _allocate(rich, 8000)
    assert len(got.get("results", "")) == 5000       # highest priority, full
    assert len(got.get("discussion", "")) == 3000    # partial — budget ran out
    assert "methods" not in got                       # nothing left
    assert "references" not in got                    # never included


def test_rich_sections_falls_back_without_full_text():
    fallback = {"results": "precapped results"}
    assert _rich_sections(fallback, None, cap=BIG) is fallback


def test_critical_render_header_demands_verbatim_numbers():
    out = TargetClaimsOutput(claims=[
        TargetClaim(type="headline", content="AUPRC 0.918",
                    importance="critical", location="Results"),
    ])
    block = render_for_author_prompt(out)
    assert "verbatim" in block.lower() and "numeric" in block.lower()
    assert "0.918" in block


def _unhealthy_paper() -> str:
    """No findings heading; a giant Introduction pushes the result to EOF."""
    return (
        "Abstract\nWe present a ranker.\n\n"
        "Introduction\n" + "background-only prose. " * 2200
        + "\nLATE_RESULT GenRec reduces the token budget to one third.\n"
        "References\nREFERENCE_DECOY should never reach a prompt.\n"
    )


def test_unhealthy_target_context_is_stratified_not_head_only():
    prompt = _build_prompt({"title": "X"}, {}, _unhealthy_paper(), 6000)
    assert "document-stratified fallback" in prompt
    assert "LATE_RESULT" in prompt
    assert "REFERENCE_DECOY" not in prompt


def test_unhealthy_author_context_is_stratified_not_head_only():
    prompt = _build_author_prompt(
        {"title": "X"}, {}, [], pdf_full_text=_unhealthy_paper(),
    )
    assert "document-stratified fallback" in prompt
    assert "LATE_RESULT" in prompt
    assert "REFERENCE_DECOY" not in prompt


def test_healthy_long_author_context_is_also_stratified():
    full = _long_paper() + ("late substantive material. " * 3000)
    sections = {
        "introduction": "Background prose.",
        "methods": "Methods detail.",
        "results": "Results include AUPRC 0.918.",
        "discussion": "Discussion limitation.",
    }
    prompt = _build_author_prompt(
        {"title": "X", "pdf_text_preview": full[:1000]}, sections, [],
        pdf_full_text=full,
    )
    assert "[document stratum 5/5" in prompt
    assert "[stratified to 30000 chars" in prompt
    assert "DECOY" not in prompt


def test_target_extraction_returns_filtered_keywords(monkeypatch):
    payload = {
        "claims": [{
            "type": "headline", "content": "The caller reaches AUPRC 0.918",
            "importance": "critical", "location": "Results",
        }],
        "keywords": [
            "VariantMedium", "variant calling", "AUPRC", "genome benchmark",
            "precision recall", "study", "variant calling",
        ],
    }
    monkeypatch.setattr(
        "researchwiki.agents.phases.target_claims.llm.call",
        lambda **kw: SimpleNamespace(
            text=json.dumps(payload), model="test-model", input_tokens=10,
            output_tokens=5, cache_read_tokens=0, cache_write_tokens=0,
        ),
    )
    out = extract_target_claims(
        metadata={"title": "VariantMedium"}, sections={"results": "AUPRC 0.918"},
    )
    assert len(out.claims) == 1
    assert out.keywords == [
        "VariantMedium", "variant calling", "AUPRC", "genome benchmark",
        "precision recall",
    ]


def test_retry_reuses_only_complete_identical_target_claims(tmp_path, monkeypatch):
    from researchwiki.agents import model_config, llm
    from researchwiki.agents.phases import target_claims

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RW_LLM_BASE_URL", "http://localhost:8080/v1")
    cfg = model_config.ModelConfig("openai-compatible", "model-a", 0.2, 2000)
    monkeypatch.setattr(model_config, "for_phase", lambda phase: cfg)
    monkeypatch.setattr(model_config, "target_claims_max_chars", lambda: 5000)
    calls = []
    data = {"claims": [{"type": "headline", "content": "A verified result.", "importance": "critical"}],
            "keywords": ["alpha", "beta", "gamma", "delta", "epsilon"], "visual_requests": []}

    def call(**kwargs):
        calls.append(kwargs)
        return llm.LLMResponse(json.dumps(data), cfg.model, 0.2, 70, 30)

    monkeypatch.setattr(llm, "call", call)
    args = dict(metadata={}, sections={"results": "A source result."}, source_hash="a" * 64)
    first = extract_target_claims(**args)
    hit = extract_target_claims(**args)
    assert hit.claims == first.claims and hit.keywords == first.keywords
    assert hit.cache_hit and hit.input_tokens == hit.output_tokens == 0 and len(calls) == 1
    receipt, = (tmp_path / ".ingest/target-claims").glob("*.json")
    record = json.loads(receipt.read_text())
    record["text"] = record["text"].replace("verified", "different")
    receipt.write_text(json.dumps(record))
    restored = extract_target_claims(**args)
    assert not restored.cache_hit and restored.claims == first.claims
    monkeypatch.setenv("RW_FRESH_TARGET_CLAIMS", "1")
    assert not extract_target_claims(**args).cache_hit
    monkeypatch.delenv("RW_FRESH_TARGET_CLAIMS")
    assert not extract_target_claims(**dict(args, source_hash="b" * 64)).cache_hit
    assert not extract_target_claims(**dict(args, metadata={"title": "Changed"})).cache_hit
    assert not extract_target_claims(**args, visual_inventory=[]).cache_hit
    assert extract_target_claims(**args, visual_inventory=[]).cache_hit
    monkeypatch.setenv("RW_LLM_BASE_URL", "http://localhost:9090/v1")
    assert not extract_target_claims(**args).cache_hit
    cfg = model_config.ModelConfig("openai-compatible", "model-b", 0.2, 2000)
    assert not extract_target_claims(**args).cache_hit
    monkeypatch.setattr(target_claims, "_JSON_SCHEMA", {"type": "object"})
    data["keywords"] = []
    assert not extract_target_claims(**args).cache_hit
    assert not extract_target_claims(**args).cache_hit
    assert len(calls) == 10
