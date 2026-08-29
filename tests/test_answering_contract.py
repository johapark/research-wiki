"""Research answers should be grounded without ritual over-retrieval or writes."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLAUDE = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
ASK = (ROOT / "prompts" / "ask-system.md").read_text(encoding="utf-8")
WORKFLOW = (ROOT / "WORKFLOW.md").read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_main_contract_uses_adaptive_minimum_sufficient_retrieval():
    assert "retrieve proportionally" in CLAUDE
    assert "minimum sufficient retrieval" in CLAUDE
    for question_shape in (
        "Known single paper",
        "Direct factual topic",
        "Comparison, landscape",
        "Corpus count/filter",
        "Ingest quality/cost/timing",
        "Follow-up within the same scope",
    ):
        assert question_shape in CLAUDE


def test_answering_does_not_implicitly_authorize_wiki_writes():
    for contract in (CLAUDE, WORKFLOW):
        prose = _flat(contract)
        assert "read-only by default" in prose
        assert "explicitly asks to file" in prose
    assert "Non-trivial cross-paper → create a synthesis page" not in CLAUDE


def test_empty_retrieval_is_not_treated_as_proof_of_absence():
    assert "not proof that the corpus has no paper" in CLAUDE
    assert "not proof of corpus absence" in ASK
    assert "If `search` returns nothing" not in ASK


def test_mcp_prompt_routes_by_question_shape_instead_of_always_calling_both():
    assert "Known single paper" in ASK
    assert "Direct factual topic" in ASK
    assert "Comparison, landscape" in ASK
    assert "do not call both mechanically for every question" in ASK
    assert "Call `claims` with a topic query" not in ASK
