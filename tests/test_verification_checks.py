"""Reporting distinguishes retrieval from entailment and cannot spend a call."""

import ast
from pathlib import Path

import pytest

from researchwiki.grade.checks import entailment_check
from researchwiki.grade.fidelity.synthesis import FidelityClaim


@pytest.mark.parametrize("scores,status,completed", [
    ({}, "not_requested", False),
    ({"support_check_complete": False}, "incomplete", False),
    ({"support_check_failed": True}, "failed", False),
    ({"support_check_complete": True, "n_unsupported": 1}, "failed", True),
    ({"support_check_complete": True, "n_unsupported": 0}, "passed", True),
])
def test_reports_existing_support_state(scores, status, completed):
    check = entailment_check(scores)
    assert check["status"] == status
    assert check["completed"] == completed


def test_supported_retrieval_verdict_does_not_claim_entailment():
    claim = FidelityClaim(text="A plausible qualitative falsehood.", line_start=1, unit_index=0,
                          cited_stems=["example"], unresolved_citations=[],
                          best_stem="example", best_bm25=8., best_semantic=.95,
                          numeric_unmatched=[], negation_mismatch=False, verdict="supported")
    result = claim.to_dict()
    assert result["verdict"] == "supported"
    assert result["checks"]["retrieval"] == "completed"
    assert result["checks"]["entailment"]["status"] == "not_requested"


def test_shared_helpers_do_not_depend_on_cli_adapters():
    root = Path(__file__).resolve().parents[1] / "researchwiki"
    for name in ("supplementary.py", "page_naming.py", "agents/runner.py", "concepts/scaffold.py"):
        tree = ast.parse((root / name).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(module in {"tasks.attach", "tasks.synthesize"} for module in imports)
    tree = ast.parse((root / "grade/checks.py").read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
