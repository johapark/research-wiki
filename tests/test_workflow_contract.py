"""Keep operational documentation aligned with executable ingest behavior."""
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = (ROOT / "WORKFLOW.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("signal", [
    "fidelity",
    "salience-recall",
    "target-claim recall",
    "coherence",
    "n_target_claims",
])
def test_workflow_names_current_fitness_signals(signal):
    assert signal in WORKFLOW


@pytest.mark.parametrize("stale_claim", [
    "0.5·semantic + 0.5·salience",
    "Every operation appends",
    "staleness by mtime",
    "reindex after every ingest batch",
])
def test_retired_workflow_claims_do_not_return(stale_claim):
    assert stale_claim not in WORKFLOW


def test_workflow_documents_incremental_agent_indexing():
    assert "agent ingest upserts incrementally" in WORKFLOW


def test_workflow_uses_the_canonical_citation_scout_command():
    assert "researchwiki scout --json" in WORKFLOW
    assert "researchwiki audit --json" not in WORKFLOW


def test_workflow_separates_agent_web_scouting_from_corpus_evidence():
    assert "researchwiki scout web request" in WORKFLOW
    assert "researchwiki scout web list" in WORKFLOW
    assert "researchwiki scout web show" in WORKFLOW
    assert "researchwiki scout web record" in WORKFLOW
    assert "normal conversational format with native citations" in WORKFLOW
    assert "stores no research prose" in WORKFLOW
    assert "--deliverable research-brief" not in WORKFLOW
    assert "performs no network access" in WORKFLOW
    assert "discovery-only" in WORKFLOW
    assert "feeds page authoring, claims, indexes, or `[[wikilink]]` generation" in WORKFLOW
