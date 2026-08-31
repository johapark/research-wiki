"""CLI semantics for successful PubMed retraction lookups."""

from __future__ import annotations

import json

from researchwiki.tasks import retraction_check


def test_successful_pubmed_no_hit_is_not_an_environment_error(monkeypatch, capsys):
    doi = "10.1000/not-in-pubmed"
    record = {
        "doi": doi,
        "pmid": None,
        "retracted": False,
        "is_retraction_notice": False,
        "pubtypes": [],
        "pubdate": "",
        "source": "pubmed",
        "fetched_at": "2026-08-31",
    }
    monkeypatch.setattr(retraction_check, "retraction_status", lambda value: record)

    assert retraction_check.main(["--doi", doi, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == record
