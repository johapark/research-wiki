"""`researchwiki synthesize` scaffolds the fixed synthesis structure.

The scaffold is the first thing an author sees, so it has to agree with the
lint contract that later checks the page — otherwise every new page starts out
in violation. And the pre-pulled claims go to a side file, because at field
scale (dozens of papers, ~29 claims each) inlining them would bury the
structure under hundreds of bullets the author has to delete.
"""
from __future__ import annotations

import pytest

from researchwiki.tasks import synthesize
from researchwiki.tasks.lint.synthesis_contract import REQUIRED_SECTIONS, check_page


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    page = tmp_path / "wiki" / "cgt" / "a-2024-x.md"
    page.parent.mkdir(parents=True)
    page.write_text("---\ntype: paper\ncategory: [cgt]\n---\n\n## Summary\n\ntext\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fake_claims(monkeypatch):
    import researchwiki.search as search
    hit = {"paper_stem": "a-2024-x", "claim_slug": "kc-1234abcd", "section": "key_contributions",
           "position": 0, "text": "A specific claim.", "graded": True}
    monkeypatch.setattr(search, "claim_lookup", lambda q, k: [hit])
    monkeypatch.setattr(search, "claims_by_stem", lambda stem: [hit])


def test_scaffold_has_the_fixed_structure_and_passes_the_contract(wiki):
    assert synthesize.main(["--title", "Off-target effects", "--papers", "a-2024-x"]) == 0
    text = (wiki / "wiki/synthesis/off-target-effects.md").read_text()
    body = text.split("---\n", 2)[2]
    h2 = [line[3:].strip().lower() for line in body.splitlines() if line.startswith("## ")]
    assert tuple(h2) == REQUIRED_SECTIONS
    assert "what would update this page" not in text.lower()
    # Findings has no H3 yet, which is the one finding an empty scaffold owes.
    kinds = {v["kind"] for v in check_page(wiki / "x.md", body)}
    assert kinds == {"synthesis_findings_without_themes"}


def test_evidence_goes_to_a_side_file_not_the_page(wiki, monkeypatch):
    _fake_claims(monkeypatch)
    rc = synthesize.main(["--title", "Off-target effects", "--topic-seed", "off-target",
                          "--papers", "a-2024-x"])
    assert rc == 0
    page = (wiki / "wiki/synthesis/off-target-effects.md").read_text()
    evidence = (wiki / ".ingest/synthesis/off-target-effects/evidence.md").read_text()
    assert "[[a-2024-x#kc-1234abcd]]" in evidence
    assert "kc-1234abcd" not in page
    assert ".ingest/synthesis/off-target-effects/evidence.md" in page
