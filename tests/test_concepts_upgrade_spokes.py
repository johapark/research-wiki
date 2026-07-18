"""Tests for `researchwiki concepts --upgrade-spokes` backfill.

Covers:
  - bare `[[stem]]` spoke → `[[stem#slug]]` when a matching claim exists
  - already-slug-cited spoke (`[[stem#slug]]`) is left untouched
  - spoke with no matching claim is left bare (counted as skipped)
  - referenced_papers: entries in frontmatter stay bare
  - dry-run mode doesn't mutate the file
  - idempotence: a second run is a no-op
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researchwiki.concepts import refresh as concepts_mod
from researchwiki.concepts.refresh import upgrade_spokes


def _stub_matching_claims(monkeypatch, per_stem_hits):
    def fake(stem, term):
        return per_stem_hits.get(stem, [])
    monkeypatch.setattr("researchwiki.concepts.term_claims._matching_claims", fake)


def _make_hub(wiki: Path, slug: str, term: str, spokes: str) -> Path:
    """Write a minimal concept hub with a given spoke block."""
    cdir = wiki / "concepts"
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"{slug}.md"
    path.write_text(
        f'---\ntitle: "{term}"\ntype: concept\ncategory: [ai]\n'
        f"referenced_papers:\n  - [[ai/paper-a]]\n"
        f'concept_span: 1\ngenerated_at: 2020-01-01\ntopic_seed: "{term}"\n'
        f"tags: [concept, {slug}]\n---\n\n"
        f"## Definition\nA thing.[^s]\n\n"
        f"## How it appears across the corpus\n{spokes}\n\n"
        f"[^s]: [[ai/paper-a]]\n"
    )
    return path


@pytest.fixture
def tmp_wiki(tmp_path, monkeypatch):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    monkeypatch.setattr("researchwiki.concepts.refresh.wiki_dir", lambda: wiki)
    # Patching wiki_dir only redirects markdown I/O. A non-dry-run upgrade
    # calls commit_page(cp), which resolves the DB path independently via
    # wiki_root() (= cwd) — without this, that write silently lands in the
    # real per-repo state.db instead of a throwaway one.
    monkeypatch.setenv("RESEARCHWIKI_DB_PATH", str(tmp_path / "state.db"))
    return wiki


def test_upgrades_bare_spoke_to_slug_form(tmp_wiki, monkeypatch):
    _stub_matching_claims(monkeypatch, {
        "paper-a": [{"section": "key_contributions", "position": 0,
                      "claim_slug": "kc-abcd1234", "text": "...",
                      "semantic_score": 0.9}],
    })
    hub = _make_hub(tmp_wiki, "raptor", "RAPTOR",
                    "- [[ai/paper-a]] — how it uses RAPTOR")
    stats = upgrade_spokes()
    assert stats["spokes_upgraded"] == 1
    assert stats["hubs_updated"] == 1
    text = hub.read_text()
    assert "[[ai/paper-a#kc-abcd1234]]" in text
    # frontmatter referenced_papers stays bare — that's whole-paper enumeration.
    assert "  - [[ai/paper-a]]\n" in text


def test_already_slug_cited_spoke_is_untouched(tmp_wiki, monkeypatch):
    _stub_matching_claims(monkeypatch, {
        "paper-a": [{"section": "key_contributions", "position": 0,
                      "claim_slug": "kc-abcd1234", "text": "...",
                      "semantic_score": 0.9}],
    })
    hub = _make_hub(tmp_wiki, "raptor", "RAPTOR",
                    "- [[ai/paper-a#kc-existing1]] — how it uses RAPTOR")
    before = hub.read_text()
    stats = upgrade_spokes()
    assert stats["spokes_upgraded"] == 0
    assert hub.read_text() == before  # byte-identical


def test_spoke_without_matching_claim_stays_bare(tmp_wiki, monkeypatch):
    _stub_matching_claims(monkeypatch, {})  # no matching claims for any stem
    hub = _make_hub(tmp_wiki, "raptor", "RAPTOR",
                    "- [[ai/paper-b]] — how it uses RAPTOR")
    stats = upgrade_spokes()
    assert stats["spokes_upgraded"] == 0
    assert stats["spokes_skipped_no_claim"] == 1
    assert "[[ai/paper-b]]" in hub.read_text()
    assert "[[ai/paper-b#" not in hub.read_text()


def test_dry_run_does_not_mutate(tmp_wiki, monkeypatch):
    _stub_matching_claims(monkeypatch, {
        "paper-a": [{"section": "key_contributions", "position": 0,
                      "claim_slug": "kc-abcd1234", "text": "...",
                      "semantic_score": 0.9}],
    })
    hub = _make_hub(tmp_wiki, "raptor", "RAPTOR",
                    "- [[ai/paper-a]] — how it uses RAPTOR")
    before = hub.read_text()
    stats = upgrade_spokes(dry_run=True)
    assert stats["spokes_upgraded"] == 1
    assert hub.read_text() == before  # dry-run leaves the file untouched


def test_upgrade_is_idempotent(tmp_wiki, monkeypatch):
    _stub_matching_claims(monkeypatch, {
        "paper-a": [{"section": "key_contributions", "position": 0,
                      "claim_slug": "kc-abcd1234", "text": "...",
                      "semantic_score": 0.9}],
    })
    _make_hub(tmp_wiki, "raptor", "RAPTOR",
              "- [[ai/paper-a]] — how it uses RAPTOR")
    upgrade_spokes()
    stats2 = upgrade_spokes()
    assert stats2["spokes_upgraded"] == 0


def test_upgrade_only_touches_the_appearance_section(tmp_wiki, monkeypatch):
    """A bare `[[stem]]` in the Definition body (or footnote) should NOT be
    upgraded — only spokes under `## How it appears across the corpus`."""
    _stub_matching_claims(monkeypatch, {
        "paper-a": [{"section": "key_contributions", "position": 0,
                      "claim_slug": "kc-abcd1234", "text": "...",
                      "semantic_score": 0.9}],
    })
    cdir = tmp_wiki / "concepts"
    cdir.mkdir()
    hub = cdir / "raptor.md"
    hub.write_text(
        '---\ntitle: "RAPTOR"\ntype: concept\ncategory: [ai]\n'
        "referenced_papers:\n  - [[ai/paper-a]]\n"
        'concept_span: 1\ngenerated_at: 2020-01-01\ntopic_seed: "RAPTOR"\ntags: [concept, raptor]\n---\n\n'
        "## Definition\nUses [[ai/paper-a]] as an example.[^s]\n\n"
        "## How it appears across the corpus\n- [[ai/paper-a]] — spoke\n\n"
        "[^s]: [[ai/paper-a]]\n"
    )
    stats = upgrade_spokes()
    text = hub.read_text()
    # Spoke was upgraded.
    assert "- [[ai/paper-a#kc-abcd1234]] — spoke" in text
    # Definition body's plain wikilink is unchanged.
    assert "Uses [[ai/paper-a]] as an example." in text
    # Footnote definition is unchanged.
    assert "[^s]: [[ai/paper-a]]" in text
    assert stats["spokes_upgraded"] == 1
