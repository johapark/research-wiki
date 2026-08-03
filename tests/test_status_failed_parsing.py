"""`researchwiki status` derives the failed-parsing list from page YAML.

It used to scan `wiki/pdfs-failed-parsing.md` for `- **{stem}**` lines. That
ledger was hand-maintained, drifted to nonexistent, and the scan's
`if not path.exists(): return []` turned that into a silent zero — status
reported no extraction failures while a page still carried the marker.

The signal now comes from each page's YAML `pdf_extraction_note:`, which
`prompts/ingest-digest.md` already tells the unextractable-PDF path to set. It
can't drift from the pages because it lives on them.
"""

from __future__ import annotations

from dataclasses import dataclass

from researchwiki.tasks.status import _failed_parsing_entries


@dataclass
class FakePage:
    """Stands in for `wiki.Page` — the scan only reads these three attrs."""
    stem: str
    category: str
    fm: dict


def test_page_with_a_note_is_listed_with_its_note():
    pages = [
        FakePage("a-2025-x", "genetics",
                 {"pdf_extraction_note": "Not text-extractable; built from abstract."}),
        FakePage("b-2025-y", "genomics", {}),
    ]
    assert _failed_parsing_entries(pages) == [
        ("genetics/a-2025-x", "Not text-extractable; built from abstract.")
    ]


def test_no_notes_yields_an_empty_list():
    assert _failed_parsing_entries([FakePage("a", "genetics", {})]) == []


def test_empty_and_whitespace_notes_do_not_count_as_failures():
    """A page carrying `pdf_extraction_note:` with no value hasn't failed —
    treating the bare key as a failure would inflate the count."""
    pages = [
        FakePage("a", "genetics", {"pdf_extraction_note": ""}),
        FakePage("b", "genetics", {"pdf_extraction_note": "   "}),
        FakePage("c", "genetics", {"pdf_extraction_note": None}),
    ]
    assert _failed_parsing_entries(pages) == []


def test_non_string_note_is_coerced_not_crashed():
    """Frontmatter is user-editable; a bare number must not raise."""
    pages = [FakePage("a", "genetics", {"pdf_extraction_note": 2025})]
    assert _failed_parsing_entries(pages) == [("genetics/a", "2025")]


def test_entries_are_sorted_by_page_key():
    pages = [
        FakePage("z-2025", "genomics", {"pdf_extraction_note": "n"}),
        FakePage("a-2025", "genetics", {"pdf_extraction_note": "n"}),
        FakePage("m-2025", "genetics", {"pdf_extraction_note": "n"}),
    ]
    assert [k for k, _ in _failed_parsing_entries(pages)] == [
        "genetics/a-2025", "genetics/m-2025", "genomics/z-2025",
    ]


def test_any_page_type_can_report_an_unextractable_pdf():
    """Reference docs (guidance, protocols) have PDFs too — the scan keys on the
    note, not on `type: paper`."""
    pages = [
        FakePage("fda-2026-guide", "references",
                 {"type": "guidance", "pdf_extraction_note": "Scanned image PDF."}),
    ]
    assert _failed_parsing_entries(pages) == [
        ("references/fda-2026-guide", "Scanned image PDF.")
    ]


def test_the_removed_ledger_file_is_not_consulted(tmp_path, monkeypatch):
    """A stale `wiki/pdfs-failed-parsing.md` left over in someone's vault must
    not contribute entries — page YAML is the only source."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "pdfs-failed-parsing.md").write_text(
        "# Failed PDFs\n\n- **ghost-2020-not-a-real-page**\n"
    )
    monkeypatch.setattr("researchwiki.paths.wiki_dir", lambda: wiki)
    assert _failed_parsing_entries([FakePage("a", "genetics", {})]) == []


def test_paths_no_longer_exposes_a_ledger_helper():
    """Guards against the ledger being reintroduced as a second source of truth."""
    import researchwiki.paths as paths
    assert not hasattr(paths, "pdfs_failed_parsing_path")
