"""The `inspect visuals` review queue.

What is worth pinning is the set of things that make the queue *mislead*, since
a wrong queue looks exactly as authoritative as a right one:

  - A superseded receipt must never be shown. A re-ingest writes another receipt
    beside the old one, and nothing on the page would tell a reviewer that the
    reading they are checking was replaced an hour ago.
  - A paper already in `wiki/` must drop out. The receipt outlives the run, so
    the wiki is the authority on whether anyone is still waiting.
  - A blocking figure with no reading must not sit in the same list as one with
    a reading: the first needs a re-run, the second needs a human. Conflating
    them is what made the raw receipts unusable.
  - A missing page render is normal (the cache is disposable and a held paper's
    PDF is still in `inbox/`), so it must degrade to a note, never an error.
  - The payload is injected into a `<script>` block, so a caption containing
    `</script>` must not be able to blank the page.

Hermetic: tmp wiki, synthetic receipts, no PDF, no network, no LLM.
"""

from __future__ import annotations

import json

import pytest

from researchwiki import paths, review_queue
from researchwiki.tasks import inspect as cli

ATTEMPTS = [
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
]


def _receipt(**over):
    """A receipt with one blocking figure that has a reading."""
    base = {
        "status": "needs_review",
        "selection_status": "complete",
        "results_source": "body",
        "source_sha256": "abc123",
        "critical_unresolved": ["fig:1"],
        "selected": [{
            "request": {"ref_id": "fig:1", "priority": "critical",
                        "reason": "visual_evidence", "question": "What does panel b show?"},
            "page": 3, "caption": "Fig. 1 | Screen design",
            "render": {"page": 3, "width": 910, "height": 1209, "dpi": 110},
        }],
        "observations": [{
            "ref_id": "fig:1", "page": 3, "answered": True, "corroborated": False,
            "observation": "Panel b shows higher expansion in the knockout arm.",
            "visible_labels": "knockout; control", "units_conditions": "fold change",
            "uncertainty": "read from the plot", "source_quote": "",
        }],
        "unresolved": [{"request": {"ref_id": "fig:1", "priority": "critical"},
                        "reason": "source_image_review"}],
        "unselected": ["fig:2"],
        "error": None, "model": "test-vision", "input_tokens": 100,
        "output_tokens": 50, "request_ms": 900, "estimated_cost_usd": None,
        "cache_hit": False, "cache_key": "k", "pixels": 1, "preparation_ms": 1,
        "request_count": 1, "usage_known": True,
    }
    base.update(over)
    return base


def _write(root, stem, attempt, receipt=None, *, draft=True):
    sandbox = root / ".agent-output"
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / f"{stem}-{attempt}.visual.json").write_text(
        json.dumps(receipt or _receipt()), encoding="utf-8")
    if draft:
        (sandbox / f"{stem}.md").write_text(
            f'---\ntitle: "A paper about {stem}"\n---\n\n## Summary\nText.\n',
            encoding="utf-8")


def _cached_png(root, sha, page):
    """A real PNG, encoded the way the ingest encodes one."""
    import numpy as np

    from researchwiki.pdf.render import encode_png
    payload, _ = encode_png(np.zeros((12, 10, 3), dtype=np.uint8))
    dest = root / ".figures-cache" / "inspection" / f"{sha}-p{page}-110.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return dest


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki" / "cgt").mkdir(parents=True)
    # No state.db in a tmp wiki, so gate-reason enrichment is exercised in its
    # degraded form throughout; one test covers it explicitly.
    _write(tmp_path, "smith-2024-held-paper", ATTEMPTS[0])
    return tmp_path


# ---------- selection ----------

def test_newest_attempt_wins(wiki):
    """Three receipts for one stem: only the newest reading is shown."""
    import os
    import time

    for i, attempt in enumerate(ATTEMPTS):
        _write(wiki, "smith-2024-held-paper", attempt,
               _receipt(observations=[dict(_receipt()["observations"][0],
                                           observation=f"reading {i}")]))
        path = wiki / ".agent-output" / f"smith-2024-held-paper-{attempt}.visual.json"
        os.utime(path, (time.time() + i, time.time() + i))

    queue = review_queue.build_queue(images=False)
    assert len(queue.papers) == 1
    assert queue.papers[0].attempt_id == ATTEMPTS[-1]
    assert queue.papers[0].items[0].observation == "reading 2"

    every = review_queue.build_queue(images=False, all_attempts=True)
    assert len(every.papers) == 3


def test_a_promoted_paper_drops_out_of_the_queue(wiki):
    assert review_queue.build_queue(images=False).meta["n_papers"] == 1
    (wiki / "wiki" / "cgt" / "smith-2024-held-paper.md").write_text("---\n---\n", encoding="utf-8")

    assert review_queue.build_queue(images=False).meta["n_papers"] == 0
    kept = review_queue.build_queue(images=False, include_resolved=True)
    assert kept.meta["n_papers"] == 1 and kept.papers[0].promoted


@pytest.mark.parametrize("status", ["inspected", "not_needed"])
def test_a_resolved_inspection_drops_out(wiki, status):
    _write(wiki, "jones-2025-passed", ATTEMPTS[1],
           _receipt(status=status, critical_unresolved=[]))
    stems = {p.stem for p in review_queue.build_queue(images=False).papers}
    assert stems == {"smith-2024-held-paper"}
    assert len(review_queue.build_queue(images=False, include_resolved=True).papers) == 2


def test_an_unreadable_receipt_is_counted_not_fatal(wiki):
    (wiki / ".agent-output" / f"broken-2024-x-{ATTEMPTS[1]}.visual.json").write_text(
        "{not json", encoding="utf-8")
    meta = review_queue.build_queue(images=False).meta
    assert meta["n_papers"] == 1 and meta["n_unreadable_receipts"] == 1


# ---------- ready vs needs-a-re-run ----------

def test_a_blocking_figure_with_no_reading_is_not_ready(wiki):
    """The distinction the raw receipts could not express."""
    _write(wiki, "lee-2025-errored", ATTEMPTS[1],
           _receipt(status="incomplete", observations=[], error="TimeoutError: timed out"))
    queue = review_queue.build_queue(images=False)
    ready = {p.stem: p.ready for p in queue.papers}
    assert ready == {"smith-2024-held-paper": True, "lee-2025-errored": False}
    # Ready papers sort first, so the reviewer meets the judgeable ones.
    assert queue.papers[0].stem == "smith-2024-held-paper"
    assert queue.meta["n_ready"] == 1
    assert queue.meta["n_blocking"] == 2 and queue.meta["n_blocking_with_reading"] == 1


def test_fewest_blocking_figures_first_among_ready(wiki):
    _write(wiki, "zhang-2025-two-figures", ATTEMPTS[1], _receipt(
        critical_unresolved=["fig:1", "fig:4"],
        selected=[
            _receipt()["selected"][0],
            {"request": {"ref_id": "fig:4", "priority": "critical",
                         "reason": "missing_detail", "question": "q2"},
             "page": 7, "caption": "Fig. 4 | Survival"},
        ],
        observations=[
            _receipt()["observations"][0],
            dict(_receipt()["observations"][0], ref_id="fig:4", page=7),
        ],
    ))
    order = [p.stem for p in review_queue.build_queue(images=False).papers]
    assert order == ["smith-2024-held-paper", "zhang-2025-two-figures"]


def test_corroborated_reading_is_reported(wiki):
    _write(wiki, "ok-2025-confirmed", ATTEMPTS[1], _receipt(
        observations=[dict(_receipt()["observations"][0], corroborated=True,
                           uncertainty="", source_quote="The knockout arm expanded more.")]))
    paper = next(p for p in review_queue.build_queue(images=False).papers
                 if p.stem == "ok-2025-confirmed")
    assert paper.items[0].corroborated
    assert review_queue.build_queue(images=False).meta["n_corroborated"] == 1


# ---------- page images ----------

def test_a_cached_render_is_embedded(wiki):
    _cached_png(wiki, "abc123", 3)
    item = review_queue.build_queue().papers[0].items[0]
    assert item.image and item.image.startswith("data:image/png;base64,")
    assert item.image_note is None
    assert review_queue.build_queue().meta["n_images"] == 1


def test_a_missing_render_degrades_to_a_note(wiki):
    """The cache is disposable and a held paper's PDF is still in inbox/."""
    item = review_queue.build_queue().papers[0].items[0]
    assert item.image is None
    assert "cached render is gone" in item.image_note
    assert review_queue.build_queue().meta["n_images"] == 0


def test_the_image_budget_drops_renders_and_says_so(wiki):
    _cached_png(wiki, "abc123", 3)
    meta = review_queue.build_queue(image_budget_bytes=1).meta
    assert meta["n_images"] == 0 and meta["n_images_omitted"] == 1
    item = review_queue.build_queue(image_budget_bytes=1).papers[0].items[0]
    assert "omitted" in item.image_note


def test_no_images_skips_the_cache_entirely(wiki):
    _cached_png(wiki, "abc123", 3)
    item = review_queue.build_queue(images=False).papers[0].items[0]
    assert item.image is None and item.image_note is None


def test_a_reference_with_no_page_is_listed_without_a_render(wiki):
    """A caption never located has no page to show, but still blocks."""
    _write(wiki, "gap-2025-unlocated", ATTEMPTS[1], _receipt(
        critical_unresolved=["fig:5"], selected=[], observations=[],
        unresolved=[{"request": {"ref_id": "fig:5", "priority": "critical"},
                     "reason": "missing_caption_location"}]))
    paper = next(p for p in review_queue.build_queue().papers
                 if p.stem == "gap-2025-unlocated")
    item = paper.items[0]
    assert item.page is None and item.image is None and item.blocking
    assert "caption never located" in item.reason_text


# ---------- labels, titles, metadata ----------

@pytest.mark.parametrize("ref_id,label", [
    ("fig:2", "Fig. 2"), ("table:1", "Table 1"),
    ("edfig:4", "Extended Data Fig. 4"), ("edtable:3", "Extended Data Table 3"),
])
def test_labels_read_the_way_papers_print_them(ref_id, label):
    assert review_queue._label_for(ref_id) == label


def test_the_draft_title_is_used_when_present(wiki):
    paper = review_queue.build_queue(images=False).papers[0]
    assert paper.title == "A paper about smith-2024-held-paper"
    assert paper.draft_path and paper.draft_path.endswith("smith-2024-held-paper.md")


def test_a_receipt_without_a_draft_still_renders(wiki):
    _write(wiki, "nodraft-2025-x", ATTEMPTS[1], draft=False)
    paper = next(p for p in review_queue.build_queue(images=False).papers
                 if p.stem == "nodraft-2025-x")
    assert paper.draft_path is None and paper.title == "nodraft 2025 x"


def test_a_legacy_receipt_without_results_source_is_tolerated(wiki):
    receipt = _receipt()
    del receipt["results_source"]
    _write(wiki, "legacy-2024-x", ATTEMPTS[1], receipt)
    paper = next(p for p in review_queue.build_queue(images=False).papers
                 if p.stem == "legacy-2024-x")
    assert paper.results_source == ""


def test_gate_reasons_are_best_effort(wiki):
    """A tmp wiki has no state.db; the queue must still build."""
    paper = review_queue.build_queue(images=False).papers[0]
    assert paper.gate_reasons == ()


def test_the_source_pdf_is_matched_by_content_hash(wiki):
    """resolve_pdf can't find a held paper, so the receipt's sha does."""
    import hashlib
    inbox = wiki / "inbox"
    inbox.mkdir()
    (inbox / "publisher-name.pdf").write_bytes(b"pdf bytes")
    sha = hashlib.sha256(b"pdf bytes").hexdigest()
    _write(wiki, "hashed-2025-x", ATTEMPTS[1], _receipt(source_sha256=sha))
    papers = {p.stem: p for p in review_queue.build_queue(images=False).papers}
    assert papers["hashed-2025-x"].source_pdf.endswith("publisher-name.pdf")
    # The fixture's receipt hashes to nothing in inbox/, so it reports none
    # rather than guessing at the only PDF there.
    assert papers["smith-2024-held-paper"].source_pdf is None


# ---------- rendering ----------

def test_render_substitutes_the_placeholder(wiki):
    html = review_queue.render_html(review_queue.build_queue(images=False))
    assert "__QUEUE_DATA__" not in html
    assert "const DATA = {" in html


def test_a_caption_cannot_terminate_the_script_block(wiki):
    _write(wiki, "evil-2025-x", ATTEMPTS[1], _receipt(selected=[{
        "request": {"ref_id": "fig:1", "priority": "critical",
                    "reason": "visual_evidence", "question": "q"},
        "page": 3, "caption": "Fig. 1 | </script><script>alert(1)</script>",
    }]))
    html = review_queue.render_html(review_queue.build_queue(images=False))
    assert "<\\/script>" in html
    assert html.count("</script>") == 1


def test_the_page_offers_both_themes(wiki):
    html = review_queue.render_html(review_queue.build_queue(images=False))
    assert "prefers-color-scheme:dark" in html
    assert "--bg:" in html


# ---------- CLI ----------

def test_cli_writes_a_page_and_reports_the_queue(wiki, capsys):
    _cached_png(wiki, "abc123", 3)
    out = wiki / "output" / "inspect" / "visuals.html"
    assert cli.main([]) == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in text
    printed = capsys.readouterr().out
    assert "1 draft(s) held" in printed and "1 ready to judge" in printed


def test_cli_accepts_the_positional_source(wiki):
    assert cli.main(["visuals"]) == 0


def test_cli_rejects_an_unknown_source(wiki, capsys):
    assert cli.main(["impact"]) == 1
    assert "unknown review source" in capsys.readouterr().err


def test_cli_rejects_a_negative_budget(wiki, capsys):
    assert cli.main(["--max-bytes", "-1"]) == 1
    assert "--max-bytes" in capsys.readouterr().err


def test_cli_json_mode_pins_the_contract(wiki, capsys):
    assert cli.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"schema_version", "meta", "papers"}
    assert payload["schema_version"] == review_queue.QUEUE_SCHEMA_VERSION
    assert set(payload["meta"]) == {
        "n_papers", "n_ready", "n_items", "n_blocking", "n_blocking_with_reading",
        "n_corroborated", "n_images", "n_images_omitted", "n_unreadable_receipts",
        "include_resolved", "all_attempts", "images",
    }
    paper = payload["papers"][0]
    assert set(paper) == {
        "stem", "title", "attempt_id", "status", "selection_status",
        "results_source", "error", "receipt_path", "draft_path", "source_pdf",
        "gate_reasons", "unselected", "model", "input_tokens", "output_tokens",
        "request_ms", "estimated_cost_usd", "promoted", "ready", "n_blocking",
        "items",
    }
    assert set(paper["items"][0]) >= {
        "ref_id", "label", "page", "priority", "reason", "reason_text", "question",
        "caption", "blocking", "has_reading", "observation", "corroborated",
    }


def test_cli_json_omits_the_data_uri_when_there_is_no_image(wiki, capsys):
    assert cli.main(["--json", "--no-images"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "image" not in payload["papers"][0]["items"][0]


def test_cli_renders_an_empty_queue_and_exits_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "wiki").mkdir()
    assert cli.main([]) == 0
    printed = capsys.readouterr().out
    assert "nothing pending" in printed
    html = (tmp_path / "output" / "inspect" / "visuals.html").read_text(encoding="utf-8")
    assert "Nothing is waiting on you" in html


def test_cli_needs_a_wiki(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "wiki_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_warns_when_images_were_dropped(wiki, capsys):
    _cached_png(wiki, "abc123", 3)
    assert cli.main(["--max-bytes", "1"]) == 0
    assert "omitted to stay under" in capsys.readouterr().out


def test_no_model_is_ever_called(wiki, monkeypatch):
    from researchwiki.agents import llm
    monkeypatch.setattr(llm, "call", lambda **kw: pytest.fail("inspect must not call a model"))
    assert cli.main([]) == 0
