"""Reject stale claims without repairing state during a query."""

from pathlib import Path
import json

from ..grade.parser import parse_claims
from ..db.pdf_hashes import PdfHashCache
from ..paths import papers_dir
from ..wiki import read_page
from ..grade.checks import entailment_check


def current_rows(conn, rows):
    """Keep claims whose source page still contains the indexed claim text."""
    hashes = PdfHashCache(conn)
    try:
        return _current_rows(conn, rows, hashes)
    finally:
        hashes.flush()


def _current_rows(conn, rows, hashes):
    sources = {}
    kept = []
    for row in rows:
        stem = row["paper_stem"]
        if stem not in sources:
            paper = conn.execute("SELECT page_path FROM papers WHERE stem = ?", (stem,)).fetchone()
            page = read_page(Path(paper["page_path"])) if paper and paper["page_path"] else None
            pdf = papers_dir() / f"{stem}.pdf"
            sources[stem] = ({
                (c.section, c.position): c.text for c in parse_claims(page)
            } if page and page.fm and page.page_type == "paper" else {},
                hashes.sha256(pdf) if pdf.is_file() else None)
        claims, digest = sources[stem]
        if claims.get((row["section"], row["position"])) == row["text"]:
            item = dict(row)
            current = digest is not None and item.get("source_pdf_sha256") == digest
            item["source_current"] = current
            graded = current and item.get("last_graded_at") is not None
            item["checks"] = {
                "source_freshness": "current" if current else (
                    "stale" if item.get("last_graded_at") is not None else "unverified"),
                "retrieval": "completed" if graded else "not_checked",
                "numeric": ("failed" if json.loads(item.get("numeric_unmatched") or "[]")
                            else "passed") if graded and item.get("bm25_top1_chunk_id") is not None
                           else "not_checked",
                "entailment": entailment_check({}),
            }
            if not current:
                for key in ("last_graded_at", "semantic_score", "bm25_top1", "bm25_top1_chunk_id",
                            "supporting_text", "supporting_provenance", "numeric_unmatched"):
                    item[key] = None
            kept.append(item)
    return kept
