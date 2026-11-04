"""Bounded PDF evidence for an existing repair call, with broad fallback.

Only repairs query the index already built by grading. Retrieval nominates
passages; it never certifies a claim or changes grading/promotion thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import time

from ...log import log
from ...pdf.sections import assess_section_health, stratified_text_sample

MAX_QUERIES = 4
CHUNKS_PER_ISSUE = 2
PACKET_CHARS = 12_000
CONTEXT_CHARS = 4_000


@dataclass
class RevisionEvidence:
    text: str
    metrics: dict = field(default_factory=dict)


def broad_revision_evidence(sections: dict, pdf_full_text: str | None) -> str:
    """Keep the existing section-health-aware revision fallback unchanged."""
    parts = []
    for key, label, budget in (
        ("introduction", "Introduction", 1500),
        ("methods", "Methods", 2500),
        ("results", "Results", 2500),
        ("discussion", "Discussion", 1500),
        ("figure_captions", "Figure / Table captions", 4000),
        ("extended_data", "Extended Data captions", 4000),
    ):
        excerpt = (sections.get(key) or "").strip()
        if excerpt:
            parts.extend([f"## {label}", excerpt[:budget], ""])
    if pdf_full_text:
        health = assess_section_health(pdf_full_text, sections)
        body = (pdf_full_text[:30_000] if health.healthy
                else stratified_text_sample(pdf_full_text, 30_000))
        label = ("Full PDF text (supplementary context)" if health.healthy
                 else "Full PDF text (document-stratified fallback)")
        parts.extend([f"## {label}", body, ""])
    return "\n".join(parts).rstrip()


def repair_issues(gaps, weak_claims) -> list[tuple[str, str]]:
    """Critical omissions first, then other gaps and existing weak claims."""
    ordered = sorted(gaps, key=lambda g: g.get("importance") != "critical")
    issues = [
        (f"gap {g.get('id', '?')}; {g.get('location') or g.get('axis', '?')}",
         g.get("text") or "") for g in ordered
    ]
    issues.extend((f"claim {c.section}#{c.position}", c.text) for c in weak_claims)
    return issues


def _usable_hit(hit, query: str) -> bool:
    """Reject empty/zero hits and matches carried only by ordinary filler.

    This cheap lexical check is deliberately not an entailment judgment.
    Incorrect quantities in a weak claim need not appear in its correction.
    """
    stop = {"the", "and", "for", "with", "that", "this", "from", "were", "was",
            "are", "has", "have", "but", "not", "into", "its", "their"}
    terms = set(re.findall(r"\b[\w-]{3,}\b", query.lower())) - stop
    words = set(re.findall(r"\b[\w-]{3,}\b", hit.text.lower())) - stop
    return (hit.score > 0 and bool(hit.text.strip()) and bool(terms)
            and len(terms & words) >= min(2, len(terms)))


def _targeted_packet(stem, pdf_path, issues, metrics) -> str | None:
    from ...index.pdf_chunks import query_pdf

    if not stem or not issues:
        metrics["evidence_fallback_reason"] = "no-specific-issues"
        return None
    if len(issues) > MAX_QUERIES:
        metrics["evidence_fallback_reason"] = "issue-limit"
        return None
    blocks = ["## Issue-specific PDF evidence (candidates; follow the repair notes)"]
    seen = set()
    for label, query in issues:
        if not query.strip():
            metrics["evidence_fallback_reason"] = "empty-issue"
            return None
        metrics["evidence_queries"] += 1
        hits = query_pdf(stem, query, topk=CHUNKS_PER_ISSUE, pdf_path=pdf_path)
        candidates = [h for h in hits[:CHUNKS_PER_ISSUE] if _usable_hit(h, query)]
        if not candidates:
            metrics["evidence_fallback_reason"] = "no-usable-hit"
            return None
        # Issue text and references count against the packet budget too.
        header = f"### {label}\nIssue: {query}"
        if len("\n\n".join([*blocks, header])) > PACKET_CHARS:
            metrics["evidence_fallback_reason"] = "packet-limit"
            return None
        blocks.append(header)
        covered = False
        for hit in candidates:
            ref = f"<Chunk {hit.chunk_id}; {hit.provenance() or 'provenance unavailable'}>"
            block = (f"Evidence: Chunk {hit.chunk_id} (included above)" if hit.chunk_id in seen
                     else f"{ref}\n{hit.text.strip()}")
            if len("\n\n".join([*blocks, block])) > PACKET_CHARS:
                continue
            blocks.append(block)
            seen.add(hit.chunk_id)
            covered = True
        if not covered:
            metrics["evidence_fallback_reason"] = "packet-limit"
            return None
    metrics["evidence_chunks"] = len(seen)
    return "\n\n".join(blocks)


def build_repair_evidence(
    *, sections: dict, pdf_full_text: str | None,
    issues: list[tuple[str, str]], stem: str | None = None,
    pdf_path: Path | str | None = None, broad_required: bool = False,
) -> RevisionEvidence:
    """Use a compact packet only when every repair issue has usable evidence."""
    started = time.monotonic()
    metrics = {"evidence_queries": 0, "evidence_chunks": 0,
               "evidence_packet_chars": 0, "evidence_context_chars": 0}
    packet = None
    if broad_required:
        metrics["evidence_fallback_reason"] = "structural-expansion"
    else:
        try:
            packet = _targeted_packet(stem, pdf_path, issues, metrics)
        except Exception as exc:
            log(f"revision evidence unavailable ({type(exc).__name__}); using broad PDF context", tag="agent")
            metrics["evidence_fallback_reason"] = "retrieval-error"
    if packet is None:
        text = broad_revision_evidence(sections, pdf_full_text)
        metrics["evidence_route"] = "broad"
    else:
        label = "## Compact supplementary context\n"
        source = pdf_full_text or "\n\n".join(str(v) for v in sections.values() if v)
        # The sampler adds stratum labels; reserve space and bound the result.
        context = (label + stratified_text_sample(source, CONTEXT_CHARS - 500))[:CONTEXT_CHARS - 2]
        text = packet + "\n\n" + context
        metrics.update(evidence_route="targeted", evidence_packet_chars=len(packet),
                       evidence_context_chars=len(context) + 2)
    metrics["evidence_chars"] = len(text)
    metrics["evidence_ms"] = round((time.monotonic() - started) * 1000, 3)
    return RevisionEvidence(text, metrics)
