"""Read-only comparator search for a proposal before an editorial decision."""

import json
import re
from pathlib import Path

from .proposal_preview import validate_receipt
from .proposals import cited_evidence_stems, find_proposal
from .search import claim_query, format_claim_ref


def compare_methods(title: str, operation: str, cited: set[str]) -> dict:
    """Retrieve by the proposed operation, independently of the original packet."""
    query = f"{title} {operation}".strip()
    candidates = []
    seen = set()
    for hit in claim_query(query, k=80):
        stem = hit["paper_stem"]
        if stem in seen or not hit.get("claim_slug"):
            continue
        seen.add(stem)
        candidates.append({"citation": format_claim_ref(hit), "paper_stem": stem,
                           "text": hit["text"], "already_cited": stem in cited})
    # A packet's own papers can occupy every leading rank. Reserve space for
    # at least one outside comparator while retaining the best overall match.
    selected = candidates[:5]
    if selected and all(item["already_cited"] for item in selected):
        outside = next((item for item in candidates if not item["already_cited"]), None)
        if outside:
            selected[-1] = outside
    return {"query": query, "candidates": selected,
            "review_question": "What can this proposal measure or decide beyond the closest "
                               "comparator? Cite its claim and defer if the distinction is unclear."}


def compare_candidate(proposal: dict, evidence: list[dict], operation: str = "") -> dict:
    evidence_ids = {eid for row in proposal["evidence_connections"] for eid in row["evidence_ids"]}
    cited = {item["paper_stem"] for item in evidence if item["id"] in evidence_ids}
    query = operation or proposal.get("mechanism") or proposal["thesis"]
    return compare_methods(proposal["title"], query, cited)


def review(identifier: str, selection: int | None, operation: str = "") -> dict:
    if selection is not None:
        receipt = json.loads(Path(identifier).read_text(encoding="utf-8"))
        validate_receipt(receipt)
        if not 1 <= selection <= len(receipt["proposals"]):
            raise ValueError("selection is outside the preview")
        proposal = receipt["proposals"][selection - 1]
        return compare_candidate(proposal, receipt["packet"]["evidence"], operation)

    record = find_proposal(identifier)
    if record is None:
        raise ValueError(f"unknown proposal: {identifier}")
    body = record.path.read_text(encoding="utf-8")
    mechanism = re.search(r"^### Mechanism\s*\n(.*?)(?=^#{1,3} |\Z)", body, re.M | re.S)
    cited = cited_evidence_stems(body)
    return compare_methods(record.title, operation or (mechanism[1].strip() if mechanism else record.thesis), cited)
