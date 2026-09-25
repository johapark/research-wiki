"""Portable review receipts. Accepted Markdown remains the canonical ledger."""

import hashlib
import json
from pathlib import Path

from .fsatomic import write_text_atomic
from .proposal_generation import paper_categories, validate_candidates
from .proposals import _proposals_on_disk, create_proposal
from .provenance import specific_author_model


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_preview(packet: dict, proposals: list[dict], usage: dict) -> Path:
    receipt = {"version": 1, "packet": packet, "proposals": proposals, "usage": usage}
    validate_receipt(receipt)
    path = Path(".proposal-cache") / f"{_digest(receipt)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return path


def validate_receipt(receipt: dict, *, check_wiki: bool = False) -> None:
    """Structural receipt check; `check_wiki` also verifies stems and categories.

    A receipt can be written by hand in the chat-authored workflow, so its
    evidence categories cannot be taken on trust: the cross-category rule reads
    them, and a hand-edited receipt could otherwise label same-category
    evidence as a transfer. Acceptance therefore checks each stem against the
    wiki. Saving a freshly generated preview skips that walk, since its packet
    came from the wiki moments earlier.
    """
    if not isinstance(receipt, dict) or receipt.get("version") != 1:
        raise ValueError("unsupported preview receipt version")
    packet = receipt.get("packet")
    if not isinstance(packet, dict) or not isinstance(packet.get("evidence"), list):
        raise ValueError("preview needs an evidence packet")
    for key in ("topic", "target_category"):
        if not isinstance(packet.get(key), str) or not packet[key].strip():
            raise ValueError(f"preview packet needs {key}")
    if packet.get("planning_pending"):
        raise ValueError(
            "preview packet is an unplanned cross-category preparation; add a search "
            "plan (`proposals generate --search-plan PLAN.json --prepare-only`) first"
        )
    ids = set()
    for item in packet["evidence"]:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("id", "paper_stem", "claim_slug", "text", "citation", "category")
        ):
            raise ValueError("preview contains malformed evidence (each item needs "
                             "id, paper_stem, claim_slug, text, citation, and category)")
        if item["id"] in ids or item["citation"] != f"[[{item['paper_stem']}#{item['claim_slug']}]]":
            raise ValueError("preview evidence ids or citation mappings are invalid")
        ids.add(item["id"])
    usage = receipt.get("usage")
    model = usage.get("model") if isinstance(usage, dict) else None
    if not specific_author_model(model):
        raise ValueError(
            "preview must record the exact author model (a placeholder or family "
            f"alias such as 'gpt-5.6' is not provenance): {model!r}"
        )
    if check_wiki:
        categories = paper_categories()
        for item in packet["evidence"]:
            actual = categories.get(item["paper_stem"])
            if actual is None:
                raise ValueError(f"preview cites a paper not in this wiki: {item['paper_stem']}")
            if actual != item["category"]:
                raise ValueError(
                    f"preview labels {item['paper_stem']} as {item['category']!r}, "
                    f"but the wiki files it under {actual!r}"
                )
    validate_candidates(receipt.get("proposals"), packet)


class PartialAccept(ValueError):
    """Some selected entries were written before a later one failed."""

    def __init__(self, written: list, cause: Exception) -> None:
        self.written = written
        self.cause = cause
        super().__init__(
            f"{cause} (after writing {len(written)} proposal(s): "
            + ", ".join(record.proposal_id for record in written) + ")"
        )


def accept_preview(path: Path, selections: list[int]) -> list:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"preview is not valid JSON: {path}: {exc}") from exc
    validate_receipt(receipt, check_wiki=True)
    proposals, packet = receipt["proposals"], receipt["packet"]
    if not selections or any(i < 1 or i > len(proposals) for i in selections):
        raise ValueError("select proposal numbers from the preview (starting at 1)")
    # One directory scan for the whole accept, not one per selected entry.
    existing = {record.proposal_id: record for record in _proposals_on_disk()}
    records = []
    for index in dict.fromkeys(selections):
        proposal = proposals[index - 1]
        acceptance_id = hashlib.sha256(f"{_digest(receipt)}:{index}".encode()).hexdigest()[:12]
        try:
            records.append(create_proposal(
                proposal, evidence_items=packet["evidence"], topic_seed=packet["topic"],
                target_category=packet["target_category"],
                author_model=receipt["usage"]["model"],
                parent_proposal=proposal.get("parent_proposal") or "",
                acceptance_id=acceptance_id, existing=existing,
            ))
        except (ValueError, OSError) as exc:
            if records:
                raise PartialAccept(records, exc) from exc
            raise
    return records
