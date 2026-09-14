"""Portable review receipts. Accepted Markdown remains the canonical ledger."""

import hashlib
import json
from pathlib import Path

from .fsatomic import write_text_atomic
from .proposal_generation import validate_candidates
from .proposals import create_proposal


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_preview(packet: dict, proposals: list[dict], usage: dict) -> Path:
    receipt = {"version": 1, "packet": packet, "proposals": proposals, "usage": usage}
    validate_receipt(receipt)
    path = Path(".proposal-cache") / f"{_digest(receipt)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return path


def validate_receipt(receipt: dict) -> None:
    if not isinstance(receipt, dict) or receipt.get("version") != 1:
        raise ValueError("unsupported preview receipt version")
    packet = receipt.get("packet")
    if not isinstance(packet, dict) or not isinstance(packet.get("evidence"), list):
        raise ValueError("preview needs an evidence packet")
    for key in ("topic", "target_category"):
        if not isinstance(packet.get(key), str) or not packet[key].strip():
            raise ValueError(f"preview packet needs {key}")
    ids = set()
    for item in packet["evidence"]:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("id", "paper_stem", "claim_slug", "text", "citation")
        ):
            raise ValueError("preview contains malformed evidence")
        if item["id"] in ids or item["citation"] != f"[[{item['paper_stem']}#{item['claim_slug']}]]":
            raise ValueError("preview evidence ids or citation mappings are invalid")
        ids.add(item["id"])
    usage = receipt.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("model"), str) or not usage["model"].strip():
        raise ValueError("preview must record the exact author model")
    validate_candidates(receipt.get("proposals"), packet)


def accept_preview(path: Path, selections: list[int]) -> list:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    proposals, packet = receipt["proposals"], receipt["packet"]
    if not selections or any(i < 1 or i > len(proposals) for i in selections):
        raise ValueError("select proposal numbers from the preview (starting at 1)")
    records = []
    for index in dict.fromkeys(selections):
        proposal = proposals[index - 1]
        acceptance_id = hashlib.sha256(f"{_digest(receipt)}:{index}".encode()).hexdigest()[:12]
        records.append(create_proposal(
            proposal, evidence_items=packet["evidence"], topic_seed=packet["topic"],
            target_category=packet["target_category"], author_model=receipt["usage"]["model"],
            parent_proposal=proposal.get("parent_proposal") or "", acceptance_id=acceptance_id,
        ))
    return records
