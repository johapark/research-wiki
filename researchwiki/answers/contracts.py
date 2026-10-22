"""Strict, portable answer receipts. Verification is a recorded check, not a guarantee."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ..evidence import EvidenceScope, validate_stem
from ..operations._wire import WireModel, nonempty
from ..provenance import specific_author_model


@dataclass(frozen=True, kw_only=True)
class Evidence(WireModel):
    evidence_id: str
    stem: str
    citation: str
    claim_text: str
    passage: str
    page_start: int
    page_end: int

    def __post_init__(self):
        super().__post_init__()
        validate_stem(self.stem)
        if not re.fullmatch(r"e[0-9]+", self.evidence_id):
            raise ValueError("invalid evidence ID")
        allowed = rf"\[\[{re.escape(self.stem)}(?:#kc-[0-9a-f]+)?\]\]"
        if not re.fullmatch(allowed, self.citation):
            raise ValueError("citation must belong to its source")
        if ("#" in self.citation) != bool(self.claim_text):
            raise ValueError("claim citations require the cited claim text")
        nonempty(self.passage, "passage")
        if len(self.passage) > 6000 or len(self.claim_text) > 4000:
            raise ValueError("answer evidence exceeds passage limits")
        if not 1 <= self.page_start <= self.page_end <= 80:
            raise ValueError("invalid PDF page range")


@dataclass(frozen=True, kw_only=True)
class Statement(WireModel):
    text: str
    evidence_id: str
    support: Literal["supported", "partial", "unsupported", "not_checked"]
    anchor_support: Literal["supported", "partial", "unsupported", "not_applicable", "not_checked"]
    numeric_drift: tuple[str, ...]

    def __post_init__(self):
        super().__post_init__()
        nonempty(self.text, "statement")
        # The renderer owns citations and Markdown. Model text is plain prose.
        if (len(self.text) > 1500 or any(c in self.text for c in "\r\n[]<>`*#")
                or re.search(r"https?://|www\.", self.text, re.I)):
            raise ValueError("statement must be bounded plain text without links or markup")


@dataclass(frozen=True, kw_only=True)
class ModelRun(WireModel):
    phase: Literal["answer", "answer_judge"]
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int

    def __post_init__(self):
        super().__post_init__()
        if not specific_author_model(self.model) or self.model.startswith("stub:") or min(self.input_tokens, self.output_tokens,
                                                self.cache_read_tokens, self.cache_write_tokens) < 0:
            raise ValueError("answer verification requires attributable, non-stub model runs")


@dataclass(frozen=True)
class AnswerReceipt:
    question: str
    scope: EvidenceScope
    evidence: tuple[Evidence, ...]
    statements: tuple[Statement, ...]
    models: tuple[ModelRun, ...]
    status: str
    reason: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported answer receipt version")
        if not isinstance(self.question, str) or not 1 <= len(self.question.strip()) <= 2000:
            raise ValueError("question must contain 1–2000 characters")
        if not isinstance(self.scope, EvidenceScope) or (self.scope.stems and not self.scope.source_revisions):
            raise ValueError("answer requires a complete source snapshot")
        if self.status not in {"verified", "rejected", "insufficient_evidence"}:
            raise ValueError("invalid answer status")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("answer outcome requires a reason")
        if not isinstance(self.created_at, str):
            raise ValueError("invalid answer timestamp")
        stamp = datetime.fromisoformat(self.created_at)
        if stamp.utcoffset() is None:
            raise ValueError("answer timestamp requires a timezone")
        for items, cls in ((self.evidence, Evidence), (self.statements, Statement), (self.models, ModelRun)):
            if not isinstance(items, tuple) or any(not isinstance(item, cls) for item in items):
                raise ValueError("invalid answer receipt records")
        if len(self.evidence) > 12 or len(self.statements) > 8 or len(self.models) > 2:
            raise ValueError("answer receipt exceeds bounded workflow limits")
        by_id = {e.evidence_id: e for e in self.evidence}
        if len(by_id) != len(self.evidence) or any(e.stem not in self.scope.stems for e in self.evidence):
            raise ValueError("duplicate or out-of-scope evidence")
        for statement in self.statements:
            if statement.evidence_id not in by_id:
                raise ValueError("statement cites unknown evidence")
            if bool(by_id[statement.evidence_id].claim_text) == (statement.anchor_support == "not_applicable"):
                raise ValueError("anchor check does not match citation")
        if self.status == "verified":
            if not self.statements or [m.phase for m in self.models] != ["answer", "answer_judge"]:
                raise ValueError("verified answers require statements and both model runs")
            if any(s.support != "supported" or s.anchor_support not in ("supported", "not_applicable")
                   or s.numeric_drift for s in self.statements):
                raise ValueError("verified answer has incomplete or failing checks")
        if self.status == "insufficient_evidence" and self.statements:
            raise ValueError("insufficient evidence cannot contain an answer")

    def to_dict(self):
        return {"schema_version": self.schema_version, "question": self.question,
                "scope": {**self.scope.to_dict(), "fingerprint": self.scope.fingerprint},
                "evidence": [e.to_dict() for e in self.evidence],
                "statements": [s.to_dict() for s in self.statements],
                "models": [m.to_dict() for m in self.models],
                "status": self.status, "reason": self.reason, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data):
        fields = {"schema_version", "question", "scope", "evidence", "statements", "models", "status", "reason", "created_at"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("invalid answer receipt fields")
        values = dict(data)
        values["scope"] = EvidenceScope.from_dict(data["scope"])
        for field, model in (("evidence", Evidence), ("statements", Statement), ("models", ModelRun)):
            if not isinstance(data[field], list):
                raise ValueError(f"{field} must be an array")
            values[field] = tuple(model.from_dict(item) for item in data[field])
        return cls(**values)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def markdown(self):
        if self.status != "verified":
            return f"No verified answer: {self.reason}\n"
        by_id = {e.evidence_id: e for e in self.evidence}
        lines = []
        for statement in self.statements:
            evidence = by_id[statement.evidence_id]
            lines.append(f"- {statement.text} {evidence.citation} (PDF pp. {evidence.page_start}–{evidence.page_end})")
        return "\n".join(lines) + "\n"
