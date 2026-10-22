"""Versioned synthesis provenance; a verified flag cannot omit any gate."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Literal

from ..answers.contracts import Evidence, Statement
from ..evidence import EvidenceScope, validate_stem
from ..grade.generated import ModelProvenance
from ..operations._wire import WireModel
from .document import Draft


@dataclass(frozen=True, kw_only=True)
class Source(WireModel):
    stem: str
    key: str

    def __post_init__(self):
        super().__post_init__()
        validate_stem(self.stem)
        if not re.fullmatch(r"[a-zA-Z0-9_-]+/" + re.escape(self.stem), self.key):
            raise ValueError("invalid synthesis source key")


@dataclass(frozen=True, kw_only=True)
class ModelRun(ModelProvenance):
    phase: Literal["synthesis_author", "synthesis_support"]


@dataclass(frozen=True, kw_only=True)
class Gates(WireModel):
    grounding_total: int
    ungrounded: int
    fidelity_graded: int
    misattributed: int
    unresolved: int
    weak: int
    dangling_anchors: int
    contract_violations: tuple[str, ...]
    coverage_omissions: tuple[str, ...]

    def __post_init__(self):
        super().__post_init__()
        if min(self.grounding_total, self.ungrounded, self.fidelity_graded,
               self.misattributed, self.unresolved, self.weak, self.dangling_anchors) < 0:
            raise ValueError("invalid synthesis gate counts")

    @property
    def passed(self):
        return (self.grounding_total > 0 and self.fidelity_graded > 0 and self.ungrounded == 0
                and self.misattributed == 0 and self.unresolved == 0 and self.dangling_anchors == 0
                and not self.contract_violations)


@dataclass(frozen=True, kw_only=True)
class Review(WireModel):
    theme_organization: bool
    cross_paper_reasoning: bool
    coverage_rationale: bool
    reason: str

    def __post_init__(self):
        super().__post_init__()
        if not self.reason.strip() or len(self.reason) > 1200:
            raise ValueError("synthesis review requires a bounded explanation")

    @property
    def passed(self):
        return self.theme_organization and self.cross_paper_reasoning and self.coverage_rationale


@dataclass(frozen=True)
class SynthesisReceipt:
    title: str
    question: str
    scope: EvidenceScope
    sources: tuple[Source, ...]
    evidence: tuple[Evidence, ...]
    draft: Draft | None
    models: tuple[ModelRun, ...]
    gates: Gates | None
    review: Review | None
    status: str
    reason: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported synthesis receipt version")
        for value, limit in ((self.title, 180), (self.question, 1500)):
            if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
                raise ValueError("invalid synthesis title/question")
            Statement(text=value, evidence_id="e0", support="not_checked",
                      anchor_support="not_applicable", numeric_drift=())
            if any(c in value for c in "\r\n[]<>`*#"):
                raise ValueError("synthesis title/question must be plain text")
        if not isinstance(self.scope, EvidenceScope) or (self.scope.stems and not self.scope.source_revisions):
            raise ValueError("synthesis requires a complete source snapshot")
        if len(self.scope.stems) > 8 or len(self.evidence) > 24 or len(self.models) > 2:
            raise ValueError("synthesis exceeds bounded workflow limits; narrow the selection")
        if not isinstance(self.reason, str) or not self.reason or not isinstance(self.created_at, str):
            raise ValueError("synthesis requires a reason and timestamp")
        if datetime.fromisoformat(self.created_at).utcoffset() is None:
            raise ValueError("synthesis timestamp requires timezone")
        for items, cls in ((self.sources, Source), (self.evidence, Evidence), (self.models, ModelRun)):
            if not isinstance(items, tuple) or any(not isinstance(item, cls) for item in items):
                raise ValueError("invalid synthesis receipt records")
        if {s.stem for s in self.sources} != self.scope.stems or len(self.sources) != len(self.scope.stems):
            raise ValueError("source keys must cover the exact selection")
        if len({e.evidence_id for e in self.evidence}) != len(self.evidence) or any(e.stem not in self.scope.stems for e in self.evidence):
            raise ValueError("duplicate or out-of-scope synthesis evidence")
        if self.status not in ("verified", "rejected", "insufficient_evidence"):
            raise ValueError("invalid synthesis status")
        if self.draft is not None:
            if not isinstance(self.draft, Draft):
                raise ValueError("invalid synthesis draft")
            self.draft.validate_sources(self.evidence, self.scope)
        if self.gates is not None and not isinstance(self.gates, Gates):
            raise ValueError("invalid synthesis gates")
        if self.review is not None and not isinstance(self.review, Review):
            raise ValueError("invalid synthesis review")
        if self.status == "verified":
            if (self.draft is None or self.gates is None or not self.gates.passed
                    or self.review is None or not self.review.passed
                    or [m.phase for m in self.models] != ["synthesis_author", "synthesis_support"]
                    or any(not unit.passed for unit in self.draft.units())):
                raise ValueError("verified synthesis requires every prose, structure, and fidelity check")
            excluded = {d.stem for d in self.draft.coverage if d.decision == "exclude"}
            if not set(self.gates.coverage_omissions).issubset(excluded):
                raise ValueError("unreviewed coverage omissions")
            by_id = {e.evidence_id: e for e in self.evidence}
            for unit in self.draft.units():
                anchored = any(by_id[eid].claim_text for eid in unit.evidence_ids)
                if unit.anchor_support != ('supported' if anchored else 'not_applicable'):
                    raise ValueError('synthesis anchor checks do not match its cited evidence')
        if self.status == "insufficient_evidence" and self.draft is not None:
            raise ValueError("insufficient evidence cannot contain a synthesis")

    def to_dict(self):
        return {"schema_version": 1, "title": self.title, "question": self.question,
                "scope": {**self.scope.to_dict(), "fingerprint": self.scope.fingerprint},
                "sources": [s.to_dict() for s in self.sources], "evidence": [e.to_dict() for e in self.evidence],
                "draft": self.draft.to_dict() if self.draft else None,
                "models": [m.to_dict() for m in self.models], "gates": self.gates.to_dict() if self.gates else None,
                "review": self.review.to_dict() if self.review else None,
                "status": self.status, "reason": self.reason, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data):
        fields = {"schema_version", "title", "question", "scope", "sources", "evidence", "draft", "models",
                  "gates", "review", "status", "reason", "created_at"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("invalid synthesis receipt fields")
        values = dict(data)
        values["scope"] = EvidenceScope.from_dict(data["scope"])
        for name, model in (("sources", Source), ("evidence", Evidence), ("models", ModelRun)):
            if not isinstance(data[name], list):
                raise ValueError(f"{name} must be an array")
            values[name] = tuple(model.from_dict(item) for item in data[name])
        for name, model in (("draft", Draft), ("gates", Gates), ("review", Review)):
            values[name] = model.from_dict(data[name]) if data[name] is not None else None
        return cls(**values)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def markdown(self):
        if self.status != "verified":
            return f"No verified synthesis: {self.reason}\n"
        from .render import render
        return render(self)
