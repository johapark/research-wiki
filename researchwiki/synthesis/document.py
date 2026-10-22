"""Structured prose; only the renderer may create Markdown and source links."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..answers.contracts import Statement
from ..evidence import validate_stem
from ..operations._wire import WireModel, nonempty

HEADINGS = ("Short answer", "Background", "Organizing framework",
            "Cross-cutting insights", "Tensions / open questions", "Outlook")


def authored_draft(data):
    draft = Draft.from_dict(data)
    units = [data['hook'], *(p for s in data['sections'] for p in s['paragraphs']),
             *(p for t in data['themes'] for p in (t['title'], *t['paragraphs']))]
    # Persisted clauses include check fields; the author contract never does.
    # Enforce this at the API boundary too, where transport schemas are hints.
    if any(set(unit) != {'text', 'evidence_ids'} for unit in units):
        raise ValueError('author may supply prose/evidence IDs only, not validation results')
    return draft


@dataclass(frozen=True, kw_only=True)
class Clause(WireModel):
    text: str
    evidence_ids: tuple[str, ...]
    support: Literal["supported", "partial", "unsupported", "not_checked"] = "not_checked"
    anchor_support: Literal["supported", "partial", "unsupported", "not_checked", "not_applicable"] = "not_checked"
    numeric_drift: tuple[str, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        Statement(text=self.text, evidence_id="e0", support=self.support,
                  anchor_support=self.anchor_support, numeric_drift=self.numeric_drift)
        if not 1 <= len(self.evidence_ids) <= 8 or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("each synthesis unit needs one to eight unique evidence IDs")

    @property
    def passed(self):
        return (self.support == "supported" and self.anchor_support in ("supported", "not_applicable")
                and not self.numeric_drift)


@dataclass(frozen=True, kw_only=True)
class Section(WireModel):
    heading: str
    paragraphs: tuple[Clause, ...]

    def __post_init__(self):
        super().__post_init__()
        if self.heading not in HEADINGS or not 1 <= len(self.paragraphs) <= 4:
            raise ValueError("invalid or empty synthesis section")


@dataclass(frozen=True, kw_only=True)
class Theme(WireModel):
    title: Clause
    paragraphs: tuple[Clause, ...]

    def __post_init__(self):
        super().__post_init__()
        if len(self.title.text) > 80 or not 1 <= len(self.paragraphs) <= 5:
            raise ValueError("invalid synthesis theme")


@dataclass(frozen=True, kw_only=True)
class Decision(WireModel):
    stem: str
    decision: Literal["cite", "exclude"]
    reason: str

    def __post_init__(self):
        super().__post_init__()
        validate_stem(self.stem)
        nonempty(self.reason, "coverage reason")
        if len(self.reason) > 600:
            raise ValueError("coverage reason is too long")


@dataclass(frozen=True, kw_only=True)
class Draft(WireModel):
    hook: Clause
    sections: tuple[Section, ...]
    themes: tuple[Theme, ...]
    coverage: tuple[Decision, ...]

    def __post_init__(self):
        super().__post_init__()
        if tuple(s.heading for s in self.sections) != HEADINGS:
            raise ValueError("synthesis must contain the complete ordered section structure")
        if not 1 <= len(self.themes) <= 4 or len({t.title.text for t in self.themes}) != len(self.themes):
            raise ValueError("synthesis requires unique idea-based themes")
        if len(self.hook.text) > 1000 or len(self.units()) > 40:
            raise ValueError("synthesis exceeds bounded composition limits")

    def units(self):
        return (self.hook, *(p for s in self.sections for p in s.paragraphs),
                *(p for t in self.themes for p in (t.title, *t.paragraphs)))

    def with_units(self, units):
        iterator = iter(units)
        hook = next(iterator)
        sections = tuple(replace(s, paragraphs=tuple(next(iterator) for _ in s.paragraphs)) for s in self.sections)
        themes = tuple(replace(t, title=next(iterator), paragraphs=tuple(next(iterator) for _ in t.paragraphs))
                       for t in self.themes)
        if next(iterator, None) is not None:
            raise ValueError("unexpected synthesis units")
        return replace(self, hook=hook, sections=sections, themes=themes)

    def validate_sources(self, evidence, scope):
        by_id = {e.evidence_id: e for e in evidence}
        if len(by_id) != len(evidence) or any(e.stem not in scope.stems for e in evidence):
            raise ValueError("duplicate or out-of-scope synthesis evidence")
        for unit in self.units():
            if any(eid not in by_id for eid in unit.evidence_ids):
                raise ValueError("synthesis cites unknown evidence")
        used = {by_id[eid].stem for unit in self.units() for eid in unit.evidence_ids}
        decisions = {d.stem: d for d in self.coverage}
        if len(decisions) != len(self.coverage) or set(decisions) != scope.stems:
            raise ValueError("coverage ledger must decide every selected paper exactly once")
        if {d.stem for d in self.coverage if d.decision == "cite"} != used:
            raise ValueError("coverage decisions disagree with citations")
        if len(used) < 2:
            raise ValueError("synthesis must cite at least two distinct papers")
        cross = next(s for s in self.sections if s.heading == "Cross-cutting insights")
        if not any(len({by_id[eid].stem for eid in p.evidence_ids}) >= 2 for p in cross.paragraphs):
            raise ValueError("cross-cutting insights must combine evidence from multiple papers")
        for theme in self.themes:
            if len({by_id[eid].stem for p in theme.paragraphs for eid in p.evidence_ids}) < 2:
                raise ValueError("each Findings theme must connect at least two papers")
