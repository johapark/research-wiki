"""Strict source-backed evidence records for synthesis composition."""
from dataclasses import dataclass
import re

from ..evidence import validate_stem
from ..operations._wire import WireModel, nonempty


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
            raise ValueError("synthesis evidence exceeds passage limits")
        if not 1 <= self.page_start <= self.page_end <= 80:
            raise ValueError("invalid PDF page range")
