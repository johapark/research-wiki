"""Shared state + exception types for the ingest agent state machine.

Extracted from `runner.py` to keep that file focused on orchestration.
The Context dataclass carries every piece of mutable state passed
through the 18-phase pipeline; the two exception types are how reconcile
signals failures the CLI handler should turn into actionable hints.

These types are imported by `runner.py` and the phase modules in
`agents/phases/`. The phases themselves take their inputs as parameters
rather than reaching into Context — Context is the runner's working
memory, not a service locator. When a phase needs Context for
persistence (writing `ingest_iterations` rows), it accepts `ctx` as an
explicit parameter alongside the values it operates on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import phases


class ReconcileFailed(RuntimeError):
    """Raised when the reconcile phase cannot derive a stem.

    Triggered when the metadata cascade — overrides → LLM extractor → DOI
    detection (incl. URL→DOI hunt) → S2 / Crossref lookup → text-scan
    fallbacks — produces no `(title, year, first-author)` triple. Without
    those three, `derive_stem` cannot run, and proceeding through the
    remaining phases would burn tokens on a draft we cannot file.

    Carries the providers tried and which fields ended up null so the CLI
    handler can surface the right `--doi`/`--year`/`--title` hint without
    a stack trace.
    """

    def __init__(self, *, sources: list[str], missing: list[str]) -> None:
        self.sources = list(sources)
        self.missing = list(missing)
        super().__init__(
            f"reconcile could not derive a stem; missing={missing}; "
            f"sources tried={sources}"
        )


class StemRenameRefused(RuntimeError):
    """Raised when reconcile finds a prior page (DOI match) at a stem that
    differs from the newly derived stem. Aborts before any state mutation
    so the prior page and PDF stay intact. Pass --allow-rename to opt in.
    """

    def __init__(self, *, prior_stem: str, new_stem: str) -> None:
        self.prior_stem = prior_stem
        self.new_stem = new_stem
        super().__init__(
            f"reconcile found prior page at stem '{prior_stem}' but new "
            f"derived stem is '{new_stem}'."
        )


@dataclass
class Context:
    """Mutable state passed through every phase of one ingest attempt."""
    attempt_id: str
    pdf_path: Path
    pdf_filename: str
    use_stub: bool = False
    use_semantic: bool = True
    verify_claim_entailment: bool = False  # opt-in per-claim entailment veto (grade.support)
    max_evolve: int = 1
    promote_mode: str = "auto"           # 'auto' | 'always' | 'never' (force-sandbox)
    max_debug: int = 1                   # max DEBUG repair passes after a structural gate fail
    iteration: int = 0
    sandbox_dir: Path = field(default_factory=lambda: Path(".agent-output"))
    doi_override: str | None = None      # bypass in-text DOI detection (manual recovery)
    title_override: str | None = None    # bypass title extraction (manual recovery)
    year_override: int | None = None     # bypass year extraction — needed for fresh preprints
    authors_override: list[str] | None = None  # bypass author extraction (manual recovery)
    author_prompt_override: str | None = None  # path to custom author system prompt (eval A/B)
    supplementary: list[Path] | None = None  # paths to supp files staged after promote
    use_llm_reconcile: bool = True            # default-on after R3 (--no-llm-reconcile to opt out)
    allow_rename: bool = False                # opt-in: allow committing a stem rename when reconcile finds a prior page at a different stem

    # Filled by phases (None until set):
    paper_stem: str | None = None
    metadata: dict | None = None        # reconcile output: {title, year, doi, venue, ...}
    sections: dict | None = None        # extract output: {introduction, methods, results, discussion, ...} (each capped at 4000 chars)
    pdf_full_text: str | None = None    # extract output: full PDF text, no cap — for keyword extraction's wider sampling window
    claims_count: int | None = None     # extract output: number of claims parsed from PDF
    target_claims: object | None = None # L3 output: TargetClaimsOutput; None when phase skipped or failed
    crosslink_candidates: list = field(default_factory=list)   # list[phases.CrosslinkCandidate]
    drafts: list[phases.Draft] = field(default_factory=list)
    winner: phases.Draft | None = None
    committed_path: Path | None = None

    def next_iter(self) -> int:
        self.iteration += 1
        return self.iteration
