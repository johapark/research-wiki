"""Phase functions for the ingest agent state machine.

Each phase is a pure function — it takes inputs, returns outputs. Persistence
to ingest_iterations happens in `runner.py` wrappers, never inside these
functions. This separation keeps phases unit-testable and the framework
auditable.

Phases (in order), by the function that implements each:
  reconcile_metadata : DOI / title / year / venue from PDF + S2
  extract_sections   : section + claim extractor (reuses tasks.ingest helpers)
  author             : LLM call producing a wiki-page draft
  grade_draft        : Phase 1 grader on the draft (BM25 + bi-encoder semantic)
  tournament         : pick the highest-scored draft (deterministic argmax)
  critic             : translate grader flags into actionable revision notes
  evolve             : revise the winning draft based on critic notes
  debug              : repair structural-gate failures (drift, KC underflow)
  persist_grades     : post-commit fidelity grading on the promoted page
  evolve_memory      : propose edits to neighbouring synthesis pages

The *phase-name strings* used for `ingest_iterations.role`, `phase=` model-config
keys, and log tags are unchanged ("reconcile", "extract", "grade", "commit", …) —
those are persisted/user-facing identifiers, deliberately decoupled from the
Python function names.

Files in this package mirror the phase grouping. Public names re-exported
here so callers can keep `from .phases import X` style imports.

**Naming convention — phase functions are `verb_object`, modules are topics.**
`reconcile.py` defines `reconcile_metadata()`, `grade.py` defines `grade_draft()`,
and so on. Follow it when adding a phase.

The rule exists so no re-exported name ever equals a submodule name. Previously
several did (`grade.py` exported `grade`), and the re-export shadowed the module —
so `phases.grade` was the function while `phases.draft` was a module, an
unpredictable split that produced confusing AttributeErrors when monkeypatching
what you assumed was a module. With the convention there is nothing to shadow:
`from .phases import grade_draft` gives the function, `from .phases import grade`
gives the module, and both are obvious from the name.

`tests/test_phases_namespace.py` enforces this as an invariant — it fails if any
`__all__` entry collides with a submodule name — so the guarantee holds for
phases added later without anyone maintaining a list.
"""

from .commit import (
    KeywordsOutput,
    ShortNameOutput,
    propose_keywords,
    propose_keywords_batch,
    MAX_KEYWORDS,
    MIN_KEYWORDS,
    propose_short_name,
    render_keywords_yaml,
)
from .crosslinks import (
    CrosslinkCandidate,
    VerificationReport,
    crosslink_candidates,
    propose_crosslinks,
    verify_crosslinks,
)
from .draft import (
    DRAFT_STANCES,
    Draft,
    _wrap_with_frontmatter,
    author,
    stance_for_slot,
    tournament,
)
from .evolution import (
    EvolutionProposal,
    propose_evolution,
    render_proposal_md,
)
from .extract import extract_sections
from .grade import (
    ClaimDetail,
    grade_draft,
)
from .grade_persist import persist_grades
from .memory_evolve import evolve_memory
from .reconcile import reconcile_metadata
from .revise import (
    CritiqueOutput,
    DebugOutput,
    EvolveOutput,
    critic,
    debug,
    detect_structural_gate_issues,
    evolve,
)

__all__ = [
    # Phase functions
    "reconcile_metadata", "extract_sections",
    "crosslink_candidates", "propose_crosslinks", "verify_crosslinks",
    "author", "tournament", "stance_for_slot", "DRAFT_STANCES",
    "grade_draft", "persist_grades",
    "critic", "evolve", "debug", "detect_structural_gate_issues",
    "propose_short_name", "propose_keywords", "propose_keywords_batch",
    "render_keywords_yaml",
    "propose_evolution", "render_proposal_md", "evolve_memory",
    # Dataclasses / outputs
    "Draft", "ClaimDetail",
    "CrosslinkCandidate", "VerificationReport",
    "CritiqueOutput", "EvolveOutput", "DebugOutput",
    "ShortNameOutput", "KeywordsOutput", "MIN_KEYWORDS", "MAX_KEYWORDS",
    "EvolutionProposal",
    # Internal but re-exported (used by runner sandbox writer)
    "_wrap_with_frontmatter",
]
