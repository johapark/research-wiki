"""Phase functions for the ingest agent state machine.

Each phase is a pure function — it takes inputs, returns outputs. Persistence
to ingest_iterations happens in `runner.py` wrappers, never inside these
functions. This separation keeps phases unit-testable and the framework
auditable.

Phases (in order):
  reconcile  : metadata reconciler — DOI / title / year / venue from PDF + S2
  extract    : section + claim extractor (reuses researchwiki.tasks.ingest helpers)
  author     : LLM call (Anthropic) producing a wiki-page draft
  grade      : Phase 1 grader on the draft (BM25 + bi-encoder semantic)
  tournament : pick the highest-scored draft (deterministic argmax)
  critic     : translate grader flags into actionable revision notes
  evolve     : revise the winning draft based on critic notes
  debug      : repair structural-gate failures (numeric drift, KC underflow)
  commit     : write the winning draft to sandbox dir as markdown

Files in this package mirror the phase grouping. Public names re-exported
here so callers can keep `from .phases import X` style imports.

**Name shadowing — read before writing an import.** Six phase functions share
a name with the module that defines them, and the re-export below wins: once
this package is imported, `phases.commit` is the *function*, not the module.
The split is not uniform — it depends purely on whether `__init__` re-exports
a same-named symbol:

    function (module shadowed): commit, extract, grade, reconcile,
                                grade_persist, memory_evolve
    module (nothing shadows it): crosslinks, draft, evolution, revise,
                                evolve_ledger, target_claims

So `from .phases import grade` hands you a function while `from .phases import
draft` hands you a module. When you specifically need the *module* — typically
to monkeypatch its internals in a test — import it by full path, which the
import system resolves independently of this namespace:

    from researchwiki.agents.phases.commit import propose_keywords  # fine
    importlib.import_module("researchwiki.agents.phases.commit")    # fine
    from researchwiki.agents.phases import commit  # the FUNCTION, not the module

`tests/test_phases_namespace.py` pins this map; update it when you add or drop
a re-export.
"""

from .commit import (
    KeywordsOutput,
    ShortNameOutput,
    commit,
    propose_keywords,
    propose_keywords_batch,
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
from .extract import extract
from .grade import (
    ClaimDetail,
    grade,
)
from .grade_persist import grade_persist
from .memory_evolve import memory_evolve
from .reconcile import reconcile
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
    "reconcile", "extract",
    "crosslink_candidates", "propose_crosslinks", "verify_crosslinks",
    "author", "tournament", "stance_for_slot", "DRAFT_STANCES",
    "grade", "grade_persist",
    "critic", "evolve", "debug", "detect_structural_gate_issues",
    "commit", "propose_short_name", "propose_keywords", "propose_keywords_batch",
    "render_keywords_yaml",
    "propose_evolution", "render_proposal_md", "memory_evolve",
    # Dataclasses / outputs
    "Draft", "ClaimDetail",
    "CrosslinkCandidate", "VerificationReport",
    "CritiqueOutput", "EvolveOutput", "DebugOutput",
    "ShortNameOutput", "KeywordsOutput",
    "EvolutionProposal",
    # Internal but re-exported (used by runner sandbox writer)
    "_wrap_with_frontmatter",
]
