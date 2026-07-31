"""Concept hub notes: candidate surfacing + scaffold + reciprocal linking.

A concept page is a mini-synthesis around a single recurring term — a bridge
node tying every wiki paper that instantiates the term into one hub.

This package splits the ~1900-LOC single-file `tasks/concepts.py` into
four concerns:

  candidates   — un-scaffolded candidate detection + persistence
                 (`researchwiki candidates concepts` reads this)
  term_claims  — term ↔ claim helpers shared by scaffold + attach
  scaffold     — the `researchwiki concepts <term>` scaffolder +
                 `attach_after_ingest` post-ingest hook
  refresh      — `refresh_concept` and `upgrade_spokes`

The CLI (`researchwiki concepts <term>` / `--upgrade-spokes` /
`refresh <slug>`) still lives at `researchwiki.tasks.concepts` and is
just a thin argparse wrapper over these entry points.
"""

from .candidates import collect_candidates, n_bridge_candidates
from .declines import add_decline, add_declines, load_declines, remove_decline
from .refresh import refresh_concept, upgrade_spokes
from .scaffold import attach_after_ingest, find_members, run
from .triage import TRIAGE_THRESHOLD, apply_triage, triage_candidates

__all__ = [
    "collect_candidates",
    "n_bridge_candidates",
    "add_decline",
    "add_declines",
    "load_declines",
    "remove_decline",
    "refresh_concept",
    "upgrade_spokes",
    "attach_after_ingest",
    "find_members",
    "run",
    "triage_candidates",
    "apply_triage",
    "TRIAGE_THRESHOLD",
]
