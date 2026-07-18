"""Phase 2 ingest agent — state-machine driver + LLM-backed phases.

The agent never logs; the framework does. See plan-v1-research-agent.md for
the full architecture and `runner.py` for the state machine.

Public entry point is `run_ingest(pdf_path)` (see runner.py).
"""

from __future__ import annotations
