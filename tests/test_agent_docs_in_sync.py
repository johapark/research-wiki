"""`AGENTS.md` must stay byte-identical to `CLAUDE.md`.

Codex CLI, Cursor, Aider, Continue, Gemini CLI and Cody all auto-load a
repo-level instruction file, but none of them follow markdown links *out* of
it. So the pointer file `AGENTS.md` used to be left every non-Claude agent
running without the Four Rules, the naming convention, or the page-type
contracts — it only told them to go read `CLAUDE.md`, which they never did.

It's a symlink now. These tests fail if someone "helpfully" replaces it with a
standalone document that can then drift out of sync.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / "AGENTS.md"
CLAUDE = ROOT / "CLAUDE.md"


def test_agents_md_content_matches_claude_md():
    assert AGENTS.read_text(encoding="utf-8") == CLAUDE.read_text(encoding="utf-8")


def test_agents_md_is_a_symlink_to_claude_md():
    """Equality alone would still permit a copy that drifts on the next edit;
    the symlink is what makes drift structurally impossible."""
    assert AGENTS.is_symlink(), (
        "AGENTS.md must be a symlink to CLAUDE.md, not a copy — see the "
        "'Editing this file' section of CLAUDE.md for why."
    )
    # Relative target, so the link resolves in any clone location.
    import os
    assert os.readlink(AGENTS) == "CLAUDE.md"
