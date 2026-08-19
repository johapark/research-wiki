"""Which `prompts/*.md` files CLAUDE.md can actually reach.

Split from `eval.triggers` so `lint` can import it without pulling in the LLM
stack: the reachability check is pure filesystem work and belongs in the
zero-token health check, while the graded routing sweep next door costs a
provider call per case.

The contract this checks is CLAUDE.md's own editing rule — "trigger-gated
procedures live in `prompts/{slug}.md`; leave a one-line pointer here with the
trigger condition." A prompt with no pointer is unreachable: the agent has no
condition under which it would read the file, so the procedure is dead weight
however well written it is. The inverse — a pointer to a file that isn't there —
sends the agent looking for something that doesn't exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..paths import wiki_root

# `[`label`](./prompts/slug.md)` or `[…](./prompts/slug.md#anchor)`.
POINTER_RE = re.compile(r"\[[^\]]*\]\(\./prompts/([a-z0-9-]+)\.md(?:#[^)]*)?\)")

_HEADING_RE = re.compile(r"^#{2,4}\s+\S")

# A section longer than this is truncated for the grader's catalogue. Generous:
# the point is to bound the catalogue when 17 sections are concatenated, not to
# trim any particular one.
MAX_SECTION_CHARS = 1200

# Prompts whose name carries `-system` are LLM *system prompts*, not procedures
# an agent reads. Most are loaded by code — `concept-triage-system` by
# `concepts.triage`, `suggest-splits-system` / `suggest-category-splits-system` by
# `tasks.suggest_splits`, `bootstrap-categories-system` by
# `tasks.bootstrap_categories`, and the `author-system-research` /
# `author-system-review` pair through `agents.prompt_lib.load_prompt`. `ask-system`
# is the exception: it is the system prompt an MCP *client* runs against
# `researchwiki mcp-serve`, so this package never loads it either. None of the
# seven is a trigger-gated procedure, and all are correctly absent from CLAUDE.md
# — counting them as orphans would report seven permanent false positives and
# make the check worthless.
#
# Substring, not suffix: two of the seven carry a variant suffix after it
# (`author-system-research`), which a `.endswith()` rule missed.
SYSTEM_PROMPT_MARKER = "-system"


@dataclass
class Pointer:
    slug: str
    line: str          # the CLAUDE.md text that gates the prompt (see collect)
    body: str          # the prompt file's own text


def is_gated_prompt(slug: str) -> bool:
    return SYSTEM_PROMPT_MARKER not in slug


def _enclosing_section(lines: list[str], idx: int) -> str:
    """The `##`/`###` block containing line `idx`, heading included.

    **Not just the link's own line.** CLAUDE.md is loaded whole on every turn,
    so what actually gates a prompt is the surrounding section, and the link is
    not always in the sentence that states the trigger — `export-bibliography`
    states its trigger ("*can I get a bib file*") one paragraph above the
    paragraph carrying the link. Extracting the link's line alone would feed a
    grader a passage about citekeys and then score the trigger as having missed,
    blaming the prompt for this function's choice.
    """
    start = 0
    for i in range(idx, -1, -1):
        if _HEADING_RE.match(lines[i]):
            start = i
            break
    end = len(lines)
    for i in range(idx + 1, len(lines)):
        if _HEADING_RE.match(lines[i]):
            end = i
            break
    section = "\n".join(ln for ln in lines[start:end] if ln.strip()).strip()
    return section[:MAX_SECTION_CHARS]


def collect(claude_md: Path | None = None,
            prompts_dir: Path | None = None) -> list[Pointer]:
    """Every `prompts/*.md` reachable from CLAUDE.md, with the text that gates it."""
    claude_md = claude_md or (wiki_root() / "CLAUDE.md")
    prompts_dir = prompts_dir or (wiki_root() / "prompts")
    if not claude_md.exists():
        return []

    lines = claude_md.read_text(encoding="utf-8").splitlines()
    seen: dict[str, Pointer] = {}
    for idx, line in enumerate(lines):
        for slug in POINTER_RE.findall(line):
            if slug in seen:
                continue
            body_path = prompts_dir / f"{slug}.md"
            if not body_path.exists():
                continue
            seen[slug] = Pointer(
                slug=slug,
                line=_enclosing_section(lines, idx),
                body=body_path.read_text(encoding="utf-8"),
            )
    return sorted(seen.values(), key=lambda p: p.slug)


def orphans(claude_md: Path | None = None,
            prompts_dir: Path | None = None) -> list[str]:
    """Trigger-gated prompt files no CLAUDE.md line points at."""
    prompts_dir = prompts_dir or (wiki_root() / "prompts")
    if not prompts_dir.is_dir():
        return []
    linked = {p.slug for p in collect(claude_md, prompts_dir)}
    return sorted(
        p.stem for p in prompts_dir.glob("*.md")
        if p.stem not in linked and is_gated_prompt(p.stem)
    )


def broken(claude_md: Path | None = None,
           prompts_dir: Path | None = None) -> list[str]:
    """Slugs CLAUDE.md points at that have no file — a link into nothing."""
    claude_md = claude_md or (wiki_root() / "CLAUDE.md")
    prompts_dir = prompts_dir or (wiki_root() / "prompts")
    if not claude_md.exists():
        return []
    found: set[str] = set()
    for slug in POINTER_RE.findall(claude_md.read_text(encoding="utf-8")):
        if not (prompts_dir / f"{slug}.md").exists():
            found.add(slug)
    return sorted(found)
