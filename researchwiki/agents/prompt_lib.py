"""Load LLM prompts from `prompts/*.md` at the repo root.

Centralizing prompts as text files (instead of Python constants) lets an A/B
run swap a prompt by passing a different file path — `agent ingest
--author-prompt-file`, or `benchmark-fixture --repeat` for a scored
comparison — with no source edit and no rebuild. It also makes prompt changes
diff-able as content rather than as code.

Conventions:
  - One prompt per file, no frontmatter.
  - File location: `prompts/{name}.md` relative to the wiki root.
  - Filenames are lowercase-kebab. Example: prompts/author-system-research.md.

Loader caches by absolute path so re-reads within a process are free,
but a process restart re-reads from disk — useful when iterating on a
prompt file mid-session.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..paths import wiki_root


def prompts_dir() -> Path:
    return wiki_root() / "prompts"


@lru_cache(maxsize=None)
def _read(path_str: str) -> str:
    """Read and cache a prompt file. Cached by absolute path so the same
    file isn't read twice within a process; raises FileNotFoundError with
    a helpful message when missing."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {p}. "
            f"Expected location: prompts/<name>.md at repo root."
        )
    return p.read_text(encoding="utf-8").rstrip() + "\n"


def load_prompt(name: str, *, override_path: str | Path | None = None) -> str:
    """Load `prompts/{name}.md`, or `override_path` if provided.

    `override_path` is the eval / `--*-prompt-file` escape hatch — pass an
    absolute or relative file path to bypass the default lookup. Useful
    for A/B testing without renaming the on-disk default.
    """
    if override_path is not None:
        return _read(str(Path(override_path).resolve()))
    return _read(str((prompts_dir() / f"{name}.md").resolve()))


def load_author_system(
    paper_type: str | None = None,
    *,
    override_path: str | Path | None = None,
) -> str:
    """Pick the author system prompt by paper_type.

    Resolution order (first match wins):
      1. `override_path` if explicitly given (eval A/B path).
      2. `prompts/author-system-{paper_type}.md` if it exists. Lets new
         paper-types ship just by adding a file — no code edit required.
         Shipped in this repo: `review` only — every other paper_type
         reconcile can emit falls through to step 3. Future: clinical-trial,
         dataset, position, theory.
      3. `prompts/author-system-research.md` as the default fallback.

    Reconcile may emit any of "research", "review", "perspective",
    "methods", "preprint", "clinical-trial". Without a dedicated file,
    a paper_type silently falls back to research — safe but loses
    type-specific structure.
    """
    if override_path is not None:
        return load_prompt("(override)", override_path=override_path)
    if paper_type:
        # Normalize: dashes preserved, slashes/spaces stripped just in case
        # an LLM emits "clinical trial" or "clinical/trial".
        normalized = paper_type.lower().strip().replace(" ", "-").replace("/", "-")
        candidate = prompts_dir() / f"author-system-{normalized}.md"
        if candidate.exists():
            return load_prompt(f"author-system-{normalized}")
    return load_prompt("author-system-research")
