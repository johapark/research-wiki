"""Do the trigger-gated prompts fire when they should?

CLAUDE.md's editing rule is "trigger-gated procedures live in `prompts/{slug}.md`;
leave a one-line pointer here with the trigger condition." That makes the pointer
load-bearing: if it doesn't fire, the procedure may as well not exist. Nothing
tested it — 23 prompt files, every one reached only through a sentence in a file
that has to stay lean.

Method, adapted from OpenKB's `skill/evaluator.py` (Apache-2.0):

1. A **generator** reads one prompt's pointer *and its body* and writes N requests
   that should route to it plus N that shouldn't. Using the body matters: prompts
   generated from the pointer alone only test whether the pointer describes
   itself, which it trivially does.
2. A **grader** sees only the pointers — *all* of them, so competing triggers can
   steal, and picking the wrong prompt is observable rather than being scored as
   a pass.
3. Score false negatives (should have fired, didn't) and false positives
   (fired when it shouldn't, or fired the wrong one).

Two details from OpenKB that make the numbers trustworthy, both carried over: the
graders run concurrently but bounded, and a grading that *errors* is excluded from
the denominator rather than counted as a failure. Without the second, one timeout
silently depresses a pass rate and the metric stops meaning anything.

The output that matters is the named misses, not the rates. A trigger scoring 7/10
tells you less than the three requests it missed.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..agents import llm
from ..paths import wiki_root

# How many graders may be in flight. The generator is one call per prompt; the
# graders are 2N per prompt, which is where the parallelism is worth having and
# where an unbounded fan-out would trip provider rate limits.
MAX_CONCURRENCY = 6
DEFAULT_COUNT = 5

# `[`label`](./prompts/slug.md)` or `[…](./prompts/slug.md#anchor)`, anywhere in
# a CLAUDE.md line. The pointer is whatever sentence carries that link.
_POINTER_RE = re.compile(r"\[[^\]]*\]\(\./prompts/([a-z0-9-]+)\.md(?:#[^)]*)?\)")


@dataclass
class Pointer:
    slug: str
    line: str          # the CLAUDE.md text that gates the prompt (see collect_pointers)
    body: str          # the prompt file's own text


@dataclass
class Case:
    slug: str          # the prompt this case is about
    request: str       # a user request
    should_fire: bool


@dataclass
class Graded:
    case: Case
    chose: str | None      # slug the grader picked, or None for "no prompt"
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.chose == self.case.slug if self.case.should_fire \
            else self.chose != self.case.slug


@dataclass
class SlugReport:
    slug: str
    should_fire_total: int = 0
    should_fire_hit: int = 0
    should_not_total: int = 0
    should_not_hit: int = 0
    errors: int = 0
    misses: list[Graded] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        return (self.should_fire_hit / self.should_fire_total
                if self.should_fire_total else None)

    @property
    def precision(self) -> float | None:
        return (self.should_not_hit / self.should_not_total
                if self.should_not_total else None)


_HEADING_RE = re.compile(r"^#{2,4}\s+\S")

# A section longer than this is truncated for the grader's catalogue. Generous:
# the point is to bound the catalogue when 17 sections are concatenated, not to
# trim any particular one.
MAX_SECTION_CHARS = 1200


def _enclosing_section(lines: list[str], idx: int) -> str:
    """The `##`/`###` block containing line `idx`, heading included.

    **Not just the link's own line.** CLAUDE.md is loaded whole on every turn,
    so what actually gates a prompt is the surrounding section, and the link is
    not always in the sentence that states the trigger — `export-bibliography`
    states its trigger ("*can I get a bib file*") one paragraph above the
    paragraph carrying the link. Extracting the link's line alone would feed the
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


def collect_pointers(claude_md: Path | None = None,
                     prompts_dir: Path | None = None) -> list[Pointer]:
    """Every `prompts/*.md` reachable from CLAUDE.md, with the text that gates it.

    "The text that gates it" is the enclosing section, not the link's line —
    see `_enclosing_section`. A prompt with no pointer at all is unreachable by
    the contract and is reported by `orphan_prompts` rather than silently
    skipped; that is itself a finding.
    """
    claude_md = claude_md or (wiki_root() / "CLAUDE.md")
    prompts_dir = prompts_dir or (wiki_root() / "prompts")
    if not claude_md.exists():
        return []

    lines = claude_md.read_text(encoding="utf-8").splitlines()
    seen: dict[str, Pointer] = {}
    for idx, line in enumerate(lines):
        for slug in _POINTER_RE.findall(line):
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


# Prompts whose name carries `-system` are LLM *system prompts* loaded by code
# through `agents.prompt_lib.load_prompt` — `ask-system`, `concept-triage-system`,
# `suggest-splits-system`, and the `author-system-research` / `author-system-review`
# pair. They are not trigger-gated procedures and are correctly absent from
# CLAUDE.md, so counting them as orphans would report seven permanent false
# positives and make the check worthless.
#
# Substring, not suffix: two of the seven carry a variant suffix after it
# (`author-system-research`), which a `.endswith()` rule missed.
SYSTEM_PROMPT_MARKER = "-system"


def is_gated_prompt(slug: str) -> bool:
    return SYSTEM_PROMPT_MARKER not in slug


def orphan_prompts(claude_md: Path | None = None,
                   prompts_dir: Path | None = None) -> list[str]:
    """Trigger-gated prompt files no CLAUDE.md line points at.

    Unreachable by the contract — the agent has no condition under which it
    would read them — which is a finding in its own right, and free to compute.
    """
    prompts_dir = prompts_dir or (wiki_root() / "prompts")
    if not prompts_dir.is_dir():
        return []
    linked = {p.slug for p in collect_pointers(claude_md, prompts_dir)}
    return sorted(
        p.stem for p in prompts_dir.glob("*.md")
        if p.stem not in linked and is_gated_prompt(p.stem)
    )


# ---------- generation ----------

_GEN_SYSTEM = (
    "You write evaluation cases for a documentation routing test. "
    "Reply with JSON only — no prose, no code fences."
)

_GEN_TEMPLATE = """\
An agent works on a research-paper wiki. Some procedures live in separate files
that it only reads when a one-line trigger in its always-loaded instructions
matches the situation.

Here is one such trigger, and the procedure it gates.

TRIGGER LINE:
{line}

PROCEDURE (`prompts/{slug}.md`):
{body}

Write {n} user requests that SHOULD make the agent read this procedure, and {n}
that should NOT.

The should-fire requests must reflect what the procedure actually covers, not
just what the trigger line says — phrase them the way a user would, without
quoting the trigger's vocabulary back.

The should-not requests are the hard part. Make them *near misses*: plausible
requests about this same wiki that a careless reader might route here but that
this procedure does not cover. Do not write off-topic filler.

JSON, exactly:
{{"should_fire": ["...", ...], "should_not_fire": ["...", ...]}}
"""


def generate_cases(pointer: Pointer, n: int = DEFAULT_COUNT,
                   *, use_stub: bool = False) -> list[Case]:
    resp = llm.call(
        prompt=_GEN_TEMPLATE.format(
            line=pointer.line, slug=pointer.slug,
            body=pointer.body[:6000], n=n,
        ),
        phase="eval_judge",
        system=_GEN_SYSTEM,
        max_tokens=1200,
        use_stub=use_stub,
    )
    data = _parse_json(resp.text) or {}
    cases = [Case(pointer.slug, r, True)
             for r in (data.get("should_fire") or []) if isinstance(r, str)]
    cases += [Case(pointer.slug, r, False)
              for r in (data.get("should_not_fire") or []) if isinstance(r, str)]
    return cases


# ---------- grading ----------

_GRADE_SYSTEM = (
    "You route a user request to at most one procedure, using only the trigger "
    "descriptions given. Reply with JSON only."
)

_GRADE_TEMPLATE = """\
An agent has these trigger-gated procedures available. Each line is the only
description it has of that procedure.

{catalogue}

USER REQUEST:
{request}

Which single procedure should the agent read before answering? Answer with its
slug, or null if none of them applies. Judge only from the trigger lines above.

JSON, exactly: {{"slug": "<slug-or-null>"}}
"""


def _catalogue(pointers: list[Pointer]) -> str:
    return "\n".join(f"- {p.slug}: {p.line}" for p in pointers)


def grade_case(case: Case, catalogue: str, *, use_stub: bool = False) -> Graded:
    try:
        resp = llm.call(
            prompt=_GRADE_TEMPLATE.format(catalogue=catalogue, request=case.request),
            phase="eval_judge",
            system=_GRADE_SYSTEM,
            max_tokens=120,
            temperature=0.0,
            use_stub=use_stub,
        )
    except Exception as e:
        return Graded(case=case, chose=None, error=f"{type(e).__name__}: {e}")
    data = _parse_json(resp.text)
    if data is None:
        return Graded(case=case, chose=None, error="unparseable grader reply")
    chose = data.get("slug")
    return Graded(case=case, chose=chose if isinstance(chose, str) else None)


def grade_all(cases: list[Case], pointers: list[Pointer],
              *, use_stub: bool = False,
              max_workers: int = MAX_CONCURRENCY) -> list[Graded]:
    """Grade every case concurrently, bounded.

    A grading that raises comes back as a `Graded` carrying `error`, never as a
    thrown exception: one provider hiccup must not discard the other N-1
    results, and an errored case is excluded from the rates rather than counted
    against the trigger.
    """
    catalogue = _catalogue(pointers)
    if not cases:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(
            lambda c: grade_case(c, catalogue, use_stub=use_stub), cases
        ))


def summarize(graded: list[Graded]) -> dict[str, SlugReport]:
    reports: dict[str, SlugReport] = {}
    for g in graded:
        rep = reports.setdefault(g.case.slug, SlugReport(slug=g.case.slug))
        if g.error:
            rep.errors += 1
            continue          # excluded from both denominators, deliberately
        if g.case.should_fire:
            rep.should_fire_total += 1
            if g.correct:
                rep.should_fire_hit += 1
            else:
                rep.misses.append(g)
        else:
            rep.should_not_total += 1
            if g.correct:
                rep.should_not_hit += 1
            else:
                rep.misses.append(g)
    return reports


def _parse_json(text: str) -> dict | None:
    """Parse a JSON object out of a model reply, tolerating code fences."""
    if not text:
        return None
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(),
                     flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except ValueError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except ValueError:
            return None
    return data if isinstance(data, dict) else None
