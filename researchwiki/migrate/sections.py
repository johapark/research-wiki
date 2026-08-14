"""H2 heading normalization for imported pages.

Claim extraction matches headings **exactly**: `grade.parser.parse_claims`
iterates `SECTION_KEYS` and does `sections.get("results")`. So a page whose
findings live under `## Findings` yields zero claims — it can't be cited, and
`lint`'s `ungraded_papers` can't see it either (that check JOINs `claims`, so a
page with none is invisible; `zero_claim_papers` is the one that catches it).
Renaming the heading is the whole fix, and it is what this module does.

Three things make it less trivial than a regex replace:

**Merge on collision.** Two source headings can map to one canonical name
(`## Findings` and `## Benchmarks` both → `Results`). Emitting the canonical name
twice is actively harmful, because three modules then disagree about the same
page:

  - `parser._split_sections` (`grade/parser.py:70-79`) — `out[name] = …`, so the
    **last** duplicate wins and the first section's claims vanish
  - `wiki.extract_section` (`wiki.py:152`) — `.search()`, so the **first** wins
  - `coherence` (`coherence.py:155`) — prefix match, so both look fine

So a collision keeps the first heading and appends the later bodies to it in
document order, dropping the later headings.

**Ambiguity is refused, not guessed.** `## Results and Discussion` mapped to
`Results` imports discussion prose as graded claims, which then scores badly
against the PDF and pollutes the corpus. `## References` renamed to
`Related Papers` fills that section with bibliography entries that aren't
wikilinks. These are reported for a human.

**Classification uses the parser's rule, not coherence's.** `coherence` matches
required sections by case-insensitive *prefix*, so `## Results (main text)` reads
as present there while yielding zero claims from the parser. Judging compliance
by coherence would mark exactly the broken pages as already fine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..grade.coherence import PAPER_REQUIRED_SECTIONS
from ..grade.parser import ON_PAGE_H2

# H2 only. H1 is the page title and H3 is sub-structure inside a section;
# rewriting either would change the document outline rather than fix a section
# name. The `(?!#)` is load-bearing: without it `### Findings` matches `##` and
# then captures `# Findings` as the heading text, so H3s get rewritten into H2s
# and the document outline collapses. A trailing colon and whitespace are
# tolerated.
_H2 = r"^[ \t]*##(?!#)[ \t]*"
_TAIL = r"[ \t]*:?[ \t]*$"

#: Confident surface-form → canonical H2. Ordered, first match wins — same shape
#: as `pdf/sections.py:34-43`, which maps many printed forms onto one name.
SECTION_ALIASES: list[tuple[str, re.Pattern[str]]] = [
    ("Summary", re.compile(
        rf"(?im){_H2}(summary|abstract|overview|tl;?dr|synopsis|in\s+brief){_TAIL}")),
    ("Key Contributions", re.compile(
        rf"(?im){_H2}(key\s+contributions?|contributions?|key\s+findings?|"
        rf"key\s+points?|key\s+takeaways?|main\s+contributions?|highlights){_TAIL}")),
    # `architect\w*` rather than `architecture`: an LLM-authored page in the
    # maintainer's corpus read "## Methodology and Architecting". The suffix is
    # decoration on a heading whose first word already decides the section, so
    # tolerating its inflections costs nothing and a typo'd heading otherwise
    # yields zero claims and stays invisible to `ungraded_papers`.
    ("Methodology and Architecture", re.compile(
        rf"(?im){_H2}(methodology(?:\s+and\s+architect\w*)?|methods?|approach|"
        rf"architect\w*|how\s+it\s+works|what\s+they\s+did|"
        rf"technical\s+(?:approach|details)|implementation){_TAIL}")),
    ("Results", re.compile(
        rf"(?im){_H2}(results?|findings?|evaluation|experiments?|benchmarks?|"
        rf"performance){_TAIL}")),
    ("Limitations", re.compile(
        rf"(?im){_H2}(limitations?|caveats?|weaknesses|shortcomings|"
        rf"threats\s+to\s+validity){_TAIL}")),
    ("Related Papers", re.compile(
        rf"(?im){_H2}(related\s+(?:papers?|work|reading)|see\s+also|connections?){_TAIL}")),
]

#: (suggestion, pattern, why) — reported, never rewritten without an explicit
#: opt-in. Each of these would import the wrong text into a graded section.
AMBIGUOUS_HEADINGS: list[tuple[str, re.Pattern[str], str]] = [
    ("Results", re.compile(rf"(?im){_H2}results?\s+and\s+discussion{_TAIL}"),
     "mapping this to Results imports discussion prose as graded claims, which "
     "then scores badly against the PDF"),
    ("Results", re.compile(rf"(?im){_H2}(discussion|conclusions?){_TAIL}"),
     "may hold the paper's findings or may be commentary — only a reader can tell"),
    ("Related Papers", re.compile(rf"(?im){_H2}(references|bibliography|citations){_TAIL}"),
     "a bibliography is not a cross-link list; renaming it fills Related Papers "
     "with entries that aren't [[wikilinks]]"),
]

#: Every canonical name this module may emit. `ON_PAGE_H2` is the four graded
#: ones; `PAPER_REQUIRED_SECTIONS` adds Summary and Related Papers, which affect
#: the coherence score but produce no claims.
CANONICAL_H2: tuple[str, ...] = tuple(
    dict.fromkeys((*ON_PAGE_H2, *PAPER_REQUIRED_SECTIONS))
)
#: The subset whose renaming creates or destroys claims.
GRADED_CANONICAL: frozenset[str] = frozenset(ON_PAGE_H2)

_ANY_H2 = re.compile(rf"{_H2}(.+?)[ \t]*$", re.MULTILINE)


@dataclass
class HeadingChange:
    original: str
    canonical: str
    graded: bool          # renaming this one moves claims
    merged_into_earlier: bool = False


@dataclass
class HeadingPlan:
    changes: list[HeadingChange] = field(default_factory=list)
    ambiguous: list[tuple[str, str, str]] = field(default_factory=list)  # (orig, suggestion, why)
    unmapped: list[str] = field(default_factory=list)
    already_canonical: list[str] = field(default_factory=list)

    @property
    def needs_human(self) -> bool:
        return bool(self.ambiguous)

    @property
    def graded_changes(self) -> list[HeadingChange]:
        return [c for c in self.changes if c.graded]


#: Decorations that never change *which* section a heading is. Stripped before
#: matching so `SECTION_ALIASES` stays a readable list of surface forms instead of
#: growing a parenthetical and slash variant of every entry.
#:
#: A parenthetical qualifier scopes the section without redefining it
#: ("Key Contributions (as a Review)", "(as stated in the abstract)"). A slashed
#: pair names the same section twice ("Results / Findings") — both halves are
#: already aliases of one canonical name, so the first is enough.
_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)\s*$")
_SLASHED_ALT_RE = re.compile(r"\s*/\s*\S.*$")


def _undecorate(heading_text: str) -> str:
    """Strip a trailing parenthetical and a slashed alternative, in that order."""
    t = _QUALIFIER_RE.sub("", heading_text.strip())
    return _SLASHED_ALT_RE.sub("", t).strip()


def canonical_for(heading_text: str) -> str | None:
    """Canonical name for one H2's text, or None if unmapped/ambiguous.

    Canonical names map to themselves — asserted in tests over `CANONICAL_H2`,
    so adding a required section can't silently escape the table.

    Matching is tried on the heading as written and then on an undecorated form
    (see `_undecorate`). The ambiguity guard runs against **both**, so stripping a
    qualifier can never smuggle a heading past it — `## Discussion (results)`
    stays ambiguous rather than becoming Results.
    """
    candidates = [heading_text.strip()]
    bare = _undecorate(heading_text)
    if bare and bare != candidates[0]:
        candidates.append(bare)

    for cand in candidates:
        line = f"## {cand}"
        for _suggestion, pattern, _why in AMBIGUOUS_HEADINGS:
            if pattern.match(line):
                return None

    for cand in candidates:
        line = f"## {cand}"
        for canonical, pattern in SECTION_ALIASES:
            if pattern.match(line):
                return canonical
    return None


def ambiguous_for(heading_text: str) -> tuple[str, str] | None:
    """-> (suggestion, why) when this heading is deliberately not auto-mapped."""
    line = f"## {heading_text.strip()}"
    for suggestion, pattern, why in AMBIGUOUS_HEADINGS:
        if pattern.match(line):
            return suggestion, why
    return None


def plan_headings(body: str) -> HeadingPlan:
    """Classify every H2 in `body` without touching it."""
    plan = HeadingPlan()
    seen_canonical: set[str] = set()
    for m in _ANY_H2.finditer(body):
        original = m.group(1).strip()
        amb = ambiguous_for(original)
        if amb is not None:
            plan.ambiguous.append((original, amb[0], amb[1]))
            continue
        canonical = canonical_for(original)
        if canonical is None:
            plan.unmapped.append(original)
            continue
        if canonical == original:
            plan.already_canonical.append(original)
            seen_canonical.add(canonical)
            continue
        plan.changes.append(HeadingChange(
            original=original,
            canonical=canonical,
            graded=canonical in GRADED_CANONICAL,
            merged_into_earlier=canonical in seen_canonical,
        ))
        seen_canonical.add(canonical)
    return plan


def _sections_in_order(body: str) -> list[tuple[str, int, int, int]]:
    """-> [(heading_text, h2_line_start, body_start, body_end)] in document order."""
    matches = list(_ANY_H2.finditer(body))
    out: list[tuple[str, int, int, int]] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        out.append((m.group(1).strip(), m.start(), body_start, body_end))
    return out


def rewrite_headings(body: str, *, accept_ambiguous: bool = False) -> tuple[str, HeadingPlan]:
    """Rename H2s to canonical names, merging bodies on collision.

    Returns `(new_body, plan)`. Text outside H2 heading lines is preserved
    byte-for-byte except where a merge moves a section body — and a merge only
    ever concatenates, never drops.

    Content before the first H2 (frontmatter has already been split off by the
    caller, so this is any lead-in prose) is preserved verbatim.
    """
    plan = plan_headings(body)
    if accept_ambiguous:
        # Fold the ambiguous ones in as ordinary changes, using the suggestion.
        for original, suggestion, _why in plan.ambiguous:
            plan.changes.append(HeadingChange(
                original=original, canonical=suggestion,
                graded=suggestion in GRADED_CANONICAL,
            ))
        plan.ambiguous = []

    rename: dict[str, str] = {c.original: c.canonical for c in plan.changes}
    if not rename:
        return body, plan

    sections = _sections_in_order(body)
    if not sections:
        return body, plan

    preamble = body[: sections[0][1]]

    # Resolve each section to its final canonical (or unchanged) name, then
    # group by that name preserving first-appearance order. Grouping is what
    # implements merge-on-collision: two sources under one name concatenate.
    order: list[str] = []
    grouped: dict[str, list[str]] = {}
    for heading, _h2_start, b_start, b_end in sections:
        final = rename.get(heading, heading)
        if final not in grouped:
            grouped[final] = []
            order.append(final)
        grouped[final].append(body[b_start:b_end])

    parts: list[str] = [preamble]
    for name in order:
        chunks = grouped[name]
        if len(chunks) == 1:
            # Single source: emit the original body slice verbatim, so a plain
            # rename touches nothing but the heading line itself.
            body_text = chunks[0]
            if not body_text.startswith("\n"):
                body_text = "\n" + body_text
        else:
            # Merge: the slices carry their own surrounding blank lines, so
            # normalize to one blank line between heading, each body, and the
            # next heading rather than concatenating raw.
            joined = "\n\n".join(c.strip("\n") for c in chunks if c.strip())
            body_text = f"\n\n{joined}\n\n"
        parts.append(f"## {name}{body_text}")

    out = "".join(parts)
    # Normalize the tail so a merge can't leave a page without its final newline.
    if body.endswith("\n") and not out.endswith("\n"):
        out += "\n"
    return out, plan
