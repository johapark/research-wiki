"""Extract atomic claims from a wiki page.

Original V1 scope: Key Contributions bullets + Results section bullets and
table rows. Skipped Summary (prose, hard to atomize), Methodology
(descriptive), Limitations (opinion-shaped).

L6 expansion (2026-06-13): also grade Limitations bullets and Methodology
prose sentences. Limitations was an ungraded refuge — the original
"specific numbers were not extractable from excerpts" cop-out survived
L1 validation because no grader signal flagged it. Extending coverage
catches that pattern post-hoc and gives the critic loop a signal where
none existed.

Methodology grading is conservative: bullets graded as bullets, prose
extracted as sentences ≥ 40 chars / ≥ 5 words to filter transitions.
Sentences are noisier than bullets — expect a small drop in mean BM25
on benchmark papers because Methodology prose paraphrases more loosely
than KC/Results bullets do. The signal that matters is per-claim weak
flags, not aggregate averages.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable

from ..wiki import Page


HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)
TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$", re.MULTILINE)
SEPARATOR_ROW_RE = re.compile(r"^\|[\s\-:|]+\|\s*$")
# Sentence boundary: end-punct + whitespace + capital letter (or the
# beginning of a quoted/parenthesized fragment). Conservative — splits
# only on clear sentence ends to avoid fragmenting "et al." or
# "i.e., the" mid-sentence.
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'`])")
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
CROSSREF_HINT_RE = re.compile(
    r"\b(cite[ds]?|cited by|references?|see\s+\[\[|"
    r"superseded by|extends?|builds?\s+on|contrasts?\s+with)\b",
    re.IGNORECASE,
)
# Sections graded for claim coverage, as (on-page H2 lowercased, DB section
# key) pairs. Order matches the typical wiki page layout. `methodology and
# architecture` is prose-heavy, so the parser runs both bullet- and
# sentence-level extraction on it.
#
# This is THE list that decides whether a page yields claims at all — matching
# is exact, so a page whose findings sit under `## Findings` produces nothing.
# `parse_claims` iterates it, and `researchwiki.migrate.sections` asserts its
# alias table covers it, so the two can't drift.
SECTION_KEYS: tuple[tuple[str, str], ...] = (
    ("key contributions", "key_contributions"),
    ("results", "results"),
    ("limitations", "limitations"),
    ("methodology and architecture", "methodology"),
)
# Canonical on-page spellings, for callers that render or validate headings.
ON_PAGE_H2: tuple[str, ...] = (
    "Key Contributions", "Results", "Limitations", "Methodology and Architecture",
)
# Derived, not hand-maintained — it was previously a second copy of the same
# four names that nothing read.
GRADED_SECTIONS = {name for name, _ in SECTION_KEYS}
PROSE_GRADED_SECTIONS = {"methodology and architecture"}

# Min sentence length for prose extraction. Cuts transitions ("This is
# important.") and short fragments without losing real claim content.
PROSE_MIN_CHARS = 40
PROSE_MIN_WORDS = 5


@dataclass
class Claim:
    section: str       # 'key_contributions', 'results'
    position: int      # 0-indexed within section
    text: str          # the claim sentence/bullet, with markdown bullet markers stripped
    is_cross_ref: bool # True if the claim looks like a cross-paper reference
                       # (skip from coverage grading; audit handles these)
    source_offset: int = 0  # Character offset in Page.body, for reader navigation.


def _split_sections(body: str) -> dict[str, tuple[str, int]]:
    """Return {lowercased-heading: (section-body-text, source offset)}."""
    out: dict[str, tuple[str, int]] = {}
    matches = list(HEADING_RE.finditer(body))
    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw = body[start:end]
        out[name] = (raw.strip(), start + len(raw) - len(raw.lstrip()))
    return out


def _strip_markdown(text: str) -> str:
    """Strip markdown emphasis markers and quotes; keep numbers and units intact."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    return text.strip()


def _looks_like_cross_ref(text: str) -> bool:
    """Heuristic: is this claim primarily about another wiki paper?

    A bullet that is dominated by a `[[wikilink]]` + a relationship verb
    is a cross-reference, not a content claim about the *current* paper.
    Examples:
      - [[cgt/foo-2024]] cites this paper
      - extends [[compbio/bar-2025]]'s tree-search idea
    """
    if not WIKILINK_RE.search(text):
        return False
    # Strip wikilinks and see how much non-wikilink prose remains.
    stripped = WIKILINK_RE.sub("", text).strip(" -—:\t")
    if not stripped:
        return True
    # Short remainder + crossref-hinting verb → cross-ref bullet
    if len(stripped) < 80 and CROSSREF_HINT_RE.search(stripped):
        return True
    return False


def _extract_bullets(section_text: str) -> list[tuple[str, int]]:
    """Return cleaned bullet texts from a section body."""
    out: list[tuple[str, int]] = []
    for m in BULLET_RE.finditer(section_text):
        line = _strip_markdown(m.group(1))
        if line:
            out.append((line, m.start(1)))
    return out


def _extract_prose_sentences(section_text: str) -> list[tuple[str, int]]:
    """Extract sentence-shaped chunks from prose, skipping bullet content
    and table rows (those are handled by their own extractors). Filters
    very short fragments under PROSE_MIN_CHARS / PROSE_MIN_WORDS.

    Bold bullet headers like `**Indexing:**` (used in Methodology pages
    to introduce sub-sections) are stripped from the start of sentences
    so the actual prose claim is graded, not the formatting label.
    """
    out: list[tuple[str, int]] = []
    # Map positions in the filtered prose back to their lines in the source.
    non_bullet_lines = []
    prose_starts = []
    source_starts = []
    prose_length = source_offset = 0
    for raw_line in section_text.splitlines(keepends=True):
        line = raw_line.splitlines()[0]
        if not (BULLET_RE.match(line) or TABLE_ROW_RE.match(line) or SEPARATOR_ROW_RE.match(line)):
            prose_starts.append(prose_length)
            source_starts.append(source_offset)
            non_bullet_lines.append(line)
            prose_length += len(line) + 1
        source_offset += len(raw_line)
    prose = "\n".join(non_bullet_lines)
    cursor = 0
    for paragraph in re.split(r"\n\s*\n", prose):
        for raw in SENTENCE_BOUNDARY_RE.split(paragraph.strip()):
            start = prose.index(raw, cursor)
            cursor = start + len(raw)
            sent = _strip_markdown(re.sub(r"\s+", " ", raw))
            # Strip a leading bold-header label like "**Indexing:**".
            sent = re.sub(r"^\*+[\w\s/&-]+:?\*+\s*", "", sent)
            sent = re.sub(r"^[A-Z][\w\s/&-]+:\s+", "", sent, count=1)
            sent = sent.strip()
            if len(sent) < PROSE_MIN_CHARS:
                continue
            if len(sent.split()) < PROSE_MIN_WORDS:
                continue
            line_index = bisect_right(prose_starts, start) - 1
            offset = source_starts[line_index] + start - prose_starts[line_index]
            out.append((sent, offset))
    return out


def _extract_table_rows(section_text: str) -> list[tuple[str, int]]:
    """Return cleaned cell contents from each markdown table data row,
    joined into a single sentence per row.

    `| Hepa1-6 | LNP-mRNA | up to 2x higher |`  →  "Hepa1-6 LNP-mRNA up to 2x higher"
    Header rows are dropped (they sit above a `|---|---|` separator);
    we use the separator as the discriminator.
    """
    rows: list[tuple[str, int]] = []
    offset = 0
    in_table = False
    for raw in section_text.splitlines(keepends=True):
        start = offset
        offset += len(raw)
        line = raw.rstrip()
        if SEPARATOR_ROW_RE.match(line):
            in_table = True
            continue
        if in_table and TABLE_ROW_RE.match(line):
            cells = [c.strip() for c in line.strip("| ").split("|")]
            text = " ".join(c for c in cells if c)
            text = _strip_markdown(text)
            if text:
                rows.append((text, start + 1))
            continue
        # Empty line breaks the table
        if in_table and not line.strip():
            in_table = False
    return rows


def parse_claims(page: Page) -> list[Claim]:
    """Extract gradable claims from a wiki page's graded sections.

    Sections graded (post-L6, 2026-06-13): Key Contributions, Results,
    Limitations, Methodology and Architecture. Each Claim carries its
    source section, position, and a heuristic flag for whether it's a
    cross-paper reference (which the coverage grader skips — audit
    handles those instead).

    Per-section extraction strategy:
      - Key Contributions, Limitations: bullets only (these sections are
        bullet-shaped by convention and the wiki author prompts).
      - Results: bullets + table rows (mixed prose+benchmark-table
        format).
      - Methodology and Architecture: bullets + prose sentences
        (≥ PROSE_MIN_CHARS, ≥ PROSE_MIN_WORDS — filters transitions and
        fragments).
    """
    sections = _split_sections(page.body)
    claims: list[Claim] = []

    for name, key in SECTION_KEYS:
        if name not in sections:
            continue
        section_body, section_start = sections[name]
        items: Iterable[tuple[str, int]]
        if name == "results":
            # Results sections often mix prose bullets with a benchmark table.
            items = _extract_bullets(section_body) + _extract_table_rows(section_body)
        elif name in PROSE_GRADED_SECTIONS:
            # Methodology: bullets first (when the author used them),
            # then prose sentences for the rest.
            items = (
                _extract_bullets(section_body)
                + _extract_prose_sentences(section_body)
            )
        else:
            items = _extract_bullets(section_body)
        for pos, (text, offset) in enumerate(items):
            if len(text) < 10:
                continue
            claims.append(
                Claim(
                    section=key,
                    position=pos,
                    text=text,
                    is_cross_ref=_looks_like_cross_ref(text),
                    source_offset=section_start + offset,
                )
            )
    return claims
