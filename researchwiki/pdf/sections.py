"""Section-anchor extraction from PDF text (engine-agnostic; works on text
returned by `researchwiki.pdf.text.extract_pdf`, currently pypdfium2-backed).

Two extractors:

  - `anchor_sections`: traditional section headings (Introduction / Methods /
    Results / Discussion / References) returned as named excerpts.

  - `extract_caption_blocks`: figure / table / Extended Data caption blocks.
    Brief Communications and short-format Articles (Nature, Cell, Science)
    push load-bearing quantitative results into figure and ED captions —
    e.g. enzyme kcat/KM tables in review-paper Fig. 3d, Lai-style trial
    benchmark tables in main-text Fig. 1, ED Fig results sections at the
    back of a paper. The traditional anchor parser doesn't see these
    because captions don't have section-style headings; this extractor
    catches them via the "Fig. N |" / "Extended Data Fig. N |" pattern
    that Nature-family journals consistently use. Returned as
    `figure_captions` and `extended_data` keys in the sections dict so the
    author phase can consume them as labeled blocks alongside the
    traditional excerpts.
"""

from __future__ import annotations

import re

# Optional leading prefix tolerated on every section heading: an outline
# number ("1. Introduction"), a line number ("26 Abstract", "44 Introduction"
# in Nature accelerated previews), or both. The `[.\s]` alternation is what
# adds line-numbered support — previous versions only tolerated "1." and
# missed line-numbered Nature/Science preview formats.
_LINE_PREFIX = r"(?:\d+[.\s]\s*)?"

SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Many Nature-family papers omit the Abstract header entirely; the
    # fallback path in `extract_abstract` handles those.
    ("abstract", re.compile(rf"(?im)^\s*{_LINE_PREFIX}abstract\s*$")),
    ("introduction", re.compile(rf"(?im)^\s*{_LINE_PREFIX}(introduction|background)\s*$")),
    ("methods", re.compile(rf"(?im)^\s*{_LINE_PREFIX}(materials\s+and\s+methods|methods?|experimental\s+procedures?)\s*$")),
    ("results", re.compile(rf"(?im)^\s*{_LINE_PREFIX}(results?)\s*$")),
    ("discussion", re.compile(rf"(?im)^\s*{_LINE_PREFIX}(discussion|conclusions?)\s*$")),
    ("references", re.compile(rf"(?im)^\s*{_LINE_PREFIX}(references|bibliography)\s*$")),
]

# Caption-start regex. Matches Nature-family pipe-style captions:
#   "Fig. 1 | Folddisco's workflow ..."
#   "Figure 2 | Design of binding ..."
#   "Extended Data Fig. 4 | Partial zinc finger motifs ..."
#   "Table 1 | Patient characteristics"
# Requires the pipe (`|`) to avoid matching inline references like "(Fig. 1c)"
# or "see Fig. 2 for details" that appear in body prose. Pipe-style is
# consistent across Nature, Cell, Science, so this catches captions
# reliably for the journals our fixtures come from. Falls through silently
# when a paper uses period-style captions ("Figure 1. Workflow.") — those
# are still in pdf_full_text via L1 path.
#
# Period-style generalization was attempted on 2026-06-13 but rolled back
# pending multi-run validation: the wider regex extracted captions for
# Science/NEJM/Mol Cell papers but single-run scores moved within the
# ~±10pp variance band, so attribution wasn't conclusive. See
# benchmark-fixtures/PLAN.md for the trajectory.
CAPTION_START_RE = re.compile(
    r"(?im)^\s*(?P<ed>Extended\s+Data\s+)?(?:Fig\.|Figure|Table)\s+\d+\w?\s*\|",
)

# Caption block ends when we hit one of these — keeps the block bounded so
# the author doesn't see the entire post-caption text as one giant caption.
# We apply this between *every* pair of captions (not just after the last
# one) because Nature accelerated-preview PDFs interleave Tables / Methods /
# Supplementary between Figure Legends and Extended Data Figure Legends:
# without a mid-block terminator, the last main caption (e.g. Table 1)
# absorbs Methods text up to the per-caption cap.
#
# Leading line-number prefix is tolerated for line-numbered preprints / Nature
# accelerated previews where every line is prefixed with a 3-4 digit line
# number ("489 Methods", "488 Tables").
CAPTION_BLOCK_TERMINATORS = re.compile(
    r"(?im)^\s*(?:\d{1,4}\s+)?(References|Bibliography|Acknowledgements|"
    r"Author\s+contributions|Competing\s+interests|Data\s+availability|"
    r"Code\s+availability|Methods?|Materials\s+and\s+Methods|"
    r"Tables?|Supplementary(?:\s+(?:Methods|Information|Note|Figures?|Tables?))?)"
    r"\s*$",
)


def _trim_title_block(paragraph: str) -> str:
    """Trim leading title / author / affiliation lines from a pre-introduction
    paragraph picked by the headerless-abstract fallback.

    The pypdfium2-extracted text typically merges the title + author block
    into the same paragraph as the abstract (no blank-line separators),
    which inflates the picked paragraph. We trim by walking lines and
    starting from the first line that looks like *prose*:

      - contains a sentence-internal punctuation pattern `. [A-Z]`, OR
      - ends with `. ! ?`,
      AND has ≥8 whitespace-separated tokens,
      AND does NOT contain author-affiliation markers (`*†‡§¶`) — these
      are the dominant signal that distinguishes author/affiliation
      lines from abstract prose, and rejecting them prevents false
      anchoring on initials like "Cory Y." in author lists.

    After finding the prose anchor line, walk backwards to include
    continuation lines (lines without affiliation markers and ≥5 tokens)
    so the wrapped first sentence isn't truncated.

    Falls through to the original paragraph if no prose line is found.
    """
    sentence_inside = re.compile(r"\.\s+[A-Z]")
    sentence_end = re.compile(r"[.!?]\s*$")
    # Asterisk / dagger / section symbols mark equal-contribution and
    # corresponding-author annotations.
    affiliation_marker = re.compile(r"[\*†‡§¶]")
    # Inline name-glued-to-digit superscripts ("Cardille2", "Co2", "Smith1,2")
    # are the strongest author-line signature: they almost never appear in
    # abstract prose. Three or more in one line → author list.
    author_superscript = re.compile(r"[a-z]\d")
    # Multi-affiliation refs like "1,2" or "2,7" are another author-list cue.
    multi_affiliation = re.compile(r"\d+,\d+")

    def _is_author_line(stripped: str) -> bool:
        if affiliation_marker.search(stripped):
            return True
        if len(author_superscript.findall(stripped)) >= 3:
            return True
        if len(multi_affiliation.findall(stripped)) >= 2:
            return True
        return False

    lines = paragraph.splitlines()

    anchor_idx: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_author_line(stripped):
            continue
        n_tokens = len(stripped.split())
        if n_tokens < 8:
            continue
        if sentence_inside.search(stripped) or sentence_end.search(stripped):
            anchor_idx = i
            break

    if anchor_idx is None:
        return paragraph.strip()

    # Walk backwards at most ONE line to capture the wrapped first sentence
    # of the abstract (e.g. "The cycle of scientific discovery is frequently
    # bottlenecked by the slow, manual\n[anchor: creation of software...]").
    # Capping at 1 avoids over-pulling preceding title-block residue
    # (corresponding-author addresses, copyright blurbs) that don't trigger
    # the author-line heuristics. The cost: papers with a first abstract
    # sentence wrapped over 3+ lines lose the earliest fragment — rare in
    # practice; double-line wrap is the common case.
    start = anchor_idx
    if anchor_idx > 0:
        prev = lines[anchor_idx - 1].strip()
        if prev and not _is_author_line(prev) and len(prev.split()) >= 5:
            start = anchor_idx - 1

    return "\n".join(lines[start:]).strip()


def extract_abstract(text: str, max_chars: int = 4000) -> str:
    """Return the abstract paragraph, or "" when none recoverable.

    Two paths:

      1. **Explicit header.** If the text contains a line "Abstract" by itself
         (case-insensitive, optional leading line-number prefix), take the
         text from that header to the next detected section heading. Covers
         most arXiv preprints and bioRxiv that put `Abstract` on its own line.

      2. **Headerless fallback.** Many Nature-family accelerated previews
         and Brief Communications have an abstract paragraph but no header.
         Take the text BEFORE the first detected non-abstract section
         heading (typically Introduction), find the largest paragraph in
         the pre-introduction region (paragraphs split by blank lines),
         and trim leading title/author/affiliation lines via
         `_trim_title_block`. A 50–2000 word size guard rejects degenerate
         picks (an author-list line on its own, a copyright blurb).

    Returns "" when neither path yields a candidate.
    """
    # Path 1: explicit header.
    abs_pat = SECTION_PATTERNS[0][1]
    abs_match = abs_pat.search(text)
    if abs_match:
        next_start = len(text)
        for name, pat in SECTION_PATTERNS[1:]:
            m = pat.search(text, abs_match.end())
            if m and m.start() < next_start:
                next_start = m.start()
        body = text[abs_match.end():next_start].strip()
        if body:
            return body[:max_chars]

    # Path 2: headerless fallback.
    first_section_pos = None
    for name, pat in SECTION_PATTERNS[1:]:
        m = pat.search(text)
        if m and (first_section_pos is None or m.start() < first_section_pos):
            first_section_pos = m.start()
    if first_section_pos is None:
        return ""

    pre = text[:first_section_pos]
    paragraphs = re.split(r"\n[ \t]*\n+", pre)
    candidates: list[tuple[int, str]] = []
    for p in paragraphs:
        clean = p.strip()
        if not clean:
            continue
        n_words = len(re.findall(r"[A-Za-z][A-Za-z0-9'\-]+", clean))
        if 50 <= n_words <= 2000:
            candidates.append((n_words, clean))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    trimmed = _trim_title_block(candidates[0][1])
    return trimmed[:max_chars]


def anchor_sections(text: str, max_chars: int = 4000) -> dict[str, str]:
    """Find section headings, extract bounded text between them.

    Also extracts an abstract via `extract_abstract` (with a headerless
    fallback for Nature-family PDFs that omit the `Abstract` header),
    and figure / extended-data captions via `extract_caption_blocks`.
    Each extractor runs independently — `anchor_sections` may return an
    empty section dict on a Brief Communication that lacks Methods/Results
    headings while abstract / caption extraction still surfaces useful
    structure.
    """
    positions: list[tuple[int, str]] = []
    for name, pat in SECTION_PATTERNS:
        for m in pat.finditer(text):
            positions.append((m.start(), name))
    out: dict[str, str] = {}
    if positions:
        positions.sort()
        seen: dict[str, int] = {}
        ordered: list[tuple[int, str]] = []
        for pos, name in positions:
            if name not in seen:
                seen[name] = pos
                ordered.append((pos, name))
        ordered.sort()
        for idx, (pos, name) in enumerate(ordered):
            end = ordered[idx + 1][0] if idx + 1 < len(ordered) else len(text)
            out[name] = text[pos:end][:max_chars].strip()

    # Abstract: explicit-header detection runs through SECTION_PATTERNS, but
    # the resulting slice starts at the word "Abstract" itself — strip that
    # leading header line so downstream sentence-split sees only the body.
    # Headerless papers fall back to the largest-paragraph-before-introduction
    # heuristic via `extract_abstract`.
    if "abstract" in out:
        out["abstract"] = re.sub(
            r"^\s*(?:\d+\s+)?abstract\s*\n+", "", out["abstract"],
            count=1, flags=re.IGNORECASE,
        ).strip()
    if "abstract" not in out or not out["abstract"]:
        fallback = extract_abstract(text, max_chars=max_chars)
        if fallback:
            out["abstract"] = fallback
        else:
            out.pop("abstract", None)

    # Caption extraction is independent of section anchoring — runs even
    # when `anchor_sections` finds no traditional section headings.
    fig_caps, ed_caps = extract_caption_blocks(text)
    if fig_caps:
        out["figure_captions"] = fig_caps
    if ed_caps:
        out["extended_data"] = ed_caps

    return out


def extract_caption_blocks(
    text: str,
    *,
    per_caption_max_chars: int = 1500,
    total_max_chars: int = 12000,
) -> tuple[str, str]:
    """Extract figure / table caption blocks from the PDF text.

    Returns (main_figure_captions, extended_data_captions) — each a
    newline-separated concatenation of caption blocks. Main captions
    include "Fig. N | ..." and "Table N | ..." for figures and tables in
    the main text; extended_data captions include all "Extended Data Fig./
    Extended Data Table N | ..." occurrences.

    Each individual caption is capped at `per_caption_max_chars` (1500
    default — long enough for a 200-300 word caption with sub-figure
    descriptions, short enough that a paper with 10 figures fits the
    aggregate budget). The total per side is capped at `total_max_chars`
    so the prompt isn't overwhelmed by caption text on review papers
    with many figures.

    Implementation notes:
      - Pipe-style (`|`) requirement avoids body-prose false positives
        ("(Fig. 1c)", "see Fig. 2"). Period-style captions ("Figure 1.")
        fall through to L1's full PDF text — accepted limitation for v1.
      - First caption start to next caption start (or to a known
        end-of-paper section) defines each block; the per-caption cap
        is a safety net for papers with sparse captions.
      - PyPDF text extraction sometimes inserts mid-caption page breaks;
        we keep these as-is (the LLM tolerates them).
    """
    matches = list(CAPTION_START_RE.finditer(text))
    if not matches:
        return "", ""

    # Find the first end-of-paper terminator AFTER the first caption start
    # so that captions appearing post-References (Extended Data figures
    # placed after the bibliography in some Brief Communications) aren't
    # accidentally cut off. We want terminators to be applied per-block
    # only when a caption block is the last one.
    main_blocks: list[str] = []
    ed_blocks: list[str] = []
    for i, m in enumerate(matches):
        block_start = m.start()
        # Upper bound: next caption start, or end-of-text for the last one.
        candidate_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Always check for an end-of-paper terminator within the candidate
        # region. Catches the accelerated-preview case where Methods / Tables
        # / Supplementary sit between two captions: the block clips at the
        # terminator instead of absorbing prose up to the per-caption cap.
        term_match = CAPTION_BLOCK_TERMINATORS.search(text, block_start + 1, candidate_end)
        block_end = term_match.start() if term_match else candidate_end
        block = text[block_start:block_end].strip()[:per_caption_max_chars]
        if not block:
            continue
        if m.group("ed"):
            ed_blocks.append(block)
        else:
            main_blocks.append(block)

    main_text = "\n\n".join(main_blocks)[:total_max_chars]
    ed_text = "\n\n".join(ed_blocks)[:total_max_chars]
    return main_text, ed_text
