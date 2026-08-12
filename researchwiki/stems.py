"""Canonical-stem derivation per CLAUDE.md "File Naming Convention" rules."""

from __future__ import annotations

import re
import unicodedata

from . import names

STOP_WORDS = {
    "a", "an", "the", "of", "for", "with", "and", "or",
    "in", "on", "at", "to", "from", "by", "as",
    "across", "over", "all", "that", "this", "these", "those",
}


# Latin letters NFKD cannot help with. An accented letter like `í` decomposes
# into `i` + a combining mark, so dropping the mark leaves the ASCII base. These
# do not decompose at all — the stroke, bar or ligature is part of the codepoint
# — so they survive NFKD intact and are then deleted by the `[^a-z0-9\s-]` pass
# in `slugify_phrase`, silently removing a letter from the middle of a name.
#
# Observed 2026-08-04: `Szałata` → `szaata`, an author page whose stem was
# missing an `l`. Left unmapped, `Đurić` → `uric` and `Løken` → `lken` — the
# first loses its leading letter, which is the one a reader scans for.
#
# Only characters with an unambiguous single-letter or standard two-letter
# romanization are listed. Anything genuinely contested stays out.
_TRANSLITERATE = str.maketrans({
    "ł": "l", "Ł": "L",     # Polish
    "ø": "o", "Ø": "O",     # Danish / Norwegian
    "đ": "d", "Đ": "D",     # Croatian / Serbian / Vietnamese
    "ħ": "h", "Ħ": "H",     # Maltese
    "ŧ": "t", "Ŧ": "T",     # Sámi
    "ı": "i", "İ": "I",     # Turkish dotless / dotted
    "ß": "ss",
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "þ": "th", "Þ": "TH",   # Icelandic thorn
    "ð": "d", "Ð": "D",     # Icelandic eth
})


def strip_diacritics(s: str) -> str:
    """Fold a string to its ASCII-letter skeleton.

    Three mechanisms, because no one of them is enough:

      1. `_TRANSLITERATE` handles the Latin letters NFKD cannot decompose.
      2. NFKD handles anything that decomposes into a base letter plus
         combining marks. Transliteration runs first so a mapped letter
         carrying an additional accent still reaches NFKD.
      3. Every Unicode dash (category `Pd` — en/em dash, figure and
         non-breaking hyphens) folds to ASCII `-`. This runs *after* NFKD
         because U+2011 NON-BREAKING HYPHEN decomposes to U+2010 HYPHEN, so
         folding first would miss the decomposition product.

    Step 3 lived only in `slugify_phrase` until 2026-08-11, which is exactly
    how the bug it prevents got into stem derivation: every caller of this
    function then needed to remember a normalization the function itself
    didn't do, and `normalize_title_word` didn't. Publisher-set titles use
    U+2010 freely, and the `[^a-z0-9-]` passes downstream *delete* an
    unfolded dash rather than preserving the word boundary — welding
    `ATAC‐seq` into `atacseq` while the same paper's ASCII-hyphen spelling
    gives `atac-seq`. Two stems, one paper, depending on which source the
    metadata came from. Measured on a real 532-item library: 15 stems.

    NBSP needs no special case — it is category `Zs` and NFKD-normalizes to a
    plain space. U+2212 MINUS SIGN is deliberately *not* folded: it is
    category `Sm`, a mathematical operator, and titles use it as one.
    """
    s = (s or "").translate(_TRANSLITERATE)
    nfkd = unicodedata.normalize("NFKD", s)
    folded = "".join(c for c in nfkd if not unicodedata.combining(c))
    return "".join("-" if unicodedata.category(c) == "Pd" else c for c in folded)


def slugify_phrase(s: str) -> str:
    """Canonical page-slug form for a free-text phrase (synthesis titles,
    concept terms). Shared so a scaffolded page's filename and the edge target
    that points at it can't drift apart.

    Both normalizations live in `strip_diacritics`: the NFKD fold that turns an
    accented letter into its ASCII base rather than deleting it (CLAUDE.md's
    `García` → `garcia`; deleting yielded `garca`), and the Unicode-dash fold
    that keeps word boundaries (`k‑mers` → `k-mers`, not `kmers`). This function
    used to apply the dash fold itself, on top of `strip_diacritics`; it is now
    inherited, so stems and slugs cannot drift apart again.

    Remaining punctuation is deleted rather than replaced with a separator,
    which is what keeps possessives and decimals intact (`Claude's` →
    `claudes`, `PubTator 3.0` → `pubtator-30`). Verified against all 33
    concept + synthesis pages on disk: no existing slug changes.
    """
    s = strip_diacritics(s or "")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def first_author_surname(authors: list[str]) -> str:
    """Extract the last name from the first author string, slugged for a stem.

    Handles three shapes:
      - "First M. Last" / "F. M. Last" → `last`
      - "Last, First" (comma = surname-first, common in bibliographic exports)
        → `last`
      - consortium bylines ("1000 Genomes Project") → the whole name slugged,
        per CLAUDE.md's consortium rule, rather than a trailing token
        (`project`).

    The *parsing* lives in `names`, shared with the bibliographic exporter, which
    needs the same boundary for CSL-JSON's `family`/`given`. What stays here is
    the part only a stem wants: fold to ASCII first (so `Ré` → `re`), then slug
    the surname to `[a-z0-9-]` with no edge or doubled separator.
    """
    if not authors:
        return "unknown"
    # Fold before parsing, so the walk sees ASCII and the slug below has nothing
    # left to strip. `names.surname_span` compares against ASCII particles, which
    # is why folding first is safe rather than merely convenient.
    raw = names.strip_et_al(strip_diacritics(authors[0]).strip())
    if not raw:
        return "unknown"

    # Consortium byline: slug the whole name (keeps hyphenated surnames working
    # since those don't carry a consortium token).
    if names.is_consortium(raw):
        return slugify_phrase(raw) or "unknown"

    # "Last, First" — the surname is the part before the first comma.
    if "," in raw:
        raw = raw.split(",", 1)[0].strip()

    parts = raw.split()
    if not parts:
        return "unknown"
    surname = "-".join(parts[names.surname_span(parts):]).lower()
    surname = re.sub(r"[^a-z0-9-]", "", surname)
    # Same invariant the title part holds: a stem component never carries an
    # edge or doubled separator. Rare here (a byline ending in a stray dash),
    # but the alternative is a stem whose author segment breaks STEM_PREFIX_RE.
    surname = re.sub(r"-{2,}", "-", surname).strip("-")
    return surname or "unknown"


def normalize_title_word(w: str) -> str:
    """One title word → its stem-safe form, or `""` if nothing survives.

    Interior hyphens are kept, because CLAUDE.md counts a hyphenated term as
    one word (`Cas-OFFinder` → `cas-offinder`). Edge hyphens are not: a
    suspended compound like *"epigenome- and transcriptome-wide"* leaves a
    dangling `-` that the join then carries into the stem, producing
    `…-in-epigenome-` and `…-long--and-short-read` — trailing and doubled
    separators that no other stem has. Collapse runs too, so a word cannot
    contribute `--` from its own interior either.
    """
    w = strip_diacritics(w).lower()
    w = re.sub(r"[^a-z0-9-]", "", w)
    return re.sub(r"-{2,}", "-", w).strip("-")


def derive_title_part(title: str) -> str:
    """First five title words, with stop-word extension, hyphen preservation,
    colon skipping, diacritic stripping, and lowercase (per CLAUDE.md rules)."""
    title = title.replace(":", " ")
    raw_words = title.split()
    words = [normalize_title_word(w) for w in raw_words]
    words = [w for w in words if w]
    if not words:
        return "untitled"
    kept = words[:5]
    i = 5
    while kept and kept[-1] in STOP_WORDS and i < len(words):
        kept.append(words[i])
        i += 1
    return "-".join(kept)


def derive_stem(authors: list[str], year: int | str, title: str) -> str:
    """Canonical `{surname}-{year}-{first-five-title-words}` stem."""
    surname = first_author_surname(authors)
    title_part = derive_title_part(title)
    return f"{surname}-{year}-{title_part}"


STEM_PREFIX_RE = re.compile(r"^([a-z][-a-z0-9]*?)-(\d{4}[a-z]?)(?:-|$)")


def stem_author_year(stem: str) -> str:
    """Shorten `garcia-lopez-2024-long-title-words` to `garcia-lopez-2024`.

    Handles hyphenated surnames and BibTeX year suffixes (2024b, 2024c).
    Returns the input unchanged if it doesn't match the canonical pattern.
    """
    m = STEM_PREFIX_RE.match(stem)
    return f"{m.group(1)}-{m.group(2)}" if m else stem
