"""Canonical-stem derivation per CLAUDE.md "File Naming Convention" rules."""

from __future__ import annotations

import re
import unicodedata

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

    Two mechanisms, because one is not enough: NFKD handles anything that
    decomposes into a base letter plus combining marks, and `_TRANSLITERATE`
    handles the Latin letters that do not decompose. Transliteration runs first
    so a mapped letter carrying an additional accent still reaches NFKD.
    """
    s = (s or "").translate(_TRANSLITERATE)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def slugify_phrase(s: str) -> str:
    """Canonical page-slug form for a free-text phrase (synthesis titles,
    concept terms). Shared so a scaffolded page's filename and the edge target
    that points at it can't drift apart.

    Applies the same two normalizations stem derivation does, in the same
    order, before reducing to `[a-z0-9-]`:

      1. NFKD-fold and drop combining marks, so an accented letter becomes its
         ASCII base rather than vanishing (CLAUDE.md's naming rule: `García` →
         `garcia`). Deleting it instead yielded `garca`.
      2. Map every Unicode dash (category Pd — en/em dash, non-breaking and
         figure hyphens) to ASCII `-`. These are common in publisher-set
         titles, and deleting them welded words together: `k‑mers` → `kmers`,
         `CRISPR–Cas9` → `crisprcas9`.

    Remaining punctuation is deleted rather than replaced with a separator,
    which is what keeps possessives and decimals intact (`Claude's` →
    `claudes`, `PubTator 3.0` → `pubtator-30`). Verified against all 33
    concept + synthesis pages on disk: no existing slug changes.
    """
    s = strip_diacritics(s or "")
    s = "".join("-" if unicodedata.category(c) == "Pd" else c for c in s)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


# Tokens that mark a consortium/collaboration byline rather than a personal
# name. Per CLAUDE.md's issuer rules, these get slugged whole
# (`1000 Genomes Project` → `1000-genomes-project`) instead of reduced to a
# trailing token (`project`).
_CONSORTIUM_TOKENS = frozenset({
    "project", "consortium", "consortia", "network", "initiative",
    "collaboration", "collaborative", "group", "alliance", "program",
    "programme",
})


# Nobiliary particles (tussenvoegsels) that belong to the surname rather than
# the given name, so CLAUDE.md's "surname as printed on p.1" keeps them:
# `S. De Winter` → `de-winter`, not `winter`. Matches the corpus precedent set
# by `van-kempen-2024-fast-and-accurate-protein-structure`.
_SURNAME_PARTICLES = frozenset({
    "de", "van", "von", "del", "della", "di", "da", "du", "la", "le",
    "den", "der", "ten", "ter", "dos", "das", "bin", "ibn",
})


def first_author_surname(authors: list[str]) -> str:
    """Extract the last name from the first author string.

    Handles three shapes:
      - "First M. Last" / "F. M. Last" → `last`
      - "Last, First" (comma = surname-first, common in bibliographic exports)
        → `last`
      - consortium bylines ("1000 Genomes Project") → the whole name slugged,
        per CLAUDE.md's consortium rule, rather than a trailing token
        (`project`).
    """
    if not authors:
        return "unknown"
    raw = strip_diacritics(authors[0]).strip()
    if not raw:
        return "unknown"

    # Consortium byline: slug the whole name (keeps hyphenated surnames working
    # since those don't carry a consortium token).
    if any(tok in _CONSORTIUM_TOKENS for tok in re.findall(r"[a-z0-9]+", raw.lower())):
        slug = slugify_phrase(raw)
        return slug or "unknown"

    # "Last, First" — the surname is the part before the first comma.
    if "," in raw:
        raw = raw.split(",", 1)[0].strip()

    parts = raw.split()
    if not parts:
        return "unknown"
    # Walk left across nobiliary particles: `S. De Winter` → de-winter,
    # `L. Van Den Berg` → van-den-berg. The `i > 1` floor never consumes the
    # first token, which is what keeps a two-token byline safe — `Bin Liu` and
    # `Di Liu` are given name + surname, not particle + surname, and `bin`/`di`
    # are in the particle set for Arabic and Italian names.
    i = len(parts) - 1
    while i > 1 and parts[i - 1].lower().strip(".") in _SURNAME_PARTICLES:
        i -= 1
    surname = "-".join(parts[i:]).lower()
    surname = re.sub(r"[^a-z0-9-]", "", surname)
    return surname or "unknown"


def normalize_title_word(w: str) -> str:
    w = strip_diacritics(w).lower()
    w = re.sub(r"[^a-z0-9-]", "", w)
    return w


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
