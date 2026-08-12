"""Author-name logic, in one place.

Three callers need to answer questions about a person's name and they used to
answer them separately: `stems.first_author_surname` (which surname goes in a
stem), `refexport` (which part is the family name, for CSL-JSON), and
`metadata_sanity.normalise_author` (do two spellings denote the same person).
The third is a *comparison* folder and stays where it is — it asks a different
question. The first two share a parser, and this is it.

Nothing here imports `stems`, deliberately: `stems` imports *this*, and the
particle walk needs only the particle set, so the dependency stays
one-directional. `strip_diacritics` therefore stays in `stems`, applied by its
own wrapper before the walk. Particle comparison is ASCII-only
(`token.lower().strip(".")`), which is safe because every particle *is* ASCII —
a folded and an unfolded `van` are the same string.

**The output convention differs by caller and that is the point.** This module
returns names in their **original spelling and case** (`van den Berg`), because
CSL-JSON has to show a reader the name as printed. Slugging to `van-den-berg` is
stem derivation's business and stays in `stems`, its only consumer that wants it.
"""

from __future__ import annotations

import re

#: Tokens that mark a byline as a consortium rather than a person, so it is never
#: split into given/family. `1000 Genomes Project` has no surname.
CONSORTIUM_TOKENS = frozenset({
    "project", "consortium", "consortia", "network", "initiative",
    "collaboration", "collaborative", "group", "alliance", "program",
    "programme",
})

#: Nobiliary particles (tussenvoegsels) that belong to the surname rather than
#: the given name, so CLAUDE.md's "surname as printed on p.1" keeps them:
#: `S. De Winter` → `De Winter`, not `Winter`. Matches the corpus precedent set
#: by `van-kempen-2024-fast-and-accurate-protein-structure`.
SURNAME_PARTICLES = frozenset({
    "de", "van", "von", "del", "della", "di", "da", "du", "la", "le",
    "den", "der", "ten", "ter", "dos", "das", "bin", "ibn",
})

#: `et al.` / `et al` / `and others`, with or without a preceding comma.
_ET_AL_RE = re.compile(r"(?i)[,;]?\s*\b(?:et\s+al\.?|and\s+others)\s*\.?\s*$")

#: A byline that is prose, not a name list. Three real pages carry one, e.g.
#: `Laura Luebbert (Anthropic Science). Based on research by Ferdous Nasri, …`,
#: which comma-splits into ten fake authors including `Based on research by
#: Ferdous Nasri`. Parentheses, a colon and a slash never appear in a personal
#: name in this corpus; a period followed by a lowercase word deliberately is
#: *not* a signal, because four real pages carry `A. van der Graaf`-shaped names
#: where that lowercase word is a nobiliary particle.
_PROSE_RE = re.compile(r"[()/:]")

#: Above this many whitespace tokens, a single "name" is a sentence.
_MAX_NAME_TOKENS = 6


def strip_et_al(raw: str) -> str:
    """Drop a trailing `et al.` / `and others`.

    An abbreviated byline means names are *missing from the source*, which is
    worth reporting rather than absorbing. Without stripping it, the surname walk
    below returns `al`, which matches no real author — four pages recorded that
    way were reported as wrong-DOI mismatches by `backfill doi --verify` when the
    DOIs were fine.
    """
    return _ET_AL_RE.sub("", (raw or "").strip()).strip().rstrip(",;")


def is_consortium(raw: str) -> bool:
    """Whether a byline names an organisation rather than a person."""
    tokens = re.findall(r"[a-z0-9]+", (raw or "").lower())
    return any(tok in CONSORTIUM_TOKENS for tok in tokens)


def looks_like_prose(raw: str) -> bool:
    """Whether a byline is a sentence rather than a name (or name list)."""
    raw = (raw or "").strip()
    if not raw:
        return False
    return bool(_PROSE_RE.search(raw)) or len(raw.split()) > _MAX_NAME_TOKENS


def surname_span(tokens: list[str]) -> int:
    """Index in `tokens` where the surname begins.

    Walks left across nobiliary particles: `S. De Winter` → 1, `L. Van Den Berg`
    → 1. The `i > 1` floor never consumes the first token, which is what keeps a
    two-token byline safe — `Bin Liu` and `Di Liu` are given name + surname, not
    particle + surname, and `bin`/`di` are in the particle set for Arabic and
    Italian names.
    """
    if not tokens:
        return 0
    i = len(tokens) - 1
    while i > 1 and tokens[i - 1].lower().strip(".") in SURNAME_PARTICLES:
        i -= 1
    return i


def split_author_field(value) -> list[str]:
    """One `authors:` frontmatter value → one string per author.

    Delimiter is chosen, not assumed. A YAML list is already split. A `;` is
    unambiguous and three pages use it. Otherwise `,`, which 418 pages use.

    Returns `[]` for a prose byline rather than guessing where the names are —
    the caller reports it and emits no author field, which every entry type that
    can carry a prose byline (`@techreport`, `@misc`) permits.
    """
    if isinstance(value, list):
        parts = [str(v) for v in value]
    else:
        raw = strip_et_al(str(value or ""))
        if not raw:
            return []
        if looks_like_prose(raw):
            return []
        parts = raw.split(";") if ";" in raw else raw.split(",")
    return [p for p in (part.strip() for part in parts) if p]


def as_family_given(raw: str) -> tuple[str, str] | None:
    """`("Berg", "A. van den")`-style split, or None when it is not safe to split.

    None is the **correct answer**, not a failure path: CSL-JSON's own `literal`
    field exists for names that have no given/family structure, and emitting one
    is faithful where inventing a split would not be. Only CSL needs this at all
    — BibTeX and RIS parse `First von Last` themselves, so they get the name
    unmodified and cannot be corrupted by a wrong guess here.

    Declines to split when: the byline is a consortium, it is a single token
    (`DeepSeek-AI`), or it has more than four tokens with no particle to anchor
    the boundary — the `Given Given Family` vs `Given Family Family` ambiguity
    that this corpus carries no marker for.
    """
    raw = (raw or "").strip()
    if not raw or is_consortium(raw):
        return None

    # "Last, First" — the comma already says where the boundary is.
    if "," in raw:
        family, _, given = raw.partition(",")
        family, given = family.strip(), given.strip()
        return (family, given) if family else None

    tokens = raw.split()
    if len(tokens) < 2:
        return None
    i = surname_span(tokens)
    if len(tokens) > 4 and i == len(tokens) - 1:
        return None
    return " ".join(tokens[i:]), " ".join(tokens[:i])
