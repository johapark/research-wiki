"""YAML frontmatter key mapping for imported pages.

Another tool's page carries the same facts under different key names
(`paper_title`, `pub_year`, `container_title`). This maps those onto this
framework's names so `db rebuild` can read them.

Two rules it never breaks:

**Never invent a value.** Rule 1's corollary: a value not legible in the page
needs a lookup plus a sanity check, not a guess. `title`/`authors`/`year` missing
→ the page is blocked (they're also exactly what `stems.derive_stem` needs).
`doi`/`venue` missing → flagged for the lookup path, which is
`backfill.lookup_doi_for_page` and is already gated by `_sanity_ok`.

**Never resolve a disagreement.** Two aliases present with different values
(`year: 2024` *and* `date: 2023-11-02`) is a real conflict; picking one is a coin
flip with no basis. Reported for a human instead.

Unknown source keys are preserved verbatim — dropping a field the other tool
used would lose the user's data, and `lint` has no complaint about extra keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..agents.promote import _yaml_dq

#: canonical name → accepted source spellings (matched on a normalized form).
#: The canonical name itself is always first.
FM_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "paper_title", "name", "headline"),
    "authors": ("authors", "author", "author_list", "creators", "by"),
    "year": ("year", "pub_year", "publication_year", "date", "published", "issued"),
    "doi": ("doi", "doi_url", "identifier"),
    "venue": ("venue", "journal", "publication", "container_title", "conference",
              "booktitle", "proceedings"),
    "keywords": ("keywords", "topics", "subject", "subjects", "kw"),
    "short_name": ("short_name", "shortname", "nickname", "handle", "alias"),
    "hook": ("hook", "gloss", "one_liner", "tagline"),
    "type": ("type", "page_type", "kind"),
    "senior_authors": ("senior_authors", "last_authors", "pi"),
    "tags": ("tags", "labels"),
}

#: Order canonical keys are rendered in — mirrors `promote._build_frontmatter`
#: so a migrated page reads like an ingested one in Obsidian's property view.
_RENDER_ORDER: tuple[str, ...] = (
    "title", "short_name", "hook", "authors", "senior_authors", "year", "doi",
    "venue", "type", "category", "pdf_path", "keywords",
    "migrated_at", "tags",
)

#: Values that must never be auto-filled.
REQUIRED = ("title", "authors", "year")
#: Values a lookup can supply (never a prompt).
LOOKUP_FILLABLE = ("doi", "venue")

#: `migrated_at` is quoted so YAML keeps it a string rather than materializing a
#: `datetime`. Local to the migrate path, and NOT the house style: `ingested_at`
#: and `generated_at` are written unquoted (`agents/promote.py`) as real YAML
#: timestamps, which is what the Dataview date column in `wiki/views.md` needs
#: to format them. Either form survives the DB — `db/rebuild.py` serializes
#: `raw_frontmatter` with `default=str`.
_QUOTED_SCALAR = frozenset({"migrated_at"})

#: Free-text scalars, always double-quoted. An unquoted `': '` inside a value is
#: the single most common cause of `lint`'s `invalid_frontmatter` — PyYAML reads
#: `authors: X, Y (senior: Z)` as a nested mapping and Obsidian's property panel
#: falls back to raw `---` display for the whole page.
_FREE_TEXT = frozenset({"title", "hook", "authors", "senior_authors", "venue"})

_ISO_DATE = re.compile(r"^\s*(\d{4})-\d{1,2}-\d{1,2}")
_BARE_YEAR = re.compile(r"^\s*(\d{4})\s*$")
_ANY_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def normalize_key(key: str) -> str:
    """Lowercase, collapse `-`/space/`.` to `_`. So `Publication Year`,
    `publication-year` and `publication_year` all land on one alias."""
    return re.sub(r"[\s\-.]+", "_", str(key).strip().lower())


@dataclass
class FrontmatterPlan:
    mapped: dict = field(default_factory=dict)          # canonical -> value
    renames: list[tuple[str, str]] = field(default_factory=list)   # (source key, canonical)
    conflicts: list[tuple[str, list[tuple[str, object]]]] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    lookup_needed: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)          # unknown keys, preserved
    notes: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.missing_required)

    @property
    def needs_human(self) -> bool:
        return bool(self.conflicts)


def _coerce_year(raw) -> tuple[int | None, str | None]:
    """-> (year, note). None year means 'could not read it without guessing'."""
    if isinstance(raw, bool):
        return None, "year was a boolean"
    if isinstance(raw, int):
        return (raw, None) if 1800 <= raw <= 2100 else (None, f"year {raw} out of range")
    s = str(raw or "").strip()
    if not s:
        return None, None
    m = _BARE_YEAR.match(s) or _ISO_DATE.match(s)
    if m:
        return int(m.group(1)), None
    # A date like 11/02/23 is genuinely ambiguous (day/month order, century).
    # Refuse rather than pick.
    if re.search(r"\d{1,2}[/.]\d{1,2}[/.]\d{2,4}", s):
        return None, f"ambiguous date {s!r} — needs a lookup or a human"
    m = _ANY_YEAR.search(s)
    if m:
        return int(m.group(1)), f"year read as {m.group(1)} from {s!r}"
    return None, f"could not read a year from {s!r}"


def _yaml_safe_bare(value) -> bool:
    """True when `value` can be written unquoted without changing its meaning.

    Conservative: anything with a `: ` (nested-mapping trap), a `#` (comment), a
    leading indicator character, or a `[[wikilink]]` (parses as a nested flow
    sequence) gets quoted instead.
    """
    if isinstance(value, (int, float, bool)) or value is None:
        return True
    s = str(value)
    if s != s.strip() or not s:
        return False
    if ": " in s or s.endswith(":") or "#" in s or "[[" in s:
        return False
    return s[0] not in "[]{}&*!|>%@`\"'"


def _as_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [t.strip() for t in re.split(r"[;,]", s) if t.strip()]


def map_keys(fm: dict) -> FrontmatterPlan:
    """Map an incoming frontmatter dict onto canonical keys."""
    plan = FrontmatterPlan()
    by_norm: dict[str, list[tuple[str, object]]] = {}
    for k, v in (fm or {}).items():
        by_norm.setdefault(normalize_key(k), []).append((k, v))

    claimed: set[str] = set()
    for canonical, aliases in FM_ALIASES.items():
        found: list[tuple[str, object]] = []
        for alias in aliases:
            for src_key, val in by_norm.get(alias, []):
                if str(val).strip() if not isinstance(val, list) else val:
                    found.append((src_key, val))
                claimed.add(alias)
        if not found:
            continue
        # Distinct non-empty values under two aliases is a real disagreement.
        distinct = {str(v).strip() for _k, v in found}
        if len(distinct) > 1:
            plan.conflicts.append((canonical, found))
            continue
        src_key, value = found[0]
        if src_key != canonical:
            plan.renames.append((src_key, canonical))
        plan.mapped[canonical] = value

    # year needs coercion; a failure to read it is not a failure to have it.
    if "year" in plan.mapped:
        year, note = _coerce_year(plan.mapped["year"])
        if note:
            plan.notes.append(note)
        if year is None:
            del plan.mapped["year"]
        else:
            plan.mapped["year"] = year

    for key in REQUIRED:
        if not plan.mapped.get(key):
            plan.missing_required.append(key)
    for key in LOOKUP_FILLABLE:
        if not plan.mapped.get(key):
            plan.lookup_needed.append(key)

    for norm, pairs in by_norm.items():
        if norm in claimed:
            continue
        for src_key, val in pairs:
            plan.extras[src_key] = val
    return plan


def render_frontmatter(
    plan: FrontmatterPlan, *, stem: str, category: str, page_type: str,
    migrated_at: str,
) -> str:
    """Render the `---` block. Excludes the fences.

    `category` is rendered to match the **directory** — `db rebuild` derives the
    real category from the parent dir and ignores YAML (`db/rebuild.py:129-131`),
    so writing anything else only trips `lint`'s `category_yaml_drift`.

    Provenance is honest: `migrated_at` + a `migrated` tag, never `ingested_at` /
    `author_model` / `ingested-via-agent`, which would claim this page came out of
    the agent pipeline. (`source_collection: migrated` used to carry this too and
    was dropped — nothing ever read the field, and `migrated_at`'s presence already
    says the page was migrated.)
    """
    vals = dict(plan.mapped)
    vals["type"] = page_type
    vals["category"] = f"[{category}]"
    vals["pdf_path"] = f'"[[{stem}.pdf]]"'
    vals["migrated_at"] = migrated_at

    tags = _as_list(vals.get("tags"))
    if "migrated" not in tags:
        tags.append("migrated")
    vals["tags"] = f"[{', '.join(tags)}]"

    kw = _as_list(vals.get("keywords"))
    if kw:
        vals["keywords"] = f"[{', '.join(kw)}]"
    else:
        vals.pop("keywords", None)

    authors = vals.get("authors")
    if isinstance(authors, list):
        vals["authors"] = ", ".join(str(a) for a in authors)
    senior = vals.get("senior_authors")
    if isinstance(senior, list):
        vals["senior_authors"] = ", ".join(str(s) for s in senior)

    lines: list[str] = []
    for key in _RENDER_ORDER:
        if key not in vals:
            continue
        value = vals[key]
        if key in _FREE_TEXT or key in _QUOTED_SCALAR:
            lines.append(f"{key}: {_yaml_dq(str(value))}")
        else:
            lines.append(f"{key}: {value}")
    # Unknown keys last, so nothing the source tool recorded is lost. Quoted when
    # the value would otherwise break the block.
    #
    # Skip any extra that collides with a key rendered above: emitting it twice
    # yields two `key:` lines and YAML keeps the LAST one, so an incoming
    # `category: [wrongcat]` would silently override the directory-derived value
    # this function just computed — and then trip lint's category_yaml_drift.
    rendered_keys = {line.split(":", 1)[0] for line in lines}
    for key, value in plan.extras.items():
        if key in rendered_keys or normalize_key(key) in rendered_keys:
            continue
        out = value if _yaml_safe_bare(value) else _yaml_dq(str(value))
        lines.append(f"{key}: {out}")
    return "\n".join(lines)
