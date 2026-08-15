"""Emit the corpus as an Open Knowledge Format bundle (OKF v0.2).

**Why this is a different job from `refexport`, which shares the same command.**

`refexport` emits a *bibliography*: a list of documents somebody else published,
so it carries only `paper`/`commentary`/`whitepaper`/`guidance`/`book` and there is
deliberately no flag to include synthesis, idea or concept pages — a BibTeX entry
for one would assert a publication that does not exist.

OKF has the opposite scope. Its unit is a **concept**: "anything you want to
capture", explicitly including abstract ideas with no underlying resource (spec
§2, §4.4). A synthesis page is not a fake publication here, it is the most
valuable thing in the bundle, and omitting it would ship a knowledge base with its
knowledge removed. So this emitter carries **every** page type, and `resource` is
simply absent on the pages that describe an idea rather than an artifact — which
is what the spec prescribes, not a gap in the data.

The two also differ in output shape, and that is forced by the formats rather than
chosen: a bibliography is one stream and can go to stdout, whereas an OKF bundle is
a directory tree (spec §3), so `--format okf` requires `--out`.

What is deliberately *not* translated:

  - **Claim anchors.** `[[stem#kc-9f3a2b1c]]` becomes `/cat/stem.md#kc-9f3a2b1c`.
    OKF has no claim-level citation concept, so the fragment is preserved as a
    plain markdown fragment: a consumer ignores it, a reader still lands on the
    right page, and nothing about the anchor is asserted to OKF that OKF defines.
  - **`verified` on synthesis / idea / concept pages.** `check-grounding` and
    `grade synthesis` persist nothing, so there is no record that a page ever
    passed a gate. Emitting `verified` for them would be an unbacked trust claim,
    so these pages carry none and the report counts them. Paper pages *do* get it,
    from `claims.last_graded_at`.

Zero tokens, no network, deterministic — two runs over an unchanged corpus are
byte-identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .paths import wiki_dir
from .wiki import Page, read_pages

#: The spec revision this emitter targets. Declared in the bundle-root
#: `index.md`, the one place OKF permits frontmatter in a reserved file (§12).
OKF_VERSION = "0.2"

#: research-wiki `type:` → OKF `type:`. Title Case because OKF's own examples are
#: (`BigQuery Table`, `Metric`, `Playbook`) and the vocabulary is not registered —
#: consumers MUST tolerate unknown values (§4.1), so this is presentation, not a
#: compatibility claim. An unmapped type is Title-Cased rather than dropped.
TYPE_MAP: dict[str, str] = {
    "paper": "Paper",
    "commentary": "Commentary",
    "synthesis": "Synthesis",
    "idea": "Idea",
    "concept": "Concept",
    "guidance": "Guidance",
    "protocol": "Protocol",
    "whitepaper": "Whitepaper",
    "book": "Book",
    "meta": "Meta",
    "dashboard": "Dashboard",
}

#: idea-page `status:` → OKF lifecycle `status:` (§5.4, `draft|stable|deprecated`).
#: The native value is kept alongside in `x_researchwiki_status`, since OKF's three
#: states cannot express the difference between `superseded` and `abandoned` and
#: producers MAY add keys (§4.1).
STATUS_MAP: dict[str, str] = {
    "open": "draft",
    "scoping": "draft",
    "validated": "stable",
    "superseded": "deprecated",
    "abandoned": "deprecated",
}

#: Frontmatter keys that map onto an OKF field or are deliberately dropped. Any
#: key NOT listed is carried through untouched under an `x_researchwiki_` prefix,
#: so an export never silently loses curation.
_MAPPED_KEYS = frozenset({
    "type", "title", "hook", "doi", "tags", "keywords",
    "author_model", "ingested_at", "generated_at", "status",
    "source_url",   # → `resource`, when the page has no DOI
    # Dropped: local-vault plumbing with no meaning outside this repo.
    "pdf_path", "category", "referenced_papers",
})

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(#[^\]|]*)?(?:\|([^\]]+))?\]\]")
_FOOTNOTE_DEF_RE = re.compile(r"^\[\^([^\]]+)\]:[ \t]*(.+)$", re.M)
_LOG_ENTRY_RE = re.compile(r"^##[ \t]*\[(\d{4}-\d{2}-\d{2})\][ \t]*(\S+)[ \t]*\|[ \t]*(.+)$")


@dataclass
class OkfReport:
    """What the run emitted, and what it could not say.

    Every list here is a to-do or a caveat rather than a statistic:
    `generated_missing_actor` names pages whose write date is known but whose
    author is not, and `verified_absent_no_gate_record` is the Phase-2 gap.
    """

    concepts: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    links_rewritten: int = 0
    links_unresolved: list[dict] = field(default_factory=list)
    sources_emitted: int = 0
    verified_emitted: int = 0
    verified_absent_no_gate_record: list[str] = field(default_factory=list)
    generated_missing_actor: list[str] = field(default_factory=list)
    description_missing: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "format": "okf",
            "okf_version": OKF_VERSION,
            "concepts": self.concepts,
            "by_type": self.by_type,
            "links_rewritten": self.links_rewritten,
            "links_unresolved": self.links_unresolved,
            "sources_emitted": self.sources_emitted,
            "verified_emitted": self.verified_emitted,
            "verified_absent_no_gate_record": self.verified_absent_no_gate_record,
            "generated_missing_actor": self.generated_missing_actor,
            "description_missing": self.description_missing,
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------- selection


#: Filenames OKF reserves at *every* level of the tree (§3.1), not just the root.
#: A concept document may not use one, so a page whose stem is `index` or `log` is
#: unemittable wherever it lives.
_RESERVED_STEMS = frozenset({"index", "log"})


def _skip_reason(page: Page) -> str | None:
    """Why this page cannot be a concept document, or None if it can.

    Two distinct exclusions, previously conflated:

    - **Root-level pages** (`wiki/index.md`, `wiki/log.md`, `wiki/views.md`) are
      the wiki's own bookkeeping. The first two are regenerated in OKF's shape
      (§8, §9) and the rest are per-user dashboards with no place in a category
      tree. Detected by comparing the parent directory against `wiki_dir()`, not
      by looking for a slash in the page key — `Page.key` is always
      `category/stem`, so a slash test is vacuous.
    - **A reserved stem inside a category dir.** `wiki/cgt/index.md` is a
      perfectly ordinary wiki page, but emitting it would put `cgt/index.md` in
      the bundle, which §3.1 forbids and which a consumer would read as a
      directory listing rather than as knowledge. Reported rather than dropped
      silently, since the fix is to rename the page.
    """
    if page.path.parent.resolve() == wiki_dir().resolve():
        return "wiki-root bookkeeping page"
    if page.stem in _RESERVED_STEMS:
        return f"stem {page.stem!r} is an OKF reserved filename (§3.1)"
    return None


# ---------------------------------------------------------------- link rewriting


def _concept_path(key: str) -> str:
    """`cat/stem` → the bundle-relative path OKF recommends (§6.1)."""
    return f"/{key}.md"


def _rewrite_links(
    body: str, titles: dict[str, str], *, page_key: str, report: OkfReport
) -> str:
    """`[[wikilink]]` → `[Title](/cat/stem.md)`.

    Bundle-relative (leading `/`) rather than relative, per §6.1: it is stable when
    a document moves within its subdirectory, and this emitter mirrors `wiki/`'s
    layout so every target is addressable from the root.

    Bare-stem links resolve the way Obsidian resolves them, matching
    `tasks.lint.walk.extract_links` so the bundle and the link graph agree. An
    unresolved target keeps its text and is reported — §6.1 requires consumers to
    tolerate broken links ("may simply represent not-yet-written knowledge"), so
    dropping it would destroy information the spec expects to survive.
    """
    bare = {k.split("/", 1)[1]: k for k in titles if "/" in k}

    def repl(m: re.Match) -> str:
        target, anchor, alias = m.group(1).strip(), m.group(2) or "", m.group(3)
        key = target if target in titles else bare.get(target)
        if key is None:
            report.links_unresolved.append({"page": page_key, "target": target})
            return alias or target.rsplit("/", 1)[-1].replace("-", " ")
        report.links_rewritten += 1
        label = alias or titles.get(key) or key.rsplit("/", 1)[-1]
        return f"[{label}]({_concept_path(key)}{anchor})"

    return _WIKILINK_RE.sub(repl, body)


def _plainify_links(text: str, titles: dict[str, str]) -> str:
    """Wikilinks → plain text, for one-line frontmatter values.

    `hook:` routinely contains `[[wikilinks]]` (CLAUDE.md says so explicitly), and
    it becomes OKF's `description` — a single sentence used in index snippets and
    previews (§4.1). Markdown link syntax in a snippet renders as noise in some
    consumers and raw brackets in the rest, so the alias (or the target's title)
    replaces the link entirely. Nothing is lost: the body still carries the link.
    """
    bare = {k.split("/", 1)[1]: k for k in titles if "/" in k}

    def repl(m: re.Match) -> str:
        target, alias = m.group(1).strip(), m.group(3)
        if alias:
            return alias
        key = target if target in titles else bare.get(target)
        if key is not None:
            return titles[key]
        return target.rsplit("/", 1)[-1].replace("-", " ")

    return _WIKILINK_RE.sub(repl, text)


def _pathify(value, titles: dict[str, str]):
    """Wikilinks → bundle-relative paths, for reference-valued frontmatter.

    `primary_paper` and `companion_synthesis` name other concepts, and §6.2 says a
    path-valued field takes a bundle-relative path. Rewriting them keeps the
    reference followable by a consumer that has never heard of `[[wikilinks]]`; a
    target outside the selection keeps its original text rather than becoming a
    path to a file that isn't in the bundle.
    """
    if isinstance(value, list):
        return [_pathify(v, titles) for v in value]
    if not isinstance(value, str) or "[[" not in value:
        return value

    bare = {k.split("/", 1)[1]: k for k in titles if "/" in k}

    def repl(m: re.Match) -> str:
        target = m.group(1).strip()
        key = target if target in titles else bare.get(target)
        return _concept_path(key) if key is not None else m.group(0)

    return _WIKILINK_RE.sub(repl, value)


# ---------------------------------------------------------------- frontmatter


def _actor_for(page: Page) -> str | None:
    """OKF actor (§7) for `generated.by`, or None when nothing recorded it.

    `author_model` is the only field naming who wrote a page's prose. Absent it,
    we return None and the whole `generated` block is omitted: `by` is REQUIRED
    inside `generated` (§5.2), and inventing `process:something` to carry a
    timestamp would assert an author we do not know.
    """
    model = page.str_field("author_model").strip()
    return f"researchwiki/{model}" if model else None


def _sources_for(page: Page, titles: dict[str, str]) -> list[dict]:
    """`sources` entries (§5.1) from whichever citation surface this type uses.

    Synthesis and idea pages define `[^id]: [[cat/stem]]` footnotes; the label is
    already a stable key rather than a positional index, which is exactly what
    OKF's `sources[].id` requires and for the same stated reason — agents rewrite
    these documents and a positional reference misattributes silently. So the
    footnote label carries straight over and the body needs no change for
    per-claim attribution to keep working.

    Concept pages use the `referenced_papers` spoke registry instead.
    """
    out: list[dict] = []
    seen: set[str] = set()

    for fid, target in _FOOTNOTE_DEF_RE.findall(page.body):
        m = _WIKILINK_RE.search(target)
        if not m:
            continue
        key = m.group(1).strip()
        if key not in titles or fid in seen:
            continue
        seen.add(fid)
        out.append({"id": fid, "resource": _concept_path(key),
                    "title": titles[key]})

    for raw in page.list_field("referenced_papers"):
        m = _WIKILINK_RE.search(raw)
        if not m:
            continue
        key = m.group(1).strip()
        if key not in titles:
            continue
        fid = key.rsplit("/", 1)[-1]
        if fid in seen:
            continue
        seen.add(fid)
        out.append({"id": fid, "resource": _concept_path(key),
                    "title": titles[key]})
    return out


def _okf_frontmatter(
    page: Page, titles: dict[str, str], graded_at: dict[str, int], report: OkfReport
) -> dict:
    """Build one concept's frontmatter. `type` is the only required key (§4.1)."""
    rw_type = page.page_type
    fm: dict = {"type": TYPE_MAP.get(rw_type, rw_type.replace("-", " ").title())}

    if (title := page.str_field("title").strip()):
        fm["title"] = title
    if (hook := page.str_field("hook").strip()):
        fm["description"] = _plainify_links(hook, titles)
    else:
        report.description_missing.append(page.key)

    # `resource` is a URI for the underlying asset, absent for abstract concepts
    # (§4.1) — which is the honest state for synthesis/idea/concept pages.
    doi = page.str_field("doi").strip().strip('"').strip("'")
    if doi and doi.lower() not in ("todo", "none"):
        fm["resource"] = f"https://doi.org/{doi}"
    elif (url := page.str_field("source_url").strip()):
        fm["resource"] = url

    # OKF has one `tags` field; research-wiki splits the same vocabulary across
    # `tags` (concept/idea/synthesis) and `keywords` (paper/reference). Union them
    # in that order, deduped, so no term is lost to the split.
    tags = list(dict.fromkeys(page.list_field("tags") + page.list_field("keywords")))
    if tags:
        fm["tags"] = tags

    actor = _actor_for(page)
    at = page.str_field("ingested_at").strip() or page.str_field("generated_at").strip()
    if actor:
        gen: dict = {"by": actor}
        if at:
            gen["at"] = at
        fm["generated"] = gen
    elif at:
        report.generated_missing_actor.append(page.key)

    ts = graded_at.get(page.stem)
    if ts:
        # Machine-confirmed tier (§5.3): a non-`human:` verifier. Truthful — the
        # claim grader scored every claim on this page against its own PDF.
        fm["verified"] = {"by": "process:researchwiki-grade-paper",
                          "at": _iso(ts)}
        report.verified_emitted += 1
    elif rw_type in ("synthesis", "idea", "concept"):
        report.verified_absent_no_gate_record.append(page.key)

    if (native_status := page.str_field("status").strip()):
        if (mapped := STATUS_MAP.get(native_status)):
            fm["status"] = mapped
        fm["x_researchwiki_status"] = native_status

    if (sources := _sources_for(page, titles)):
        fm["sources"] = sources
        report.sources_emitted += len(sources)

    # Preserve unmapped curation under the extension prefix rather than dropping
    # it — §4.1 lets producers add keys and asks consumers to keep them.
    for k, v in sorted(page.fm.items()):
        if k in _MAPPED_KEYS or v in (None, "", [], {}):
            continue
        fm[f"x_researchwiki_{k}"] = _pathify(v, titles)

    fm["x_researchwiki_stem"] = page.stem
    return fm


def _iso(epoch: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dump_frontmatter(fm: dict) -> str:
    body = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True,
                          default_flow_style=False, width=10_000)
    return f"---\n{body}---\n"


# ---------------------------------------------------------------- reserved files


def render_index(
    pages: list[Page], titles: dict[str, str], descriptions: dict[str, str]
) -> str:
    """Bundle-root `index.md` in OKF's shape (§8).

    `# Section` headings with `* [Title](path) - description` bullets, which is a
    different shape from research-wiki's own `index.md` (`## category` headings and
    `[[wikilinks]]`). Carries `okf_version` — the only frontmatter OKF permits in a
    reserved file (§12).

    `descriptions` comes from the frontmatter pass rather than being re-derived from
    `hook:`, because §8 says an entry SHOULD carry the linked concept's description
    and a second derivation could drift from the one in the file.
    """
    out = [_dump_frontmatter({"okf_version": OKF_VERSION}), "", "# Bundle", "",
           f"{len(pages)} concepts, grouped by the category directory they live in.",
           ""]
    by_cat: dict[str, list[Page]] = {}
    for p in pages:
        by_cat.setdefault(p.category, []).append(p)
    for cat in sorted(by_cat):
        out += [f"# {cat}", ""]
        for p in sorted(by_cat[cat], key=lambda x: x.stem):
            desc = " ".join(descriptions.get(p.key, "").split())
            label = titles.get(p.key) or p.stem
            bullet = f"* [{label}]({_concept_path(p.key)})"
            out.append(f"{bullet} - {desc}" if desc else bullet)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_log(source_log: str | None) -> str | None:
    """Translate `wiki/log.md` into OKF's log shape (§9), or None if unavailable.

    research-wiki writes one H2 per entry, `## [YYYY-MM-DD] kind | subject`. OKF
    wants ISO date headings grouping bullets, with the leading bold word a
    convention. The operation kind becomes that word, so nothing is lost.
    """
    if not source_log:
        return None
    grouped: dict[str, list[str]] = {}
    for line in source_log.splitlines():
        if (m := _LOG_ENTRY_RE.match(line)):
            date, kind, subject = m.groups()
            grouped.setdefault(date, []).append(
                f"* **{kind.capitalize()}**: {subject.strip()}")
    if not grouped:
        return None
    out = ["# Directory Update Log", ""]
    for date in sorted(grouped, reverse=True):     # newest first, per §9
        out += [f"## {date}", "", *grouped[date], ""]
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------- collect


def _graded_at() -> dict[str, int]:
    """stem → newest `claims.last_graded_at`, for `verified` (§5.2).

    Degrades to `{}` on a fresh clone or an unreadable DB: the bundle then simply
    carries no `verified` keys, which OKF treats as the unverified tier rather
    than as an error (§5.3).
    """
    from .db.safe import safe_read

    def q(conn):
        rows = conn.execute(
            "SELECT paper_stem, MAX(last_graded_at) FROM claims "
            " WHERE last_graded_at IS NOT NULL GROUP BY paper_stem"
        ).fetchall()
        return {r[0]: int(r[1]) for r in rows if r[1]}

    return safe_read(q, default={}, label="okfexport.graded_at")


def collect_bundle(
    *,
    categories: list[str] | None = None,
    years: tuple[int, int] | None = None,
    stems: list[str] | None = None,
    include_log: bool = True,
) -> tuple[dict[str, str], OkfReport]:
    """Build the whole bundle in memory: `{bundle-relative path: text}`.

    Titles resolve against the **selected** set, so a filtered export reports the
    links it could not resolve instead of emitting paths to absent concepts.
    """
    report = OkfReport()

    all_pages: list[Page] = []
    for page in read_pages():
        reason = _skip_reason(page)
        if reason is None:
            all_pages.append(page)
        elif page.stem in _RESERVED_STEMS and page.path.parent.resolve() != wiki_dir().resolve():
            # A wiki-root bookkeeping page is expected and uninteresting; a
            # reserved stem in a category dir is a page the user should rename.
            report.skipped.append({"page": page.key, "reason": reason})

    selected: list[Page] = []
    for p in all_pages:
        if categories and p.category not in categories:
            continue
        if stems and p.stem not in stems:
            continue
        if years is not None:
            y = p.year_int()
            if y is None or not (years[0] <= y <= years[1]):
                report.skipped.append({"page": p.key, "reason": "year filter"})
                continue
        selected.append(p)

    titles = {p.key: (p.str_field("title").strip() or p.stem) for p in selected}
    graded = _graded_at()

    files: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    for p in sorted(selected, key=lambda x: x.key):
        fm = _okf_frontmatter(p, titles, graded, report)
        body = _rewrite_links(p.body, titles, page_key=p.key, report=report)
        files[f"{p.key}.md"] = _dump_frontmatter(fm) + "\n" + body.lstrip("\n")
        descriptions[p.key] = fm.get("description", "")
        okf_type = fm["type"]
        report.by_type[okf_type] = report.by_type.get(okf_type, 0) + 1

    report.concepts = len(files)
    files["index.md"] = render_index(selected, titles, descriptions)

    if include_log:
        from .paths import log_path
        src = log_path()
        text = src.read_text(encoding="utf-8") if src.exists() else None
        if (rendered := render_log(text)) is not None:
            files["log.md"] = rendered

    return files, report


# ---------------------------------------------------------------- write


def looks_like_okf_bundle(out: Path) -> bool:
    """True when `out` holds a bundle this emitter wrote.

    Used to decide whether overwriting is safe. Keyed on the `okf_version`
    declaration in a root `index.md`, which only a bundle root carries (§12).
    """
    idx = out / "index.md"
    if not idx.is_file():
        return False
    try:
        head = idx.read_text(encoding="utf-8")[:400]
    except OSError:
        return False
    return "okf_version" in head


def write_bundle(files: dict[str, str], out: Path) -> list[str]:
    """Write the bundle under `out`. Returns paths of pre-existing files left behind.

    Nothing is deleted: a stale concept from an earlier export (a page since
    removed or filtered out) is *reported* so the operator decides, because
    silently pruning a directory the user pointed us at is the wrong default.
    """
    from .fsatomic import write_text_atomic

    out.mkdir(parents=True, exist_ok=True)
    written = set()
    for rel, text in sorted(files.items()):
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(dest, text)
        written.add(dest.resolve())

    stale = [
        str(p.relative_to(out))
        for p in sorted(out.rglob("*.md"))
        if p.resolve() not in written
    ]
    return stale
