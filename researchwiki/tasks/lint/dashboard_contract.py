"""Advisory contract checks for the hand-editable Obsidian dashboard.

``wiki/views.md`` is scaffolded once and deliberately left user-editable, so
comparing it byte-for-byte with the init template would turn harmless prose or
extra columns into noise.  These checks pin only the semantics that other repo
contracts rely on: table order, provenance timestamps, limits, and the distinct
membership models for synthesis and concept pages.

The check is lightweight and runs only when ``researchwiki lint`` is invoked.
Findings are advisory: lint keeps its exit-code-0 reporting contract.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...paths import wiki_dir


_H2_RE = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_DATAVIEW_RE = re.compile(r"```dataview[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_DATAVIEWJS_RE = re.compile(r"```dataviewjs\b", re.IGNORECASE)

_SECTIONS = (
    ("papers", "recent papers"),
    ("ideas", "recent ideas"),
    ("synthesis", "recent synthesis"),
    ("concepts", "recent concept hubs"),
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _dashboard_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Return recognized H2 bodies and their keys in document order."""
    headings = list(_H2_RE.finditer(text))
    found: dict[str, str] = {}
    order: list[str] = []
    for index, match in enumerate(headings):
        heading = _normalize(match.group(1))
        key = next((key for key, prefix in _SECTIONS if heading.startswith(prefix)), None)
        if key is None or key in found:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        found[key] = text[match.end():end]
        order.append(key)
    return found, order


def _query(section: str) -> str:
    match = _DATAVIEW_RE.search(section)
    return _normalize(match.group(1)) if match else ""


def _violation(path: Path, kind: str, detail: str) -> dict:
    return {"page": path, "kind": kind, "detail": detail}


def find_dashboard_contract_violations(views_path: Path | None = None) -> list[dict]:
    """Return customization-tolerant ``views.md`` contract findings."""
    path = views_path if views_path is not None else wiki_dir() / "views.md"
    if not path.is_file():
        return [_violation(path, "dashboard_missing", "wiki/views.md does not exist")]

    text = path.read_text(encoding="utf-8")
    violations: list[dict] = []
    sections, order = _dashboard_sections(text)

    for key, heading in _SECTIONS:
        if key not in sections:
            violations.append(_violation(
                path, "dashboard_section_missing", f"missing an H2 beginning `## {heading.title()}`",
            ))

    expected_order = [key for key, _heading in _SECTIONS]
    if len(order) == len(expected_order) and order != expected_order:
        violations.append(_violation(
            path, "dashboard_section_order",
            "expected papers → ideas → synthesis → concept hubs",
        ))

    if _DATAVIEWJS_RE.search(text):
        violations.append(_violation(
            path, "dashboard_dataviewjs", "dashboard should use standard Dataview blocks only",
        ))

    all_queries = " ".join(_query(section) for section in sections.values())
    if "file.mtime" in all_queries or "file.ctime" in all_queries:
        violations.append(_violation(
            path, "dashboard_filesystem_time",
            "filesystem times rank edits, not additions; use ingested_at/generated_at",
        ))

    queries = {key: _query(section) for key, section in sections.items()}
    for key, query in queries.items():
        if not query:
            violations.append(_violation(
                path, "dashboard_query_missing", f"{key} section has no standard Dataview block",
            ))

    paper = queries.get("papers", "")
    if paper:
        required_prefix = (
            'table without id link(file.link, file.name) as "stem", '
            'join(category, ", ") as "category", venue as "journal",'
        )
        if not paper.startswith(required_prefix):
            violations.append(_violation(
                path, "dashboard_paper_columns",
                "paper columns must begin Stem → Category → Journal",
            ))
        if ('where type = "paper" and ingested_at' not in paper
                or "sort ingested_at desc" not in paper or "limit 15" not in paper):
            violations.append(_violation(
                path, "dashboard_paper_query",
                "papers require ingested_at, descending sort, and LIMIT 15",
            ))

    idea = queries.get("ideas", "")
    if idea and ('where type = "idea" and generated_at' not in idea
                 or "sort generated_at desc" not in idea or "limit 10" not in idea):
        violations.append(_violation(
            path, "dashboard_idea_query",
            "ideas require generated_at, descending sort, and LIMIT 10",
        ))

    synthesis = queries.get("synthesis", "")
    if synthesis:
        if 'as "members"' in synthesis or "referenced_papers" in synthesis:
            violations.append(_violation(
                path, "dashboard_synthesis_members",
                "synthesis has no frontmatter member registry; omit its Members column",
            ))
        if ('where type = "synthesis" and generated_at' not in synthesis
                or "sort generated_at desc" not in synthesis or "limit 10" not in synthesis):
            violations.append(_violation(
                path, "dashboard_synthesis_query",
                "synthesis requires generated_at, descending sort, and LIMIT 10",
            ))

    concept = queries.get("concepts", "")
    if concept:
        if 'length(referenced_papers) as "members"' not in concept:
            violations.append(_violation(
                path, "dashboard_concept_members",
                "concept Members must count the referenced_papers spoke registry",
            ))
        if ('where type = "concept" and generated_at' not in concept
                or "sort generated_at desc" not in concept or "limit 10" not in concept):
            violations.append(_violation(
                path, "dashboard_concept_query",
                "concept hubs require generated_at, descending sort, and LIMIT 10",
            ))

    return violations
