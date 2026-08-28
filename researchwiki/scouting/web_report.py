"""Deterministic source-ledger rendering for web-scout receipts."""

from __future__ import annotations

import html
import re
from pathlib import Path

from ..fsatomic import exclusive_lock, write_text_atomic
from ..paths import wiki_root
from . import web


def markdown_text(value: object) -> str:
    """Render untrusted source metadata inert inside Markdown."""
    escaped = html.escape(str(value), quote=True)
    return re.sub(r"([\\`*_{}\[\]()<>#+\-.!|])", r"\\\1", escaped)


# Constructs that can restructure a line from inside a URL: link and image
# brackets, code spans, autolink angles, emphasis runs, and the escape
# character itself. Deliberately narrower than `markdown_text`: `.`, `-`, `+`,
# `#` and `|` cannot open a construct mid-line here (there are no tables in
# this ledger), and `html.escape` is skipped entirely because CommonMark
# renders `\<` as a literal `<` — so HTML still cannot be injected, while `&`
# survives, and every second query string contains one. A source ledger exists
# to make URLs copy-pasteable, so escaping is kept to what safety requires.
_URL_MD_SPECIALS = re.compile(r"([\\`*_\[\]()<>])")


def markdown_url(value: object) -> str:
    """Render a validated source URL inert in Markdown, still copy-pasteable."""
    return _URL_MD_SPECIALS.sub(r"\\\1", str(value))


def report_text(request: dict, receipt: dict, manifest: dict) -> str:
    lines = [
        "# Web Scout — Source Ledger",
        "",
        "> **DISCOVERY ONLY — NOT WIKI EVIDENCE.** The chat agent's native "
        "answer is not stored here. Ingest the underlying PDF before using a "
        "source in wiki prose, claims, or wikilinks.",
        "",
        "## Scope",
        "",
        f"- Run: `{request['run_id']}`",
        f"- Query: {markdown_text(request['query'])}",
        f"- Requested: {request['created_at']}",
        f"- Recorded: {manifest['recorded_at']}",
        f"- Harness: {markdown_text(manifest['harness'])}",
        f"- Sources: {manifest['source_count']} "
        f"({manifest['fetched_count']} harness-reported opened)",
    ]
    if request["constraints"].get("since"):
        dated = sum(
            1 for source in receipt.get("sources", []) if source.get("published_at")
        )
        unknown = manifest["source_count"] - dated
        lines.append(
            f"- Since bound: {request['constraints']['since']} "
            f"({dated} dated; {unknown} without a publication date)"
        )
    lines.extend(["", "## Sources", ""])
    sources = receipt.get("sources") or []
    if not sources:
        lines.extend(["_No sources recorded._", ""])
    for number, source in enumerate(sources, start=1):
        # Both fields are agent-supplied and equally untrusted; they differ only
        # in what they are for. The heading is display text, so a title takes
        # the full escape. The URL line is the artifact a reader copies, so it
        # takes the narrow one — and it previously took neither, leaving a
        # validated-but-attacker-shaped URL able to render as a link to
        # somewhere else entirely.
        title = source.get("title")
        heading = markdown_text(title) if title else markdown_url(source["url"])
        lines.extend([
            f"### {number}. {heading}",
            "",
            f"- URL: {markdown_url(source['url'])}",
            "- Access: " + (
                "harness-reported opened" if source["fetched"]
                else "search result only; page not opened"
            ),
        ])
        if source.get("published_at"):
            lines.append(f"- Published: {source['published_at']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def report_path(run_id: str) -> Path:
    root = wiki_root()
    output = root / "output"
    scout = output / "scout"
    run = scout / web._validate_run_id(run_id)
    if any(path.is_symlink() for path in (output, scout, run)):
        raise web.ScoutInputError("web-scout report directories must not be symlinks")
    return run / "report.md"


def render_report(run_id: str) -> tuple[str, Path, dict]:
    """Render and persist the deterministic discovery-only source ledger."""
    request, receipt, manifest = web._load_recorded(run_id)
    text = report_text(request, receipt, manifest)
    path = report_path(run_id)
    try:
        with exclusive_lock(path):
            write_text_atomic(path, text)
    except OSError as exc:
        raise web.ScoutStorageUnavailable(f"cannot write scout report: {exc}") from exc
    summary = {
        "schema_version": web.SCHEMA_VERSION,
        "run_id": run_id,
        "evidence_class": "discovery-only",
        "source_count": manifest["source_count"],
        "fetched_count": manifest["fetched_count"],
        "report_path": str(path),
    }
    return text, path, summary
