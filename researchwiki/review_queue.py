"""Collect what a gate held back for human review, and render it as one page.

✅ Use when: `agent ingest --visual-inspection` left drafts in `.agent-output/`
   and you want to clear the backlog — the page puts the rendered PDF page
   beside the model's reading of it, which is the comparison the receipt asks a
   human to make but cannot show.
❌ Don't use: to answer a factual question, or to decide anything about a paper
   already in `wiki/`. This reports a queue; `claims` and `search` are for
   content.

The receipts are already JSON, so this module exists for the two things reading
them by hand cannot do: it joins each flagged reference to the *picture* the
model looked at, and it separates the items a human can judge now from the ones
that need a re-run first.

**The page image is a cache read, not a render.** `agents/phases/visuals.py`
writes every inspected page to `.figures-cache/inspection/{sha}-p{N}-110.png`
during the ingest, so the common path touches no PDF. That matters more than it
sounds: `paths.resolve_pdf` looks only in `papers/` and the bundled fixtures,
and a *sandboxed* paper is by definition still in `inbox/` under its publisher
filename — so for exactly the papers in this queue, re-rendering is not
available. Measured on the 2026-11-06 backlog: 0 of 7 pending stems resolved
through `resolve_pdf`, while 16 of 16 receipts had their pages cached. A missing
image is therefore a designed state (`image_note`), never an error.

`source_sha256` is what makes the fallback honest: when the cache has been
cleared, hashing `inbox/` finds the exact source bytes the receipt was built
from (0.07 s for 7 files on that backlog) rather than guessing from a filename.

Read-only, zero tokens, no network. The page records no decisions — a reviewer
acts with the ordinary commands, and an approval a gate could *read* would need
a durable ledger, which is deliberately not this.

**Experimental surface** (`visual-inspection` in `data/deprecations.yaml`): the
`--json` contract here, and the receipt shape it reads, are exempt from the beta
compatibility window. The missing approval ledger is the reason — building it
will change both.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .paths import figures_cache_dir, sandbox_dir, wiki_dir, wiki_root

# Bump on a breaking change to the `--json` shape: adding a key is safe,
# renaming or removing one is not. Unlike `visualize` (whose payload only ever
# feeds its own template), this contract is meant to gain review sources, so a
# reader needs to know which shape it is holding.
QUEUE_SCHEMA_VERSION = 1

# A receipt filename is `<stem>-<attempt uuid>.visual.json`. The stem itself can
# contain hyphens, so the uuid is matched by shape and the stem is whatever
# precedes it.
_RECEIPT_RE = re.compile(
    r"^(?P<stem>.+)-(?P<attempt>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
    r"-[0-9a-f]{4}-[0-9a-f]{12})\.visual\.json$"
)

# Statuses that mean nobody is waiting on a human: `inspected` passed, and
# `not_needed` never needed a look. `needs_review` and `incomplete` are the
# queue. A promoted paper is also excluded regardless of status — a receipt
# outlives the run, so the wiki is the authority on whether it still matters.
RESOLVED_STATUSES = frozenset({"inspected", "not_needed"})

# Embedded-image budget for the whole page. The images are the point of the
# page, so the default is generous, but a 40-paper backlog at ~300-500 KB of
# base64 per page would otherwise build a file nothing opens comfortably. The
# cap drops images (reporting how many) rather than shrinking them, because
# re-encoding smaller is exactly what this queue cannot do.
DEFAULT_IMAGE_BUDGET = 8_000_000

_REASON_TEXT = {
    "source_image_review": "read from the image alone; no prose sentence confirmed it",
    "missing_caption_location": "caption never located, so the page is unknown",
    "ambiguous_location": "the label repeats on several pages; none is unambiguous",
    "page_pixel_budget": "not inspected — the two-page/pixel budget was already spent",
    "pixel_budget": "not inspected — the page exceeded the pixel budget",
    "selection_incomplete": "nomination failed, so no request was sent",
    "preparation_failed": "the page could not be prepared for inspection",
    "deadline": "the inspection deadline expired before this reference",
}


def _label_for(ref_id: str) -> str:
    """`edfig:4` → `Extended Data Fig. 4`, the form the papers print."""
    kind, _, number = ref_id.partition(":")
    extended = kind.startswith("ed")
    base = "Table" if kind.endswith("table") else "Fig."
    return ("Extended Data " if extended else "") + f"{base} {number}"


@dataclass(frozen=True)
class ReviewItem:
    """One flagged figure or table: what was asked, what came back, the page."""

    ref_id: str
    label: str
    page: int | None
    priority: str
    reason: str
    reason_text: str
    question: str
    caption: str
    blocking: bool
    has_reading: bool
    observation: str = ""
    visible_labels: str = ""
    units_conditions: str = ""
    uncertainty: str = ""
    source_quote: str = ""
    corroborated: bool = False
    image: str | None = None       # data: URI, or None
    image_note: str | None = None  # why there is no image
    image_width: int | None = None
    image_height: int | None = None

    def as_json(self) -> dict[str, Any]:
        out = {
            "ref_id": self.ref_id, "label": self.label, "page": self.page,
            "priority": self.priority, "reason": self.reason,
            "reason_text": self.reason_text, "question": self.question,
            "caption": self.caption, "blocking": self.blocking,
            "has_reading": self.has_reading, "observation": self.observation,
            "visible_labels": self.visible_labels,
            "units_conditions": self.units_conditions,
            "uncertainty": self.uncertainty, "source_quote": self.source_quote,
            "corroborated": self.corroborated,
            "image_note": self.image_note,
            "image_width": self.image_width, "image_height": self.image_height,
        }
        # The data URI is the bulk of the payload; carry it only when present so
        # `--json` stays readable on a terminal.
        if self.image:
            out["image"] = self.image
        return out


@dataclass(frozen=True)
class ReviewPaper:
    """One held paper and its flagged references."""

    stem: str
    title: str
    attempt_id: str
    status: str
    selection_status: str
    results_source: str
    error: str | None
    receipt_path: str
    draft_path: str | None
    source_pdf: str | None
    gate_reasons: tuple[str, ...]
    items: tuple[ReviewItem, ...]
    unselected: tuple[str, ...]
    model: str
    input_tokens: int
    output_tokens: int
    request_ms: int
    estimated_cost_usd: float | None
    promoted: bool

    @property
    def ready(self) -> bool:
        """True when a human has something to compare right now.

        An errored inspection, or one whose blocking references came back with
        no reading, needs the ingest re-run before anyone can judge it. Keeping
        those apart is the difference between a queue and a list.
        """
        if self.error:
            return False
        return any(item.has_reading for item in self.items if item.blocking) or not any(
            item.blocking for item in self.items
        )

    @property
    def n_blocking(self) -> int:
        return sum(1 for item in self.items if item.blocking)

    def as_json(self) -> dict[str, Any]:
        return {
            "stem": self.stem, "title": self.title, "attempt_id": self.attempt_id,
            "status": self.status, "selection_status": self.selection_status,
            "results_source": self.results_source, "error": self.error,
            "receipt_path": self.receipt_path, "draft_path": self.draft_path,
            "source_pdf": self.source_pdf,
            "gate_reasons": list(self.gate_reasons),
            "unselected": list(self.unselected), "model": self.model,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "request_ms": self.request_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "promoted": self.promoted, "ready": self.ready,
            "n_blocking": self.n_blocking,
            "items": [item.as_json() for item in self.items],
        }


@dataclass
class Queue:
    papers: list[ReviewPaper] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "meta": self.meta,
            "papers": [paper.as_json() for paper in self.papers],
        }


def _receipts(source_dir: Path, *, all_attempts: bool) -> list[tuple[str, str, Path]]:
    """`(stem, attempt_id, path)` for each receipt, newest attempt per stem.

    A re-ingest writes another receipt beside the old one — the 2026-11-06
    backlog had three for one stem — so the newest wins by default. Showing a
    superseded reading is worse than showing none: a reviewer cannot tell it is
    stale from the page.
    """
    found: list[tuple[float, str, str, Path]] = []
    for path in sorted(source_dir.glob("*.visual.json")):
        match = _RECEIPT_RE.match(path.name)
        if not match:
            continue
        found.append(
            (path.stat().st_mtime, match["stem"], match["attempt"], path)
        )
    found.sort(key=lambda row: (row[1], row[0]))
    if all_attempts:
        return [(stem, attempt, path) for _, stem, attempt, path in found]
    newest: dict[str, tuple[str, str, Path]] = {}
    for _, stem, attempt, path in found:  # sorted oldest-first per stem
        newest[stem] = (stem, attempt, path)
    return [newest[stem] for stem in sorted(newest)]


def _page_image(
    sha: str, page: int | None, *, budget: list[int]
) -> tuple[str | None, str | None, int | None, int | None]:
    """The cached render for one page as a data URI, or a note saying why not.

    `budget` is a one-element list used as a mutable counter so the caller can
    cap the whole page without threading a running total through every item.
    """
    if page is None:
        return None, None, None, None
    cached = figures_cache_dir() / "inspection" / f"{sha}-p{page}-110.png"
    if not cached.exists():
        return (
            None,
            "the cached render is gone — `.figures-cache/` is disposable, and "
            "this paper's PDF is still in `inbox/`, so it cannot be re-rendered "
            "from the wiki",
            None,
            None,
        )
    payload = cached.read_bytes()
    encoded_size = len(payload) * 4 // 3
    if encoded_size > budget[0]:
        return (
            None,
            "omitted to keep the page openable — raise `--max-bytes` or narrow "
            "the queue to include it",
            None,
            None,
        )
    budget[0] -= encoded_size
    uri = "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
    return uri, None, None, None


def _gate_reasons(attempt_id: str) -> tuple[str, ...]:
    """Every gate that held this attempt, from the ingest log.

    The receipt knows only its own vetoes; the commit row carries the full list,
    including non-visual ones ("1 critical target claim missing") that change
    what a reviewer has to do. Best-effort by design — this is a file reader
    first, and a locked or absent `state.db` must not turn it into exit 2.
    """
    try:
        from .db.iterations import read_attempt

        rows = [row for row in read_attempt(attempt_id) if row.role == "commit"]
    except Exception:
        return ()
    for row in rows:
        _, _, tail = (row.decision_reason or "").partition("gates_failed:")
        if tail:
            return tuple(part.strip() for part in tail.split(";") if part.strip())
    return ()


def _inbox_by_hash() -> dict[str, Path]:
    """Map `sha256 -> inbox PDF`, so a receipt finds its exact source bytes."""
    out: dict[str, Path] = {}
    inbox = wiki_root() / "inbox"
    if not inbox.is_dir():
        return out
    for path in sorted(inbox.glob("*.pdf")):
        try:
            out[hashlib.sha256(path.read_bytes()).hexdigest()] = path
        except OSError:
            continue
    return out


def _title_and_draft(stem: str) -> tuple[str, str | None]:
    """The draft's title and path, falling back to the stem."""
    draft = sandbox_dir() / f"{stem}.md"
    if not draft.exists():
        return stem.replace("-", " "), None
    try:
        head = draft.read_text(encoding="utf-8")[:2000]
    except OSError:
        return stem.replace("-", " "), str(draft)
    match = re.search(r'^title:\s*"?(.+?)"?\s*$', head, re.MULTILINE)
    return (match.group(1) if match else stem.replace("-", " ")), str(draft)


def _items_for(
    receipt: dict[str, Any], *, budget: list[int], images: bool
) -> tuple[ReviewItem, ...]:
    selected = {
        item["request"]["ref_id"]: item
        for item in receipt.get("selected") or []
        if isinstance(item, dict) and isinstance(item.get("request"), dict)
    }
    observations = {
        obs["ref_id"]: obs
        for obs in receipt.get("observations") or []
        if isinstance(obs, dict) and "ref_id" in obs
    }
    # Why each reference is unresolved. `source_image_review` is the common case
    # and is the least informative, so a more specific reason wins.
    reasons: dict[str, str] = {}
    for entry in receipt.get("unresolved") or []:
        ref = (entry.get("request") or {}).get("ref_id")
        if not ref:
            continue
        if ref not in reasons or reasons[ref] == "source_image_review":
            reasons[ref] = entry.get("reason", "")
    blocking = set(receipt.get("critical_unresolved") or [])
    sha = receipt.get("source_sha256") or ""

    items = []
    for ref in sorted(selected.keys() | blocking, key=lambda r: (r not in blocking, r)):
        request = (selected.get(ref) or {}).get("request") or {}
        obs = observations.get(ref)
        page = (selected.get(ref) or {}).get("page")
        image = note = width = height = None
        if images:
            image, note, width, height = _page_image(sha, page, budget=budget)
        reason = reasons.get(ref, "")
        items.append(ReviewItem(
            ref_id=ref,
            label=_label_for(ref),
            page=page,
            priority=request.get("priority", "critical" if ref in blocking else "high"),
            reason=reason,
            reason_text=_REASON_TEXT.get(reason, reason.replace("_", " ")),
            question=request.get("question", ""),
            caption=(selected.get(ref) or {}).get("caption", ""),
            blocking=ref in blocking,
            has_reading=obs is not None,
            observation=(obs or {}).get("observation", ""),
            visible_labels=(obs or {}).get("visible_labels", ""),
            units_conditions=(obs or {}).get("units_conditions", ""),
            uncertainty=(obs or {}).get("uncertainty", ""),
            source_quote=(obs or {}).get("source_quote", ""),
            corroborated=bool((obs or {}).get("corroborated")),
            image=image, image_note=note, image_width=width, image_height=height,
        ))
    return tuple(items)


def build_queue(
    *,
    source_dir: Path | None = None,
    include_resolved: bool = False,
    all_attempts: bool = False,
    images: bool = True,
    image_budget_bytes: int = DEFAULT_IMAGE_BUDGET,
    gate_reasons: bool = True,
) -> Queue:
    """Collect the pending visual-inspection reviews.

    Resolved receipts are dropped by default: a paper already in `wiki/`, or one
    whose inspection passed, has nobody waiting on it. `include_resolved` keeps
    them for auditing a past run.
    """
    source = source_dir if source_dir is not None else sandbox_dir()
    budget = [max(0, image_budget_bytes)]
    by_hash = _inbox_by_hash() if source.is_dir() else {}
    papers: list[ReviewPaper] = []
    unreadable = 0

    for stem, attempt, path in (_receipts(source, all_attempts=all_attempts)
                                if source.is_dir() else []):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        if not isinstance(receipt, dict):
            unreadable += 1
            continue
        promoted = any(wiki_dir().glob(f"*/{stem}.md"))
        resolved = promoted or receipt.get("status") in RESOLVED_STATUSES
        if resolved and not include_resolved:
            continue
        title, draft = _title_and_draft(stem)
        pdf = by_hash.get(receipt.get("source_sha256") or "")
        papers.append(ReviewPaper(
            stem=stem,
            title=title,
            attempt_id=attempt,
            status=receipt.get("status", ""),
            selection_status=receipt.get("selection_status", ""),
            # Receipts written before `results_source` existed simply omit it.
            results_source=receipt.get("results_source", ""),
            error=receipt.get("error"),
            receipt_path=str(path),
            draft_path=draft,
            source_pdf=str(pdf) if pdf else None,
            gate_reasons=_gate_reasons(attempt) if gate_reasons else (),
            items=_items_for(receipt, budget=budget, images=images),
            unselected=tuple(receipt.get("unselected") or []),
            model=receipt.get("model", ""),
            input_tokens=int(receipt.get("input_tokens") or 0),
            output_tokens=int(receipt.get("output_tokens") or 0),
            request_ms=int(receipt.get("request_ms") or 0),
            estimated_cost_usd=receipt.get("estimated_cost_usd"),
            promoted=promoted,
        ))

    # Cheapest first among the judgeable: a paper with one blocking figure is a
    # single comparison, and clearing it is visible progress.
    papers.sort(key=lambda p: (not p.ready, p.n_blocking, p.stem))
    items = [item for paper in papers for item in paper.items]
    omitted = sum(1 for item in items if item.image_note and "omitted" in item.image_note)
    return Queue(papers=papers, meta={
        "n_papers": len(papers),
        "n_ready": sum(1 for paper in papers if paper.ready),
        "n_items": len(items),
        "n_blocking": sum(1 for item in items if item.blocking),
        "n_blocking_with_reading": sum(
            1 for item in items if item.blocking and item.has_reading
        ),
        "n_corroborated": sum(1 for item in items if item.corroborated),
        "n_images": sum(1 for item in items if item.image),
        "n_images_omitted": omitted,
        "n_unreadable_receipts": unreadable,
        "include_resolved": include_resolved,
        "all_attempts": all_attempts,
        "images": images,
    })


def render_html(queue: Queue) -> str:
    """Inject the queue as JSON into the self-contained template.

    `</` is escaped because the payload lands inside a `<script>` block, where a
    `</script>` inside a string literal — a figure caption or a model's reading
    could contain one — would terminate the block and blank the page.
    """
    template = (
        resources.files("researchwiki")
        .joinpath("templates/inspect.html")
        .read_text(encoding="utf-8")
    )
    payload = json.dumps(queue.as_json(), ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__QUEUE_DATA__", payload)


def write_html(queue: Queue, out: Path) -> Path:
    from .fsatomic import write_text_atomic

    write_text_atomic(out, render_html(queue))
    return out
