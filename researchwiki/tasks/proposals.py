"""Generate, review, and remember synthesis/idea proposals as synced Markdown.

Usage:
  researchwiki proposals generate "question or topic" [--papers STEM ...]
      [--target-category CAT --cross-category [--search-plan PLAN.json]]
      [--prepare-only] [--write] [--json]
  researchwiki proposals accept PREVIEW.json --select 1 [2 3]
  researchwiki proposals list [--status STATUS] [--json]
  researchwiki proposals feedback ID --decision STATUS --reason TEXT [--actor NAME]
      [--resulting-page LINK]

Markdown under ``wiki/proposals/`` is canonical. The proposal and feedback
tables in ``state.db`` are derived views rebuilt by ``researchwiki db rebuild``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..categories import content_categories
from ..log import append_log_md
from ..proposal_generation import (
    apply_search_plan,
    build_evidence_packet,
    expand_cross_category,
    generate_proposals,
)
from ..proposal_preview import PartialAccept, accept_preview, save_preview
from ..proposals import STATUSES, append_feedback, load_proposals


def _log_line(text: str, limit: int = 160) -> str:
    """One line for a `log.md` entry, whose format is one `## ` H2 per event.

    A multi-line reason can carry its own `## ` headings, and pasting it in
    verbatim created spurious log entries. The full text stays in the
    proposal page's ledger; the log only needs to point at it.
    """
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _log_written(records: list, topic: str, verb: str) -> None:
    if records:
        append_log_md(
            "proposal",
            _log_line(f"{verb} {len(records)} proposal(s) for {topic}"),
            ", ".join(record.proposal_id for record in records),
        )


def _generate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki proposals generate",
        description="Generate up to three question-driven page proposals from wiki claims.",
    )
    parser.add_argument("topic", help="Question, limitation, or topic to investigate.")
    parser.add_argument("--papers", nargs="*", default=[],
                        help="Paper stems that must enter the packet (up to 8; cross-category: "
                             "up to 4 target-category papers). Not every proposal must cite all.")
    parser.add_argument("--target-category", default=None,
                        help="Content category containing the target problem.")
    parser.add_argument("--cross-category", action="store_true",
                        help="Search other categories for methods matching the target problem.")
    parser.add_argument("--search-plan", type=Path, default=None,
                        help="Cross-category only: a JSON plan ({target_problem, "
                             "required_capabilities, queries}) written by the chat "
                             "agent. Replaces the planner call; with --prepare-only "
                             "the packet then carries source evidence too.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Print the retrieved evidence packet without generating proposals.")
    parser.add_argument("--write", action="store_true",
                        help="Save generated proposals under wiki/proposals/. "
                             "Otherwise preview only.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    if args.prepare_only and args.write:
        parser.error("--prepare-only cannot be combined with --write")

    if args.target_category and args.target_category not in content_categories():
        print(f"researchwiki proposals: unknown content category: {args.target_category}",
              file=sys.stderr)
        return 1
    if args.cross_category and not args.target_category:
        print("researchwiki proposals: --cross-category requires --target-category",
              file=sys.stderr)
        return 1
    if args.search_plan and not args.cross_category:
        print("researchwiki proposals: --search-plan requires --cross-category",
              file=sys.stderr)
        return 1
    plan = None
    if args.search_plan:
        try:
            plan = json.loads(args.search_plan.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"researchwiki proposals: cannot read --search-plan: {exc}",
                  file=sys.stderr)
            return 1
    # Everything but --prepare-only makes a model call. Fail on missing
    # credentials or an unattributable author model now, before retrieval, as
    # `agent ingest` does — otherwise an alias like `gpt-5.6` is written into
    # every saved proposal's `author_model:`.
    if not args.prepare_only:
        from ..agents.llm import preflight_providers
        preflight_providers()

    try:
        packet = build_evidence_packet(
            args.topic,
            papers=args.papers,
            target_category=args.target_category,
            cross_category=args.cross_category,
        )
    except (ValueError, OSError) as exc:
        print(f"researchwiki proposals: {exc}", file=sys.stderr)
        return 1

    try:
        if plan is not None:
            packet = apply_search_plan(packet, plan)
    except (ValueError, OSError) as exc:
        print(f"researchwiki proposals: {exc}", file=sys.stderr)
        return 1

    if args.prepare_only:
        print(json.dumps(packet, ensure_ascii=False, indent=2))
        return 0

    try:
        if packet.get("planning_pending"):
            packet = expand_cross_category(packet)
        generated, usage = generate_proposals(packet)
        preview = save_preview(packet, generated, {**packet.get("usage", {}), **usage})
    except (ValueError, OSError) as exc:
        print(f"researchwiki proposals: {exc}", file=sys.stderr)
        return 1

    written = []
    if args.write and generated:
        try:
            records = accept_preview(preview, list(range(1, len(generated) + 1)))
        except PartialAccept as exc:
            # Earlier entries are on disk; record them before reporting, so
            # log.md does not silently miss pages that exist.
            _log_written(exc.written, args.topic, "generated")
            print(f"researchwiki proposals: {exc}", file=sys.stderr)
            print(f"  saved before the failure: "
                  f"{', '.join(str(r.path) for r in exc.written)}", file=sys.stderr)
            print(f"  review the rest in {preview}", file=sys.stderr)
            return 1
        except (ValueError, OSError) as exc:
            print(f"researchwiki proposals: {exc}", file=sys.stderr)
            print(f"  nothing was saved; the preview is at {preview}", file=sys.stderr)
            return 1
        written = [
            {"proposal_id": r.proposal_id, "path": str(r.path), "title": r.title}
            for r in records
        ]
        _log_written(records, args.topic, "generated")

    result = {
        "topic": args.topic,
        "preview": str(preview),
        "packet": packet,
        "cross_category": args.cross_category,
        "n_evidence": len(packet["evidence"]),
        "proposals": generated,
        "written": written,
        "usage": {**packet.get("usage", {}), **usage},
    }
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if not generated:
        print("No worthwhile proposal found in the supplied evidence.")
        return 0
    for i, proposal in enumerate(generated, 1):
        print(f"{i}. [{proposal['page_type']}/{proposal['direction']}] {proposal['title']}")
        print(f"   Question: {proposal['question']}")
        print(f"   Thesis: {proposal['thesis']}")
        if args.write:
            print(f"   Saved: {written[i - 1]['path']}")
        print()
    if not args.write:
        print(f"Full preview and evidence: {preview}")
        print(f"Accept exact selections: researchwiki proposals accept {preview} --select 1")
    return 0


def _accept(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="researchwiki proposals accept")
    parser.add_argument("preview", type=Path, help="Reviewed preview JSON, including chat-authored receipts.")
    parser.add_argument("--select", type=int, nargs="+", required=True,
                        help="One-based proposal numbers to save, without model calls.")
    args = parser.parse_args(argv)
    try:
        records = accept_preview(args.preview, args.select)
    except PartialAccept as exc:
        _log_written(exc.written, str(args.preview), "accepted")
        for record in exc.written:
            print(f"{record.proposal_id}: {record.path}")
        print(f"researchwiki proposals: {exc}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"researchwiki proposals: {exc}", file=sys.stderr)
        return 1
    _log_written(records, str(args.preview), "accepted")
    for record in records:
        print(f"{record.proposal_id}: {record.path}")
    return 0


def _list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="researchwiki proposals list")
    parser.add_argument("--status", choices=sorted(STATUSES), default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    records = [r for r in load_proposals() if not args.status or r.status == args.status]
    if args.as_json:
        print(json.dumps([
            {
                "proposal_id": r.proposal_id,
                "stem": r.stem,
                "title": r.title,
                "status": r.status,
                "direction": r.direction,
                "proposed_page_type": r.proposed_page_type,
                "question": r.question,
                "topic_seed": r.topic_seed,
                "target_category": r.target_category,
                "updated_at": r.updated_at,
                "feedback_count": len(r.feedback),
                "latest_feedback": r.feedback[-1].reason if r.feedback else "",
                "parent_proposal": r.parent_proposal,
                "resulting_page": r.resulting_page,
            }
            for r in records
        ], ensure_ascii=False, indent=2))
        return 0
    if not records:
        print("No proposals found.")
        return 0
    for record in records:
        print(f"{record.proposal_id}  [{record.status}] {record.title}")
        print(f"  {record.proposed_page_type}/{record.direction} — {record.question}")
        if record.resulting_page:
            print(f"  Result: {record.resulting_page}")
        print(f"  {record.path}")
    return 0


def _feedback(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="researchwiki proposals feedback")
    parser.add_argument("identifier", help="Proposal ID or Markdown stem.")
    parser.add_argument("--decision", required=True, choices=sorted(STATUSES))
    parser.add_argument("--reason", required=True,
                        help="Why this proposal was selected, deferred, or rejected.")
    parser.add_argument("--actor", default="user")
    parser.add_argument("--resulting-page", default="",
                        help="Optional resulting wiki page link/path for drafted "
                             "or published proposals.")
    args = parser.parse_args(argv)
    try:
        feedback = append_feedback(
            args.identifier,
            decision=args.decision,
            reason=args.reason,
            actor=args.actor,
            resulting_page=args.resulting_page,
        )
    except ValueError as exc:
        print(f"researchwiki proposals: {exc}", file=sys.stderr)
        return 1
    append_log_md(
        "proposal-feedback",
        _log_line(f"{args.identifier} → {feedback.decision}"),
        _log_line(feedback.reason),
    )
    print(f"recorded {feedback.feedback_id}: {args.identifier} → {feedback.decision}")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print((__doc__ or "").strip())
        return 0
    action, rest = argv[0], argv[1:]
    if action == "generate":
        return _generate(rest)
    if action == "accept":
        return _accept(rest)
    if action == "list":
        return _list(rest)
    if action == "feedback":
        return _feedback(rest)
    print(f"researchwiki proposals: unknown action: {action}", file=sys.stderr)
    return 1
