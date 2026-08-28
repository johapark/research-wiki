"""CLI for the agent-native web-scout handoff protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import web


def _request(args) -> int:
    request, path = web.create_request(
        args.query,
        max_results=args.max_results,
        max_fetches=args.max_fetches,
        domains=args.domain,
        since=args.since,
    )
    if args.as_json:
        print(json.dumps({**request, "request_path": str(path)}, indent=2))
    else:
        print(f"Web-scout request: {request['run_id']}")
        print(f"Request artifact: {path}")
        print("Use the chat agent's native web-search harness for the answer.")
        print("Then record the URLs it used with:")
        print(f"  researchwiki scout web record {request['run_id']} --harness NAME ...")
    return 0


def _print_recorded(manifest: dict, path: Path, *, as_json: bool) -> int:
    if as_json:
        print(json.dumps({**manifest, "manifest_path": str(path)}, indent=2))
    else:
        print(f"Recorded {manifest['source_count']} discovery-only source(s).")
        print(f"Manifest: {path}")
        print(f"Inspect cache: researchwiki scout web show {manifest['run_id']}")
    return 0


def _record(args) -> int:
    published_pairs = args.published_at or []
    if len({pair[0] for pair in published_pairs}) != len(published_pairs):
        raise web.ScoutInputError("--published-at may be given once per URL")
    published_at = dict(published_pairs)
    manifest, path = web.record_sources(
        args.run_id,
        harness=args.harness,
        fetched_urls=args.fetched,
        snippet_urls=args.snippet,
        published_at=published_at,
    )
    return _print_recorded(manifest, path, as_json=args.as_json)


def _accept(args) -> int:
    if args.receipt_file == "-":
        try:
            incoming = json.load(sys.stdin)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise web.ScoutInputError("stdin is not valid JSON") from exc
        manifest, path = web.accept_submission(args.run_id, incoming)
    else:
        manifest, path = web.accept_receipt(args.run_id, Path(args.receipt_file))
    return _print_recorded(manifest, path, as_json=args.as_json)


def _show(args) -> int:
    run = web.load_run(args.run_id)
    if args.as_json:
        print(json.dumps(run, indent=2))
    else:
        print(f"Run:      {run['run_id']}")
        print(f"State:    {run['state']}")
        print(f"Query:    {run['query']}")
        print(f"Created:  {run['created_at']}")
        print(f"Bounds:   {json.dumps(run['constraints'], ensure_ascii=False)}")
        print(f"Request:  {run['request_path']}")
        cached = run["cached_result"]
        if cached is None:
            print("Cached result: none — this request is awaiting the agent")
            return 0
        receipt = cached["receipt"]
        manifest = cached["manifest"]
        print(f"Recorded: {receipt['recorded_at']}")
        print(f"Harness:  {receipt['harness']}")
        print(
            f"Sources:  {manifest['source_count']} "
            f"({manifest['fetched_count']} harness-reported opened)"
        )
        for source in receipt["sources"]:
            access = "opened" if source["fetched"] else "search-only"
            published = source.get("published_at") or "date-unverified"
            title = f" — {source['title']}" if source.get("title") else ""
            print(f"  - [{access}; {published}] {source['url']}{title}")
        print(f"Receipt:  {cached['receipt_path']}")
        print(f"Manifest: {cached['manifest_path']}")
    return 0


def _list(args) -> int:
    rows = web.list_runs(state=args.state)
    if args.as_json:
        print(json.dumps({"schema_version": web.SCHEMA_VERSION, "runs": rows}, indent=2))
        return 0
    if not rows:
        qualifier = f" in state {args.state!r}" if args.state else ""
        print(f"No web-scout runs{qualifier}.")
        return 0
    print("STATE          CREATED                   RUN ID / QUERY")
    for row in rows:
        query = row.get("query") or row.get("error") or "(unreadable)"
        if len(query) > 72:
            query = query[:69].rstrip() + "..."
        print(
            f"{row['state']:<14} {(row.get('created_at') or '-'):<25} "
            f"{row['run_id']}"
        )
        print(f"{'':<40}{query}")
        if row.get("next_command"):
            print(f"{'':<40}→ {row['next_command']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchwiki scout web",
        description=(
            "Create a bounded handoff for a chat agent's native web-search harness "
            "and retain only a minimal discovery-only source receipt. The CLI "
            "performs no web search and stores no research prose."
        ),
    )
    subs = parser.add_subparsers(dest="action", required=True)

    request = subs.add_parser("request", help="Create an agent handoff request.")
    request.add_argument("query")
    request.add_argument("--max-results", type=int, default=web.DEFAULT_MAX_RESULTS)
    request.add_argument("--max-fetches", type=int, default=web.DEFAULT_MAX_FETCHES)
    request.add_argument("--domain", action="append", default=[])
    request.add_argument("--since")
    request.add_argument("--json", dest="as_json", action="store_true")
    request.set_defaults(func=_request)

    record = subs.add_parser(
        "record", help="Record used URLs directly, without writing JSON."
    )
    record.add_argument("run_id")
    record.add_argument("--harness", required=True)
    record.add_argument("--fetched", action="append", default=[], metavar="URL")
    record.add_argument("--snippet", action="append", default=[], metavar="URL")
    record.add_argument(
        "--published-at", action="append", nargs=2, default=[],
        metavar=("URL", "YYYY-MM-DD"),
        help="Attach a known publication date to a URL (useful with --since).",
    )
    record.add_argument("--json", dest="as_json", action="store_true")
    record.set_defaults(func=_record)

    accept = subs.add_parser("accept", help="Validate a minimal JSON source receipt.")
    accept.add_argument("run_id")
    accept.add_argument("receipt_file", help="JSON file, or - to read stdin")
    accept.add_argument("--json", dest="as_json", action="store_true")
    accept.set_defaults(func=_accept)

    show = subs.add_parser(
        "show", help="Show a request and its cached result, when recorded."
    )
    show.add_argument("run_id")
    show.add_argument("--json", dest="as_json", action="store_true")
    show.set_defaults(func=_show)

    listing = subs.add_parser("list", help="List resumable local web-scout runs.")
    listing.add_argument("--state", choices=web.RUN_STATES)
    listing.add_argument("--json", dest="as_json", action="store_true")
    listing.set_defaults(func=_list)
    return parser


def main(argv: list[str]) -> int:
    # Friendly shorthand: `scout web "query"` means `scout web request "query"`.
    # Keep the removed report action reserved so an old invocation fails with a
    # useful message instead of silently creating a new request for the query
    # "report".
    if argv and argv[0] == "report":
        print(
            "researchwiki scout web: `report` was removed; use `show <run-id>` "
            "to inspect the cached result",
            file=sys.stderr,
        )
        return 1
    actions = {
        "request", "record", "accept", "show", "list", "-h", "--help"
    }
    if argv and argv[0] not in actions:
        argv = ["request", *argv]
    try:
        args = _parser().parse_args(argv)
        return args.func(args)
    except web.ScoutInputError as exc:
        print(f"researchwiki scout web: {exc}", file=sys.stderr)
        return 1
