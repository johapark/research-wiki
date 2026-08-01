"""Author disambiguation via ORCID Public API.

✅ Use when: normalising a YAML `authors` field against a canonical
   source, confirming that two paper-author strings refer to the same
   person, or recording a time-anchored affiliation for a paper's
   senior author. Rule-1 structured-API carve-out.
❌ Don't use: as a general author-biography source. ORCID returns
   biography and researcher-url fields — those are prose and are not
   re-exposed by this tool. Don't paraphrase from what you see in the
   ORCID web page either; the prose ban applies equally.

Two entry points:
  --orcid 0000-0000-0000-0000      Direct ID lookup.
  --name "Given Family"            Name search; returns top-5 candidates.
  --given X --family Y             Same search, pre-split.

Exit codes:
  0 — ran to completion (zero results is still 0)
  1 — user-input error (no flag, or malformed ORCID ID)
  2 — environment error (ORCID unreachable after retries)
"""

from __future__ import annotations

import argparse
import json

from ..providers.orcid import ORCID_ID_RE, lookup_by_id, search_by_name


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki orcid-lookup",
        description="Look up an ORCID record by ID, or search by author name.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--orcid", help="ORCID ID (0000-0000-0000-0000)")
    group.add_argument("--name", help="Full name 'Given Family' — split on last space")
    parser.add_argument("--given", help="Given name (with --family)")
    parser.add_argument("--family", help="Family name (with --given)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max candidates to return on name search (default 5)")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit structured JSON. Schema: "
                             "{orcid, given_names, family_name, credit_name, "
                             "other_names, latest_affiliation, source, fetched_at}.")
    args = parser.parse_args(argv)

    if args.orcid:
        orcid = args.orcid.strip().upper()
        if not ORCID_ID_RE.match(orcid):
            print(f"error: '{args.orcid}' is not a valid ORCID ID "
                  f"(expected 0000-0000-0000-0000 format)", flush=True)
            return 1
        record = lookup_by_id(orcid)
        if args.as_json:
            print(json.dumps(record, indent=2))
        else:
            _print_prose([record])
        return 0

    # Name search
    given = args.given or ""
    family = args.family or ""
    if args.name and not (given or family):
        parts = args.name.strip().rsplit(" ", 1)
        if len(parts) == 2:
            given, family = parts[0], parts[1]
        else:
            family = parts[0]
    if not (given or family):
        print("error: --name or (--given --family) required", flush=True)
        return 1

    records = search_by_name(given=given, family=family, limit=args.limit)

    if args.as_json:
        print(json.dumps({
            "query": {"given": given, "family": family, "limit": args.limit},
            "results": records,
        }, indent=2))
        return 0

    print(f"Search: given={given!r} family={family!r} → {len(records)} candidate(s)")
    print()
    if records:
        _print_prose(records)
    else:
        print("_no matches._")
    return 0


def _print_prose(records: list[dict]) -> None:
    for r in records:
        if not r["orcid"]:
            continue
        name_parts = [r["given_names"], r["family_name"]]
        name = " ".join(p for p in name_parts if p) or "(no name)"
        credit = f" (credit: {r['credit_name']})" if r["credit_name"] else ""
        aff = r["latest_affiliation"]
        aff_str = ""
        if aff and aff["organization"]:
            role = f", {aff['role']}" if aff["role"] else ""
            span = ""
            if aff["start_year"]:
                span = f" [{aff['start_year']}–{aff['end_year'] or 'present'}]"
            aff_str = f" · {aff['organization']}{role}{span}"
        print(f"- **{r['orcid']}** — {name}{credit}{aff_str}")
