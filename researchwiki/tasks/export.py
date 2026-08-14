"""Emit the corpus — as a bibliography (BibTeX, RIS, CSL-JSON) or as an OKF bundle.

✅ Use when: the user wants their wiki library in a reference manager (Zotero,
   Paperpile, Mendeley, EndNote), a `.bib` for a manuscript in progress, or a
   portable Open Knowledge Format bundle another agent or tool can consume.
   Zero tokens, no network, deterministic — two runs are byte-identical.
❌ Don't use: to turn one synthesis or idea page into a document for a human
   reader — that is `prompts/share-page.md`. Nor to bring a library *in*, which
   is `researchwiki import`, the inverse of this command.

    researchwiki export --format bibtex --category cgt > cgt.bib
    researchwiki export --format ris --out library.ris
    researchwiki export --format okf --out output/okf     # a directory, not a file
    researchwiki export --json                        # what an export would contain

## Two formats, two scopes — this is the thing to get right

The bibliography formats and OKF disagree about *which pages belong*, and both are
correct for their own target. The difference is not configurable, because it
follows from what each format asserts.

**Bibliography scope — published documents only.** `paper`, `commentary`,
`whitepaper`, `guidance`, `book`. Synthesis, idea and concept pages are the user's
own unpublished analysis with no DOI, venue or year of record, so a BibTeX entry
for one would assert a publication that does not exist. There is deliberately no
flag to include them.

**OKF scope — every page.** OKF's unit is a *concept*, defined as "anything you
want to capture" and explicitly covering abstract ideas that have no underlying
resource (spec §2, §4.4). A synthesis page is not a fake publication in that
frame; it is the most valuable thing in the bundle, and omitting it would ship a
knowledge base with its knowledge removed. Pages with nothing published behind
them simply carry no `resource` key, which is what the spec prescribes rather than
a hole in the data. There is deliberately no flag to exclude them.

So: a page absent from the `.bib` and present in the bundle is not an
inconsistency to fix. Anyone tempted to "align" the two lists should change
neither.

**Output shape follows the same way.** A bibliography is one stream and can go to
stdout, so `--out` is optional. An OKF bundle is a directory tree (§3) whose file
paths *are* the concept identities (§2), so `--format okf` requires `--out` and
writes a tree there. `--json` still prints a report — a different report, keyed to
what a bundle can be short of rather than what a citation can.

The citekey is the page stem, because a citekey sitting in someone's manuscript
must never change; any key computed at export time renumbers its disambiguating
letters when a sibling paper is ingested. OKF needs no citekey: the concept's
identity is its path.

Where the corpus is short of data the run reports it rather than guessing: a paper
with no recorded venue is emitted as `@misc` rather than as an `@article` with no
`journal`, and a venue that is really typesetting furniture is dropped. Data
quality never fails the run.

No `log.md` entry: that is for operations that mutate the corpus. `db papers`,
`search` and `lint` do not log either.

Exit codes: 0 success (including zero matches — a filter that matches nothing is
a result), 1 bad argument (including `--format okf` without `--out`), 2 the output
path can't be written or is a non-empty directory that isn't an OKF bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..errors import EnvironmentFailure

_BIB_FORMATS = ("bibtex", "ris", "csl-json")
_FORMATS = (*_BIB_FORMATS, "okf")


def _parse_year(raw: str) -> tuple[int, int] | None:
    """`YYYY` or `YYYY-YYYY` → an inclusive range, or None if malformed.

    Same spellings and the same reversed-range refusal as `db papers`, so the two
    commands cannot disagree about what `--year 2024-2026` selects.
    """
    y = raw.strip()
    if "-" in y and all(p.strip().isdigit() for p in y.split("-", 1)):
        lo, hi = (int(p) for p in y.split("-", 1))
        if lo > hi:
            print(f"researchwiki export: --year range is reversed ({y}); "
                  f"did you mean {hi}-{lo}?", file=sys.stderr)
            return None
        return lo, hi
    if y.isdigit():
        return int(y), int(y)
    print(f"researchwiki export: --year expects YYYY or YYYY-YYYY (got {y!r})",
          file=sys.stderr)
    return None


def _summary(report, fmt: str) -> str:
    bits = [f"{report.records} records"]
    with_doi = report.records - len(report.doi_missing)
    bits.append(f"{with_doi} with DOI")
    bits += [f"{n} @{entry}" for entry, n in sorted(report.by_entry_type.items())]
    if report.venue_missing:
        bits.append(f"{len(report.venue_missing)} with no venue")
    if report.venue_furniture:
        bits.append(f"{len(report.venue_furniture)} venue values suppressed")
    if report.authors_unparseable:
        bits.append(f"{len(report.authors_unparseable)} without a usable byline")
    return f"{fmt}: " + " · ".join(bits)


def _okf_summary(report) -> str:
    bits = [f"{report.concepts} concepts",
            *(f"{n} {t}" for t, n in sorted(report.by_type.items())),
            f"{report.links_rewritten} links rewritten"]
    if report.sources_emitted:
        bits.append(f"{report.sources_emitted} source entries")
    if report.verified_emitted:
        bits.append(f"{report.verified_emitted} machine-verified")
    if report.links_unresolved:
        bits.append(f"{len(report.links_unresolved)} links unresolved")
    return "okf: " + " · ".join(bits)


def _run_okf(args) -> int:
    """`--format okf` — write a bundle directory. See the module docstring for why
    this format's scope and output shape both differ from the bibliography path."""
    from ..okfexport import collect_bundle, looks_like_okf_bundle, write_bundle

    if not args.out:
        print("researchwiki export: --format okf writes a directory tree, so it "
              "needs --out (e.g. --out output/okf). A bundle cannot go to stdout: "
              "each concept's identity is its file path.", file=sys.stderr)
        return 1

    out = Path(args.out)
    if out.exists() and not out.is_dir():
        print(f"researchwiki export: --out {out} exists and is not a directory.",
              file=sys.stderr)
        return 1
    if out.is_dir() and any(out.iterdir()) and not looks_like_okf_bundle(out):
        raise EnvironmentFailure(
            f"refusing to write into {out}: it is not empty and carries no bundle-root "
            f"index.md, so it is not a bundle this command wrote. Point --out at a new "
            f"directory, or clear that one yourself."
        )

    years = None
    if args.year:
        years = _parse_year(args.year)
        if years is None:
            return 1

    files, report = collect_bundle(categories=args.category, years=years,
                                   stems=args.stem)
    try:
        stale = write_bundle(files, out)
    except OSError as e:
        raise EnvironmentFailure(f"cannot write {out}: {e}") from e

    if args.as_json:
        payload = report.as_dict()
        payload["stale_files"] = stale
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(_okf_summary(report), file=sys.stderr)
    print(f"  wrote {len(files)} file(s) under {out}/", file=sys.stderr)
    if report.verified_absent_no_gate_record:
        # Not a defect in the bundle — a gap in what this repo records. Named here
        # because a reader would otherwise read "unverified" as "ungraded".
        print(f"  {len(report.verified_absent_no_gate_record)} synthesis/idea/concept "
              f"page(s) carry no `verified`: gate runs aren't persisted, so no trust "
              f"tier can be claimed for them", file=sys.stderr)
    if report.generated_missing_actor:
        print(f"  {len(report.generated_missing_actor)} page(s) have a write date but "
              f"no recorded author; `generated` omitted rather than invented",
              file=sys.stderr)
    if report.links_unresolved:
        print(f"  {len(report.links_unresolved)} wikilink(s) had no target in the "
              f"selection (kept as text; OKF tolerates broken links)", file=sys.stderr)
    if stale:
        print(f"  {len(stale)} pre-existing file(s) left in place (not written by this "
              f"run): {', '.join(stale[:3])}{' …' if len(stale) > 3 else ''}",
              file=sys.stderr)
    if not report.concepts:
        print("  0 concepts matched the filters.", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="researchwiki export",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--format", default="bibtex", choices=_FORMATS)
    p.add_argument("--out",
                   help="write here instead of stdout (atomic). Required for "
                        "--format okf, where it names a directory.")
    p.add_argument("--category", action="append",
                   help="repeatable; matches the page's directory")
    p.add_argument("--year", help="YYYY or YYYY-YYYY (inclusive)")
    p.add_argument("--stem", action="append",
                   help="repeatable; export only these pages")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="print the report instead of the bibliography; "
                        "combine with --out to get both")
    args = p.parse_args(argv)

    if args.format == "okf":
        return _run_okf(args)

    years = None
    if args.year:
        years = _parse_year(args.year)
        if years is None:
            return 1

    from ..fsatomic import write_text_atomic
    from ..refexport import RENDERERS, collect

    records, report = collect(categories=args.category, years=years,
                              stems=args.stem)
    text = RENDERERS[args.format](records)

    if args.out:
        try:
            write_text_atomic(args.out, text)
        except OSError as e:
            raise EnvironmentFailure(f"cannot write {args.out}: {e}") from e

    # stdout carries exactly one payload. Without `--out` the bibliography owns
    # it, so the summary goes to stderr — which is what keeps `> refs.bib` clean.
    # `--json` claims stdout for the report instead, and then `--out` is the only
    # way to also get the bibliography.
    if args.as_json:
        print(json.dumps(report.as_dict(args.format), indent=2, ensure_ascii=False))
    else:
        if not args.out:
            sys.stdout.write(text)
        print(_summary(report, args.format), file=sys.stderr)
        if args.out:
            print(f"  wrote {args.out}", file=sys.stderr)

    if not records:
        print("  0 records matched the filters.", file=sys.stderr)
    return 0
