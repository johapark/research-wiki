"""Emit the corpus as a bibliography — BibTeX, RIS or CSL-JSON.

✅ Use when: the user wants their wiki library in a reference manager (Zotero,
   Paperpile, Mendeley, EndNote), or a `.bib` for a manuscript in progress.
   Zero tokens, no network, deterministic — two runs are byte-identical.
❌ Don't use: to turn one synthesis or idea page into a document for a human
   reader — that is `prompts/share-page.md`. Nor to bring a library *in*, which
   is `researchwiki import`, the inverse of this command.

    researchwiki export --format bibtex --category cgt > cgt.bib
    researchwiki export --format ris --out library.ris
    researchwiki export --json                        # what an export would contain

Only pages describing a document somebody else published are exported — paper,
commentary, whitepaper, guidance, book. Synthesis, idea and concept pages are the
user's own unpublished analysis with no DOI, venue or year of record, so an entry
for one would assert a publication that does not exist. There is deliberately no
flag to include them.

The citekey is the page stem, because a citekey sitting in someone's manuscript
must never change; any key computed at export time renumbers its disambiguating
letters when a sibling paper is ingested.

Where the corpus is short of data the run reports it rather than guessing: a paper
with no recorded venue is emitted as `@misc` rather than as an `@article` with no
`journal`, and a venue that is really typesetting furniture is dropped. Data
quality never fails the run.

No `log.md` entry: that is for operations that mutate the corpus. `db papers`,
`search` and `lint` do not log either.

Exit codes: 0 success (including zero matches — a filter that matches nothing is
a result), 1 bad argument, 2 the output path can't be written.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..errors import EnvironmentFailure

_FORMATS = ("bibtex", "ris", "csl-json")


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


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="researchwiki export",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--format", default="bibtex", choices=_FORMATS)
    p.add_argument("--out", help="write here instead of stdout (atomic)")
    p.add_argument("--category", action="append",
                   help="repeatable; matches the page's directory")
    p.add_argument("--year", help="YYYY or YYYY-YYYY (inclusive)")
    p.add_argument("--stem", action="append",
                   help="repeatable; export only these pages")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="print the report instead of the bibliography; "
                        "combine with --out to get both")
    args = p.parse_args(argv)

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
