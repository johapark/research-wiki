"""Import LLM-generated paper pages from an older or simpler wiki.

✅ Use when: bringing in one-paper-per-PDF markdown produced by an older release
   of this framework, or by a simpler "PDF in → summary page out" generator.
   Adds this framework's contract without re-authoring prose.
❌ Don't use: for arbitrary note vaults (no claims to extract, no PDF to grade
   against), or to fix bad prose — that's `prompts/recovery.md`.

    researchwiki migrate preflight               # deps + disk, no writes
    researchwiki migrate inspect  <src-dir>      # read-only classification
    researchwiki migrate apply    [--run DIR]    # land pages; no tokens
    researchwiki migrate verify   [--run DIR]    # did it work?

Phases are ordered so everything free happens before anything paid. `inspect`,
`apply` and `verify` cost **zero tokens**; so does grading, which is local
pypdfium2 + BM25 + a CPU bi-encoder. `apply` prints the token-spending follow-ups
(`backfill hook` / `keywords` / `doi`) rather than running them, so the user
decides when to spend.

Full workflow, failure modes and rollback: `prompts/migration-backfill.md`.

**Coverage gap, stated plainly.** `tests/test_migrate.py` exercises only the
pure modules — `sections.py`, `frontmatter.py`, `classify.py`. Nothing in the
suite imports `migrate/manifest.py`, `migrate/apply.py` or this file, so the run
directory, the journal, the seven-step apply, the pre-apply tarball and `verify`
have never been executed by a test. The phase layer has also seen little use on
real corpora. Treat a `migrate` run as needing your own verification, and read
`inspect`'s output rather than trusting `apply` blindly.

This is recorded rather than fixed because partial coverage on a command with no
known users would buy confidence it hasn't earned; the useful signal is that the
gap exists. `tests/test_refimport_phases.py` is the template if you do backfill
it — same phase-level shape, against a temporary wiki.

Exit codes: 0 success, 1 bad input / nothing actionable, 2 environment
(embedding model unavailable, unreadable source).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
import tarfile
from pathlib import Path

from ..errors import EnvironmentFailure
from ..paths import papers_dir, wiki_dir


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def _iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ---------- preflight ----------

def _run_preflight(args: argparse.Namespace) -> int:
    from ..index import embeddings

    print("# migrate preflight\n")
    ok = True

    # The one hard gate. Grading is free of tokens but leans entirely on this
    # model, and when it's missing `_get_model` swallows the error and grading
    # silently degrades to BM25-only — which means re-grading every paper later.
    available = embeddings.is_available()
    if available:
        print(f"  embedding model    OK   {embeddings.DEFAULT_MODEL} "
              f"on {embeddings.DEFAULT_DEVICE}")
    else:
        ok = False
        print("  embedding model    FAIL unavailable")
        print("       The BGE weights are NOT installed by `pip install -e .` —")
        print("       they're a ~133 MB HuggingFace download on first use. Run any")
        print("       command online once to fetch them, or `pip install -e .` if")
        print("       sentence-transformers itself is missing. Without it every")
        print("       claim grades BM25-only and would need re-grading.")

    if shutil.which("curl"):
        print("  curl               OK")
    else:
        print("  curl               WARN not found — `backfill doi`'s Crossref")
        print("       fallback shells out to it and fails silently without it.")

    n = 0
    if args.src_dir:
        src = Path(args.src_dir)
        if not src.is_dir():
            print(f"\nresearchwiki migrate preflight: no such directory: {src}",
                  file=sys.stderr)
            return 1
        n = len(list(src.glob("*.md")))
        print(f"\n  source pages       {n} under {src}")

    if n:
        # Measured on this corpus: 152 MB / 410 papers.
        est_mb = n * 0.38
        free_mb = shutil.disk_usage(wiki_dir().parent).free / 1e6
        print(f"  .grade-cache est.  ~{est_mb:.0f} MB   (free: {free_mb:.0f} MB)")
        if free_mb < est_mb * 3:
            print("  WARN low disk headroom for the chunk indexes")

    print()
    if not ok:
        raise EnvironmentFailure(
            "embedding model unavailable — see the note above; grading would "
            "silently degrade to BM25-only"
        )
    print("Preflight passed. Next: `researchwiki migrate inspect <src-dir>`")
    return 0


# ---------- inspect ----------

def _run_inspect(args: argparse.Namespace) -> int:
    from ..migrate.classify import assess_all
    from ..migrate.manifest import new_run_dir

    src = Path(args.src_dir)
    if not src.is_dir():
        print(f"researchwiki migrate inspect: no such directory: {src}", file=sys.stderr)
        return 1
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else src

    if args.category and not (wiki_dir() / args.category).is_dir():
        print(f"researchwiki migrate inspect: category '{args.category}' has no "
              f"wiki/{args.category}/ directory — create it first "
              f"(categories are explicit; see CLAUDE.md → Categories).",
              file=sys.stderr)
        return 1

    assessments = assess_all(src, pdf_dir=pdf_dir, category=args.category or "other",
                             require_pdf=not args.allow_no_pdf)
    if args.limit:
        assessments = assessments[: args.limit]
    if not assessments:
        print(f"No .md pages under {src}. Nothing to migrate.")
        return 1

    records = [a.as_dict() for a in assessments]
    run = new_run_dir(_stamp(), base=Path(args.run_dir) if args.run_dir else None)
    run.write_manifest(records, src_dir=src, pdf_dir=pdf_dir,
                       category=args.category or "other", created_at=_iso())

    by_verdict: dict[str, list] = {}
    for a in assessments:
        by_verdict.setdefault(a.verdict, []).append(a)

    if args.as_json:
        print(json.dumps({"run_dir": str(run.root), "items": records}, indent=2))
    else:
        print(f"# migrate inspect — {len(assessments)} page(s) under {src}\n")
        for verdict in ("compliant", "fixable", "needs-human", "blocked", "duplicate"):
            group = by_verdict.get(verdict) or []
            if not group:
                continue
            print(f"## {verdict} ({len(group)})\n")
            for a in group:
                delta = f"{a.claims_before} → {a.claims_after}"
                flag = "  ⚠ still 0" if a.claims_after == 0 and a.page_type == "paper" else ""
                print(f"- {a.src_page.name}")
                print(f"    stem   {a.derived_stem or '(none)'}")
                print(f"    claims {delta}{flag}")
                if a.headings and a.headings.graded_changes:
                    ren = ", ".join(f"{c.original}→{c.canonical}"
                                    for c in a.headings.graded_changes)
                    print(f"    graded renames: {ren}")
                for r in a.reasons:
                    print(f"    ! {r}")
            print()
        n_act = sum(len(by_verdict.get(v) or []) for v in ("compliant", "fixable"))
        print(f"Actionable: {n_act}    manifest: {run.manifest_path}")
        if n_act:
            print(f"\nNext: `researchwiki migrate apply --run {run.root} --dry-run`")

    return 0


# ---------- apply ----------

def _resolve_run(args: argparse.Namespace):
    from ..migrate.manifest import latest_run_dir, open_run_dir
    if args.run:
        return open_run_dir(Path(args.run))
    run = latest_run_dir()
    if run is None:
        print("researchwiki migrate apply: no migrate-* run directory found — "
              "run `researchwiki migrate inspect <src-dir>` first.", file=sys.stderr)
        return None
    return run


def _already_graded(stems: list[str]) -> dict[str, int]:
    """{stem: n_graded_claims} for stems that already carry grading work.

    Re-applying over these would rewrite bodies and mass-NULL grader columns on
    the next `db rebuild`, forcing a full re-grade. Refused unless the user
    explicitly opts in.
    """
    from ..db.safe import safe_read

    if not stems:
        return {}

    def _q(conn):
        marks = ",".join("?" * len(stems))
        return conn.execute(
            f"""SELECT paper_stem, COUNT(*) AS n
                  FROM claims
                 WHERE last_graded_at IS NOT NULL AND paper_stem IN ({marks})
                 GROUP BY paper_stem""",
            stems,
        ).fetchall()

    rows = safe_read(_q, default=[], label="migrate.already_graded")
    return {r["paper_stem"]: r["n"] for r in rows}


def _backup_targets(run, records: list[dict]) -> int:
    """Tar every existing path `apply` would overwrite. Returns the count."""
    victims: list[Path] = []
    for r in records:
        stem = r.get("derived_stem")
        if not stem:
            continue
        page = wiki_dir() / (r.get("target_category") or "other") / f"{stem}.md"
        pdf = papers_dir() / f"{stem}.pdf"
        victims += [p for p in (page, pdf) if p.exists()]
    if not victims:
        return 0
    with tarfile.open(run.backup_path, "w:gz") as tf:
        for p in victims:
            tf.add(p, arcname=str(p.relative_to(wiki_dir().parent)))
    return len(victims)


def _run_apply(args: argparse.Namespace) -> int:
    from ..migrate.apply import apply_page, build_rename_map

    run = _resolve_run(args)
    if run is None:
        return 1
    data = run.read_manifest()
    records = [r for r in data["items"] if r["verdict"] in ("compliant", "fixable")]
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("No actionable pages in the manifest (need verdict compliant/fixable). "
              "Re-run `migrate inspect` after fixing the blockers it listed.")
        return 1

    stems = [r["derived_stem"] for r in records if r.get("derived_stem")]
    graded = _already_graded(stems)
    if graded and not args.force_regrade_ok:
        total = sum(graded.values())
        print(f"researchwiki migrate apply: {len(graded)} stem(s) already carry "
              f"grading work ({total} graded claims).", file=sys.stderr)
        print("Re-applying rewrites their bodies, and the next `db rebuild` NULLs "
              "every grader column whose claim text changed — a full re-grade with "
              "no shortcut. Pass --force-regrade-ok to accept that.", file=sys.stderr)
        for stem, n in sorted(graded.items())[:10]:
            print(f"    {stem}: {n} graded claims", file=sys.stderr)
        return 1

    rename_map, category_of = build_rename_map(records)
    journal = run.read_journal()

    if not args.dry_run:
        n_backed = _backup_targets(run, records)
        if n_backed:
            print(f"Backed up {n_backed} existing target(s) → {run.backup_path}\n")

    migrated_at = _iso()
    landed = skipped = failed = 0
    print(f"# migrate apply — {len(records)} page(s)"
          f"{' (dry run — staging only)' if args.dry_run else ''}\n")
    for i, rec in enumerate(records, 1):
        res = apply_page(rec, run, journal, rename_map=rename_map,
                         category_of=category_of, migrated_at=migrated_at,
                         dry_run=args.dry_run,
                         accept_ambiguous=args.accept_ambiguous)
        mark = {"landed": "ok", "skipped": "staged", "failed": "FAIL"}[res.status]
        print(f"[{i}/{len(records)}] {mark:<6} {res.stem}")
        if res.reason:
            print(f"          {res.reason}")
        landed += res.status == "landed"
        skipped += res.status == "skipped"
        failed += res.status == "failed"

    print(f"\nlanded: {landed}   staged: {skipped}   failed: {failed}")
    if args.dry_run:
        print(f"\nStaged copies: {run.staged_dir}")
        print(f"Review every rewrite:  diff -u {data['src_dir']}/<name>.md "
              f"{run.staged_dir}/<stem>.md")
        print(f"Then: `researchwiki migrate apply --run {run.root}`")
        return 0

    if landed:
        print("\nNext — free (no tokens):")
        print("  researchwiki db rebuild && researchwiki reindex")
        print("  researchwiki lint --json     # zero_claim_papers MUST be 0")
        print("  researchwiki grade regression --missing-only --no-salience")
        print("\nThen — spends tokens:")
        print("  researchwiki backfill doi                 # 0 tokens, lookup only")
        print("  researchwiki backfill keywords            # ~1 call per 10 pages")
        print("  researchwiki backfill hook -w 6           # ~1 cheap call per page")
        print("  researchwiki db rebuild && researchwiki reindex")
        print("\nDeferred, opt-in (dominant token cost):")
        print("  researchwiki claim-overlap <stem> --top 4   # per new stem")
        print(f"\nRollback: `researchwiki migrate verify --run {run.root}` first; "
              f"the journal and staged copies are under {run.root}")
    return 0 if failed == 0 else 1


# ---------- verify ----------

def _run_verify(args: argparse.Namespace) -> int:
    from ..migrate.manifest import open_run_dir, latest_run_dir
    from ..tasks.lint.db_checks import find_zero_claim_papers

    run = open_run_dir(Path(args.run)) if args.run else latest_run_dir()
    if run is None:
        print("researchwiki migrate verify: no run directory found.", file=sys.stderr)
        return 1
    data = run.read_manifest()
    journal = run.read_journal()
    stems = [r["derived_stem"] for r in data["items"]
             if r.get("derived_stem") and r["verdict"] in ("compliant", "fixable")]

    from ..db.safe import safe_read

    def _q(conn):
        if not stems:
            return []
        marks = ",".join("?" * len(stems))
        return conn.execute(
            f"""SELECT p.stem,
                       (SELECT COUNT(*) FROM claims c WHERE c.paper_stem = p.stem) AS n_claims,
                       (SELECT COUNT(*) FROM claims c WHERE c.paper_stem = p.stem
                          AND c.last_graded_at IS NOT NULL) AS n_graded,
                       json_extract(p.raw_frontmatter, '$.hook')     AS hook,
                       json_extract(p.raw_frontmatter, '$.keywords') AS keywords
                  FROM papers p WHERE p.stem IN ({marks}) ORDER BY p.stem""",
            stems,
        ).fetchall()

    rows = safe_read(_q, default=[], label="migrate.verify")
    in_db = {r["stem"] for r in rows}
    missing = [s for s in stems if s not in in_db]
    zero_claims = [r["stem"] for r in rows if not r["n_claims"]]
    ungraded = [r["stem"] for r in rows if r["n_claims"] and not r["n_graded"]]
    no_hook = [r["stem"] for r in rows if not r["hook"]]
    no_kw = [r["stem"] for r in rows if not r["keywords"]]
    committed = sum(
        1 for k in (journal.get("items") or {})
        if (journal["items"][k].get("steps") or {}).get("commit") == "done"
    )

    payload = {
        "run_dir": str(run.root),
        "manifest_actionable": len(stems),
        "journal_committed": committed,
        "in_db": len(in_db),
        "missing_from_db": missing,
        "zero_claim_stems": zero_claims,
        "ungraded_stems": ungraded,
        "missing_hook": no_hook,
        "missing_keywords": no_kw,
        "corpus_zero_claim_papers": [p["stem"] for p in find_zero_claim_papers()],
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
        return 0 if not (missing or zero_claims) else 1

    print(f"# migrate verify — {run.root}\n")
    print(f"  manifest actionable   {len(stems)}")
    print(f"  journal committed     {committed}")
    print(f"  present in DB         {len(in_db)}")
    for label, items, hint in (
        ("missing from DB", missing, "run `researchwiki db rebuild`"),
        ("ZERO claims", zero_claims, "non-canonical headings — fix and db rebuild"),
        ("ungraded", ungraded, "`grade regression --missing-only`"),
        ("missing hook:", no_hook, "`backfill hook`"),
        ("missing keywords:", no_kw, "`backfill keywords`"),
    ):
        if items:
            print(f"\n  {label} ({len(items)}) — {hint}")
            for s in items[:10]:
                print(f"    - {s}")
            if len(items) > 10:
                print(f"    ... ({len(items) - 10} more)")
    if not (missing or zero_claims):
        print("\nEvery migrated page is in the DB with claims extracted.")
    return 0 if not (missing or zero_claims) else 1


# ---------- CLI ----------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki migrate",
        description="Import LLM-generated paper pages from an older or simpler wiki.",
    )
    subs = parser.add_subparsers(dest="phase", required=True, metavar="PHASE")

    pf = subs.add_parser("preflight", help="Check deps and disk before a bulk run.")
    pf.add_argument("src_dir", nargs="?", help="Source dir, to size the run.")
    pf.set_defaults(func=_run_preflight)

    ins = subs.add_parser("inspect", help="Classify an incoming corpus. Read-only.")
    ins.add_argument("src_dir", help="Directory of *.md pages (PDFs beside them).")
    ins.add_argument("--pdf-dir", help="Where the PDFs are, if not beside the pages.")
    ins.add_argument("--category", help="Target wiki/<category>/ (must already exist).")
    ins.add_argument("--limit", type=int, default=0, help="Assess at most N pages.")
    ins.add_argument("--allow-no-pdf", action="store_true",
                     help="Don't block pages whose PDF is missing (they can never "
                          "be graded, so they can't ground a citation).")
    ins.add_argument("--run-dir", help="Where to write the run directory.")
    ins.add_argument("--json", dest="as_json", action="store_true")
    ins.set_defaults(func=_run_inspect)

    ap = subs.add_parser("apply", help="Land assessed pages. No tokens.")
    ap.add_argument("--run", help="Run directory (default: most recent).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Stage and rewrite, but write nothing into wiki/ or papers/.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--accept-ambiguous", action="store_true",
                    help="Also rewrite headings flagged ambiguous (e.g. 'Results "
                         "and Discussion'), using the reported suggestion.")
    ap.add_argument("--force-regrade-ok", action="store_true",
                    help="Proceed even where stems already carry grading work, "
                         "accepting that grader columns will be wiped.")
    ap.set_defaults(func=_run_apply)

    ver = subs.add_parser("verify", help="Check a completed migration.")
    ver.add_argument("--run", help="Run directory (default: most recent).")
    ver.add_argument("--json", dest="as_json", action="store_true")
    ver.set_defaults(func=_run_verify)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)
