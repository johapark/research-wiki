"""Run the Phase 2 ingest agent on one or more PDFs.

The agent runs reconcile → extract → author × N → grade × N → tournament →
commit, persisting every step to ingest_iterations. The output wiki page is
written to .agent-output/{stem}.md (NOT wiki/) so you can review before
promoting it manually.

With ≥2 PDFs (or an explicit --workers/--resume), the invocation auto-enters
crash-safe batch mode: workers run in parallel, a checkpoint under
.ingest/batch-<ts>/ records each completion atomically, and a mid-batch
crash is recoverable via `--resume <batch-dir>`.

Usage:
  researchwiki agent ingest <pdf-path>              # single PDF, real Anthropic API
  researchwiki agent ingest <pdf-path> --stub       # offline framework test
  researchwiki agent ingest <pdf-path> -n 3         # 3 drafts in tournament
  researchwiki agent ingest inbox/*.pdf             # batch, 4 workers, checkpoint
  researchwiki agent ingest inbox/*.pdf -w 2        # batch, 2 workers
  researchwiki agent ingest --resume .ingest/batch-.../   # resume after a crash
  researchwiki agent trace <attempt-id>             # print one attempt's iteration log
  researchwiki agent eval --label baseline          # A/B regression eval over a fixed paper set
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

from ..agents.runner import ReconcileFailed, StemRenameRefused, run_ingest
from ..db.iterations import read_attempt
from ..paths import papers_dir, wiki_root
from ..log import log


_BATCH_INCOMPATIBLE_FLAGS = (
    ("doi", "--doi"),
    ("title", "--title"),
    ("authors", "--authors"),
    ("year", "--year"),
    ("author_prompt_file", "--author-prompt-file"),
    ("supplementary", "--supplementary"),
    ("allow_rename", "--allow-rename"),
)


def _batch_passthrough_args(args) -> list[str]:
    """Emit only flags whose value diverges from the argparse default, so the
    per-worker `python -m researchwiki agent ingest` gets the same effective
    behavior without a verbose command line."""
    out: list[str] = []
    if args.stub:
        out.append("--stub")
    if args.no_semantic:
        out.append("--no-semantic")
    if args.verify_claim_entailment:
        out.append("--verify-claim-entailment")
    if args.no_cross_link:
        out.append("--no-cross-link")
    if args.auto_promote:
        out.append("--auto-promote")
    if args.force_sandbox:
        out.append("--force-sandbox")
    # n_drafts is already resolved to a concrete int by the time batch mode
    # reconstructs per-PDF commands (see _cmd_ingest), so pass it through
    # verbatim — the whole batch then uses exactly what the parent resolved
    # rather than each subprocess re-resolving against config.
    out += ["-n", str(args.n_drafts)]
    if args.max_evolve != 1:
        out += ["--max-evolve", str(args.max_evolve)]
    if not args.llm_reconcile:
        out.append("--no-llm-reconcile")
    return out


def _cmd_ingest(args) -> int:
    # --resume takes over completely: the batch dir's plan.json is the
    # source of truth for subcommand + passthrough flags.
    if args.resume:
        from . import _ingest_batch
        return _ingest_batch.resume_batch(
            Path(args.resume).expanduser().resolve(),
            no_retry=args.no_retry,
            workers_override=args.workers,
        )

    if not args.pdfs:
        print("researchwiki agent ingest: need PDF path(s) (or --resume BATCH_DIR)",
              file=sys.stderr)
        return 2

    # Resolve n_drafts once, before batch reconstruction or single-PDF run:
    # CLI `-n` wins; else the models config's `ingest.n_drafts`; else 1 (single
    # draft — the multi-draft author tournament is opt-in, since it roughly
    # doubles author-call cost for a marginal, ~1-in-3 quality gain). Batch
    # mode passes the resolved value through to each per-PDF subprocess.
    if args.n_drafts is None:
        from ..agents import model_config
        cfg_default = model_config.default_n_drafts()
        args.n_drafts = cfg_default if cfg_default is not None else 1

    # Batch mode triggers on multi-PDF OR explicit --workers. --workers=1 with
    # one PDF also enters batch mode — useful for getting a checkpoint dir on
    # a serial run.
    batch_mode = len(args.pdfs) > 1 or args.workers is not None
    if batch_mode:
        # Per-PDF overrides don't have a meaning on N PDFs. Reject with a
        # clear list of what's in conflict rather than silently ignoring.
        conflicts = [flag for attr, flag in _BATCH_INCOMPATIBLE_FLAGS
                     if getattr(args, attr, None)]
        if conflicts:
            print(f"researchwiki agent ingest: cannot combine batch mode "
                  f"({len(args.pdfs)} PDFs / --workers set) with per-PDF flags: "
                  f"{', '.join(conflicts)}",
                  file=sys.stderr)
            print("  fix: run these PDFs one at a time, or drop the flags.",
                  file=sys.stderr)
            return 2
        if args.auto_promote and args.force_sandbox:
            print("researchwiki agent ingest: --auto-promote and --force-sandbox "
                  "are mutually exclusive", file=sys.stderr)
            return 2
        from . import _ingest_batch
        return _ingest_batch.new_batch(
            args.pdfs, ["agent", "ingest"],
            _batch_passthrough_args(args),
            workers=args.workers if args.workers is not None else 4,
        )

    # Single-PDF path — unchanged from before.
    pdf = args.pdfs[0]
    promote_mode = "auto"
    if args.auto_promote and args.force_sandbox:
        print("researchwiki agent ingest: --auto-promote and --force-sandbox are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.auto_promote:
        promote_mode = "always"
    elif args.force_sandbox:
        promote_mode = "never"

    try:
        ctx = run_ingest(
            pdf,
            use_stub=args.stub,
            use_semantic=not args.no_semantic,
            verify_claim_entailment=args.verify_claim_entailment,
            n_drafts=args.n_drafts,
            max_evolve=args.max_evolve,
            promote_mode=promote_mode,
            doi_override=args.doi,
            title_override=args.title,
            year_override=args.year,
            authors_override=(
                [a.strip() for a in args.authors.split(";") if a.strip()]
                if args.authors else None
            ),
            author_prompt_override=args.author_prompt_file,
            supplementary=(
                [Path(s) for s in args.supplementary] if args.supplementary else None
            ),
            use_llm_reconcile=args.llm_reconcile,
            allow_rename=args.allow_rename,
        )
    except ReconcileFailed as e:
        # Focused diagnostic for the common case where the metadata
        # cascade gave up — print the providers tried + which fields
        # ended up null, and a copy-pasteable retry hint. No stack
        # trace; ReconcileFailed is a known-failure mode, not a bug.
        print(
            "researchwiki agent ingest: reconcile failed — could not derive a paper stem.",
            file=sys.stderr,
        )
        print(f"  sources tried: {', '.join(e.sources) or '(none)'}", file=sys.stderr)
        print(f"  missing fields: {', '.join(e.missing) or '(unknown)'}", file=sys.stderr)
        # Suggest the most-targeted fix the user can apply. DOI is the
        # highest-leverage override — once a DOI lands, S2/Crossref
        # generally fill in title + year + authors automatically.
        if "doi" in e.missing:
            print(
                "  fix: re-run with `--doi <doi>` if you can find it in the PDF "
                "(SSRN/arXiv URL footer, journal page header, or "
                "https://doi.org link in the references list).",
                file=sys.stderr,
            )
        elif "year" in e.missing:
            print("  fix: re-run with `--year YYYY` (and `--title` if needed).", file=sys.stderr)
        else:
            print(
                "  fix: re-run with `--doi`/`--title`/`--year`/`--authors` overrides "
                "(see `researchwiki agent ingest --help`).",
                file=sys.stderr,
            )
        return 2
    except StemRenameRefused as e:
        # Known-failure mode: re-ingest would orphan a prior page. Surface
        # the prior + new stems and the two ways forward, no stack trace.
        print(
            "researchwiki agent ingest: refusing to rename an existing page.",
            file=sys.stderr,
        )
        print(f"  prior stem: {e.prior_stem}", file=sys.stderr)
        print(f"  new stem:   {e.new_stem}", file=sys.stderr)
        print(
            "  fix: pass --year/--title/--authors to lock the new stem to "
            "the prior one,",
            file=sys.stderr,
        )
        print(
            "       or pass --allow-rename if the rename is intentional "
            "(prior page will be moved).",
            file=sys.stderr,
        )
        return 2
    except Exception as e:
        print(f"researchwiki agent ingest: error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2

    print()
    log(f"attempt_id   = {ctx.attempt_id}", tag="agent")
    log(f"paper_stem   = {ctx.paper_stem}", tag="agent")
    log(f"committed    = {ctx.committed_path}", tag="agent")
    print()
    print("Inspect the trace with:")
    print(f"  researchwiki agent trace {ctx.attempt_id}")

    # Proactive claim-grounded cross-linking. Runs off the committed page's
    # claims (they aren't in the DB until `db rebuild`), auto-adds reciprocal
    # Related-Papers links for LLM-confirmed matches. Opt out with
    # --no-cross-link; naturally no-ops under --stub / --no-semantic (no judge /
    # no bi-encoder) and when the page was sandboxed rather than promoted.
    # Post-commit hooks — skip when the ingest refused to promote (duplicate PDF,
    # sandbox, etc.). committed_path is None in those cases and the hooks would
    # crash on the None path since they all take an os.PathLike.
    if (not args.no_cross_link and not args.stub
            and ctx.paper_stem and ctx.committed_path):
        from . import claim_overlap
        claim_overlap.run_after_ingest(ctx.paper_stem, ctx.committed_path)
        # Attach the new paper to any existing concept hub whose term it
        # mentions (spoke + reciprocal link). No-ops until concept pages exist.
        from .. import concepts
        concepts.attach_after_ingest(ctx.paper_stem, ctx.committed_path)
        # Contradiction alert: any claim in the new paper that disagrees
        # with a graded existing claim surfaces as `⚠ contradicts [[stem#slug]]`.
        # Silent no-op when the LLM judge / bi-encoder isn't reachable.
        from ..claim_graph.alert import alert_after_ingest
        alert_after_ingest(ctx.paper_stem, ctx.committed_path)

    # Saturation check. If this paper landed in wiki/other/ and pushed the
    # bucket over threshold, surface the suggest-splits nudge once. The
    # decay-stamp inside the helper suppresses repeats for 7 days across
    # status / digest-ingest / agent-ingest, so a `for f in ...; do
    # researchwiki agent ingest "$f"; done` loop only nags once.
    from ..categories import other_saturation_warning
    msg = other_saturation_warning()
    if msg:
        print()
        print(msg)
    return 0


def _cmd_trace(args) -> int:
    rows = read_attempt(args.attempt_id)
    if not rows:
        print(f"no rows for attempt_id={args.attempt_id}", file=sys.stderr)
        return 1

    log(f"attempt_id={args.attempt_id}  ({len(rows)} rows)", tag="trace")
    print(f"  {'#':<3} {'role':<10} {'sect':<5} {'model':<22} {'in':>5} {'out':>5} "
          f"{'sem':>5} {'BM25':>5}  decision")
    for r in rows:
        sec = (r.section or "")[:5]
        model = (r.model_used or "")[:22]
        ti = r.cost_input_tokens if r.cost_input_tokens is not None else ""
        to = r.cost_output_tokens if r.cost_output_tokens is not None else ""
        sem = ""
        bm25 = ""
        if r.grader_scores:
            s = r.grader_scores.get("semantic_score")
            b = r.grader_scores.get("mean_bm25")
            sem = f"{s:.2f}" if s is not None else ""
            bm25 = f"{b:.1f}" if b is not None else ""
        decision = r.decision or ""
        if r.decision_reason:
            decision = f"{decision} — {r.decision_reason[:60]}"
        print(f"  {r.iteration:<3} {r.role:<10} {sec:<5} {model:<22} "
              f"{str(ti):>5} {str(to):>5} {sem:>5} {bm25:>5}  {decision}")

    # Critic notes are interesting — surface them by default.
    critic_rows = [r for r in rows if r.role == "critic" and r.critic_notes]
    if critic_rows:
        print()
        log("critic notes:", tag="trace")
        for r in critic_rows:
            print(f"\n  --- iter #{r.iteration} (id={r.id}) ---")
            print("  " + r.critic_notes[:800].replace("\n", "\n  "))

    if args.show_drafts:
        print()
        log("drafts (truncated to 400 chars each):", tag="trace")
        for r in rows:
            if r.role == "author" and r.draft_text:
                t_str = f", t={r.temperature:.2f}" if r.temperature is not None else ""
                print(f"\n  --- iter #{r.iteration} (id={r.id}{t_str}) ---")
                print("  " + r.draft_text[:400].replace("\n", "\n  "))

    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="researchwiki agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = subs.add_parser("ingest", help="Run the ingest agent on one or more PDFs")
    p_ingest.add_argument("pdfs", nargs="*",
                          help="Paths to PDFs (in inbox/ or papers/). Passing two or "
                               "more auto-activates crash-safe batch mode with a "
                               "checkpoint under .ingest/batch-<ts>/.")
    p_ingest.add_argument("--workers", "-w", type=int, default=None,
                          help="Concurrent subprocesses in batch mode. Implies batch "
                               "mode when set — passing `-w 1` with a single PDF gives "
                               "you a checkpoint dir for a serial run. Default 4.")
    p_ingest.add_argument("--resume", metavar="BATCH_DIR", default=None,
                          help="Continue an interrupted batch. Reads its plan.json + "
                               "checkpoint.json and re-runs only what's pending.")
    p_ingest.add_argument("--no-retry", action="store_true",
                          help="With --resume: skip PDFs that previously failed "
                               "(default is to retry them).")
    p_ingest.add_argument("--stub", action="store_true",
                          help="Use deterministic stub LLM (no Anthropic API call)")
    p_ingest.add_argument("--no-semantic", action="store_true",
                          help="Skip the bi-encoder semantic scorer in tournament "
                               "ranking (BM25 + numeric drift + negation only — "
                               "faster, less reliable)")
    p_ingest.add_argument("--verify-claim-entailment", action="store_true",
                          help="Gate auto-promotion on a per-claim entailment check: "
                               "each claim is judged against its cited source passage "
                               "(one batched LLM call), and any claim the source does "
                               "not entail vetoes promotion — the qualitative analogue "
                               "of the numeric-drift veto. Off by default (opt-in).")
    p_ingest.add_argument("-n", "--n-drafts", type=int, default=None,
                          help="Number of author drafts. >1 runs a tournament "
                               "(pick the best-graded draft); 1 skips it. Overrides "
                               "the models config's `ingest.n_drafts`; falls back to 1 "
                               "(tournament off) when neither is set — pass -n 2+ to "
                               "opt into the tournament.")
    p_ingest.add_argument("--max-evolve", type=int, default=1,
                          help="Max critic+evolve rounds after tournament (default 1, 0 to disable)")
    p_ingest.add_argument("--auto-promote", action="store_true",
                          help="Always promote to wiki/ even if gates fail (use cautiously)")
    p_ingest.add_argument("--force-sandbox", action="store_true",
                          help="Always write to .agent-output/, never promote to wiki/")
    p_ingest.add_argument("--no-cross-link", action="store_true",
                          help="Skip the post-promote claim-overlap cross-linking pass.")
    p_ingest.add_argument("--doi", default=None,
                          help="Override the DOI (skip in-text DOI detection — use when "
                               "the PDF contains fragments of neighboring articles)")
    p_ingest.add_argument("--title", default=None,
                          help="Override the title (use when pypdf metadata title is wrong "
                               "and S2 lookup is unavailable)")
    p_ingest.add_argument("--authors", default=None,
                          help="Override the author list (semicolon-separated). Use when "
                               "body-text author parsing picks the wrong name and S2 has "
                               "no record of the paper.")
    p_ingest.add_argument("--year", type=int, default=None,
                          help="Override the publication year (4-digit integer). Use when "
                               "the PDF lacks an in-text 'Accepted:'/'Published:' header "
                               "and S2 has no record (common for fresh preprints with no "
                               "DOI). Without a year, stem derivation can't run.")
    p_ingest.add_argument("--author-prompt-file", default=None,
                          help="Override the author system prompt with a custom file "
                               "(e.g., prompts/author-system-experimental.md). Used by "
                               "the A/B regression eval to test prompt variants without "
                               "editing source.")
    p_ingest.add_argument("--allow-rename", action="store_true", default=False,
                          help="Allow committing when reconcile finds an existing "
                               "wiki page (DOI match) at a different stem. By default "
                               "the runner refuses to silently rename — orphaning the "
                               "prior page and moving the PDF — and tells you which "
                               "override flags would lock the stem.")
    p_ingest.add_argument("--llm-reconcile", action=argparse.BooleanOptionalAction,
                          default=True,
                          help="Use the configured `extractor` role (config/"
                               "models.yaml — default Haiku 4.5) to extract title "
                               "/ authors / year / DOI / paper_type from the first "
                               "1-2 PDF pages. ON by default after R3 dogfooding "
                               "(10 papers, 0 fabrications). Use --no-llm-reconcile "
                               "for offline/stub mode or to opt out of the ~$0.001 "
                               "per-ingest cost. S2 still authoritative on its "
                               "fields when DOI resolves; LLM fills gaps.")
    p_ingest.add_argument("--supplementary", dest="supplementary", action="append",
                          default=None,
                          help="Path to a supplementary file to attach alongside the "
                               "primary PDF. Repeat for multiple files. Files are staged "
                               "into papers/{stem}.supp/ AFTER promote_to_wiki succeeds, "
                               "and entries are appended to the new page's `supplementary:` "
                               "YAML block. Defaults: PDF→kind=methods, xlsx/csv/tsv→"
                               "kind=data, other→kind=other.")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_trace = subs.add_parser("trace", help="Print every iteration row for an attempt_id")
    p_trace.add_argument("attempt_id")
    p_trace.add_argument("--show-drafts", action="store_true", help="Also print draft text")
    p_trace.set_defaults(func=_cmd_trace)

    p_eval = subs.add_parser(
        "eval",
        help="A/B regression eval — run ingest with --force-sandbox over a "
             "fixed paper set, dump JSON of grader scores per paper.",
    )
    p_eval.add_argument(
        "--papers", default="eval/agent-fixtures/papers.txt",
        help="File listing paper stems (one per line, # comments allowed). "
             "Stems must exist as papers/{stem}.pdf. "
             "Default: eval/agent-fixtures/papers.txt.",
    )
    p_eval.add_argument(
        "--label", default=None,
        help="Tag for this eval run, used in the output filename. "
             "Default: timestamp.",
    )
    p_eval.add_argument(
        "-n", "--n-drafts", type=int, default=1,
        help="Author drafts per paper (default 1, tournament off). "
             "Match the prod default.",
    )
    p_eval.add_argument(
        "--max-evolve", type=int, default=1,
        help="Max critic+evolve rounds (default 1). Match the prod default.",
    )
    p_eval.add_argument(
        "--no-semantic", action="store_true",
        help="Skip the bi-encoder semantic scorer (BM25 + negation only). "
             "Faster, less reliable.",
    )
    p_eval.add_argument(
        "--out-dir", default=".eval-output",
        help="Directory for the JSON summary (default: .eval-output, gitignored).",
    )
    p_eval.add_argument(
        "--author-prompt-file", default=None,
        help="Override the author system prompt with a custom file. "
             "Use to A/B-test prompt variants: run with --label baseline, "
             "edit/copy the prompt, run with --label variant "
             "--author-prompt-file <path>, then diff the JSONs.",
    )
    p_eval.add_argument(
        "--max-workers", type=int, default=2,
        help="Run papers concurrently in N subprocesses (default 2). Each "
             "worker loads its own bi-encoder embedding model (~133 MB, CPU); "
             "raise as far as your CPU/RAM tolerates. Each paper always runs "
             "in a fresh subprocess (maxtasksperchild=1) to keep allocator "
             "growth from leaking across papers.",
    )
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def _eval_one_paper(stem: str, args, pdfs_root: Path) -> dict:
    """Run the ingest agent on one paper and return its eval-row dict.

    Each paper runs in a fresh subprocess so the embedding model and any
    other process-local state are reset between runs — keeps allocator
    growth from leaking across papers in long batch evals.
    """
    pdf_path = pdfs_root / f"{stem}.pdf"
    if not pdf_path.exists():
        return {"stem": stem, "error": f"pdf not found at {pdf_path}"}

    t0 = time.time()
    try:
        ctx = run_ingest(
            pdf_path,
            use_stub=False,
            use_semantic=not args.no_semantic,
            n_drafts=args.n_drafts,
            max_evolve=args.max_evolve,
            promote_mode="never",   # force-sandbox so wiki/ is never touched
            author_prompt_override=args.author_prompt_file,
        )
    except Exception as e:
        return {
            "stem": stem,
            "error": f"ingest raised: {e!s}",
            "elapsed_s": round(time.time() - t0, 1),
        }
    elapsed = time.time() - t0

    from ..agents.promote import _count_key_contributions, should_auto_promote
    from ..agents.phases import verify_crosslinks

    winner_scores = ctx.winner.scores if ctx.winner else {}
    first_author_id = (
        ctx.drafts[0].iteration_id if ctx.drafts and len(ctx.drafts) >= 1 else None
    )
    evolved_swap = bool(
        ctx.winner and first_author_id is not None
        and ctx.winner.iteration_id != first_author_id
        and ctx.winner.iteration_id != (ctx.drafts[1].iteration_id if len(ctx.drafts) >= 2 else None)
    )
    total_in = sum((d.input_tokens or 0) for d in ctx.drafts)
    total_out = sum((d.output_tokens or 0) for d in ctx.drafts)
    n_kc = _count_key_contributions(ctx.winner.text) if ctx.winner else 0

    if ctx.winner:
        # Re-run the gate against the winner's scores. ctx.committed_path
        # always lives under .agent-output/ in promote_mode="never", so the
        # path can't tell us whether the structural gate would have passed.
        _, verification = verify_crosslinks(ctx.winner.text, ctx.crosslink_candidates)
        gate = should_auto_promote(
            scores=winner_scores,
            verification=verification,
            n_key_contributions=n_kc,
            paper_type=(ctx.metadata or {}).get("paper_type", "research"),
        )
        gate_passed = gate.promoted
        gate_reasons = gate.reasons
    else:
        gate_passed = False
        gate_reasons = ["no winner"]

    return {
        "stem": stem,
        "attempt_id": ctx.attempt_id,
        "paper_type": (ctx.metadata or {}).get("paper_type"),
        "winner_scores": {
            "semantic_score": winner_scores.get("semantic_score"),
            "mean_bm25": winner_scores.get("mean_bm25"),
            "n_drift": winner_scores.get("n_drift"),
            "n_negation_mismatches": winner_scores.get("n_negation_mismatches"),
            "n_graded": winner_scores.get("n_graded"),
            "semantic_available": winner_scores.get("semantic_available"),
        },
        "n_kc": n_kc,
        "evolved_swap": evolved_swap,
        "gate_passed": gate_passed,
        "gate_reasons": gate_reasons,
        "n_drafts_total": len(ctx.drafts),
        "tokens_drafts_total": {"input": total_in, "output": total_out},
        "elapsed_s": round(elapsed, 1),
    }


def _cmd_eval(args) -> int:
    papers_file = Path(args.papers)
    if not papers_file.exists():
        print(f"researchwiki agent eval: papers file not found: {papers_file}",
              file=sys.stderr)
        return 2

    stems: list[str] = []
    for line in papers_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            stems.append(line)
    if not stems:
        print(f"researchwiki agent eval: {papers_file} has no stems.", file=sys.stderr)
        return 2

    label = args.label or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"agent-eval-{label}.json"
    pdfs_root = papers_dir()

    # multiprocessing.Pool over concurrent.futures because we need
    # maxtasksperchild=1 (recycle the worker after each paper to defend
    # against the MiniCheck/MPS allocator leak), and ProcessPoolExecutor's
    # max_tasks_per_child only landed in 3.11.
    from multiprocessing import Pool
    n_workers = max(1, args.max_workers)
    log(f"running {len(stems)} paper(s) with {n_workers} worker(s); "
          f"each paper in a fresh subprocess (maxtasksperchild=1)", tag="eval")

    results_by_stem: dict[str, dict] = {}
    with Pool(processes=n_workers, maxtasksperchild=1) as pool:
        # apply_async returns immediately; workers run in parallel.
        async_results = [
            (stem, pool.apply_async(_eval_one_paper, (stem, args, pdfs_root)))
            for stem in stems
        ]
        # Drain in submission order — simpler than as_completed-style
        # tracking, and at fixture scale (single-digit papers) the
        # head-of-line waits don't actually delay anything: workers run
        # concurrently underneath regardless of report order.
        for i, (stem, ar) in enumerate(async_results, 1):
            try:
                row = ar.get()
            except Exception as e:
                row = {"stem": stem, "error": f"worker raised: {e!s}"}
            results_by_stem[stem] = row
            err = row.get("error")
            elapsed = row.get("elapsed_s", "?")
            tag = f"ERROR — {err}" if err else "ok"
            print(f"[eval {i}/{len(stems)}] {stem} → {tag} ({elapsed}s)",
                  file=sys.stderr)

    # Preserve fixture order in the JSON.
    results = [results_by_stem[stem] for stem in stems]

    successful = [r for r in results if "error" not in r and r.get("winner_scores")]
    summary = {
        "label": label,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "fixture": str(papers_file),
        "config": {
            "schema_version": 2,           # bumped: nli_* fields → semantic_* / negation_*
            "n_drafts": args.n_drafts,
            "max_evolve": args.max_evolve,
            "use_semantic": not args.no_semantic,
            "max_workers": n_workers,
            "author_prompt_file": args.author_prompt_file,
        },
        "n_papers": len(results),
        "n_successful": len(successful),
        "aggregate": _aggregate(successful),
        "papers": results,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(file=sys.stderr)
    log(f"wrote {out_path}", tag="eval")
    log(f"aggregate: {json.dumps(summary['aggregate'], indent=None)}", tag="eval")
    return 0


def _aggregate(rows: list[dict]) -> dict:
    """Mean of the headline scores across successful runs."""
    if not rows:
        return {}

    def _mean(key_path: list[str]) -> float | None:
        vals = []
        for r in rows:
            v = r
            for k in key_path:
                v = (v or {}).get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                vals.append(v)
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "semantic_score_avg":           _mean(["winner_scores", "semantic_score"]),
        "mean_bm25_avg":               _mean(["winner_scores", "mean_bm25"]),
        "n_drift_avg":                 _mean(["winner_scores", "n_drift"]),
        "n_negation_mismatches_avg":   _mean(["winner_scores", "n_negation_mismatches"]),
        "n_graded_avg":                _mean(["winner_scores", "n_graded"]),
        "n_kc_avg":                    _mean(["n_kc"]),
        "gate_pass_rate":  round(sum(1 for r in rows if r.get("gate_passed")) / len(rows), 3),
        "evolved_swap_rate": round(sum(1 for r in rows if r.get("evolved_swap")) / len(rows), 3),
        "elapsed_s_avg":   _mean(["elapsed_s"]),
    }
