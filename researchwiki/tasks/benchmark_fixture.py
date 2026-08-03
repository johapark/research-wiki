"""Score a wiki page against its eval fixture.

Fixtures live under `benchmark-fixtures/{stem}.yaml` (see benchmark-fixtures/README.md
for the schema). Each fixture declares the headline claims, capabilities,
limitations, and related-paper links a thorough wiki page should capture.

This task scores a wiki page along five axes — headline_claims,
capabilities, limitations, related_papers, comparator_fidelity — and
reports per-axis recall plus an importance-weighted aggregate.

Two scoring modes:

  - heuristic (default): token-overlap + numeric-presence per item, plus
    mechanical wikilink and ratio↔comparator checks. Offline, deterministic,
    directional but imperfect.
  - llm-judge (--llm): one LLM call per verbalization-based item (~30
    calls per fixture). Tolerant of paraphrase; better recall.

Replication mode (`--repeat N`): runs the author phase N times against
the fixture's PDF and scores each draft independently, reporting mean ± SD
per axis. Use this to measure whether a pipeline change moved the needle —
single-run scores have ~±10pp variance from author-side stochasticity, so
single-shot comparisons can't attribute lever effects below that magnitude.
Reconcile / extract / crosslink resolution all run ONCE per replicate
session and are reused across the N author calls — the LLM cost is roughly
N author drafts (≈$0.06 each at Sonnet 4.6) plus N × per-item judge calls
when --llm is on. This is opt-in for methodology validation; normal
ingestion (`researchwiki agent ingest`) is unaffected and stays one-shot.

Usage:
  researchwiki benchmark-fixture <stem>                    # score current wiki page
  researchwiki benchmark-fixture <stem> --page <path>      # score an arbitrary page
  researchwiki benchmark-fixture <stem> --llm              # use the LLM judge
  researchwiki benchmark-fixture <stem> --repeat 5         # run author × 5; report mean ± SD
  researchwiki benchmark-fixture <stem> --repeat 5 --llm   # replication + LLM judging
  researchwiki benchmark-fixture <stem> --json             # machine-readable
  researchwiki benchmark-fixture --list                    # list available fixtures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..benchmark import (
    ContentFixture,
    RetrievalFixture,
    compute_style,
    diff_retrieval_scores,
    find_fixtures,
    load_fixture,
    score_page,
)
from ..benchmark.content_reports import (
    format_compare_report,
    format_replicate_report,
    format_report,
    paired_t,
)
# Direct import — `replicate.py` depends on agents.phases, so re-exporting
# it from `benchmark/__init__.py` would create a circular import. Callers
# that need the replication driver import it directly.
from ..benchmark.replicate import replicate_score
from ..benchmark.retrieval_reports import (
    print_retrieval_diff,
    print_retrieval_score,
    score_retrieval,
)
from ..paths import resolve_pdf, wiki_dir


def _run_replicate_under_config(fixture: ContentFixture, *, config_path: str | None,
                                n: int, use_llm: bool, verbose: bool):
    """Run replicate_score with RW_MODELS_CONFIG set (or unset for the
    process default) and the model_config LRU cache cleared so the change
    takes effect. Restores the prior env after the call."""
    import os
    from ..agents import model_config as _mc
    prior = os.environ.get("RW_MODELS_CONFIG")
    try:
        if config_path is None:
            os.environ.pop("RW_MODELS_CONFIG", None)
        else:
            os.environ["RW_MODELS_CONFIG"] = config_path
        _mc._config.cache_clear()
        return replicate_score(fixture, n=n, use_llm=use_llm, verbose=verbose)
    finally:
        if prior is None:
            os.environ.pop("RW_MODELS_CONFIG", None)
        else:
            os.environ["RW_MODELS_CONFIG"] = prior
        _mc._config.cache_clear()


def _warn_if_judge_matches_author() -> None:
    """Warn on stderr when the resolved author and eval_judge phases share
    (provider, model). Same-model self-grading is a real bias in LLM-config
    A/Bs — a Sonnet author graded by a Sonnet judge can score higher than
    an Opus author graded by the same Sonnet judge for reasons unrelated
    to draft quality (in-family preference, shared blind spots). Warning
    only; the user may still choose to run this way."""
    try:
        from ..agents.model_config import for_phase
        author = for_phase("author")
        judge = for_phase("eval_judge")
    except Exception:
        return
    if (author.provider, author.model) == (judge.provider, judge.model):
        print(
            f"warning: eval_judge resolves to the same model as author "
            f"({author.model}@{author.provider}). --llm scores may be "
            f"biased in favor of same-family drafts. For LLM-config A/Bs, "
            f"pin `eval_judge` to a different model in config/models.yaml.",
            file=sys.stderr,
        )


def _resolve_page_path(stem: str) -> Path:
    """Find the wiki page for `stem` under any category directory.

    Mirrors `grade.fidelity.paper._resolve_page` rather than depending on it,
    because the eval task is read-only and wants to stay light on imports.
    """
    matches = list(wiki_dir().rglob(f"{stem}.md"))
    if not matches:
        raise FileNotFoundError(
            f"No wiki page found for stem={stem}. "
            f"Run `researchwiki agent ingest` first, or pass `--page <path>`."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple wiki pages match stem={stem}: {matches}. "
            f"Resolve the duplicate before scoring."
        )
    return matches[0]


def _run_retrieval_fixture(fixture: RetrievalFixture, args) -> int:
    """Dispatch entry for retrieval fixtures. Three modes:
      A. --embedding NAME              → single-embedding score
      B. --baseline-embedding X
         --candidate-embedding Y       → A/B diff
      C. neither                       → use framework default (BGE-small)
    """
    # Warn on coverage-only flags that don't apply.
    for flag in ("page", "repeat", "with_grader", "with_style", "llm"):
        if getattr(args, flag, None):
            print(
                f"warning: --{flag.replace('_', '-')} doesn't apply to "
                f"retrieval fixtures (ignored)", file=sys.stderr,
            )

    backend = args.retrieval_backend

    # Mode B: A/B diff. Per-side prefix / trust_remote_code flags so two
    # models with different conventions are compared fairly. The bare
    # --embedding-* flags don't apply here.
    if args.baseline_embedding and args.candidate_embedding:
        baseline_extra = {
            "trust_remote_code": args.baseline_trust_remote_code,
            "doc_prefix": args.baseline_doc_prefix or "",
            "query_prefix": args.baseline_query_prefix or "",
        }
        candidate_extra = {
            "trust_remote_code": args.candidate_trust_remote_code,
            "doc_prefix": args.candidate_doc_prefix or "",
            "query_prefix": args.candidate_query_prefix or "",
        }
        try:
            baseline = score_retrieval(
                fixture, args.baseline_embedding, backend, **baseline_extra,
            )
            candidate = score_retrieval(
                fixture, args.candidate_embedding, backend, **candidate_extra,
            )
            diff = diff_retrieval_scores(baseline, candidate)
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        if args.json:
            out = {
                "baseline": baseline.to_dict(),
                "candidate": candidate.to_dict(),
                "diff": diff.to_dict(),
            }
            print(json.dumps(out, indent=2))
        else:
            print_retrieval_diff(fixture, baseline, candidate, diff)
        return 0

    if args.baseline_embedding or args.candidate_embedding:
        print(
            "error: --baseline-embedding and --candidate-embedding must be "
            "used together (A/B mode)", file=sys.stderr,
        )
        return 1

    # Mode A / C: single-embedding score. --embedding-* flags apply here.
    embedding = args.embedding or "BAAI/bge-small-en-v1.5"
    extra = {
        "trust_remote_code": args.embedding_trust_remote_code,
        "doc_prefix": args.embedding_doc_prefix or "",
        "query_prefix": args.embedding_query_prefix or "",
    }
    try:
        score = score_retrieval(fixture, embedding, backend, **extra)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(score.to_dict(), indent=2))
    else:
        print_retrieval_score(fixture, score)
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki benchmark-fixture",
        description=__doc__.strip().split("\n", 1)[0] if __doc__ else None,
    )
    parser.add_argument(
        "stem",
        nargs="?",
        help="Paper stem matching benchmark-fixtures/{stem}.yaml.",
    )
    parser.add_argument(
        "--page",
        help="Path to a wiki page to score (default: wiki/<category>/<stem>.md).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM judge for verbalization-based axes (one call per item).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        metavar="N",
        help="Replication mode — run author phase N times against the fixture's "
             "PDF, score each independently, report mean ± SD. Reconcile / "
             "extract / crosslinks run ONCE and are reused. Recommended N=3-5 "
             "for lever-comparison validation. Cost: ~N × $0.06 author + "
             "N × judge calls.",
    )
    parser.add_argument(
        "--with-grader",
        action="store_true",
        help="Also run the agent's reference-free grader (BM25 + bi-encoder + "
             "numeric-integrity + negation-parity per claim) and report its "
             "scores alongside fixture-based recall. Two complementary signals: "
             "fixture recall = 'is the right content covered?'; grader = 'is "
             "what's there supported by the PDF?'. Disagreements are the "
             "diagnostic — high recall + low grader = covered topics with "
             "drift; low recall + high grader = faithful but missing fixture "
             "targets. Single-shot only (not yet wired for --repeat).",
    )
    parser.add_argument(
        "--with-style",
        action="store_true",
        help="Also report page-level style metrics — compression "
             "(page-tokens / paper-tokens) and extractiveness (fraction of "
             "page sentences with verbatim ≥10-word spans from the PDF). "
             "Catches 'too compressed', 'verbose padding', 'heavy paraphrase' "
             "(drift risk), and 'cargo-culted excerpts'. Mechanical, no LLM "
             "calls. Orthogonal to fixture recall — a page can hit 95% recall "
             "AND be too extractive.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a human report.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available fixtures and exit.",
    )
    parser.add_argument(
        "--exclude-contaminated",
        metavar="YYYY-MM-DD",
        default=None,
        help="Skip a fixture (exit 0 with stderr note) when its "
             "`published_at:` is earlier than this date. Use in CI to "
             "guard against re-introducing pre-cutoff fixtures whose "
             "high scores would reflect LLM memorisation. In --list "
             "mode, contaminated fixtures are omitted from the listing "
             "instead of skipped.",
    )
    parser.add_argument(
        "--compare-config",
        metavar="CONFIG.yaml",
        default=None,
        help="A/B two model configs against the same fixture. Baseline "
             "arm uses whatever RW_MODELS_CONFIG currently resolves to; "
             "candidate arm uses this file (bare name resolves under "
             "config/, path-shaped values used verbatim). Requires "
             "--repeat N — each arm runs replicate_score independently, "
             "then per-axis Δmean and a paired t-statistic are printed. "
             "Use `researchwiki status` to confirm each arm's resolved "
             "models.",
    )

    # ── Retrieval-fixture flags (active only when the fixture's
    # `fixture_type:` is `claims` or `pages`; quietly ignored otherwise).
    retr = parser.add_argument_group("retrieval-fixture options")
    retr.add_argument(
        "--embedding",
        metavar="MODEL",
        help="Sentence-transformers model id for single-embedding "
             "retrieval evaluation. Per-model semantic cache built on "
             "first use. Default: framework's configured embedding "
             "(BAAI/bge-small-en-v1.5).",
    )
    retr.add_argument(
        "--baseline-embedding",
        metavar="MODEL",
        help="A/B mode: model id to use as baseline. "
             "Pair with --candidate-embedding to emit per-fixture and "
             "aggregate Δ MRR / nDCG / recall.",
    )
    retr.add_argument(
        "--candidate-embedding",
        metavar="MODEL",
        help="A/B mode: model id to compare against the baseline.",
    )
    retr.add_argument(
        "--retrieval-backend",
        choices=("semantic", "bm25", "hybrid"),
        default="semantic",
        help="Which retrieval surface to evaluate (default: semantic). "
             "Pure 'semantic' isolates the embedding's contribution; "
             "'hybrid' measures end-to-end user-facing quality.",
    )
    # Single-model mode: --embedding-* applies to the one model.
    # A/B mode: per-side --baseline-*-/--candidate-*- override (see below).
    retr.add_argument(
        "--embedding-trust-remote-code",
        action="store_true",
        help="Pass `trust_remote_code=True` when loading the model. Required "
             "by some models that ship custom HF code (Nomic family). Off "
             "by default — opt in for trusted model providers only. "
             "Applies in single-model mode only; use --baseline-/--candidate- "
             "variants for A/B mode.",
    )
    retr.add_argument(
        "--embedding-doc-prefix",
        metavar="STRING",
        default="",
        help="Instruction prefix prepended to every indexed text. Used by "
             "models that distinguish indexing vs querying via prefixes "
             "(e.g. 'search_document: ' for Nomic, 'passage: ' for E5). "
             "Cache is invalidated when this changes. Single-model mode only.",
    )
    retr.add_argument(
        "--embedding-query-prefix",
        metavar="STRING",
        default="",
        help="Instruction prefix prepended to every query. Pair with "
             "--embedding-doc-prefix. Single-model mode only.",
    )

    # A/B mode: per-side prefix / trust_remote_code flags so two models with
    # different conventions can be compared fairly. Each set overrides the
    # bare --embedding-* defaults for its side.
    for side in ("baseline", "candidate"):
        retr.add_argument(
            f"--{side}-trust-remote-code",
            action="store_true",
            help=f"`trust_remote_code=True` for the {side} model only.",
        )
        retr.add_argument(
            f"--{side}-doc-prefix", metavar="STRING", default="",
            help=f"Indexing-time prefix for the {side} model only "
                 f"(empty by default — use when the {side} model needs one).",
        )
        retr.add_argument(
            f"--{side}-query-prefix", metavar="STRING", default="",
            help=f"Query-time prefix for the {side} model only.",
        )

    args = parser.parse_args(argv)

    if args.list:
        stems = find_fixtures()
        if not stems:
            print("(no fixtures yet — see benchmark-fixtures/README.md)", file=sys.stderr)
            return 1
        cutoff = args.exclude_contaminated
        for s in stems:
            if cutoff:
                # Best-effort filter — retrieval fixtures have no
                # published_at and always pass through. Malformed content
                # fixtures also pass through (they'll fail loudly at load
                # time if the user actually tries to run them).
                try:
                    f = load_fixture(s)
                    pub = getattr(f, "published_at", None)
                    if pub and pub < cutoff:
                        continue
                except Exception:
                    pass
            print(s)
        return 0

    if not args.stem:
        # Nothing to score — a missing positional, so code 1. The sibling sites
        # below already return 1 for `--compare-config` without `--repeat` and
        # for an unknown page.
        parser.print_help(sys.stderr)
        return 1

    try:
        fixture = load_fixture(args.stem)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: invalid fixture: {e}", file=sys.stderr)
        return 1

    # Contamination gate — skip pre-cutoff fixtures with exit 0 so CI
    # loops using --exclude-contaminated don't fail on retired anchors.
    # Only meaningful for content fixtures; retrieval fixtures have no
    # published_at and are always kept.
    if args.exclude_contaminated and isinstance(fixture, ContentFixture):
        pub = fixture.published_at
        if pub and pub < args.exclude_contaminated:
            print(
                f"skipped: {args.stem} — published_at {pub} precedes "
                f"--exclude-contaminated cutoff {args.exclude_contaminated}",
                file=sys.stderr,
            )
            return 0

    # Retrieval fixtures dispatch to a separate scoring path. Coverage-only
    # flags (--page, --repeat, --with-grader, --with-style, --llm) don't
    # apply; warn if the user combined them.
    if isinstance(fixture, RetrievalFixture):
        return _run_retrieval_fixture(fixture, args)

    # ── from here, content-coverage path (the original behavior) ─────
    # A real precondition, not a narrowing hint: `fixture` comes from parsing a
    # YAML file, so a third fixture type would reach here and be scored by the
    # wrong path. Raised rather than asserted so `python -O` can't strip it.
    if not isinstance(fixture, ContentFixture):
        raise TypeError(
            f"unsupported fixture type {type(fixture).__name__}; the "
            "content-coverage path needs a ContentFixture"
        )

    if args.llm:
        _warn_if_judge_matches_author()

    # --compare-config: requires --repeat N; each arm runs replicate_score
    # under its own RW_MODELS_CONFIG. Print a paired A/B diff. Mirrors the
    # --baseline-embedding / --candidate-embedding pattern on the retrieval
    # side, adapted for content fixtures where the config is the lever.
    if args.compare_config:
        if not args.repeat or args.repeat < 2:
            print(
                "error: --compare-config requires --repeat N (N >= 2) — "
                "each arm needs multiple replicates for the paired diff.",
                file=sys.stderr,
            )
            return 1
        if args.page:
            print(
                "warning: --compare-config ignores --page (each arm re-drafts "
                "from the fixture's PDF).", file=sys.stderr,
            )

        import os as _os
        baseline_env = _os.environ.get("RW_MODELS_CONFIG") or "(default: config/models.yaml)"
        candidate_arg = args.compare_config

        try:
            if args.json:
                import contextlib as _ctx
                with _ctx.redirect_stdout(sys.stderr):
                    baseline_rep = _run_replicate_under_config(
                        fixture, config_path=_os.environ.get("RW_MODELS_CONFIG"),
                        n=args.repeat, use_llm=args.llm, verbose=False,
                    )
                    candidate_rep = _run_replicate_under_config(
                        fixture, config_path=candidate_arg,
                        n=args.repeat, use_llm=args.llm, verbose=False,
                    )
            else:
                print(f"[compare-config] baseline arm ({baseline_env})", file=sys.stderr)
                baseline_rep = _run_replicate_under_config(
                    fixture, config_path=_os.environ.get("RW_MODELS_CONFIG"),
                    n=args.repeat, use_llm=args.llm, verbose=True,
                )
                print(f"[compare-config] candidate arm ({candidate_arg})", file=sys.stderr)
                candidate_rep = _run_replicate_under_config(
                    fixture, config_path=candidate_arg,
                    n=args.repeat, use_llm=args.llm, verbose=True,
                )
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"error: --compare-config failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 2

        if args.json:
            per_slot_deltas = [
                c.overall - b.overall
                for b, c in zip(baseline_rep.runs, candidate_rep.runs)
            ]
            t, df = paired_t(per_slot_deltas)
            out = {
                "paper_stem": baseline_rep.paper_stem,
                "n_runs": baseline_rep.n_runs,
                "use_llm": baseline_rep.use_llm,
                "baseline_label": baseline_env,
                "candidate_label": candidate_arg,
                "baseline": baseline_rep.to_dict(),
                "candidate": candidate_rep.to_dict(),
                "diff": {
                    "delta_overall": candidate_rep.mean - baseline_rep.mean,
                    "delta_axis": {
                        ax: candidate_rep.axis_mean.get(ax, 0.0) - baseline_rep.axis_mean[ax]
                        for ax in baseline_rep.axis_mean
                    },
                    "paired_t": t,
                    "df": df,
                    "per_slot_deltas": per_slot_deltas,
                },
            }
            print(json.dumps(out, indent=2))
        else:
            print(format_compare_report(
                baseline_rep, candidate_rep,
                baseline_label=baseline_env, candidate_label=candidate_arg,
                fixture=fixture,
            ))
        return 0

    # Replication mode: ignore --page; we'll author drafts fresh against the
    # fixture's source PDF, no committed page involved.
    if args.repeat and args.repeat > 1:
        if args.page:
            print(
                "warning: --repeat ignores --page (replication runs the author "
                "phase against the fixture's PDF, not an existing page)",
                file=sys.stderr,
            )
        # In --json mode, the replicate driver and its callees (reconcile,
        # propose_crosslinks, etc.) print progress to stdout and would
        # corrupt the JSON. Redirect stdout to stderr for the duration of
        # the call so the user still sees progress (on stderr) and the
        # final JSON ends up alone on stdout.
        try:
            if args.json:
                import contextlib
                with contextlib.redirect_stdout(sys.stderr):
                    rep_report = replicate_score(
                        fixture, n=args.repeat, use_llm=args.llm,
                        verbose=False,
                    )
            else:
                rep_report = replicate_score(
                    fixture, n=args.repeat, use_llm=args.llm,
                    verbose=True,
                )
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"error: replication failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(rep_report.to_dict(), indent=2))
        else:
            print(format_replicate_report(rep_report, fixture=fixture))
        return 0

    if args.page:
        page_path = Path(args.page)
        if not page_path.exists():
            print(f"error: page not found: {page_path}", file=sys.stderr)
            return 1
    else:
        try:
            page_path = _resolve_page_path(args.stem)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    report = score_page(fixture, page_path, use_llm=args.llm)

    grader_report = None
    if args.with_grader:
        if args.repeat:
            print(
                "warning: --with-grader is not yet wired into --repeat mode "
                "(replicate drafts aren't on disk for the grader to index)",
                file=sys.stderr,
            )
        else:
            try:
                from ..grade.fidelity.paper import grade_page as _grade_page
                # persist=False: this is a read-only score against the page
                # and PDF; don't pollute the per-paper claim grade columns
                # in the DB with eval-time scores.
                grader_report = _grade_page(
                    args.stem, page_path=str(page_path), persist=False,
                )
            except Exception as e:
                print(
                    f"warning: --with-grader failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

    style_report = None
    if args.with_style:
        if args.repeat:
            print(
                "warning: --with-style is not yet wired into --repeat mode "
                "(would need PDF text + per-draft body; future iteration)",
                file=sys.stderr,
            )
        else:
            try:
                pdf_path = resolve_pdf(args.stem)
                from ..pdf.text import extract_pdf
                paper_text, _ = extract_pdf(pdf_path, max_pages=80)
                page_body = page_path.read_text(encoding="utf-8")
                style_report = compute_style(page_body, paper_text)
            except Exception as e:
                print(
                    f"warning: --with-style failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )

    if args.json:
        out = report.to_dict()
        if grader_report is not None:
            out["grader"] = {
                "n_claims": grader_report.n_claims,
                "n_graded": grader_report.n_graded,
                "mean_top1_bm25": grader_report.mean_top1,
                "median_top1_bm25": grader_report.median_top1,
                "semantic_score": grader_report.semantic_score,
                "semantic_median": grader_report.semantic_median,
                "weakest_score": grader_report.weakest_score,
                "n_with_numeric_drift": grader_report.n_with_numeric_drift,
                "n_negation_mismatches": grader_report.n_negation_mismatches,
            }
        if style_report is not None:
            out["style"] = style_report.to_dict()
        print(json.dumps(out, indent=2))
    else:
        print(format_report(
            report, fixture=fixture,
            grader_report=grader_report, style_report=style_report,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
