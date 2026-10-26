"""Validate the already-selected single-library worker runtime without mutating it."""

from pathlib import Path
import os

from ._wire import ContractError


def validate_runtime(request) -> None:
    """A service call never changes its caller's CWD or provider environment.

    The caller starts one configured process per library/profile. Loading a
    second profile in a long-lived UI process is deliberately unsupported.
    """
    from ..env_profiles import ACTIVE_ENV_FILE_VAR

    if Path(request.library_root).resolve() != Path.cwd().resolve():
        raise ContractError("library_root must match the configured worker's current directory")
    selected = os.environ.get(ACTIVE_ENV_FILE_VAR)
    expected = request.provider_profile
    if (Path(selected).resolve() if selected else None) != (Path(expected).resolve() if expected else None):
        raise ContractError("provider_profile must match the configured worker's active profile")
    if not Path(request.source_pdf).is_file():
        raise FileNotFoundError(request.source_pdf)
    for value in (*request.options.supplementary, request.options.author_prompt_file):
        if value is not None and not Path(value).is_file():
            raise FileNotFoundError(value)


def recover_ingest_mutations() -> tuple[str, ...]:
    """Do not start paid work while a prior canonical mutation is unresolved."""
    from ..errors import EnvironmentFailure
    from ..mutation import corpus_lock, pending_journals, recover_pending

    with corpus_lock():
        try:
            notes = tuple(recover_pending())
        except Exception as exc:
            raise EnvironmentFailure("Corpus recovery failed; inspect pending mutation receipts") from exc
        if pending_journals():
            raise EnvironmentFailure("Unresolved corpus mutation; recover it before ingesting another paper")
    return notes


def runner_options(options) -> dict:
    from ..agents import model_config

    drafts = options.n_drafts
    if drafts is None:
        configured = model_config.default_n_drafts()
        drafts = configured if configured is not None else 1
    return {
        'use_stub': options.use_stub, 'use_semantic': options.use_semantic,
        'verify_claim_entailment': options.verify_claim_entailment,
        'n_drafts': drafts, 'max_evolve': options.max_evolve, 'max_debug': options.max_debug,
        'promote_mode': options.promotion,
        'doi_override': options.doi, 'title_override': options.title,
        'year_override': options.year,
        'authors_override': list(options.authors) if options.authors is not None else None,
        'author_prompt_override': options.author_prompt_file,
        'supplementary': [Path(p) for p in options.supplementary] or None,
        'use_llm_reconcile': options.use_llm_reconcile, 'allow_rename': options.allow_rename,
        'max_model_calls': options.max_model_calls, 'max_tokens': options.max_tokens,
        'max_cost_usd': options.max_cost_usd, 'max_wall_seconds': options.max_wall_seconds,
        'run_memory_evolve': options.memory_evolve,
    }
