"""Resolve once and validate queue inputs without loading a model or profile."""

from dataclasses import replace
from pathlib import Path

from ..operations import ContractError


def normalize_request(request):
    options = replace(
        request.options,
        supplementary=tuple(str(Path(p).resolve()) for p in request.options.supplementary),
        author_prompt_file=str(Path(request.options.author_prompt_file).resolve()) if request.options.author_prompt_file else None,
    )
    return replace(request, library_root=str(Path(request.library_root).resolve()),
                   source_pdf=str(Path(request.source_pdf).resolve()), options=options,
                   provider_profile=str(Path(request.provider_profile).resolve()) if request.provider_profile else None)


def validate_inputs(request):
    from ..env_profiles import snapshot_profile
    from ..paths import missing_checkout_assets

    root = Path(request.library_root)
    missing = missing_checkout_assets(root)
    if missing or not (root / "wiki").is_dir():
        raise ContractError("library_root must be an initialized checkout with runtime assets")
    for value in (request.source_pdf, *request.options.supplementary, request.options.author_prompt_file):
        if value is not None and not Path(value).is_file():
            raise ContractError(f"Input file is missing: {value}")
    profile = Path(request.provider_profile) if request.provider_profile else root / ".env"
    snapshot = snapshot_profile(profile, reject_duplicates=True)
    if request.provider_profile and not snapshot.existed:
        raise ContractError("selected provider profile does not exist")
