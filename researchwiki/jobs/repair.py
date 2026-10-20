"""Local projection repair from durable publication evidence; no authoring/hooks."""

from pathlib import Path
import re

from ..fsatomic import file_sha256


def validate_artifacts(root, artifacts):
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "paper_stem", "attempt_id", "page_path", "pdf_path", "page_sha256", "pdf_sha256",
        "gate_reasons", "checks",
    }:
        raise ValueError("invalid publication artifacts")
    stem = artifacts["paper_stem"]
    if not isinstance(stem, str) or not stem or Path(stem).name != stem:
        raise ValueError("invalid publication stem")
    if not isinstance(artifacts["attempt_id"], str) or not artifacts["attempt_id"]:
        raise ValueError("invalid publication attempt")
    from ..operations import CheckResult

    if not isinstance(artifacts["gate_reasons"], list) or not all(
        isinstance(reason, str) for reason in artifacts["gate_reasons"]
    ) or not isinstance(artifacts["checks"], list):
        raise ValueError("invalid publication gate evidence")
    for check in artifacts["checks"]:
        CheckResult.from_dict(check)
    for kind, folder, suffix in (("page", "wiki", ".md"), ("pdf", "papers", ".pdf")):
        path = Path(artifacts[kind + "_path"])
        if (not path.is_absolute() or path.name != stem + suffix
                or not path.resolve().is_relative_to((root / folder).resolve())):
            raise ValueError("publication artifact is outside its canonical library")
        if not isinstance(artifacts[kind + "_sha256"], str) or not re.fullmatch(
            "[a-f0-9]{64}", artifacts[kind + "_sha256"],
        ):
            raise ValueError("invalid publication fingerprint")


def repair_projections(record, ledger):
    """Caller holds owner and corpus locks. Never restore stale canonical bytes."""
    from ..grade.fidelity.paper import grade_page
    from ..index.incremental import update_page_indexes
    from ..mutation import BackedUpPath
    from ..projections import reconcile
    from ..wiki import read_page

    root = Path(record.request.library_root)
    artifacts = ledger["artifacts"]
    validate_artifacts(root, artifacts)
    for kind in ("page", "pdf"):
        path = Path(artifacts[kind + "_path"])
        if not path.is_file() or file_sha256(path) != artifacts[kind + "_sha256"]:
            raise ValueError("canonical artifacts changed since publication; manual review required")
    paths = []
    for raw in ledger["paths"]:
        path = Path(raw).resolve()
        if path.suffix == ".md" and path.is_relative_to((root / "wiki").resolve()):
            paths.append(path)
    if Path(artifacts["page_path"]).resolve() not in paths:
        raise ValueError("publication receipt lacks its canonical page")
    reconcile([BackedUpPath(path, None) for path in paths], "promote",
              {"stem": artifacts["paper_stem"], "repair_projections": True})
    outcome = update_page_indexes(paths)
    if (outcome.get("bm25_mode") not in {"incremental", "rebuilt"}
            or outcome.get("semantic") is None):
        raise RuntimeError("required index maintenance incomplete")
    page = read_page(Path(artifacts["page_path"]))
    if page is None:
        raise ValueError("canonical page cannot be read")
    if page.fm.get("type") != "commentary":
        grade_page(artifacts["paper_stem"], persist=True, require_persistence=True)
