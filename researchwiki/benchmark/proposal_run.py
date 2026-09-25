"""Plan, approve, and run a development-only fixed-packet proposal comparison.

`plan` is offline. `run` sends the exact approved requests to the configured
provider. No retrieval, automatic judging, wiki writes, or runner-level retries.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import time

from .proposals import AXES, _hash, _json, check_pack
from .. import __version__, proposal_generation as generation
from ..agents import llm, model_config
from ..errors import EnvironmentFailure


BASELINE_SYSTEM = """Use the supplied evidence to suggest zero to three distinct,
useful synthesis or idea pages addressing the request. Connect at least two
source papers per proposal using supplied evidence IDs. Treat proposed mechanisms
as hypotheses, not established results. Existing pages are overlap context, not
evidence. Respect scope and feedback; identify the supplied parent when revising
a prior proposal and explain the new contribution. Abstain with {"proposals": []}
when the requested topic or a distinct proposal is unsupported.
Use the exact JSON schema. For every idea include mechanism, assumptions_to_test,
baseline, and first_experiment as strings, with a rejection criterion and comparable
budgets including offline costs. Follow any cross-category mode requirements in
the request. Return JSON only.
"""


def current_route() -> dict:
    """Resolve the actual dispatch destination without probing or exposing keys."""
    model_config.clear_caches()
    cfg = model_config.for_phase("proposal_generation")
    if cfg.provider in model_config.OPENAI_COMPAT_PROVIDERS:
        endpoint = llm.resolve_openai_endpoint().url
    elif cfg.provider == "anthropic":
        endpoint = llm._validated_anthropic_base_url() or "https://api.anthropic.com"
    else:
        raise ValueError("benchmark runner requires a synchronous API provider, not chat-relay")
    path = model_config.config_path()
    return {**asdict(cfg), "endpoint": endpoint, "config_path": str(path),
            "config_sha256": _hash(path.read_bytes()) if path.exists() else None}


def _implementation() -> dict:
    return {name: _hash((Path(generation.__file__).parent / name).read_bytes())
            for name in ("proposal_generation.py", "proposals.py")}


def _identity(value: dict, field: str) -> str:
    return _hash(json.dumps({k: v for k, v in value.items() if k != field}, sort_keys=True).encode())


def plan_comparison(pack: Path, out: Path) -> dict:
    """Freeze two request variants for every development case, never holdouts."""
    if out.exists():
        raise ValueError("plan output already exists; use a new directory")
    manifest = check_pack(pack)
    cases = [case for case in manifest["cases"] if case["split"] == "development"]
    if not 1 <= len(cases) <= 8 or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("expected one to eight unique development cases")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]) for case in cases):
        raise ValueError("case IDs must be safe filenames")
    if manifest["max_proposals"] != generation.MAX_PROPOSALS:
        raise ValueError("pack and generator proposal limits differ")
    route = current_route()
    variants = ["baseline", "production"]
    random.SystemRandom().shuffle(variants)
    aliases = dict(zip(("A", "B"), variants))  # Kept out of reviewer artifacts.
    payloads, entries = {}, []
    for index, case in enumerate(cases):
        packet = json.loads((pack / case["input_path"]).read_text())
        packet_path = f"inputs/{case['id']}.json"
        payloads[packet_path] = packet
        production = generation.proposal_request(packet)
        # Alternate A/B order to balance the within-case order across the split.
        order = ["A", "B"] if index % 2 == 0 else ["B", "A"]
        for alias in order:
            request = dict(production)
            if aliases[alias] == "baseline":
                request["system"] = generation._with_contract(BASELINE_SYSTEM, request["schema"])
            request.update({k: route[k] for k in
                            ("model", "provider", "temperature", "max_tokens", "reasoning_effort")})
            request_id = f"{case['id']}-{alias}"
            path = f"requests/{request_id}.json"
            payloads[path] = request
            entries.append({"request_id": request_id, "case_id": case["id"],
                            "candidate": alias, "input_path": packet_path, "request_path": path})
    encoded = {p: (json.dumps(v, indent=2, ensure_ascii=False) + "\n").encode()
               for p, v in payloads.items()}
    plan = {"version": 1, "framework_version": __version__, "pack_id": manifest["pack_id"],
            "suite_id": manifest["suite_id"], "rubric_version": manifest["rubric_version"],
            "split": "development", "execution_mode": "fixed-packet", "repeat": 1,
            "route": route, "implementation": _implementation(), "candidate_identity": aliases,
            "entries": entries, "files": {p: _hash(b) for p, b in encoded.items()},
            "transport_note": "Existing provider retries/parameter negotiation remain enabled; wire-attempt counts and usage for failed attempts are unavailable."}
    plan["plan_id"] = _identity(plan, "plan_id")
    out.mkdir(parents=True)
    for relative, data in encoded.items():
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _json(out / "plan.json", plan)
    return plan


def check_plan(directory: Path) -> dict:
    plan = json.loads((directory / "plan.json").read_text())
    if plan.get("version") != 1 or _identity(plan, "plan_id") != plan["plan_id"]:
        raise ValueError("plan identity mismatch")
    if plan["split"] != "development" or plan["execution_mode"] != "fixed-packet":
        raise ValueError("only development fixed-packet runs are supported")
    if plan["implementation"] != _implementation():
        raise ValueError("proposal implementation changed; prepare a new plan")
    for relative, digest in plan["files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or _hash(path.read_bytes()) != digest:
            raise ValueError(f"plan file hash mismatch: {relative}")
    for entry in plan["entries"]:
        if entry["input_path"] not in plan["files"] or entry["request_path"] not in plan["files"]:
            raise ValueError("unhashed request or input path")
    return plan


def _review_artifacts(directory: Path, plan: dict, receipts: list[dict]) -> None:
    """Keep identities/prompts/usage out of the identity-hidden review folder."""
    for alias in ("A", "B"):
        reviews, outputs = [], []
        for entry, receipt in zip(plan["entries"], receipts):
            if entry["candidate"] != alias:
                continue
            outputs.append({"case_id": entry["case_id"], "raw_response": receipt["raw_response"],
                            "proposals": receipt["proposals"],
                            "operational_success": receipt["operational_success"],
                            "failure_stage": receipt.get("failure_stage")})
            review = {"case_id": entry["case_id"], "operational_success": receipt["operational_success"],
                      "response_path": f"outputs-{alias}.json", "proposals": [], "feedback_checks": []}
            for i, proposal in enumerate(receipt["proposals"] or []):
                review["proposals"].append({"proposal_index": i + 1, "scores": dict.fromkeys(AXES),
                    "reasons": dict.fromkeys(AXES, ""), "critical_factual_failure": None,
                    "constraint_failure": None, "decision": None, "decision_reason": "",
                    "parent_proposal": proposal.get("parent_proposal")})
            reviews.append(review)
        _json(directory / "review" / f"outputs-{alias}.json", outputs)
        _json(directory / "review" / f"reviews-{alias}.json", {
            "pack_id": plan["pack_id"], "candidate": alias, "repeat": 1, "split": "development",
            "execution_mode": "fixed-packet", "reviewer": "", "reviewer_kind": None,
            "reviewer_model": None, "reviews": reviews})


def run_comparison(directory: Path, approval: str) -> dict:
    plan = check_plan(directory)
    if approval != plan["plan_id"]:
        raise ValueError("run requires --approve-plan matching the reviewed plan_id")
    if current_route() != plan["route"]:
        raise ValueError("provider/config changed since approval; prepare a new plan")
    out = directory / "run"
    if out.exists():
        raise ValueError("run already exists; completed or interrupted runs are never replayed")
    out.mkdir()
    receipts = []
    for entry in plan["entries"]:
        if check_plan(directory)["plan_id"] != plan["plan_id"]:
            raise ValueError("approved plan changed during the run")
        if current_route() != plan["route"]:
            raise ValueError("provider/config changed during the run; stopping before dispatch")
        request = json.loads((directory / entry["request_path"]).read_text())
        packet = json.loads((directory / entry["input_path"]).read_text())
        path = out / f"{entry['request_id']}.json"
        receipt = {"request_id": entry["request_id"], "status": "started",
                   "started_at": datetime.now(timezone.utc).isoformat(), "plan_id": plan["plan_id"],
                   "request_sha256": plan["files"][entry["request_path"]],
                   "operational_success": False, "raw_response": None, "proposals": None,
                   "usage": None, "wire_attempts": None}
        _json(path, receipt)  # An interrupted dispatch is ambiguous; never silently repeat it.
        start = time.monotonic()
        try:
            response = llm.call(**request)
        except Exception as exc:
            receipt.update(status="failed", failure_stage="provider", error_type=type(exc).__name__,
                           error=str(exc), elapsed_seconds=time.monotonic() - start)
            _json(path, receipt)
            # Known provider failures are reviewable failed attempts. Unexpected
            # errors retain their receipt but propagate instead of hiding a bug.
            if not isinstance(exc, (EnvironmentFailure, OSError, RuntimeError)):
                raise
        else:
            receipt.update(raw_response=response.text, usage=llm.response_usage(response),
                           elapsed_seconds=time.monotonic() - start, status="returned")
            _json(path, receipt)  # Preserve text and usage BEFORE attempting validation.
            try:
                proposals = generation.parse_proposal_response(response.text, packet)
            except ValueError as exc:
                receipt.update(status="failed", failure_stage="validation", error=str(exc))
            else:
                receipt.update(status="valid", operational_success=True, proposals=proposals)
            _json(path, receipt)
        receipts.append(receipt)
        print(f"{entry['request_id']}: {receipt['status']}", flush=True)
    _review_artifacts(out, plan, receipts)
    summary = {"plan_id": plan["plan_id"], "status": "complete", **_metrics(receipts),
               "candidates": {alias: _metrics([r for e, r in zip(plan["entries"], receipts)
                                                if e["candidate"] == alias]) for alias in ("A", "B")}}
    _json(out / "summary.json", summary)
    return summary


def _metrics(receipts: list[dict]) -> dict:
    return {"attempted": len(receipts),
               "operational_success": sum(r["operational_success"] for r in receipts),
               "provider_failures": sum(r.get("failure_stage") == "provider" for r in receipts),
               "validation_failures": sum(r.get("failure_stage") == "validation" for r in receipts),
               "elapsed_seconds": sum(r["elapsed_seconds"] for r in receipts),
               "all_final_responses_have_usage": all(r["usage"] is not None for r in receipts),
               "reported_input_tokens": sum((r["usage"] or {}).get("input_tokens", 0) for r in receipts),
               "reported_output_tokens": sum((r["usage"] or {}).get("output_tokens", 0) for r in receipts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--approve-plan")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    from ..__main__ import _load_dotenv
    try:
        _load_dotenv(args.env_file)
        if args.action == "plan":
            if not args.pack or not args.out:
                parser.error("plan requires --pack and --out")
            result = plan_comparison(args.pack, args.out)
            print(json.dumps({"plan_id": result["plan_id"], "route": result["route"],
                              "planned_generations": len(result["entries"]),
                              "input_files": sorted(p for p in result["files"] if p.startswith("inputs/")),
                              "path": str(args.out)}, indent=2))
        else:
            if not args.plan or not args.approve_plan:
                parser.error("run requires --plan and --approve-plan")
            result = run_comparison(args.plan, args.approve_plan)
            print(json.dumps(result, indent=2))
            if result["provider_failures"]:
                parser.exit(2)
            if result["validation_failures"]:
                parser.exit(1)
    except EnvironmentFailure as exc:
        parser.exit(2, f"proposal benchmark: {exc}\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"proposal benchmark: {exc}\n")
    except Exception as exc:
        parser.exit(3, f"proposal benchmark internal error: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
