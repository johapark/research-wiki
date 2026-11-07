"""Validate, prepare blinded packets, and mechanically score saved responses.

This utility has no network/model transport. Run from any working directory.
"""

import argparse
import base64
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import statistics
import struct


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
INSPECTION_SYSTEM = """You inspect the supplied paper images only. Do not use remembered paper results or external sources.
Return JSON with case_id and answers, an array containing exactly one item for each question_id.
Each item has question_id, status (observed, not_shown, or unreadable), value, unit, and evidence.
Value is a number, string, boolean, or null. Use null for absent, masked, or unreadable requested data.
Preserve row/column associations, signs, denominators, units, legends and qualifications. Never digitize an exact number from an unlabeled plot.
Evidence is an array of {image_id, pdf_page, ref_id, region}; cite the actual provided image(s) supporting that answer.
Region briefly names the relevant table cell, panel, caption or missing region. For comparisons across images, cite both.
Do not silently reconcile displayed values with remembered prose. Preserve an explicitly requested unit label exactly as printed.
Return only the requested answers, not a paper summary. Optional uncertainty is an array of short strings. No guessed answers."""
NOMINATION_SYSTEM = """You are extending target-claim extraction with visual inspection requests.
Use only the supplied source text and local inventory. Nominate inspection questions only for requested facts not settled by that text.
Return JSON with case_id and visual_requests, an array of {ref_id, priority, reason, question}.
Use inventory IDs only, priority critical or high, and reason missing_detail, table_alignment, visual_evidence, or conflict.
At most three requests, each question at most 160 characters. Questions are not factual answers.
If the requested facts are fully supplied, return an empty list. Do not invent artwork pages, bounding boxes, or answers.
A caption page is not proof that the artwork is on that page."""


def read(name):
    return json.loads((ROOT / name).read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class SuiteIntegrityError(Exception):
    """A suite file fails validation. Raised explicitly so `python -O` cannot strip it."""


def require(condition, detail):
    if not condition:
        raise SuiteIntegrityError(str(detail))


def validate():
    manifest = read("manifest.json")
    cases = read("cases.json")
    nominations = read("nomination-cases.json")
    gold = read("reference/expectations.json")
    require(len(cases) == manifest["inspection_cases"] == len(gold), "inspection case count disagrees across manifest, cases and gold")
    require(len(nominations) == manifest["nomination_cases"], "nomination case count disagrees with manifest")
    require(len({c["id"] for c in cases + nominations}) == len(cases + nominations), "duplicate case id")
    for source in manifest["sources"].values():
        path = REPO / source["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], f"{path}: source hash mismatch")
    for asset in manifest["assets"].values():
        path = ROOT / asset["path"]
        data = path.read_bytes()
        require(data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: not a PNG")
        require(struct.unpack(">II", data[16:24]) == (asset["width"], asset["height"]), f"{path}: dimensions mismatch")
        require(hashlib.sha256(data).hexdigest() == asset["sha256"], f"{path}: image hash mismatch")
    for case in cases:
        require(len(case["images"]) <= 2, f"{case['id']}: more than two images")
        require(sum(a["width"] * a["height"] for a in case["images"]) <= 3_000_000, f"{case['id']}: images exceed 3,000,000 pixels")
        question_ids = {q["id"] for q in case["questions"]}
        expected = gold[case["id"]]["answers"]
        require(question_ids == {a["question_id"] for a in expected}, f"{case['id']}: questions and gold answers disagree")
        require(len(question_ids) == len(case["questions"]), f"{case['id']}: duplicate question id")
        image_ids = {a["id"] for a in case["images"]}
        for answer in expected:
            require(set(answer["required_image_ids"]) <= image_ids, f"{case['id']}: gold cites an image the case lacks")
            require(answer["required_image_ids"], f"{case['id']}: gold answer cites no image")
    return {"inspection_cases": len(cases), "nomination_cases": len(nominations),
            "answer_atoms": sum(len(g["answers"]) for g in gold.values()),
            "assets": len(manifest["assets"]), "source_hashes_verified": True,
            "gold_status": manifest["gold_status"]}


def prepare(args):
    validate()
    if args.track == "nomination":
        cases = read("nomination-cases.json")
    else:
        cases = [c for c in read("cases.json") if args.suite == "full" or c["smoke"]]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    packets, templates = [], []
    for case in cases:
        if args.track == "nomination":
            text = {k: case[k] for k in ("id", "source_text", "inventory", "instruction")}
            system = NOMINATION_SYSTEM
            templates.append({"case_id": case["id"], "visual_requests": []})
        else:
            text = {"case_id": case["id"], "ref_id": case["ref_id"], "context": case["context"],
                    "questions": case["questions"], "provided_images": []}
            if args.mode == "vision":
                text["provided_images"] = [{"image_id": a["id"], "pdf_page": a["pdf_page"]} for a in case["images"]]
            else:
                text["context"] += " No source image is provided in this text-only control. Do not use remembered values."
            system = INSPECTION_SYSTEM
            templates.append({"case_id": case["id"], "answers": [
                {"question_id": q["id"], "status": None, "value": None, "unit": None, "evidence": []}
                for q in case["questions"]], "telemetry": {"wall_ms": None, "input_tokens": None,
                                                          "output_tokens": None, "request_count": None, "cache_hit": None}})
        content = [{"type": "text", "text": json.dumps(text, ensure_ascii=False)}]
        if args.track == "inspection" and args.mode == "vision":
            for asset in case["images"]:
                data = base64.b64encode((ROOT / asset["path"]).read_bytes()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}})
        packets.append({"case_id": case["id"], "messages": [{"role": "system", "content": system},
                       {"role": "user", "content": content}],
                        "limits": {"max_completion_tokens": 1200, "deadline_seconds": 30, "max_outbound_requests": 1}})
    (out / "requests.jsonl").write_text("".join(json.dumps(p) + "\n" for p in packets))
    dump(out / "response-template.json", templates)
    dump(out / "packet-manifest.json", {"track": args.track, "mode": args.mode,
                                         "suite": args.suite, "case_ids": [c["id"] for c in cases],
                                         "contains_reference_answers": False, "model_calls_made": 0})
    return {"prepared_cases": len(cases), "output": str(out), "model_calls_made": 0}


def equal_value(actual, expected):
    if expected is None or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        if isinstance(actual, bool):
            return False
        try:
            a = Decimal(str(actual).replace("−", "-").replace(",", ""))
            e = Decimal(str(expected))
            return a.is_finite() and a == e
        except InvalidOperation:
            return False
    if not isinstance(actual, str):
        return False
    return "".join(actual.casefold().split()) == "".join(expected.casefold().split())


def score_inspection(response, case, expected):
    submitted = response.get("answers")
    if not isinstance(submitted, list):
        submitted = []
    ids = [a.get("question_id") for a in submitted if isinstance(a, dict)]
    expected_ids = {a["question_id"] for a in expected["answers"]}
    shape_ok = all(isinstance(i, str) for i in ids) and len(ids) == len(submitted) == len(expected_ids) and set(ids) == expected_ids
    answers = {a["question_id"]: a for a in submitted if isinstance(a, dict) and isinstance(a.get("question_id"), str)}
    results = []
    for reference in expected["answers"]:
        actual = answers.get(reference["question_id"], {})
        value_ok = any(equal_value(actual.get("value"), v) for v in [reference["value"]] + reference["accepted_values"])
        unit_ok = actual.get("unit") == reference["unit"]
        status_ok = actual.get("status") == reference["status"]
        evidence = actual.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        valid_evidence = [e for e in evidence if isinstance(e, dict)
                          and isinstance(e.get("region"), str) and e["region"].strip()
                          and any(all(e.get(k) == allowed[k] for k in ("image_id", "pdf_page", "ref_id"))
                                  for allowed in reference["allowed_evidence"])]
        cited = {e["image_id"] for e in valid_evidence}
        provenance_ok = set(reference["required_image_ids"]) <= cited
        results.append({"question_id": reference["question_id"], "value_ok": value_ok,
                        "unit_ok": unit_ok, "status_ok": status_ok, "provenance_ok": provenance_ok,
                        "mechanical_pass": shape_ok and value_ok and unit_ok and status_ok and provenance_ok,
                        "unsupported_value_assertion": reference["value"] is None and actual.get("value") is not None,
                        "expected_status": reference["status"], "critical": reference["critical"]})
    correct = sum(a["mechanical_pass"] for a in results)
    return {"case_id": case["id"], "track": "inspection", "answer_shape_ok": shape_ok,
            "correct_atoms": correct, "total_atoms": len(results), "accuracy": correct / len(results),
            "mechanical_pass": shape_ok and correct == len(results), "atoms": results,
            "manual_review_required": True, "telemetry": response.get("telemetry") if isinstance(response.get("telemetry"), dict) else {},
            "telemetry_shape_ok": response.get("telemetry") is None or isinstance(response.get("telemetry"), dict)}


def score_nomination(response, case, reference):
    requests = response.get("visual_requests")
    shape_ok = isinstance(requests, list)
    requests = requests if shape_ok else []
    ids = [r.get("ref_id") for r in requests if isinstance(r, dict)]
    ids_are_strings = all(isinstance(i, str) for i in ids)
    valid = shape_ok and ids_are_strings and len(ids) == len(requests) and len(ids) == len(set(ids))
    safe_ids = [i for i in ids if isinstance(i, str)]
    valid = valid and len(requests) <= reference["max_requests"]
    valid = valid and set(safe_ids) <= set(reference["allowed_refs"])
    for request in requests:
        if not isinstance(request, dict):
            valid = False
            continue
        valid = valid and set(request) == {"ref_id", "priority", "reason", "question"}
        valid = valid and request.get("priority") in ("critical", "high")
        valid = valid and request.get("reason") in reference.get("allowed_reasons", [])
        question = request.get("question")
        valid = valid and isinstance(question, str) and 0 < len(question) <= 160
    missing = sorted(set(reference["required_refs"]) - set(safe_ids))
    return {"case_id": case["id"], "track": "nomination", "mechanical_pass": valid and not missing,
            "missing_refs": missing, "requested_refs": ids, "manual_question_review_required": True}


def score(args):
    raw = args.responses.read_text()
    responses = [json.loads(line) for line in raw.splitlines() if line.strip()] if args.responses.suffix == ".jsonl" else json.loads(raw)
    if not isinstance(responses, list) or not all(isinstance(r, dict) for r in responses):
        raise ValueError("Responses must be a JSON array or JSONL objects")
    ids = [r.get("case_id") for r in responses]
    if not all(isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Case IDs must be unique strings")
    cases = {c["id"]: c for c in read("cases.json")}
    nominations = {c["id"]: c for c in read("nomination-cases.json")}
    gold = read("reference/expectations.json")
    nom_gold = read("reference/nomination-expectations.json")
    unknown = set(ids) - (cases.keys() | nominations.keys())
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    selected = read("manifest.json")["smoke_case_ids"] if args.suite == "smoke" else list(cases)
    if args.track == "nomination":
        selected = list(nominations)
    elif args.track == "both":
        selected += list(nominations)
    by_id = {r["case_id"]: r for r in responses}
    results = []
    for case_id in selected:
        response = by_id.get(case_id, {"case_id": case_id})
        result = score_inspection(response, cases[case_id], gold[case_id]) if case_id in cases else score_nomination(response, nominations[case_id], nom_gold[case_id])
        result["response_present"] = case_id in by_id
        results.append(result)
    durations = [r["telemetry"].get("wall_ms") for r in results if r["track"] == "inspection"]
    durations = [v for v in durations if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < float("inf")]
    atom_total = sum(r.get("total_atoms", 0) for r in results)
    atom_correct = sum(r.get("correct_atoms", 0) for r in results)
    usage = {}
    violations = []
    for field in ("input_tokens", "output_tokens", "request_count"):
        values = [r["telemetry"].get(field) for r in results if r["track"] == "inspection"]
        values = [v for v in values if isinstance(v, int) and not isinstance(v, bool) and v >= 0]
        usage[field] = {"recorded_samples": len(values), "sum": sum(values) if values else None}
    for result in results:
        telemetry = result.get("telemetry", {})
        for field, cap in (("wall_ms", 30000), ("request_count", 1), ("output_tokens", 1200)):
            value = telemetry.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > cap:
                violations.append({"case_id": result["case_id"], "field": field, "value": value, "cap": cap})
    report = {"suite": args.suite, "track": args.track, "cases": len(results),
              "mechanically_passing_cases": sum(r["mechanical_pass"] for r in results),
              "answer_atoms": atom_total, "correct_atoms": atom_correct,
              "atom_accuracy": atom_correct / atom_total if atom_total else None,
              "inspection_case_mean_accuracy": statistics.mean([r["accuracy"] for r in results if r["track"] == "inspection"]) if atom_total else None,
              "unsupported_value_assertions": sum(a["unsupported_value_assertion"] for r in results for a in r.get("atoms", [])),
              "recorded_usage": usage, "budget_violations": violations,
              "latency": {"recorded_samples": len(durations), "mean_ms": statistics.mean(durations) if durations else None,
                          "median_ms": statistics.median(durations) if durations else None,
                          "over_30_second_samples": sum(v > 30000 for v in durations)},
              "missing_responses": [r["case_id"] for r in results if not r["response_present"]],
              "limits": "Mechanical matching does not verify region semantics, extra assertions, or independent scientific fidelity. Gold requires independent human adjudication. Missing telemetry is unknown, not zero.",
              "results": results}
    dump(args.output, report)
    return {k: v for k, v in report.items() if k != "results"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    export = sub.add_parser("prepare")
    export.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    export.add_argument("--track", choices=["inspection", "nomination"], default="inspection")
    export.add_argument("--mode", choices=["vision", "text-only"], default="vision")
    export.add_argument("--output", type=Path, required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--responses", type=Path, required=True)
    scoring.add_argument("--suite", choices=["smoke", "full"], default="full")
    scoring.add_argument("--track", choices=["inspection", "nomination", "both"], default="inspection")
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate() if args.command == "validate" else prepare(args) if args.command == "prepare" else score(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
