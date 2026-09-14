"""Offline proposal benchmark preparation and attributed reviewer scoring.

Run from the repository root:
  python -m researchwiki.benchmark.proposals prepare --out output/proposal-benchmark-v2
  python -m researchwiki.benchmark.proposals check --pack output/proposal-benchmark-v2
  python -m researchwiki.benchmark.proposals score --pack output/proposal-benchmark-v2 --reviews reviews.json

No generation calls, model judging, wiki writes, or database schema additions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

AXES = ("evidence", "insight", "contribution", "decision_value", "proportionality")
FIXTURES = Path("benchmark-fixtures/proposals")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_suite(directory: Path = FIXTURES) -> tuple[dict, dict]:
    inputs = yaml.safe_load((directory / "inputs.yaml").read_text())
    expected = yaml.safe_load((directory / "expectations.yaml").read_text())
    validate_suite(inputs, expected)
    return inputs, expected


def case_sources(case: dict, inputs: dict) -> dict[str, list[str]]:
    return {alias: slugs for alias, slugs in inputs["evidence_sets"][case["evidence_set"]].items()
            if alias not in case.get("exclude_sources", [])}


def validate_suite(inputs: dict, expected: dict) -> None:
    if inputs.get("version") != 1 or expected.get("version") != 1:
        raise ValueError("unsupported fixture version")
    if inputs["suite_id"] != expected["suite_id"]:
        raise ValueError("suite IDs differ")
    cases = inputs["cases"]
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases) or set(by_id) != set(expected["cases"]):
        raise ValueError("duplicate case IDs or unmatched expectations")
    stems_by_split: dict[str, set[str]] = {"development": set(), "heldout": set()}
    groups_by_split: dict[str, set[str]] = {"development": set(), "heldout": set()}
    for case in cases:
        if case["split"] not in stems_by_split:
            raise ValueError("unknown split")
        sources = case_sources(case, inputs)
        group = inputs["source_groups"][case["group"]]
        if not sources or not sources.keys() <= group.keys():
            raise ValueError(f"invalid sources for {case['id']}")
        for slugs in sources.values():
            if not slugs or len(set(slugs)) != len(slugs):
                raise ValueError("empty or duplicate claim selection")
        stems_by_split[case["split"]].update(group[alias] for alias in sources)
        groups_by_split[case["split"]].add(case["group"])
        if case.get("paired_with") and by_id[case["paired_with"]]["split"] != case["split"]:
            raise ValueError("paired cases cross splits")
        rubric = expected["cases"][case["id"]]
        if rubric["expectation"] not in {"opportunity", "abstain"}:
            raise ValueError("unknown expectation")
        refs = {f"{alias}#{slug}" for alias, slugs in sources.items() for slug in slugs}
        if not set(rubric.get("essential_refs", [])) <= refs:
            raise ValueError(f"essential refs absent from {case['id']}")
        if rubric.get("expected_parent"):
            history = inputs["histories"].get(case.get("history"), [])
            if rubric["expected_parent"] not in {p["proposal_id"] for p in history}:
                raise ValueError("expected parent absent from generator history")
    if (stems_by_split["development"] & stems_by_split["heldout"]
            or groups_by_split["development"] & groups_by_split["heldout"]):
        raise ValueError("source or group leakage between development and heldout")


def prepare(out: Path, directory: Path = FIXTURES) -> dict:
    """Freeze local corpus data. Refuse overwrites and missing source evidence."""
    from ..paths import resolve_pdf
    from ..grade.parser import parse_claims
    from ..search import claims_by_stem, format_claim_ref
    from ..wiki import read_pages
    import pypdfium2 as pdfium

    inputs, expected = load_suite(directory)
    if out.exists():
        raise ValueError("output already exists; select a new pack directory")
    pages = {page.stem: page for page in read_pages()}
    claims: dict[str, dict] = {}
    page_claims: dict[str, set[tuple[str, str]]] = {}
    payloads: dict[str, bytes] = {}
    sources: dict[str, dict] = {}
    case_entries = []
    unavailable = []
    for case in inputs["cases"]:
        group = inputs["source_groups"][case["group"]]
        evidence, corpus_paths = [], []
        for alias, slugs in case_sources(case, inputs).items():
            stem = group[alias]
            if stem not in pages:
                unavailable.append(f"{case['id']}: wiki page missing: {stem}")
                continue
            if stem not in claims:
                claims[stem] = {c["claim_slug"]: c for c in claims_by_stem(stem, include_context=True)}
                page_claims[stem] = {(c.section, c.text) for c in parse_claims(pages[stem])}
            page = pages[stem]
            corpus_path = f"corpus/{stem}.md"
            payloads[corpus_path] = page.path.read_bytes()
            corpus_paths.append(corpus_path)
            if stem not in sources:
                try:
                    pdf = resolve_pdf(stem)
                    pdf_bytes = pdf.read_bytes()
                    chunks = []
                    with pdfium.PdfDocument(pdf_bytes) as document:
                        for i in range(len(document)):
                            pdf_page = document[i]
                            try:
                                text_page = pdf_page.get_textpage()
                                try:
                                    chunks.append(f"\n--- PDF page {i + 1} ---\n" + text_page.get_text_range())
                                finally:
                                    text_page.close()
                            finally:
                                pdf_page.close()
                    if not any(chunk.split("---\n", 1)[-1].strip() for chunk in chunks):
                        raise ValueError("PDF has no extractable text")
                    text_path = f"sources/{stem}.txt"
                    payloads[text_path] = "\n".join(chunks).encode("utf-8")
                    sources[stem] = {"pdf_sha256": _hash(pdf_bytes), "text_path": text_path,
                                     "wiki_path": corpus_path, "category": page.category}
                except (OSError, ValueError) as exc:
                    unavailable.append(f"{case['id']}: {stem}: {exc}")
                    continue
            for slug in slugs:
                hit = claims[stem].get(slug)
                if not hit:
                    unavailable.append(f"{case['id']}: claim missing: {stem}#{slug}")
                    continue
                if (hit["section"], hit["text"]) not in page_claims[stem]:
                    unavailable.append(f"{case['id']}: stale DB claim differs from wiki: {stem}#{slug}")
                    continue
                evidence.append({"id": f"e{len(evidence) + 1:02}", "paper_stem": stem,
                                 "category": page.category, "claim_slug": slug,
                                 "section": hit["section"], "citation": format_claim_ref(hit),
                                 "text": hit["text"], "supporting_text": hit.get("supporting_text") or "",
                                 "supporting_provenance": hit.get("supporting_provenance") or ""})
        context = []
        for reference in case.get("existing_pages", []):
            page = pages.get(reference.rsplit("/", 1)[-1])
            if not page or f"{page.category}/{page.stem}" != reference:
                unavailable.append(f"{case['id']}: existing page missing: {reference}")
                continue
            context.append({"page": reference, "title": str(page.fm.get("title") or page.stem),
                            "summary": page.body})  # Full content; same for every generator.
        packet = {"topic": case["topic"], "target_category": case["target_category"],
                  "cross_category": case.get("cross_category", False), "search_plan": None,
                  "evidence": evidence, "existing_pages": context,
                  "prior_proposals": inputs["histories"].get(case.get("history"), [])}
        path = f"inputs/{case['id']}.json"
        payloads[path] = (json.dumps(packet, indent=2, ensure_ascii=False) + "\n").encode()
        case_entries.append({"id": case["id"], "split": case["split"], "group": case["group"],
                             "family": case["family"], "input_path": path,
                             "corpus_paths": corpus_paths,
                             "paired_with": case.get("paired_with"),
                             "pair_change": case.get("pair_change")})
    if unavailable:
        raise ValueError("suite unavailable; no pack written:\n" + "\n".join(unavailable))
    # The manifest hashes the specifications; evaluator instructions are never
    # included in generator packets. Keep evaluation files in their own directory.
    payloads["evaluator/expectations.yaml"] = (directory / "expectations.yaml").read_bytes()
    payloads["evaluator/inputs-spec.yaml"] = (directory / "inputs.yaml").read_bytes()
    payloads["evaluator/SCORING.md"] = (directory / "SCORING.md").read_bytes()
    calibration = directory / "CALIBRATION.md"
    if calibration.exists():
        payloads["evaluator/CALIBRATION.md"] = calibration.read_bytes()
    manifest = {"version": 1, "suite_id": inputs["suite_id"], "max_proposals": inputs["max_proposals"],
                "rubric_version": expected["rubric_version"],
                "review_status": expected["review_status"], "cases": case_entries, "sources": sources,
                "files": {path: _hash(content) for path, content in sorted(payloads.items())}}
    manifest["pack_id"] = _hash(json.dumps(manifest, sort_keys=True).encode())
    out.mkdir(parents=True)
    for path, content in payloads.items():
        destination = out / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    _json(out / "manifest.json", manifest)
    return manifest


def check_pack(pack: Path) -> dict:
    manifest = json.loads((pack / "manifest.json").read_text())
    identity = {k: v for k, v in manifest.items() if k != "pack_id"}
    if _hash(json.dumps(identity, sort_keys=True).encode()) != manifest["pack_id"]:
        raise ValueError("manifest hash mismatch")
    for relative, digest in manifest["files"].items():
        path = (pack / relative).resolve()
        if not path.is_relative_to(pack.resolve()) or _hash(path.read_bytes()) != digest:
            raise ValueError(f"pack file hash mismatch: {relative}")
    for case in manifest["cases"]:
        if any(path not in manifest["files"] for path in
               [case["input_path"], *case.get("corpus_paths", [])]):
            raise ValueError("case references an unhashed pack file")
        packet = json.loads((pack / case["input_path"]).read_text())
        if "shortlist_conditions" in packet or "expectation" in packet:
            raise ValueError("evaluator fields leaked into input")
        for item in packet["evidence"]:
            if item["citation"] != f"[[{item['paper_stem']}#{item['claim_slug']}]]":
                raise ValueError("evidence citation mismatch")
    return manifest


def score_reviews(reviews: list[dict], expected: dict, case_ids: set[str]) -> dict:
    """Score one candidate/repeat block with complete, adjudicated reviews.

    No pooled pseudo-replicates. Missing cases are an error; no-proposal
    operational failures stay in the denominator and never count as abstention.
    """
    by_id = {review["case_id"]: review for review in reviews}
    if len(by_id) != len(reviews) or set(by_id) != case_ids:
        raise ValueError("reviews must contain each selected case exactly once")
    counters = dict(opportunities=0, useful=0, emitted=0, shortlist=0, abstention_cases=0,
                    correct_abstention=0, unnecessary_abstention=0, critical=0,
                    operational_success=0, feedback_cases=0, feedback_pass=0)
    details = []
    for case_id, review in by_id.items():
        rubric = expected["cases"][case_id]
        ok = review["operational_success"]
        if type(ok) is not bool:
            raise ValueError("operational_success must be boolean")
        counters["operational_success"] += ok
        proposals = review["proposals"]
        if not isinstance(proposals, list) or (ok and len(proposals) > 3):
            raise ValueError("a successful run may emit at most three proposals")
        counters["emitted"] += len(proposals)
        useful = 0
        for proposal in proposals:
            scores = proposal["scores"]
            reasons = proposal["reasons"]
            if set(scores) != set(AXES) or set(reasons) != set(AXES):
                raise ValueError("five axis scores and reasons are required")
            if any(type(score) is not int or score not in (0, 1, 2) for score in scores.values()):
                raise ValueError("axis scores must be integers from zero to two")
            if any(not isinstance(reason, str) or not reason.strip() for reason in reasons.values()):
                raise ValueError("every score requires a reason")
            for flag in ("critical_factual_failure", "constraint_failure"):
                if type(proposal[flag]) is not bool:
                    raise ValueError(f"{flag} must be boolean")
            decision = proposal["decision"]
            reason = proposal.get("decision_reason")
            if (decision not in {"shortlist", "defer", "reject"}
                    or not isinstance(reason, str) or not reason.strip()):
                raise ValueError("reviewer decision and reason required")
            critical = proposal["critical_factual_failure"]
            blocked = critical or proposal["constraint_failure"]
            if decision == "shortlist" and blocked:
                raise ValueError("adjudicate shortlist versus critical/constraint failure first")
            total = sum(scores.values())
            provisional = total >= 8 and min(scores.values()) > 0 and not blocked
            # Direct reviewer decisions define utility; CLI reports reviewer
            # provenance separately from the uncalibrated threshold.
            useful += decision == "shortlist"
            counters["shortlist"] += ok and decision == "shortlist"
            counters["critical"] += critical
            details.append({"case_id": case_id, "total": total, "decision": decision,
                            "threshold_shortlist": provisional,
                            "threshold_disagrees": provisional != (decision == "shortlist")})
        positive = rubric["expectation"] == "opportunity"
        if not positive and useful:
            raise ValueError("shortlisted output conflicts with abstention label; adjudicate the case")
        counters["opportunities"] += positive
        counters["useful"] += positive and ok and useful > 0
        counters["unnecessary_abstention"] += positive and ok and not proposals
        counters["abstention_cases"] += not positive
        counters["correct_abstention"] += not positive and ok and not proposals
        if rubric.get("feedback_requirements"):
            verdicts = review.get("feedback_checks")
            if (not isinstance(verdicts, list) or len(verdicts) != len(rubric["feedback_requirements"])
                    or any(type(v.get("passed")) is not bool or not v.get("reason") for v in verdicts)):
                raise ValueError("feedback requires one boolean verdict and reason per requirement")
            counters["feedback_cases"] += 1
            parent_ok = bool(proposals) and all(
                p.get("parent_proposal") == rubric["expected_parent"] for p in proposals)
            counters["feedback_pass"] += ok and parent_ok and all(v["passed"] for v in verdicts)
    def rate(numerator: str, denominator: str) -> float | None:
        return counters[numerator] / counters[denominator] if counters[denominator] else None
    return {"counts": counters, "useful_proposal_yield": rate("useful", "opportunities"),
            "proposal_precision": rate("shortlist", "emitted"),
            "appropriate_abstention": rate("correct_abstention", "abstention_cases"),
            "unnecessary_abstention": rate("unnecessary_abstention", "opportunities"),
            "critical_factual_failure_rate": rate("critical", "emitted"),
            "feedback_compliance": rate("feedback_pass", "feedback_cases"),
            "operational_success": counters["operational_success"] / len(reviews) if reviews else None,
            "proposal_scores": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check", "score"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            if not args.out:
                parser.error("prepare requires --out")
            result = prepare(args.out)
            print(json.dumps({"pack_id": result["pack_id"], "cases": len(result["cases"]),
                              "sources": len(result["sources"]), "path": str(args.out)}, indent=2))
        elif args.action == "check":
            if not args.pack:
                parser.error("check requires --pack")
            manifest = check_pack(args.pack)
            print(f"Verified {manifest['pack_id']}: {len(manifest['cases'])} cases")
        else:
            if not args.reviews or not args.pack:
                parser.error("score requires --reviews and --pack")
            manifest = check_pack(args.pack)
            expected = yaml.safe_load((args.pack / "evaluator/expectations.yaml").read_text())
            block = json.loads(args.reviews.read_text())
            if block["pack_id"] != manifest["pack_id"]:
                raise ValueError("review belongs to a different pack")
            if block["split"] not in {"development", "heldout"}:
                raise ValueError("select development or heldout")
            if block["execution_mode"] not in {"fixed-packet", "end-to-end"}:
                raise ValueError("select fixed-packet or end-to-end execution_mode")
            if block.get("reviewer_kind") not in {"human", "agent"}:
                raise ValueError("reviewer_kind must be human or agent; do not pool reviewer kinds")
            if not isinstance(block.get("reviewer"), str) or not block["reviewer"].strip():
                raise ValueError("reviewer identity required")
            case_ids = {c["id"] for c in manifest["cases"] if c["split"] == block["split"]}
            result = score_reviews(block["reviews"], expected, case_ids)
            print(json.dumps({"pack_id": block["pack_id"], "split": block["split"],
                              "execution_mode": block["execution_mode"],
                              "reviewer_kind": block["reviewer_kind"], "reviewer": block["reviewer"],
                              "reviewer_model": block.get("reviewer_model"),
                              "rubric_version": manifest.get("rubric_version"),
                              "candidate": block["candidate"], "repeat": block["repeat"], **result}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"proposal benchmark: {exc}\n")


if __name__ == "__main__":
    main()
