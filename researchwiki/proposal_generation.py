"""Evidence assembly and bounded LLM generation for page proposals."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .proposals import DIRECTIONS, load_proposals, validate_proposal
from .search import claim_query, claims_by_stem, format_claim_ref
from .wiki import read_pages, extract_section


MAX_PAPERS = 8
MAX_CLAIMS_PER_PAPER = 5
MAX_PROPOSALS = 3

# Shared lexical policy for editorial context, not a corpus-evidence cutoff.
_CONTEXT_STOPWORDS = frozenset("""
a an the and or of for to in on at by from with without as into across beyond
is are was were be been being do does did can could would should will may might
how what when where which who why this that these those it its their them they
we our you your have has had not no more most other same such than then also
current compare comparison using use used give get question questions
""".split())


def _topic_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _CONTEXT_STOPWORDS


def _context_score(topic: str, primary: str, summary: str) -> int:
    """Require substantive overlap, anchored in title/question/topic metadata.

    Two distinct terms are required for multi-term queries; a one-term query
    must match the primary fields. Summary vocabulary cannot qualify on its own.
    """
    terms = _topic_terms(topic)
    core = terms & _topic_terms(primary)
    extra = terms & _topic_terms(summary)
    if not core or len(core | extra) < min(2, len(terms)):
        return 0
    return 3 * len(core) + len(extra - core)


def _rank_explicit_claims(topic: str, explicit: list[dict], ranked: list[dict]) -> list[dict]:
    """Reuse query ranking before quotas; lexical fallback covers unranked claims.

    Explicit papers still precede discovered papers, ensuring membership even
    when none of their claims occurs in the bounded global retrieval results.
    """
    def identity(hit: dict) -> tuple[str, str, str]:
        return (str(hit.get("paper_stem") or ""), str(hit.get("claim_slug") or ""),
                str(hit.get("text") or ""))

    ranks = {identity(hit): rank for rank, hit in reversed(list(enumerate(ranked)))}
    terms = _topic_terms(topic)
    return sorted(explicit, key=lambda hit: (
        ranks.get(identity(hit), len(ranked)),
        -len(terms & _topic_terms(str(hit.get("text") or ""))),
    ))


_PROPOSAL_SCHEMA = {
    "type": "object",
    "required": ["proposals"],
    "properties": {
        "proposals": {
            "type": "array",
            "maxItems": MAX_PROPOSALS,
            "items": {
                "type": "object",
                "required": [
                    "title", "page_type", "direction", "question", "thesis",
                    "why_it_matters", "evidence_connections",
                    "distinct_from_existing", "decisive_uncertainty", "outline",
                ],
                "properties": {
                    "title": {"type": "string"},
                    "page_type": {"type": "string", "enum": ["synthesis", "idea"]},
                    "direction": {"type": "string", "enum": sorted(DIRECTIONS)},
                    "question": {"type": "string"},
                    "thesis": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "evidence_connections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["insight", "evidence_ids"],
                            "properties": {
                                "insight": {"type": "string"},
                                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                    "distinct_from_existing": {"type": "string"},
                    "decisive_uncertainty": {"type": "string"},
                    "outline": {"type": "array", "items": {"type": "string"}},
                    "source_method": {"type": ["string", "null"]},
                    "target_problem": {"type": ["string", "null"]},
                    "transfer_mapping": {"type": ["string", "null"]},
                    "mechanism": {"type": ["string", "null"]},
                    "assumptions_to_test": {"type": ["string", "null"]},
                    "necessary_adaptations": {"type": ["string", "null"]},
                    "baseline": {"type": ["string", "null"]},
                    "first_experiment": {"type": ["string", "null"]},
                    "parent_proposal": {"type": ["string", "null"]},
                },
            },
        },
    },
}

_SEARCH_PLAN_SCHEMA = {
    "type": "object",
    "required": ["target_problem", "required_capabilities", "queries"],
    "properties": {
        "target_problem": {"type": "string"},
        "required_capabilities": {"type": "array", "items": {"type": "string"}},
        "queries": {"type": "array", "minItems": 1, "maxItems": 3,
                    "items": {"type": "string"}},
    },
}

_PROPOSAL_SYSTEM = """\
You are an editorial research partner proposing insightful synthesis and idea
pages from a private, source-grounded wiki. Use only the supplied evidence for
claims about papers. Proposed explanations and applications may be creative,
but label them as hypotheses in their wording.

Return zero to three genuinely distinct proposals. Prefer one strong proposal
to several generic surveys. Explore only directions supported by the packet:
tension, shared-mechanism, complementary-limitations, boundary-condition, and
cross-category-application. A synthesis must offer an explanation, comparison,
or decision rule. An idea must name a mechanism, assumptions, a baseline, and a
test that could reject it. Evidence connections must cite only supplied IDs.
Treat prior rejected/deferred proposals as negative constraints and do not repeat
published or unchanged proposals. If feedback motivates a material revision,
set parent_proposal to that supplied proposal ID and explain the distinction.
Do not claim novelty outside this wiki. Return strict JSON only.
Existing pages are overlap context, not evidence. Omit covered proposals unless
distinct_from_existing names a concrete new question or grounded distinction.
If the packet does not support the requested topic, return {"proposals": []}.
Do not substitute an unrelated topic. Use exactly the field names in the schema,
not synonyms such as type, hypothesis, claim, assumptions, or test. All design
fields are strings, not nested objects. Every idea requires mechanism,
assumptions_to_test, baseline, and first_experiment; the experiment must state a
rejection criterion and define comparable budgets, including offline costs.
Distinguish design-fit hypotheses from demonstrated comparative superiority.
Before returning, check each prior proposal: if your proposal revises its
question in response to feedback, parent_proposal MUST be that supplied ID,
even if you changed the title or narrowed it to an experiment. Use null only
for independent proposals. Explain the revision in distinct_from_existing.
"""

_SEARCH_PLAN_SYSTEM = """\
Translate a problem in one research category into method capabilities that can
be searched in other categories. Use only the supplied target evidence for the
problem statement. Produce one to three short queries describing operations,
constraints, or evaluation needs; do not insert a guessed method name. Return
strict JSON only. Keep the target-specific problem in target_problem, but
translate it into domain-independent required_capabilities and search queries.
Do not repeat target-specific formats or domain vocabulary in every query.
"""


def _with_contract(system: str, schema: dict) -> str:
    # API transports do not all forward schema; every model must see it.
    return system + "\nREQUIRED OUTPUT JSON SCHEMA:\n" + json.dumps(schema)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)[:-3].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("model returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def paper_categories(pages: list | None = None) -> dict[str, str]:
    """`{stem: category}` for every paper page (the directory is canonical)."""
    return {
        page.stem: page.category
        for page in (pages if pages is not None else read_pages())
        if page.page_type == "paper"
    }


def _as_evidence(hit: dict, category: str) -> dict[str, Any]:
    stem = str(hit.get("paper_stem") or "")
    slug = str(hit.get("claim_slug") or "")
    return {
        "id": "",  # assigned after de-duplication
        "paper_stem": stem,
        "category": category,
        "section": str(hit.get("section") or ""),
        "claim_slug": slug,
        "citation": format_claim_ref(hit),
        "text": str(hit.get("text") or "").strip(),
        "supporting_text": str(hit.get("supporting_text") or "").strip(),
        "supporting_provenance": str(hit.get("supporting_provenance") or "").strip(),
    }


def _select_diverse(
    hits: list[dict],
    categories: dict[str, str],
    *,
    eligible_category: str | None = None,
    excluded_category: str | None = None,
    max_papers: int = MAX_PAPERS,
    per_paper: int = MAX_CLAIMS_PER_PAPER,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for hit in hits:
        stem = str(hit.get("paper_stem") or "")
        category = categories.get(stem, "")
        if not category:
            continue
        if eligible_category and category != eligible_category:
            continue
        if excluded_category and category == excluded_category:
            continue
        # A claim without a slug has no durable `[[stem#slug]]` citation, and
        # receipt validation requires one. Admitting it here spent the paid
        # generation call and then failed the save, discarding the result.
        if not str(hit.get("claim_slug") or "").strip():
            continue
        ident = (stem, str(hit.get("claim_slug") or ""), str(hit.get("text") or ""))
        if ident in seen or counts[stem] >= per_paper:
            continue
        if stem not in counts and len(counts) >= max_papers:
            continue
        seen.add(ident)
        counts[stem] += 1
        chosen.append(_as_evidence(hit, category))
    return chosen


def _assign_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, item in enumerate(items, 1):
        item["id"] = f"e{i:02d}"
    return items


def _claims_for_papers(stems: list[str]) -> list[dict]:
    out: list[dict] = []
    for raw in stems:
        stem = raw.rsplit("/", 1)[-1]
        hits = claims_by_stem(stem, include_context=True)
        if not hits:
            raise ValueError(f"explicit paper has no claim evidence: {stem}")
        out.extend(hits)
    return out


def _evidence_prompt(items: list[dict[str, Any]]) -> str:
    blocks = []
    for item in items:
        block = (
            f"[{item['id']}] {item['category']}/{item['paper_stem']} "
            f"§{item['section']}\nCLAIM: {item['text']}"
        )
        if item.get("supporting_text"):
            where = item.get("supporting_provenance") or "source PDF"
            block += f"\nSOURCE ({where}): {item['supporting_text']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _relevant_history(
    topic: str, limit: int = 5, pages: list | None = None,
) -> list[dict[str, str]]:
    scored = []
    for record in load_proposals(pages):
        overlap = _context_score(topic, f"{record.title} {record.question} {record.topic_seed}",
                                 record.thesis)
        if overlap:
            latest = record.feedback[-1].reason if record.feedback else ""
            scored.append((overlap, {
                "proposal_id": record.proposal_id,
                "title": record.title,
                "question": record.question,
                "thesis": record.thesis,
                "status": record.status,
                "latest_feedback": latest,
            }))
    return [item for _, item in sorted(scored, key=lambda x: -x[0])[:limit]]


def _existing_pages(topic: str, limit: int = 3, pages: list | None = None) -> list[dict]:
    scored = []
    for page in pages if pages is not None else read_pages():
        if page.page_type not in {"synthesis", "idea"}:
            continue
        summary = "\n".join(extract_section(page.body, heading) or "" for heading in
                            ("Question", "Short answer", "Verdict"))[:2000]
        title = str(page.fm.get("title") or page.stem)
        primary = f"{title} {page.fm.get('topic_seed') or ''} {extract_section(page.body, 'Question') or ''}"
        overlap = _context_score(topic, primary, summary)
        if overlap:
            scored.append((overlap, {"page": f"{page.category}/{page.stem}",
                                     "title": title, "summary": summary}))
    return [item for _, item in sorted(scored, key=lambda x: -x[0])[:limit]]


def build_evidence_packet(
    topic: str,
    *,
    papers: list[str] | None = None,
    target_category: str | None = None,
    cross_category: bool = False,
) -> dict[str, Any]:
    """Prepare evidence locally; cross-category planning is a separate step."""
    if not topic.strip():
        raise ValueError("topic must not be empty")
    # One wiki walk serves categories, history, and existing-page context;
    # each helper used to re-parse every page.
    pages = read_pages()
    categories = paper_categories(pages)
    stems = list(dict.fromkeys(raw.rsplit("/", 1)[-1] for raw in papers or []))
    if len(stems) > (4 if cross_category else MAX_PAPERS):
        raise ValueError("too many explicit papers: limit is 4 for cross-category, otherwise 8")
    for stem in stems:
        if stem not in categories:
            raise ValueError(f"unknown explicit paper: {stem}")
        if cross_category and categories[stem] != target_category:
            raise ValueError("cross-category --papers must belong to the target category")
    explicit = _claims_for_papers(stems)
    topic_hits = claim_query(topic, k=80, mode="hybrid", include_context=True)
    initial = _rank_explicit_claims(topic, explicit, topic_hits) + topic_hits

    if not cross_category:
        evidence = _select_diverse(initial, categories)
        inferred_category = target_category or (
            Counter(item["category"] for item in evidence if item["category"])
            .most_common(1)[0][0]
            if any(item["category"] for item in evidence) else "other"
        )
        return {
            "topic": topic,
            "target_category": inferred_category,
            "cross_category": False,
            "search_plan": None,
            "evidence": _assign_ids(evidence),
            "prior_proposals": _relevant_history(topic, pages=pages),
            "existing_pages": _existing_pages(topic, pages=pages),
        }

    if not target_category:
        raise ValueError("--cross-category requires --target-category")
    target = _select_diverse(
        initial, categories, eligible_category=target_category,
        max_papers=4, per_paper=3,
    )
    if not target:
        raise ValueError(f"no target evidence found in category: {target_category}")

    return {"topic": topic, "target_category": target_category,
            "cross_category": True, "search_plan": None,
            "evidence": _assign_ids(target),
            "prior_proposals": _relevant_history(topic, pages=pages),
            "existing_pages": _existing_pages(topic, pages=pages), "planning_pending": True}


def validate_search_plan(plan: Any) -> dict[str, Any]:
    """Structural check shared by the planner call and chat-authored plans."""
    if not isinstance(plan, dict):
        raise ValueError("cross-category search plan must be a JSON object")
    if not isinstance(plan.get("target_problem"), str) or not plan["target_problem"].strip():
        raise ValueError("cross-category planner must state target_problem")
    capabilities = plan.get("required_capabilities")
    if (not isinstance(capabilities, list) or not capabilities
            or any(not isinstance(c, str) or not c.strip() for c in capabilities)):
        raise ValueError("cross-category planner must state required_capabilities")
    queries = plan.get("queries")
    if (not isinstance(queries, list) or not 1 <= len(queries) <= 3
            or any(not isinstance(q, str) or not q.strip() for q in queries)):
        raise ValueError("cross-category planner returned no usable queries")
    return plan


def apply_search_plan(
    packet: dict[str, Any], plan: dict[str, Any], usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve source-category claims for a validated plan. Makes no model call.

    Split from the planner call so a chat agent can write the plan itself
    (`generate --search-plan`) and still get a packet with source evidence.
    Without this, the documented chat path could only produce a target-only
    packet, which no cross-category proposal can pass.
    """
    validate_search_plan(plan)
    topic, target_category = packet["topic"], packet["target_category"]
    target = packet["evidence"]
    pages = read_pages()
    categories = paper_categories(pages)
    source_hits: list[dict] = []
    for query in plan["queries"]:
        source_hits.extend(claim_query(query, k=60, mode="hybrid", include_context=True))
    target_papers = {item["paper_stem"] for item in target}
    source = _select_diverse(
        source_hits, categories, excluded_category=target_category,
        max_papers=min(5, MAX_PAPERS - len(target_papers)), per_paper=3,
    )
    if not source:
        raise ValueError("no method evidence found outside the target category")
    evidence = _assign_ids(target + source)
    return {
        **packet,
        "topic": topic,
        "target_category": target_category,
        "cross_category": True,
        "search_plan": plan,
        "planning_pending": False,
        "evidence": evidence,
        "prior_proposals": _relevant_history(topic, pages=pages),
        "usage": dict(usage or {}),
    }


def expand_cross_category(packet: dict[str, Any]) -> dict[str, Any]:
    """Make the explicit planner call, then retrieve source-category claims."""
    topic, target_category = packet["topic"], packet["target_category"]
    target = packet["evidence"]

    from .agents import llm
    plan_resp = llm.call(
        phase="proposal_search_plan",
        system=_with_contract(_SEARCH_PLAN_SYSTEM, _SEARCH_PLAN_SCHEMA),
        prompt=(f"TARGET CATEGORY: {target_category}\nTOPIC: {topic}\n\n"
                f"TARGET EVIDENCE:\n{_evidence_prompt(_assign_ids(target))}"),
        schema=_SEARCH_PLAN_SCHEMA,
    )
    plan = validate_search_plan(_parse_json(plan_resp.text))
    return apply_search_plan(packet, plan, usage={
        "search_plan_model": plan_resp.model,
        "search_plan_input_tokens": plan_resp.input_tokens,
        "search_plan_output_tokens": plan_resp.output_tokens,
    })


def proposal_request(packet: dict[str, Any]) -> dict[str, Any]:
    """Build the production request without retrieval, provider calls, or writes."""
    evidence = packet.get("evidence") or []
    if not evidence:
        raise ValueError("no evidence available for proposal generation")
    history = json.dumps(packet.get("prior_proposals") or [], ensure_ascii=False, indent=2)
    search_plan = json.dumps(packet.get("search_plan"), ensure_ascii=False, indent=2)
    prompt = (
        f"TOPIC OR QUESTION: {packet['topic']}\n"
        f"TARGET CATEGORY: {packet.get('target_category') or '(unspecified)'}\n"
        f"CROSS-CATEGORY SEARCH PLAN: {search_plan}\n\n"
        f"PRIOR PROPOSALS AND USER FEEDBACK:\n{history}\n\n"
        f"EXISTING PAGES (overlap context only):\n{json.dumps(packet.get('existing_pages', []))}\n\n"
        f"EVIDENCE PACKET:\n{_evidence_prompt(evidence)}"
    )
    if packet.get("cross_category"):
        prompt += (
            "\n\nMODE: CROSS-CATEGORY APPLICATION ONLY. Every proposal must have "
            "page_type=idea and direction=cross-category-application. Name a method "
            "from a NON-target category in source_method; cite that method's evidence "
            "and target-category evidence in evidence_connections. Explain which "
            "source operation maps to which target operation in transfer_mapping. "
            "Include target_problem, mechanism, assumptions_to_test, necessary_adaptations, "
            "baseline, and first_experiment (with a rejection criterion). "
            "A target-domain design alone is NOT a transfer. If no supplied source "
            "method supports a defensible transfer, return {\"proposals\": []}."
        )
    return {"phase": "proposal_generation",
            "system": _with_contract(_PROPOSAL_SYSTEM, _PROPOSAL_SCHEMA),
            "prompt": prompt, "schema": _PROPOSAL_SCHEMA}


def parse_proposal_response(text: str, packet: dict[str, Any]) -> list[dict]:
    """Apply the same parsing and validation to ordinary and benchmark runs."""
    parsed = _parse_json(text)
    proposals = parsed.get("proposals")
    if not isinstance(proposals, list) or len(proposals) > MAX_PROPOSALS:
        raise ValueError("proposal response must contain at most three proposals")
    validate_candidates(proposals, packet)
    return proposals


def generate_proposals(packet: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    from .agents import llm
    response = llm.call(**proposal_request(packet))
    proposals = parse_proposal_response(response.text, packet)
    usage = {
        "model": response.model,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }
    return proposals, usage


def validate_candidates(proposals: list[dict], packet: dict[str, Any]) -> None:
    if not isinstance(proposals, list) or len(proposals) > MAX_PROPOSALS:
        raise ValueError("proposal response must contain at most three proposals")
    evidence = packet["evidence"]
    evidence_ids = {str(item["id"]) for item in evidence}
    evidence_by_id = {str(item["id"]): item for item in evidence}
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("each proposal must be an object")
        validate_proposal(proposal, evidence_ids)
        used_ids = {
            str(evidence_id)
            for connection in proposal.get("evidence_connections") or []
            for evidence_id in connection.get("evidence_ids") or []
        }
        used_stems = {
            str(evidence_by_id[evidence_id].get("paper_stem") or "")
            for evidence_id in used_ids
        }
        if len(used_stems - {""}) < 2:
            raise ValueError("proposal must connect evidence from at least two papers")
        if packet.get("cross_category") and proposal["direction"] != "cross-category-application":
            raise ValueError("cross-category generation returned a non-transfer proposal")
        if proposal["direction"] == "cross-category-application":
            used_categories = {
                str(evidence_by_id[evidence_id].get("category") or "")
                for evidence_id in used_ids
            }
            target = str(packet.get("target_category") or "")
            if target not in used_categories or not (used_categories - {target, ""}):
                raise ValueError(
                    "cross-category proposal must cite target and source-category evidence"
                )
        parent = str(proposal.get("parent_proposal") or "").strip()
        known_parents = {
            str(item.get("proposal_id") or "")
            for item in packet.get("prior_proposals") or []
        }
        if parent and parent not in known_parents:
            raise ValueError(f"proposal references unknown parent proposal: {parent}")
