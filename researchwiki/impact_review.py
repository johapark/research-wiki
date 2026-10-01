"""Local, review-only impact scan from a newly ingested paper to authored pages.

The scan is deliberately separate from model-backed memory evolution. Its YAML
receipt survives a process restart and ``db rebuild``; no wiki prose is edited.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import yaml

from .fsatomic import write_text_atomic
from .paths import ingest_dir, wiki_dir
from .wiki import Page, extract_section, read_page, read_pages


REVIEW_DIR = "impact-review"
TARGET_TYPES = frozenset({"synthesis", "idea", "proposal"})
DECISIONS = frozenset({"pending", "incorporated", "not_relevant", "deferred"})
# A candidate needs strong page similarity, or moderate similarity corroborated
# by a distinctive two-term phrase. All target scores remain in the receipt.
SEMANTIC_FLOOR = 0.83
CORROBORATED_FLOOR = 0.76
SAME_CATEGORY_FLOOR = 0.80
MIN_SHARED_TERMS = 5
_COMMON = frozenset("""
the and for with from into across using used use study paper method methods
analysis analyses data dataset datasets model models gene genes cell cells type
types single rna seq sequencing benchmark benchmarking result results approach
approaches prediction predictive human based relevant reference references
single-cell comparison comparisons
""".split())


def review_path(stem: str) -> Path:
    return ingest_dir() / REVIEW_DIR / f"{stem}.yaml"


def _source_text(page: Page) -> str:
    return "\n".join(filter(None, [
        page.str_field("title"), page.str_field("keywords"),
        extract_section(page.body, "Summary"),
        extract_section(page.body, "Key Contributions"),
        extract_section(page.body, "Methodology and Architecture"),
        extract_section(page.body, "Results"),
    ]))


def _target_text(page: Page) -> str:
    headings = {
        "synthesis": ("Question", "Short answer", "What would update this page"),
        "idea": ("Verdict", "Background", "Opportunities", "Plans", "Caveats"),
        "proposal": ("Question", "Provisional thesis or hypothesis",
                     "Why this connection matters", "Decisive uncertainty"),
    }[page.page_type]
    parts = [page.str_field("title"), page.str_field("topic_seed")]
    parts.extend(extract_section(page.body, h) for h in headings)
    # The full substantive body supplies lexical evidence even when a synthesis
    # uses a thematic middle heading not covered by its general semantic index.
    body = _substantive_body(page)
    parts.append(body)
    return "\n".join(p for p in parts if p)


def _substantive_body(page: Page) -> str:
    """Exclude citation lists and proposal feedback from content fingerprints."""
    return re.split(r"(?m)^## (?:References|Feedback)\s*$", page.body,
                    maxsplit=1)[0]


def _source_passages(page: Page) -> list[str]:
    summary = extract_section(page.body, "Summary")
    contributions = extract_section(page.body, "Key Contributions")
    methodology = extract_section(page.body, "Methodology and Architecture")
    bullets = [line.strip().lstrip("- ") for line in contributions.splitlines()
               if line.lstrip().startswith("-")]
    return [page.str_field("title") + "\n" + summary[:1200],
            *bullets[:10], methodology[:1200]]


def _target_passages(page: Page, source_terms: set[str]) -> list[str]:
    core = "\n".join(filter(None, [
        page.str_field("title"), page.str_field("topic_seed"),
        extract_section(page.body, "Question"),
        extract_section(page.body, "Short answer"),
        extract_section(page.body, "Verdict"),
        extract_section(page.body, "Provisional thesis or hypothesis"),
        extract_section(page.body, "What would update this page"),
        extract_section(page.body, "Decisive uncertainty"),
    ]))[:2000]
    body = _substantive_body(page)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body)
                  if len(p.strip()) >= 60]
    ranked = sorted(paragraphs, key=lambda p: len(_terms(p) & source_terms), reverse=True)
    chosen = [p[:2000] for p in ranked if _terms(p) & source_terms][:2]
    return [core, *chosen] if core else chosen


def _passage_scores(source: Page, targets: list[Page], source_terms: set[str]) -> dict:
    """Best source-contribution ↔ target-section match for every target."""
    import numpy as np
    from .index import embeddings

    src = _source_passages(source)
    target_groups = {p.key: _target_passages(p, source_terms) for p in targets}
    texts = [*src, *(text for group in target_groups.values() for text in group)]
    vectors = embeddings.embed_texts(texts)
    if vectors is None or len(vectors) != len(texts):
        raise RuntimeError("impact passage embeddings unavailable")
    src_vectors = vectors[:len(src)]
    offset = len(src)
    out = {}
    for key, group in target_groups.items():
        if not group:
            continue
        similarity = src_vectors @ vectors[offset:offset + len(group)].T
        i, j = np.unravel_index(similarity.argmax(), similarity.shape)
        out[key] = (float(similarity[i, j]), src[int(i)][:240], group[int(j)][:240])
        offset += len(group)
    return out


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower()) if t not in _COMMON}


def _pairs(text: str) -> set[str]:
    terms = [t for t in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower())
             if t not in _COMMON]
    return {f"{a} {b}" for a, b in zip(terms, terms[1:]) if a != b}


def _eligible(page: Page) -> bool:
    if page.page_type not in TARGET_TYPES:
        return False
    status = page.str_field("status").lower()
    if page.page_type == "idea" and status in {"superseded", "abandoned"}:
        return False
    if page.page_type == "proposal" and status in {"rejected", "drafted", "published"}:
        return False
    return True


def _load_previous(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _unscanned_result(source_key: str, error: str, previous: dict) -> dict:
    """Keep earlier decisions available while making the failed scan visible."""
    return {
        "source": source_key,
        "source_fingerprint": previous.get("source_fingerprint", ""),
        "scanned_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "state": "unscanned",
        "error": error,
        "candidates": previous.get("candidates", []),
        "audit": previous.get("audit", []),
    }


def scan(source_key: str) -> dict:
    """Scan every eligible authored page and persist a durable review receipt.

    A missing index is ``unscanned``, never a clean zero-candidate result.
    Existing dispositions survive an identical rescan; substantive source or
    target changes reopen them. The scan never calls a remote model.
    """
    from .index import pages_semantic

    source = read_page(wiki_dir() / f"{source_key}.md")
    if source is None or source.page_type != "paper":
        raise ValueError(f"not a paper page: {source_key}")
    source_text = _source_text(source)
    source_hash = _fingerprint(source_text)
    path = review_path(source.stem)
    previous = _load_previous(path)
    source_changed = bool(previous.get("source_fingerprint") and
                          previous.get("source_fingerprint") != source_hash)
    old_rows = {r.get("target"): r for r in previous.get("candidates", [])
                if isinstance(r, dict) and r.get("target")}
    result = {
        "source": source_key,
        "source_fingerprint": source_hash,
        "scanned_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "state": "complete",
        "error": "",
        "candidates": [],
        "audit": [],
    }
    if not pages_semantic.index_exists():
        result = _unscanned_result(
            source_key, "semantic page index unavailable; run researchwiki reindex",
            previous)
    else:
        all_pages = read_pages()
        targets = [p for p in all_pages if _eligible(p) and p.key != source_key]
        hits = pages_semantic.query_text(
            pages_semantic.page_index_text(source),
            k=len(all_pages), page_types=tuple(TARGET_TYPES),
        )
        # An empty result with eligible pages is an index/embedding failure.
        if targets and not hits:
            result = _unscanned_result(
                source_key, "semantic page query returned no targets; check index health",
                previous)
        else:
            scores = {h.key: h.score for h in hits}
            source_terms = _terms(source.str_field("title") + " " + source.str_field("keywords"))
            source_pairs = _pairs(source.str_field("title") + " " + source.str_field("keywords"))
            passage_scores = _passage_scores(source, targets, source_terms)
            for target in targets:
                target_text = _target_text(target)
                target_hash = _fingerprint(target_text)
                shared = sorted(source_terms & _terms(target_text))
                shared_phrases = sorted(source_pairs & _pairs(target_text))
                page_score = scores.get(target.key, 0.0)
                passage_score, source_excerpt, target_excerpt = passage_scores.get(
                    target.key, (0.0, "", ""))
                score = round(max(page_score, passage_score), 4)
                cited = bool(re.search(r"\[\[(?:[^\]#|]*/)?" + re.escape(source.stem)
                                       + r"(?:#|\]|\|)", target.body))
                audit = {"target": target.key, "type": target.page_type,
                         "target_fingerprint": target_hash,
                         "semantic_score": score,
                         "page_score": round(page_score, 4),
                         "passage_score": round(passage_score, 4),
                         "source_excerpt": source_excerpt,
                         "target_excerpt": target_excerpt,
                         "shared_terms": shared[:8],
                         "shared_phrases": shared_phrases[:8],
                         "already_cited": cited}
                result["audit"].append(audit)
                old = old_rows.get(target.key, {})
                qualifies = (
                    score >= SEMANTIC_FLOOR
                    or (score >= CORROBORATED_FLOOR and bool(shared_phrases))
                    or (score >= SAME_CATEGORY_FLOOR
                        and source.category in target.list_field("category")
                        and len(shared) >= MIN_SHARED_TERMS)
                )
                if cited and not (old or source_changed):
                    continue
                if not (qualifies or old or (cited and source_changed)):
                    continue
                unchanged = (previous.get("source_fingerprint") == source_hash
                             and old.get("target_fingerprint") == target_hash)
                decision = old.get("decision") if unchanged else "pending"
                if decision not in DECISIONS:
                    decision = "pending"
                result["candidates"].append({
                    **audit, "target_fingerprint": target_hash,
                    "decision": decision,
                    "reason": old.get("reason", "") if unchanged else "",
                })
            result["audit"].sort(key=lambda r: (-r["semantic_score"], r["target"]))
            result["candidates"].sort(key=lambda r: (-r["semantic_score"], r["target"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    return result


def pending_reviews() -> list[tuple[Path, dict]]:
    """Return receipts with unfinished decisions or an incomplete scan."""
    root = ingest_dir() / REVIEW_DIR
    out = []
    for path in sorted(root.glob("*.yaml")) if root.is_dir() else []:
        data = _load_previous(path)
        if data.get("state") == "complete":
            source = read_page(wiki_dir() / f"{data.get('source', '')}.md")
            if source is None or _fingerprint(_source_text(source)) != data.get("source_fingerprint"):
                data["state"] = "stale"
            else:
                for row in data.get("audit", []):
                    if not isinstance(row, dict):
                        continue
                    target = read_page(wiki_dir() / f"{row.get('target', '')}.md")
                    if target is None or _fingerprint(_target_text(target)) != row.get("target_fingerprint"):
                        data["state"] = "stale"
                        break
        if data.get("state") != "complete" or any(
            not _resolved(r) for r in data.get("candidates", []) if isinstance(r, dict)
        ):
            out.append((path, data))
    return out


def _resolved(row: dict) -> bool:
    return (row.get("decision") in DECISIONS - {"pending"}
            and bool(str(row.get("reason") or "").strip()))


def unresolved_count(data: dict) -> int:
    return sum(not _resolved(row) for row in data.get("candidates", [])
               if isinstance(row, dict))


def record_unscanned(source_key: str, error: str) -> dict:
    """Persist a visible failure without changing the promoted paper outcome."""
    stem = source_key.split("/")[-1]
    path = review_path(stem)
    result = _unscanned_result(source_key, error, _load_previous(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    return result
