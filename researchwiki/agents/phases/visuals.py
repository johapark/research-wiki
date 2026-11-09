"""Select and inspect at most two source pages; keep transcripts out of PDF text.

Experimental (`visual-inspection` in `data/deprecations.yaml`): the `.visual.json`
receipt shape, its `visual_*` gate metrics and `POLICY_VERSION` are exempt from
the beta compatibility window while the review workflow settles. The
`visual_inspect` phase-role identifier is **not** exempt — persisted roles are
append-only permanently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time

import pypdfium2
from filelock import FileLock

from ...errors import EnvironmentFailure
from ...fsatomic import file_sha256, _write_text_atomic_unlocked
from ...pdf.figures import ARTWORK_FLOOR, page_artwork_fraction
from ...pdf.render import render_page
from .. import llm, model_config
from ..llm_images import DEADLINE_SECONDS, MAX_COMPLETION_TOKENS, ImageInput
from ..visual_evidence import candidate_quotes, parse_observations
from ..visual_selection import choose_requests, critical_claim_refs, missing_result_refs

POLICY_VERSION = 'visual-v3-review-only-unless-exact-prose'
SYSTEM = """Inspect only the supplied source pages and specified reference/questions.
Verify the requested label is on the image. Do not guess a plate's identity.
Report visible labels, units, conditions, qualifications and uncertainty. Do not
invent exact values from unlabeled plots, significance, statistics or causality.
Return JSON only: {"observations":[{"ref_id":"fig:2","page":2,"answered":true,
"observation":"...","visible_labels":"...","units_conditions":"...",
"uncertainty":"","source_quote":"..."}]}.
Use answered=false for absent, ambiguous or unreadable evidence. uncertainty is
an empty string when there is none. source_quote: copy one sentence verbatim from
prose_candidates[ref_id] when that sentence states the whole observation including
association/qualifications, and copy that same sentence into observation. A
paraphrased observation remains review-only. When no sentence fully supports the
observation, use an empty source_quote.
No self-certification: visual-only observations require source-image review.
"""


@dataclass
class VisualResult:
    status: str = 'not_needed'
    selection_status: str = 'complete'
    selected: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)
    unselected: list[str] = field(default_factory=list)
    critical_unresolved: list[str] = field(default_factory=list)
    source_sha256: str = ''
    cache_key: str = ''
    cache_hit: bool = False
    pixels: int = 0
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_known: bool = True
    estimated_cost_usd: float | None = None
    model: str = ''
    preparation_ms: int = 0
    request_ms: int = 0
    error: str | None = None
    results_source: str = ''    # how the Results section was found: heading | body | none

    def metrics(self) -> dict:
        return {'visual_selected': len(self.selected), 'visual_inspected': len(self.observations),
            'visual_critical_unresolved': len(self.critical_unresolved),
            'visual_unresolved': len(self.unresolved), 'visual_unselected': len(self.unselected),
            'visual_cache_hit': self.cache_hit, 'visual_pixels': self.pixels,
            'visual_requests': self.request_count, 'visual_usage_unknown': not self.usage_known,
            'visual_preparation_ms': self.preparation_ms, 'visual_request_ms': self.request_ms,
            'visual_status': self.status, 'visual_selection_status': self.selection_status,
            'visual_results_source': self.results_source,
            'visual_cost_usd': self.estimated_cost_usd}


def resolve_page(doc, entry) -> int | None:
    """Check only labeled candidates; never seek an unlabeled appended plate."""
    ref = entry.source
    candidates = [ref.page, *ref.also_on]
    if len(candidates) > 3:
        return None  # bounded resolution; repeated labels remain ambiguous
    artwork = []
    for number in candidates:
        page = doc[number - 1]
        try:
            if ref.kind == 'Table' or page_artwork_fraction(page) >= ARTWORK_FLOOR:
                artwork.append(number)
        finally:
            page.close()
    return artwork[0] if len(artwork) == 1 else None


def prepare_images(pdf_path: Path, entries: list, requests: list, result: VisualResult,
                   cache_dir: Path, deadline: float) -> list[ImageInput]:
    by_id = {entry.ref_id: entry for entry in entries}
    images, pages = [], {}
    doc = pypdfium2.PdfDocument(str(pdf_path))
    try:
        for request in requests:
            if time.monotonic() >= deadline:
                result.unresolved.append({'request': request.to_dict(), 'reason': 'deadline'})
                continue
            entry = by_id[request.ref_id]
            number = resolve_page(doc, entry)
            if number is None:
                result.unresolved.append({'request': request.to_dict(), 'reason': 'ambiguous_location'})
                continue
            if number not in pages:
                page = doc[number - 1]
                try:
                    width, height = (math.ceil(size * 110 / 72) for size in page.get_size())
                finally:
                    page.close()
                if (len(pages) >= 2 or result.pixels + width * height > 3_000_000
                        or max(width, height) / min(width, height) > 4):
                    result.unresolved.append({'request': request.to_dict(), 'reason': 'page_pixel_budget'})
                    continue
                dest = cache_dir / f'{result.source_sha256}-p{number}-110.png'
                with FileLock(str(dest) + ".lock", timeout=max(0, deadline - time.monotonic())):
                    rendered = render_page(pdf_path, number, dest, dpi=110)
                    image_data = rendered.path.read_bytes()
                if result.pixels + rendered.width * rendered.height > 3_000_000:
                    result.unresolved.append({'request': request.to_dict(), 'reason': 'pixel_budget'})
                    continue
                result.pixels += rendered.width * rendered.height
                pages[number] = {'page': number, 'width': rendered.width, 'height': rendered.height, 'dpi': 110}
                images.append(ImageInput(image_data, rendered.width,
                                         rendered.height, f'Source page {number}'))
            result.selected.append({'request': request.to_dict(), 'page': number,
                                    'caption': entry.source.caption, 'render': pages[number]})
    finally:
        doc.close()
    return images


def _finish_coverage(result: VisualResult) -> VisualResult:
    answered = {obs['ref_id'] for obs in result.observations
                if obs['answered'] and obs['corroborated']}
    for selected in result.selected:
        if selected['request']['ref_id'] not in answered:
            result.unresolved.append({'request': selected['request'], 'reason': 'source_image_review'})
    result.critical_unresolved = sorted({item['request']['ref_id'] for item in result.unresolved
                                        if item['request']['priority'] == 'critical'})
    if result.error:
        result.status = 'incomplete'
    elif result.critical_unresolved or result.selection_status != 'complete':
        result.status = 'needs_review'
    elif result.selected:
        result.status = 'inspected'
    return result


def inspect_visuals(*, pdf_path: Path, pages: list[str], sections: dict, inventory: list,
                    target_claims, use_stub: bool = False, cache_dir: Path | None = None,
                    expected_source_hash: str | None = None) -> VisualResult:
    started = time.monotonic()
    deadline = started + DEADLINE_SECONDS
    from ..budget import current_tracker
    tracker = current_tracker()
    if tracker:
        tracker.check_wall()
        if tracker.budget.max_wall_seconds is not None:
            deadline = min(deadline, tracker.started + tracker.budget.max_wall_seconds)
    result = VisualResult(selection_status=getattr(target_claims, 'visual_selection_status', 'incomplete'),
                          source_sha256=expected_source_hash or '')
    claims = getattr(target_claims, 'claims', [])
    requests = choose_requests(inventory, getattr(target_claims, 'visual_requests', []), claims)
    # Locating an ancillary Results visual cannot resolve a missing visual
    # named by a critical target, even when only the target's location names it.
    critical_refs = critical_claim_refs(claims)
    missing_refs = set(missing_result_refs(sections, inventory)) | (
        critical_refs - {entry.ref_id for entry in inventory})
    gap_priority = 'high' if any(entry.main_result for entry in inventory) else 'critical'
    result.unresolved.extend({'request': {'ref_id': ref,
                               'priority': 'critical' if ref in critical_refs else gap_priority},
                              'reason': 'missing_caption_location'}
                             for ref in sorted(missing_refs))
    result.unselected = [entry.ref_id for entry in inventory
                         if entry.ref_id not in {req.ref_id for req in requests}]
    if requests and result.selection_status != 'complete':
        # The publication gate already vetoes an incomplete selection, so a
        # paid image request could not change the outcome. Record the review.
        result.unresolved.extend({'request': req.to_dict(), 'reason': 'selection_incomplete'}
                                 for req in requests)
        requests = []
    if not requests:
        if result.unresolved and not result.source_sha256:
            result.source_sha256 = file_sha256(pdf_path)
        return _finish_coverage(result)
    cfg = model_config.for_phase('visual_inspect')
    result.model = cfg.model
    result.source_sha256 = file_sha256(pdf_path)
    if expected_source_hash is not None and result.source_sha256 != expected_source_hash:
        # Same condition and class as phase_extract's check: exit 2, not a bug.
        raise EnvironmentFailure('PDF source changed after text extraction; re-run the ingest')
    cache_dir = cache_dir or Path('.figures-cache') / 'inspection'
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        images = prepare_images(pdf_path, inventory, requests, result, cache_dir, deadline)
    except Exception as exc:
        result.error = f'Visual preparation failed: {type(exc).__name__}'
        result.unresolved.extend({'request': req.to_dict(), 'reason': 'preparation_failed'}
                                 for req in requests)
        return _finish_coverage(result)
    result.preparation_ms = int((time.monotonic() - started) * 1000)
    if not images:
        return _finish_coverage(result)
    # Keyed by reference, not page: two references on one page each keep
    # their own caption window.
    context, candidates = {}, {}
    for item in result.selected:
        ref_id = item['request']['ref_id']
        text = pages[item['page'] - 1]
        offset = text.find(item.get('caption', ''))
        start = max(0, offset - 1000) if offset >= 0 else 0
        context[ref_id] = text[start:start + 2400]
        candidates[ref_id] = candidate_quotes(sections, ref_id)
    prompt = json.dumps({'selected': result.selected, 'source_context': context,
                         'prose_candidates': candidates}, ensure_ascii=False)
    endpoint = '(stub)' if use_stub else llm.resolve_openai_endpoint().url
    identity = {'pdf': result.source_sha256, 'selected': result.selected, 'context': context,
                'prose_candidates': candidates,
                'model_config': asdict(cfg), 'completion_cap': min(MAX_COMPLETION_TOKENS, cfg.max_tokens),
                'endpoint_sha256': hashlib.sha256(endpoint.encode()).hexdigest(),
                'image_sha256': [hashlib.sha256(image.data).hexdigest() for image in images],
                'system': SYSTEM, 'policy': POLICY_VERSION, 'schema_version': 3, 'stub': use_stub}
    result.cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    receipt = cache_dir / f'{result.cache_key}.json'
    # The cache lock is inside its ignored namespace, with a bounded wait.
    request_state = {}
    try:
        with FileLock(str(receipt) + '.lock', timeout=max(0, deadline - time.monotonic())):
            if receipt.exists():
                try:
                    cached = json.loads(receipt.read_text())
                    if cached.get('identity') != identity:
                        raise ValueError('visual cache identity mismatch')
                    # Revalidate against current source; a cached transcript isn't evidence.
                    result.observations = parse_observations(json.dumps(cached['response']),
                                                              result.selected, sections)
                    result.model = cached['model']
                    result.cache_hit = True
                    result.estimated_cost_usd = 0.0
                except (ValueError, KeyError, TypeError):
                    receipt.unlink(missing_ok=True)
            if not result.cache_hit:
                t0 = time.monotonic()
                response = llm.call(phase='visual_inspect', prompt=prompt, system=SYSTEM,
                    images=images, max_tokens=min(MAX_COMPLETION_TOKENS, cfg.max_tokens), use_stub=use_stub,
                    deadline_seconds=deadline - t0, request_state=request_state)
                result.request_count = request_state.get("outbound_requests", 0)
                result.request_ms = int((time.monotonic() - t0) * 1000)
                result.input_tokens, result.output_tokens = response.input_tokens, response.output_tokens
                result.usage_known = getattr(response, 'usage_known', True)
                result.model = response.model
                if result.usage_known and model_config.rate_for(result.model):
                    result.estimated_cost_usd = model_config.estimate_usd(result.model,
                        response.input_tokens, response.output_tokens)
                result.observations = parse_observations(response.text, result.selected, sections)
                if time.monotonic() > deadline:
                    raise TimeoutError('visual inspection deadline expired during parsing')
                if (len(result.observations) == len({item['request']['ref_id'] for item in result.selected})
                        and all(obs['answered'] for obs in result.observations)):
                    # The cache namespace lock already serializes this target;
                    # avoid an unbounded second lock inside the generic writer.
                    cached_record = {'identity': identity, 'model': result.model,
                        'response': {'observations': result.observations},
                        'usage': {'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens,
                                  'known': result.usage_known}, 'policy': POLICY_VERSION}
                    _write_text_atomic_unlocked(receipt, json.dumps(cached_record, indent=2))
    except EnvironmentFailure:
        # House rule 1 (errors.py): a provider/environment fault propagates as
        # exit 2 so batch --resume retries it, instead of becoming a review veto.
        raise
    except Exception as exc:
        result.request_count = request_state.get("outbound_requests", 0)
        result.request_ms = int((time.monotonic() - t0) * 1000) if "t0" in locals() else 0
        result.error = f'{type(exc).__name__}: {exc}'[:240]
        if "response" not in locals():
            result.usage_known = False
    return _finish_coverage(result)
