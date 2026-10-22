"""Compose a full structured synthesis and require page gates plus model support."""
from dataclasses import replace
from datetime import datetime, timezone
import json

from ..agents import llm
from ..answers.sources import snapshot_scope
from ..grade.generated import preflight, strict_json
from .document import authored_draft
from .evidence import gather, sources_for
from .gates import page_gates
from .prompts import AUTHOR_SCHEMA, AUTHOR_SYSTEM, JUDGE_SCHEMA, JUDGE_SYSTEM
from .receipt import ModelRun, SynthesisReceipt
from .render import render
from .verification import numeric_checks, judge_items, apply_judgment


def compose(title, question, scope):
    if len(scope.stems) > 8:
        raise ValueError('composition supports at most eight selected papers; narrow the selection explicitly')
    scope = snapshot_scope(scope)
    result = SynthesisReceipt(title, question, scope, sources_for(scope), (), None, (), None, None,
                              'insufficient_evidence', 'At least two selected papers are required.',
                              datetime.now(timezone.utc).isoformat())
    if len(scope.stems) < 2:
        return result
    result = replace(result, evidence=gather(question, scope))
    snapshot_scope(scope)
    if len({e.stem for e in result.evidence}) < 2:
        return replace(result, reason='The bounded retrieval did not find evidence from two papers.')
    preflight(('synthesis_author', 'synthesis_support'))
    response = llm.call(phase='synthesis_author', system=AUTHOR_SYSTEM, schema=AUTHOR_SCHEMA,
                        prompt=json.dumps({'title': title, 'question': question,
                                           'sources': [s.to_dict() for s in result.sources],
                                           'evidence': [e.to_dict() for e in result.evidence]}))
    try:
        model = ModelRun(phase='synthesis_author', **llm.response_usage(response))
        result = replace(result, models=(model,))
        data = strict_json(response.text)
        if not isinstance(data, dict) or set(data) != {'draft'}:
            raise ValueError('author must return a draft object or null')
        if data['draft'] is None:
            snapshot_scope(scope)
            return replace(result, reason='The source evidence does not support a complete synthesis.')
        draft = authored_draft(data['draft'])
        draft.validate_sources(result.evidence, scope)
        draft = numeric_checks(draft, result.evidence)
        result = replace(result, draft=draft, status='rejected', reason='Synthesis checks have not passed.')
    except (ValueError, TypeError) as exc:
        snapshot_scope(scope)
        return replace(result, status='rejected', reason=f'Invalid synthesis author response: {exc}')
    if any(unit.numeric_drift for unit in draft.units()):
        snapshot_scope(scope)
        return replace(result, reason='A synthesis quantity is absent from its cited evidence.')
    snapshot_scope(scope)
    gates = page_gates(render(result), question, scope)
    result = replace(result, gates=gates)
    if not gates.passed:
        snapshot_scope(scope)
        return replace(result, reason='Synthesis page structure/grounding/fidelity gates failed or graded zero claims.')
    items, positions = judge_items(draft, result.evidence)
    response = llm.call(phase='synthesis_support', system=JUDGE_SYSTEM, schema=JUDGE_SCHEMA,
                        prompt=json.dumps({'question': question, 'draft': draft.to_dict(),
                                           'evidence': [e.to_dict() for e in result.evidence], 'items': items}))
    try:
        model = ModelRun(phase='synthesis_support', **llm.response_usage(response))
        result = replace(result, models=(*result.models, model))
        checked, review = apply_judgment(response.text, draft, positions)
        result = replace(result, draft=checked, review=review)
    except (ValueError, TypeError) as exc:
        snapshot_scope(scope)
        return replace(result, reason=f'Invalid synthesis judge response: {exc}')
    snapshot_scope(scope)
    if result.sources != sources_for(scope):
        raise ValueError('synthesis citation paths changed while composing; regenerate the document')
    if not review.passed or any(not unit.passed for unit in checked.units()):
        return replace(result, reason='Synthesis support or editorial/coverage review failed.')
    return replace(result, status='verified', reason='All prose, theme, coverage, grounding, and fidelity checks passed.')
