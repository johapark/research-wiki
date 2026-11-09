"""Publication and accounting regressions from the PR #28 follow-up review."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from researchwiki.agents import llm, llm_images, model_config
from researchwiki.agents.budget import BudgetExhausted, BudgetTracker, IngestBudget, activate
from researchwiki.agents.llm_images import ImageInput
from researchwiki.agents.phases import visuals
from researchwiki.agents.phases.target_claims import TargetClaim, TargetClaimsOutput
from researchwiki.agents.promote import GateResult
from researchwiki.agents.visual_evidence import apply_evidence, parse_observations
from researchwiki.agents.visual_selection import VisualEntry, VisualRequest
from researchwiki.agents.visual_support import final_publication_gate
from researchwiki.pdf.figures import FigureRef


QUOTE = 'The measured accuracy was 90% for condition A and 80% for condition B (Fig. 2).'
REQUEST = VisualRequest('fig:2', 'critical', 'visual_evidence', 'Which condition performed better?')


def selected():
    return [{'request': REQUEST.to_dict(), 'page': 1, 'caption': 'Figure 2. Accuracy'}]


def observation(text=QUOTE):
    return dict(ref_id='fig:2', page=1, answered=True, observation=text,
                visible_labels='condition A; condition B', units_conditions='accuracy (%)',
                uncertainty='none', source_quote=QUOTE)


def publication_gate(result):
    ctx = SimpleNamespace(visual_inspection=True, visual_result=result,
                          winner=SimpleNamespace(model='gpt-6.1-sol', scores={}))
    return final_publication_gate(ctx, GateResult(True))


@pytest.mark.parametrize('text', [
    'Condition A reached 80% accuracy and condition B reached 90%.',
    'Condition A caused a statistically significant improvement over condition B.',
    'Condition A had 90% mortality and condition B had 80% mortality.',
    'Condition A reached 90% accuracy, above condition B at 80%.',
])
def test_unverified_paraphrases_cannot_clear_critical_visual_gate(text):
    parsed = parse_observations(json.dumps({'observations': [observation(text)]}),
                                selected(), {'results': QUOTE})
    result = visuals._finish_coverage(visuals.VisualResult(selected=selected(), observations=parsed))
    targets = TargetClaimsOutput()
    apply_evidence(targets, result)
    assert not parsed[0]['corroborated']
    assert result.critical_unresolved == ['fig:2']
    assert not publication_gate(result).promoted
    assert not targets.claims and not targets.visual_evidence


def test_verbatim_observation_allows_only_whitespace_normalization():
    parsed = parse_observations(json.dumps({'observations': [observation(QUOTE.replace(' ', '  '))]}),
                                selected(), {'results': QUOTE})
    assert parsed[0]['corroborated']


@pytest.fixture
def inspection(tmp_path, monkeypatch):
    monkeypatch.setenv('RESEARCHWIKI_LOCK_DIR', str(tmp_path / 'locks'))
    cfg = model_config.ModelConfig('openai-compatible', 'requested-vision', 0, 1200,
                                   image_token_bound=32768)
    monkeypatch.setattr(model_config, 'for_phase', lambda *a: cfg)
    monkeypatch.setattr(llm, 'resolve_openai_endpoint',
                        lambda: SimpleNamespace(url='https://model.test/v1'))
    def prepare(pdf, entries, requests, result, cache, deadline):
        result.selected = selected()
        return [ImageInput(b'png', 900, 1200, 'Page 1')]
    monkeypatch.setattr(visuals, 'prepare_images', prepare)
    pdf = tmp_path / 'source.pdf'
    pdf.write_bytes(b'source')
    entry = VisualEntry('fig:2', FigureRef('Figure', 2, 1, False, 'Figure 2. Accuracy'), True)
    def inspect(targets, results=QUOTE):
        return visuals.inspect_visuals(pdf_path=pdf, pages=[QUOTE], sections={'results': results},
            inventory=[entry], target_claims=targets, cache_dir=tmp_path / 'cache')
    return inspect, cfg


@pytest.mark.parametrize('field', ['location', 'content'])
@pytest.mark.parametrize('mentioned_in_results', [True, False])
def test_missing_critical_claim_reference_stays_blocking(inspection, monkeypatch,
                                                        field, mentioned_in_results):
    inspect, cfg = inspection
    monkeypatch.setattr(llm, 'call', lambda **kw: llm.LLMResponse(
        json.dumps({'observations': [observation()]}), cfg.model, 0, 100, 10))
    claim = TargetClaim('headline', 'Survival doubled after treatment.', 'critical', 'Results')
    setattr(claim, field, getattr(claim, field) + ' (Figure 4)')
    targets = TargetClaimsOutput(visual_selection_status='complete', claims=[claim])
    results = QUOTE + (' The headline survival result is shown in Figure 4.' if mentioned_in_results else '')
    result = inspect(targets, results)
    assert result.observations[0]['corroborated']
    assert result.critical_unresolved == ['fig:4']
    assert not publication_gate(result).promoted


@pytest.mark.parametrize('label', ['Extended Data Fig.', 'ED Fig.'])
def test_located_extended_data_reference_does_not_create_missing_main_gap(tmp_path, monkeypatch, label):
    monkeypatch.setattr(llm, 'call', lambda **kw: pytest.fail('incomplete selection sends no request'))
    pdf = tmp_path / 'source.pdf'
    pdf.write_bytes(b'source')
    inventory = [VisualEntry('edfig:4', FigureRef('Figure', 4, 1, True,
                             'Extended Data Figure 4. Survival'), False)]
    targets = TargetClaimsOutput(visual_selection_status='incomplete', claims=[
        TargetClaim('headline', 'Survival doubled.', 'critical', f'Results ({label} 4)')])
    result = visuals.inspect_visuals(pdf_path=pdf, pages=['Extended Data Figure 4. Survival'],
        sections={}, inventory=inventory, target_claims=targets, cache_dir=tmp_path / 'cache')
    assert not [gap for gap in result.unresolved if gap['reason'] == 'missing_caption_location']


def mock_rates(monkeypatch, served_priced):
    rates = {'requested-vision': model_config.Rate('requested-vision', 1, 1)}
    if served_priced:
        rates['served-vision'] = model_config.Rate('served-vision', 2, 2)
    monkeypatch.setattr(model_config, 'rate_for', lambda name, **kw: rates.get(name))


@pytest.mark.parametrize('served_priced', [True, False])
@pytest.mark.parametrize('local', [True, False])
def test_image_settlement_uses_served_price_and_preserves_unknown_cost(monkeypatch, served_priced, local):
    mock_rates(monkeypatch, served_priced)
    cfg = model_config.ModelConfig('openai-compatible', 'requested-vision', 0, 1200,
                                   image_token_bound=32768)
    monkeypatch.setattr(model_config, 'for_phase', lambda *a: cfg)
    endpoint = 'http://localhost:1234/v1' if local else 'https://model.test/v1'
    monkeypatch.setattr(llm, 'resolve_openai_endpoint', lambda: SimpleNamespace(url=endpoint))
    def post(payload, endpoint, deadline, state):
        state['outbound_requests'] = 1
        return {'model': 'served-vision', 'choices': [{'message': {'content': '{"observations":[]}'}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}
    monkeypatch.setattr(llm_images, '_post_once', post)
    tracker = BudgetTracker(IngestBudget(max_cost_usd=0.05))
    reserved_cost = model_config.estimate_usd(cfg.model, 32769, 1200)
    with activate(tracker):
        response = llm.call(phase='visual_inspect', prompt='context',
                            images=[ImageInput(b'png', 900, 1200, 'Page 1')])
    assert response.model == 'served-vision'
    snapshot = tracker.snapshot()
    assert snapshot['tokens'] == 110 and snapshot['reserved_tokens'] == 0
    assert snapshot['reserved_cost_usd'] == pytest.approx(0)
    if local:
        assert snapshot['estimated_cost_usd'] == 0
    elif served_priced:
        assert snapshot['estimated_cost_usd'] == pytest.approx(0.00022)
    else:
        assert snapshot['estimated_cost_usd'] is None
        assert tracker.cost_usd == pytest.approx(reserved_cost)
        with pytest.raises(BudgetExhausted) as exc:
            tracker.reserve_call(model=cfg.model, provider=cfg.provider, prompt_chars=7,
                                 max_tokens=1200, image_tokens=32768, free=False)
        assert exc.value.dimension == 'estimated_cost_usd'


@pytest.mark.parametrize('served_priced', [True, False])
def test_inspection_receipts_and_cache_keep_served_model(inspection, monkeypatch, served_priced):
    inspect, _ = inspection
    mock_rates(monkeypatch, served_priced)
    calls = []
    def call(**kw):
        calls.append(kw)
        kw['request_state']['outbound_requests'] = 1
        return llm.LLMResponse(json.dumps({'observations': [observation()]}), 'served-vision', 0, 100, 10)
    monkeypatch.setattr(llm, 'call', call)
    targets = TargetClaimsOutput(visual_selection_status='complete')
    cold, warm = inspect(targets), inspect(targets)
    assert cold.model == warm.model == 'served-vision'
    if served_priced:
        assert cold.estimated_cost_usd == pytest.approx(0.00022)
    else:
        assert cold.estimated_cost_usd is None
    assert warm.cache_hit and warm.estimated_cost_usd == 0 and len(calls) == 1


def test_sentence_ending_on_ed_abbreviation_is_not_extended_data():
    """"ED" as emergency department ending a sentence must stay main Table 2."""
    from researchwiki.agents.visual_selection import mentioned_refs, missing_result_refs, build_inventory
    prose = 'Most patients were discharged from the ED. Table 2 lists the admission outcomes.'
    assert mentioned_refs(prose) == {'table:2'}
    inventory = build_inventory(['Table 2. Admission outcomes'], {'results': prose})
    assert inventory[0].main_result
    assert missing_result_refs({'results': prose}, inventory) == []


def test_chat_relay_calls_settle_free_under_a_cost_budget(monkeypatch):
    """A relay label is never priced; that must not exhaust a dollar budget."""
    cfg = model_config.ModelConfig('chat-relay', 'gpt-5.6-luna', 0, 8000)
    monkeypatch.setattr(model_config, 'for_phase', lambda *a: cfg)
    monkeypatch.setattr(llm, 'call_chat_relay',
                        lambda **kw: llm.LLMResponse('ok', 'chat-relay:claude-code', 0, 10000, 2000))
    tracker = BudgetTracker(IngestBudget(max_cost_usd=0.05))
    with activate(tracker):
        for _ in range(20):
            llm.call(phase='target_claims', prompt='x' * 40000)
    snapshot = tracker.snapshot()
    assert snapshot['estimated_cost_usd'] == 0 and snapshot['reserved_cost_usd'] == pytest.approx(0)


@pytest.mark.parametrize('page, drift', [
    ('Survival was 96.5% (95% CI 94.1–98.0).', []),   # extractor's point form
    ('Survival was 96·5% (95% CI 94·1–98·0).', []),   # copied as printed
    ('Survival was 97.5%.', ['97.5']),                 # a wrong value still drifts
])
def test_middle_dot_decimals_in_the_pdf_match_point_decimals(page, drift):
    """Target claims rewrite `96·5%` as `96.5%`; that must not drift."""
    from researchwiki.grade.primitives import check_numerics
    pdf = 'Overall survival at 12 months was 96·5% (95% CI 94·1–98·0); 10 withdrew.'
    assert check_numerics(page, pdf, pdf)[1] == drift


def test_dot_operator_multiplication_is_not_read_as_a_decimal():
    from researchwiki.grade.primitives import check_numerics
    pdf = 'Cells were seeded at 2⋅10 per well.'
    assert check_numerics('Seeded at 2.10 per well.', pdf, pdf)[1] == ['2.10']
