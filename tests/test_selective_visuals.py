"""Hermetic source identity, resource limits and visual-grounding boundaries."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from researchwiki.agents import llm, model_config
from researchwiki.agents.budget import BudgetExhausted, BudgetTracker, IngestBudget, activate
from researchwiki.agents.llm_images import ImageInput, _post_once as REAL_POST_ONCE
from researchwiki.agents.phases import extract
from researchwiki.agents.phases.target_claims import TargetClaimsOutput, extract_target_claims
from researchwiki.agents.phases import visuals
from researchwiki.agents.promote import GateResult
from researchwiki.agents.visual_evidence import parse_observations, apply_evidence
from researchwiki.agents.visual_selection import (
    VisualEntry, VisualRequest, build_inventory, choose_requests, inventory_prompt, parse_requests,
)
from researchwiki.agents.visual_support import final_publication_gate
from researchwiki.pdf.figures import FigureRef



def entry(ref_id='fig:2', page=1, main=True, repeats=()):
    return VisualEntry(ref_id, FigureRef('Table' if 'table' in ref_id else 'Figure',
        int(ref_id.split(':')[1]), page, False, 'Figure 2. Measured result', repeats), main)


def request(ref_id='fig:2', priority='critical', reason='visual_evidence', question='What do the labels show?'):
    return VisualRequest(ref_id, priority, reason, question)


def selected(req=None):
    return [{'request': (req or request()).to_dict(), 'page': 1}]


@pytest.fixture(autouse=True)
def isolate_locks(tmp_path, monkeypatch):
    monkeypatch.setenv('RESEARCHWIKI_LOCK_DIR', str(tmp_path/'locks'))


QUOTE = 'The measured accuracy was 90% for condition A and 80% for condition B (Fig. 2).'


def observation(**changes):
    result = dict(ref_id='fig:2', page=1, answered=True, observation=QUOTE,
                  visible_labels='condition A; condition B', units_conditions='accuracy (%)',
                  uncertainty='', source_quote=QUOTE)
    result.update(changes)
    return result


def test_rich_extraction_is_one_pass_and_compatible(monkeypatch):
    calls = []
    monkeypatch.setattr(extract, 'extract_pdf_page_texts', lambda *a, **k: calls.append(1) or
                        ['Abstract\nSome text.', 'Results\nMore text.'])
    sections, text, pages = extract.extract_source_pages(Path('paper.pdf'))
    assert calls == [1] and text == '\n\n'.join(pages)
    assert extract.extract_sections(Path('paper.pdf')) == (sections, text)


def test_inventory_retains_repeated_labels_and_prioritizes_results():
    pages = ['Figure 1. Method', 'Figure 2: Results', 'Figure 2. Results repeated']
    entries = build_inventory(pages, {'results': 'Accuracy improved (Fig. 2).\nFigure 1. Method'})
    assert entries[0].ref_id == 'fig:2' and entries[0].source.also_on == (3,)
    assert entries[0].main_result and not entries[1].main_result
    inventory = inventory_prompt(entries * 80)
    assert len(inventory) <= 2000 and 'Omitted entries:' in inventory


@pytest.mark.parametrize('invalid', [None, {}, '[]', [None], [dict(ref_id='fig:99',
    priority='critical', reason='visual_evidence', question='x')], [dict(ref_id='fig:2',
    priority='urgent', reason='visual_evidence', question='x')], [dict(ref_id='fig:2',
    priority='high', reason='unknown', question='x')], [dict(ref_id='fig:2',
    priority='high', reason='visual_evidence', question='x'*161)]])
def test_bad_nominations_are_incomplete(invalid):
    assert parse_requests({'visual_requests': invalid}, [entry()]) == ([], 'incomplete')


def test_duplicate_nominations_do_not_displace_distinct_questions():
    raw = [request(question='First?').to_dict()] * 8 + [request(question='Second?').to_dict()]
    parsed, status = parse_requests({'visual_requests': raw}, [entry()])
    assert len(parsed) == 2 and status == 'complete'
    grouped = choose_requests([entry()], parsed)
    assert len(grouped) == 1 and 'First?' in grouped[0].question and 'Second?' in grouped[0].question


def test_main_fallback_does_not_assume_figure_one():
    entries = [entry('fig:1', main=False), entry('fig:3', main=True)]
    fallback = choose_requests(entries, [])
    assert fallback[0].ref_id == 'fig:3' and fallback[0].priority == 'critical'
    nominated = choose_requests(entries, [request('fig:3', priority='normal')])
    assert nominated[0].priority == 'critical'


@pytest.mark.parametrize('optional', [None, 'malformed', [{'ref_id': 7}]])
def test_bad_optional_nominations_preserve_valid_claims(monkeypatch, optional):
    data = {'claims': [{'type':'headline', 'content':QUOTE, 'importance':'critical', 'location':'Results'}],
            'keywords':['alpha beta', 'condition A', 'condition B', 'test accuracy', 'known labels'],
            'visual_requests':optional}
    captured = []
    monkeypatch.setattr(llm, 'call', lambda **kw: captured.append(kw) or
                        llm.LLMResponse(json.dumps(data), 'model-1', 0, 10, 10))
    out = extract_target_claims(metadata={}, sections={'results':QUOTE}, visual_inventory=[entry()])
    assert out.claims[0].content == QUOTE and out.visual_selection_status == 'incomplete'
    assert 'visual_requests' in captured[0]['schema']['properties']
    off = extract_target_claims(metadata={}, sections={'results':QUOTE})
    assert off.claims == out.claims and off.visual_selection_status == 'off'
    assert 'visual_requests' not in captured[1]['schema']['properties']
    assert 'Visual inventory' not in captured[1]['prompt']


def test_truncated_json_recovers_claims_without_optional_objects(monkeypatch):
    raw = '{"claims":[{"type":"headline","content":"Real fact","importance":"high"}],"keywords":[],"visual_requests":[{"type":"headline","content":"Fake nomination","importance":"critical"},'
    monkeypatch.setattr(llm, 'call', lambda **kw: llm.LLMResponse(raw, 'model-1', 0, 10, 10))
    out = extract_target_claims(metadata={}, sections={}, visual_inventory=[entry()])
    assert [claim.content for claim in out.claims] == ['Real fact']
    assert out.visual_selection_status == 'incomplete'


@pytest.mark.parametrize('change', [dict(source_quote='90%'),
    dict(observation='Accuracy was 95% for condition A.'), dict(observation=''),
    dict(uncertainty='ambiguous'), dict(answered=False), dict(visible_labels='')])
def test_visual_transcripts_cannot_certify_numeric_or_uncertain_claims(change):
    parsed = parse_observations(json.dumps({'observations':[observation(**change)]}),
                                selected(), {'results':QUOTE})
    assert not parsed[0]['corroborated']
    out = TargetClaimsOutput()
    apply_evidence(out, SimpleNamespace(observations=parsed))
    assert not out.claims and not out.visual_evidence


def test_full_prose_association_can_enrich_fixed_target_packet():
    parsed = parse_observations(json.dumps({'observations':[observation()]}), selected(), {'results':QUOTE})
    out = TargetClaimsOutput()
    apply_evidence(out, SimpleNamespace(observations=parsed))
    assert parsed[0]['corroborated'] and out.claims[0].content == QUOTE
    assert out.visual_evidence == [QUOTE]
    apply_evidence(out, SimpleNamespace(observations=parsed))
    assert len(out.claims) == 1


def test_table_association_requires_review_even_if_number_is_in_prose():
    parsed = parse_observations(json.dumps({'observations':[observation()]}),
        selected(request(reason='table_alignment')), {'results':QUOTE})
    assert not parsed[0]['corroborated']


@pytest.mark.parametrize('change', [dict(page=2), dict(ref_id='fig:99'), dict(answered='true'),
                                  dict(units_conditions=[]), dict(observation='x'*1201)])
def test_bad_visual_responses_fail_closed(change):
    with pytest.raises(ValueError):
        parse_observations(json.dumps({'observations':[observation(**change)]}), selected(), {'results':QUOTE})


def test_visual_only_critical_result_veto_survives_forced_promotion():
    result = visuals.VisualResult(status='needs_review', critical_unresolved=['fig:2'])
    ctx = SimpleNamespace(visual_inspection=True, visual_result=result,
                          winner=SimpleNamespace(model='gpt-6.1-sol', scores={}), promote_mode='always')
    gate = final_publication_gate(ctx, GateResult(True))
    assert not gate.promoted and any('critical visual' in reason for reason in gate.reasons)
    assert ctx.winner.scores['visual_coverage']['visual_critical_unresolved'] == 1
    ctx.visual_inspection = False
    assert final_publication_gate(ctx, GateResult(True)).promoted


def test_missing_results_label_is_a_critical_gap_without_a_call(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, 'call', lambda **kw: pytest.fail('no label: no image call'))
    pdf=tmp_path/'paper.pdf';pdf.write_bytes(b'source')
    result = visuals.inspect_visuals(pdf_path=pdf, pages=['Results (Figure 4).'],
        sections={'results':'Results (Figure 4).'}, inventory=[],
        target_claims=TargetClaimsOutput(visual_selection_status='complete'), cache_dir=tmp_path)
    assert result.critical_unresolved == ['fig:4'] and result.request_count == 0
    assert result.source_sha256


class FakePage:
    def __init__(self, artwork=True, width=612, height=792):
        self.artwork, self.width, self.height = artwork, width, height
    def get_size(self):
        return self.width, self.height
    def get_objects(self, **kw):
        if self.artwork:
            yield SimpleNamespace(type=visuals.pypdfium2.raw.FPDF_PAGEOBJ_IMAGE,
                                  get_bounds=lambda:(0,0,400,500))
    def close(self):
        pass


class FakeDoc:
    def __init__(self, pages):
        self.pages = pages
    def __getitem__(self, index):
        return self.pages[index]
    def close(self):
        pass


def test_separated_plate_needs_repeated_label():
    doc = FakeDoc([FakePage(False), FakePage(True), FakePage(True)])
    assert visuals.resolve_page(doc, entry(repeats=(2,))) == 2
    assert visuals.resolve_page(doc, entry()) is None
    assert visuals.resolve_page(doc, entry(repeats=(2,3))) is None


def test_page_grouping_and_pixel_caps(tmp_path, monkeypatch):
    doc = FakeDoc([FakePage() for _ in range(3)])
    monkeypatch.setattr(visuals.pypdfium2, 'PdfDocument', lambda *a:doc)
    renders = []
    def render(path, number, dest, dpi):
        renders.append(number)
        dest.write_bytes(b'png')
        return SimpleNamespace(path=dest, width=935, height=1210)
    monkeypatch.setattr(visuals, 'render_page', render)
    entries = [entry('fig:1',1), entry('fig:2',1), entry('fig:3',2), entry('fig:4',3)]
    result = visuals.VisualResult(source_sha256='source')
    images = visuals.prepare_images(Path('paper.pdf'), entries, [request(e.ref_id) for e in entries],
                                   result, tmp_path, float('inf'))
    assert renders == [1,2] and len(images) == 2 and len(result.selected) == 3
    assert result.pixels <= 3_000_000 and result.unresolved[0]['reason'] == 'page_pixel_budget'
    doc.pages = [FakePage(width=2000,height=2000)]
    result = visuals.VisualResult(source_sha256='large')
    assert not visuals.prepare_images(Path('paper.pdf'), [entry()], [request()], result, tmp_path, float('inf'))


@pytest.fixture
def image_call(monkeypatch):
    from researchwiki.agents import llm_images
    cfg = model_config.ModelConfig('openai-compatible', 'gpt-6.1-sol', 0, 1200, image_token_bound=32768)
    monkeypatch.setattr(model_config, 'for_phase', lambda *a:cfg)
    monkeypatch.setattr(llm, 'resolve_openai_endpoint', lambda:SimpleNamespace(url='http://localhost:1234/v1'))
    sent = []
    def post(payload, endpoint, deadline, state):
        state['outbound_requests'] = 1
        sent.append(payload)
        return {'choices':[{'message':{'content':'{"observations":[]}'}}],
                'usage':{'prompt_tokens':101,'completion_tokens':9}}
    monkeypatch.setattr(llm_images, '_post_once', post)
    return llm_images, sent


def test_images_reach_common_dispatch_and_shared_budget(image_call):
    _, sent = image_call
    tracker = BudgetTracker(IngestBudget(max_tokens=34000,max_model_calls=1))
    state = {}
    with activate(tracker):
        result = llm.call(phase='visual_inspect', prompt='source context',
            images=[ImageInput(b'png',935,1210,'Page 1')], request_state=state)
    content = sent[0]['messages'][-1]['content']
    assert content[0]['type'] == 'text' and content[-1]['type'] == 'image_url'
    assert content[-1]['image_url']['url'].startswith('data:image/png;base64,')
    assert state['outbound_requests'] == 1 and result.usage_known
    assert tracker.calls == 1 and tracker.tokens == 110


def test_image_budget_declines_before_outbound_request(image_call):
    _, sent = image_call
    tracker = BudgetTracker(IngestBudget(max_tokens=100))
    with activate(tracker), pytest.raises(BudgetExhausted):
        llm.call(phase='visual_inspect', prompt='context', images=[ImageInput(b'png',935,1210,'Page 1')])
    assert not sent


def test_unknown_image_model_requires_bound_for_strict_budget(image_call, monkeypatch):
    _, sent = image_call
    cfg = model_config.ModelConfig('openai-compatible','unknown-vision',0,1200)
    monkeypatch.setattr(model_config,'for_phase',lambda *a:cfg)
    with activate(BudgetTracker(IngestBudget(max_tokens=100000))), pytest.raises(ValueError,match='bound'):
        llm.call(phase='visual_inspect',prompt='context',images=[ImageInput(b'png',900,1200,'Page 1')])
    assert not sent
    monkeypatch.setattr(model_config,'for_phase',lambda *a:replace(cfg,image_token_bound=8000))
    with activate(BudgetTracker(IngestBudget(max_tokens=100000))):
        llm.call(phase='visual_inspect',prompt='context',images=[ImageInput(b'png',900,1200,'Page 1')])
    assert len(sent) == 1


def test_unsupported_provider_does_not_drop_attachments(image_call, monkeypatch):
    _, sent = image_call
    monkeypatch.setattr(model_config,'for_phase',lambda *a:model_config.ModelConfig('chat-relay','model-1',0,1200))
    with pytest.raises(llm.ProviderUnavailable,match='image transport'):
        llm.call(phase='visual_inspect',prompt='context',images=[ImageInput(b'png',900,1200,'Page 1')])
    assert not sent


def test_missing_usage_retains_reservation(image_call, monkeypatch):
    module, _ = image_call
    def post(*args):
        return {'choices':[{'message':{'content':'{"observations":[]}'}}]}
    monkeypatch.setattr(module,'_post_once',post)
    tracker = BudgetTracker(IngestBudget(max_tokens=100000))
    with activate(tracker):
        response = llm.call(phase='visual_inspect',prompt='context',images=[ImageInput(b'png',900,1200,'Page 1')])
    assert not response.usage_known and tracker.tokens > 32768 and not tracker.reserved_tokens


@pytest.mark.parametrize('http_code', [400,401,429,500])
def test_image_http_errors_including_effort_rejection_never_retry(image_call, monkeypatch, http_code):
    module, _ = image_call
    monkeypatch.setattr(module,'_post_once',REAL_POST_ONCE)
    requests = []
    def send(req, timeout):
        requests.append(req)
        assert 0 < timeout <= 30
        raise urllib.error.HTTPError(req.full_url,http_code,'reasoning_effort rejected',{},None)
    monkeypatch.setattr(module,'_open_once',send)
    with pytest.raises(llm.ProviderUnavailable):
        llm.call(model='gpt-6.1-sol',provider='openai-compatible',prompt='context',reasoning_effort='minimal',
                 images=[ImageInput(b'png',900,1200,'Page 1')])
    assert len(requests) == 1


def test_throttle_deadline_prevents_request(image_call, monkeypatch):
    import time
    module, sent = image_call
    cfg = model_config.ModelConfig('openai-compatible','gpt-6.1-sol',0,1200,rpm=1)
    monkeypatch.setattr(model_config,'for_phase',lambda *a:cfg)
    monkeypatch.setitem(llm._last_call_at,cfg.model,time.monotonic())
    with pytest.raises(TimeoutError,match='throttle'):
        llm.call(phase='visual_inspect',prompt='context',deadline_seconds=.02,
                 images=[ImageInput(b'png',900,1200,'Page 1')])
    assert not sent


def test_text_transport_receives_no_new_keywords(image_call, monkeypatch):
    _, sent = image_call
    captured=[]
    monkeypatch.setattr(llm,'call_openai_compatible',lambda **kw:captured.append(kw) or
                        llm.LLMResponse('text','gpt-6.1-sol',0,1,1))
    llm.call(phase='target_claims',prompt='source text')
    assert captured and not sent
    assert not {'images','deadline_seconds','request_state'} & captured[0].keys()


def test_off_cli_option_and_batch_forwarding():
    from researchwiki.tasks import agent
    from researchwiki.tasks._ingest_observation import request_from_args
    parser = agent.build_parser() if hasattr(agent,'build_parser') else agent._build_parser()
    off = parser.parse_args(['ingest','paper.pdf','--stub'])
    on = parser.parse_args(['ingest','paper.pdf','--stub','--visual-inspection'])
    assert not request_from_args(off).options.visual_inspection
    assert request_from_args(on).options.visual_inspection
    assert '--visual-inspection' not in agent._batch_passthrough_args(off)
    assert '--visual-inspection' in agent._batch_passthrough_args(on)


def test_cache_reuses_success_and_invalidates_questions_source_and_model(tmp_path, monkeypatch):
    pdf = tmp_path/'paper.pdf'
    pdf.write_bytes(b'source1')
    monkeypatch.setattr(llm,'resolve_openai_endpoint',lambda:SimpleNamespace(url='http://localhost:1234/v1'))
    cfg = model_config.ModelConfig('openai-compatible','gpt-6.1-sol',0,1200)
    monkeypatch.setattr(model_config,'for_phase',lambda *a:cfg)
    def prepare(pdf,entries,requests,result,cache,deadline):
        result.selected = selected(requests[0])
        result.pixels = 1080000
        return [ImageInput(b'png',900,1200,'Page 1')]
    monkeypatch.setattr(visuals,'prepare_images',prepare)
    calls=[]
    def call(**kw):
        kw['request_state']['outbound_requests']=1
        calls.append(kw)
        return llm.LLMResponse(json.dumps({'observations':[observation()]}),'gpt-6.1-sol',0,100,10)
    monkeypatch.setattr(llm,'call',call)
    targets = TargetClaimsOutput(visual_selection_status='complete',visual_requests=[request()])
    def run():
        return visuals.inspect_visuals(pdf_path=pdf,pages=[QUOTE],sections={'results':QUOTE},
            inventory=[entry(main=False)],target_claims=targets,cache_dir=tmp_path/'cache')
    cold,warm = run(),run()
    assert cold.request_count == 1 and warm.cache_hit and warm.request_count == 0 and len(calls)==1
    targets.visual_requests = [request(question='Different question?')]
    assert not run().cache_hit
    pdf.write_bytes(b'source2')
    assert not run().cache_hit
    cfg=replace(cfg,model='gpt-6-sol')
    assert not run().cache_hit
    assert len(calls)==4
    receipts=list((tmp_path/'cache').glob('*.json'))
    assert len(receipts)==4 and all('base64' not in path.read_text() for path in receipts)
    from researchwiki.errors import EnvironmentFailure
    with pytest.raises(EnvironmentFailure,match='changed'):
        visuals.inspect_visuals(pdf_path=pdf,pages=[QUOTE],sections={},inventory=[entry()],
            target_claims=targets,expected_source_hash='outdated',cache_dir=tmp_path/'cache')


def test_failed_observations_are_not_cached(tmp_path,monkeypatch):
    pdf=tmp_path/'paper.pdf';pdf.write_bytes(b'source')
    monkeypatch.setattr(llm,'resolve_openai_endpoint',lambda:SimpleNamespace(url='http://localhost:1234/v1'))
    def prepare(pdf,entries,requests,result,cache,deadline):
        result.selected=selected()
        return [ImageInput(b'png',900,1200,'Page 1')]
    monkeypatch.setattr(visuals,'prepare_images',prepare)
    calls=[]
    monkeypatch.setattr(llm,'call',lambda **kw:calls.append(1) or
                        llm.LLMResponse('not json','model-1',0,5,2))
    for _ in range(2):
        result=visuals.inspect_visuals(pdf_path=pdf,pages=[QUOTE],sections={},inventory=[entry()],
            target_claims=TargetClaimsOutput(visual_selection_status='complete'),cache_dir=tmp_path/'cache')
        assert result.status=='incomplete' and result.critical_unresolved==['fig:2']
    assert len(calls)==2 and not list((tmp_path/'cache').glob('*.json'))


def test_runner_opt_in_is_per_attempt_and_visual_gate_survives_score_replacement(tmp_path,monkeypatch):
    from researchwiki.agents import runner, phases
    from researchwiki.agents.phases.draft import Draft
    from researchwiki.agents.context import Context
    from researchwiki.db.iterations import read_attempt
    from researchwiki.agents.runner_support import phase_extract
    from researchwiki.agents import visual_support
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('RESEARCHWIKI_LOCK_DIR',str(tmp_path/'locks'))
    (tmp_path/'wiki/other').mkdir(parents=True)
    pdf=tmp_path/'paper.pdf';pdf.write_bytes(b'source')
    monkeypatch.setattr(runner,'_phase_reconcile',lambda *a:({'stem':'smith-2026-test',
        'title':'Test','year':2026,'authors':'Smith','paper_type':'research'},None))
    monkeypatch.setattr(runner,'_find_exact_duplicate',lambda *a:None)
    monkeypatch.setattr(runner,'_phase_crosslinks',lambda *a:[])
    monkeypatch.setattr(phases,'extract_sections',lambda *a:({'results':QUOTE},QUOTE))
    calls=[]
    def rich(*a):
        calls.append('pages')
        return {'results':QUOTE+' (Fig. 2).'},QUOTE,['Figure 2. Measured result']
    monkeypatch.setattr(extract,'extract_source_pages',rich)
    monkeypatch.setattr('researchwiki.pdf.sections.anchor_sections',lambda *a,**k:{'results':QUOTE+' (Fig. 2).'})
    def author(*a,**k):
        from researchwiki.db.iterations import write_iteration
        ctx=a[0]
        iteration_id=write_iteration(attempt_id=ctx.attempt_id,pdf_filename=ctx.pdf_filename,
            paper_stem=ctx.paper_stem,iteration=ctx.iteration,role='author',decision='drafted')
        return Draft(text='## Summary\nMeasured result.\n\n## Key Contributions\n- Test.\n',
            model='gpt-6.1-sol',temperature=0,input_tokens=0,output_tokens=0,iteration_id=iteration_id)
    monkeypatch.setattr(runner,'_phase_author',author)
    def grade(ctx,conn,draft):
        draft.scores={'n_graded':5,'n_drift':0,'semantic_available':True,'semantic_score':.95}
    monkeypatch.setattr(runner,'_phase_grade',grade)
    monkeypatch.setattr(runner,'_phase_tournament',lambda ctx,conn:ctx.drafts[0])
    def inspect(**kw):
        calls.append('inspect')
        assert kw['pages']==['Figure 2. Measured result']
        return visuals.VisualResult(status='needs_review',critical_unresolved=['fig:2'])
    monkeypatch.setattr(visuals,'inspect_visuals',inspect)
    # Real commit path with forced mode: visual evidence still vetoes publication.
    monkeypatch.setattr('researchwiki.agents.promote.promote_to_wiki',lambda **kw:pytest.fail('visual gate bypassed'))
    on=runner.run_ingest(pdf,use_stub=True,visual_inspection=True,max_evolve=0,max_debug=0,promote_mode='always')
    assert on.outcome=='sandboxed' and calls==['pages','inspect']
    rows=read_attempt(on.attempt_id)
    assert any(row.role=='visual_inspect' for row in rows)
    receipt=next((tmp_path/'.agent-output').glob('*.visual.json'))
    assert json.loads(receipt.read_text())['critical_unresolved']==['fig:2']
    # Second invocation is off by default; never reuses the previous opt-in.
    calls.clear()
    off=runner.run_ingest(pdf,use_stub=True,max_evolve=0,max_debug=0,promote_mode='never')
    assert not off.visual_inspection and off.visual_result is None and not calls
    assert not any(row.role=='visual_inspect' for row in read_attempt(off.attempt_id))
    ctx=Context(attempt_id='single-pass',pdf_path=pdf,pdf_filename='paper.pdf',visual_inspection=True)
    phase_extract(ctx,None)
    assert calls==['pages']
    assert not off.source_pages
    assert visual_support.final_publication_gate(off,GateResult(True)).promoted


def test_visual_review_status_and_unknown_usage_are_explicit_in_events(image_call,monkeypatch):
    from researchwiki.operations.events import EventRecorder
    from researchwiki.operations.results import checks_from_scores
    module,_=image_call
    monkeypatch.setattr(module,'_post_once',lambda *a:{'choices':[{'message':{'content':'{}'}}]})
    events=[]
    recorder=EventRecorder('visual-events',events.append)
    llm.call(phase='visual_inspect',prompt='context',event_sink=recorder,
             images=[ImageInput(b'png',900,1200,'Page 1')])
    finished=[event for event in events if event.kind=='model_call_finished']
    assert finished and finished[0].usage.status=='unavailable'
    checks=checks_from_scores({'visual_coverage':{'visual_status':'needs_review'}})
    assert next(check for check in checks if check.name=='visual_inspection').status=='incomplete'


def test_fallback_uses_critical_claim_location_before_earlier_results_visual():
    from researchwiki.agents.phases.target_claims import TargetClaim
    entries=[entry('fig:1',1),entry('fig:3',3)]
    chosen=choose_requests(entries,[],[TargetClaim('headline',QUOTE,'critical','Fig. 3d')])
    assert chosen[0].ref_id=='fig:3'


def test_external_supplementary_table_reference_is_not_a_main_table_gap():
    from researchwiki.agents.visual_selection import missing_result_refs
    assert missing_result_refs({'results':'See Supplementary Table 9 and Figure 2.'},[entry()])==[]


def test_unknown_image_pricing_does_not_report_free_work(image_call,monkeypatch):
    _,_=image_call
    monkeypatch.setattr(llm,'resolve_openai_endpoint',lambda:SimpleNamespace(url='http://model.test/v1'))
    monkeypatch.setattr(model_config,'rate_for',lambda *a,**kw:None)
    tracker=BudgetTracker(IngestBudget(max_model_calls=1))
    with activate(tracker):
        llm.call(phase='visual_inspect',prompt='context',images=[ImageInput(b'png',900,1200,'Page 1')])
    assert tracker.snapshot()['estimated_cost_usd'] is None


def test_future_proxy_alias_needs_explicit_image_bound():
    from researchwiki.agents.llm_images import image_token_bound
    assert image_token_bound('gpt-6.1-sol',None) is None
    assert image_token_bound('gpt-4o-mini',None)==100000
    assert image_token_bound('gpt-6.1-sol',32768)==32768


def test_prose_number_without_selected_visual_association_cannot_resolve_question():
    quote='The measured accuracy was 90% for condition A and 80% for condition B (Fig. 5).'
    parsed=parse_observations(json.dumps({'observations':[observation(observation=quote,source_quote=quote)]}),
                              selected(),{'results':quote})
    assert not parsed[0]['corroborated']


def test_concurrent_inspection_workers_share_one_successful_reading(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time
    pdf=tmp_path/'paper.pdf';pdf.write_bytes(b'source')
    monkeypatch.setattr(llm,'resolve_openai_endpoint',lambda:SimpleNamespace(url='http://localhost:1234/v1'))
    def prepare(pdf,entries,requests,result,cache,deadline):
        result.selected=selected()
        return [ImageInput(b'png',900,1200,'Page 1')]
    monkeypatch.setattr(visuals,'prepare_images',prepare)
    calls=[]
    def call(**kw):
        calls.append(1)
        kw['request_state']['outbound_requests']=1
        time.sleep(.03)
        return llm.LLMResponse(json.dumps({'observations':[observation()]}),'model-1',0,100,10)
    monkeypatch.setattr(llm,'call',call)
    def run():
        return visuals.inspect_visuals(pdf_path=pdf,pages=[QUOTE],sections={'results':QUOTE},inventory=[entry()],
            target_claims=TargetClaimsOutput(visual_selection_status='complete'),cache_dir=tmp_path/'cache')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:run(),range(2)))
    assert len(calls)==1 and sum(result.cache_hit for result in results)==1
    assert all(result.observations[0]['corroborated'] for result in results)


# --- Review fixes (PR #28) -------------------------------------------------


@pytest.mark.parametrize('prose', ['Expression remained stable 24 h after dosing.',
                                   'This was notable 3 of 5 donors responded.',
                                   'A comfortable 2 fold margin was kept.'])
def test_words_ending_in_table_or_fig_are_not_references(prose):
    from researchwiki.agents.visual_selection import mentioned_refs, missing_result_refs
    assert mentioned_refs(prose) == set()
    assert missing_result_refs({'results': prose}, []) == []
    assert not build_inventory(['Table 3. Donor counts'], {'results': prose})[0].main_result


def test_uppercase_captions_are_located_and_not_reported_missing():
    pages = ['FIGURE 2 | Accuracy by donor', 'TABLE 1 Baseline characteristics']
    results = 'Accuracy improved (Figure 2) across cohorts (Table 1).'
    entries = build_inventory(pages, {'results': results})
    assert {e.ref_id for e in entries} == {'fig:2', 'table:1'}
    assert all(e.main_result for e in entries)
    from researchwiki.agents.visual_selection import missing_result_refs
    assert missing_result_refs({'results': results}, entries) == []


def test_unlocated_reference_beside_a_located_main_result_is_not_a_veto(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, 'call', lambda **kw: pytest.fail('no request expected'))
    pdf = tmp_path/'paper.pdf'; pdf.write_bytes(b'source')
    result = visuals.inspect_visuals(pdf_path=pdf, pages=['Results (Figure 4).'],
        sections={'results': 'Results (Figure 4) and (Fig. 2).'}, inventory=[entry()],
        target_claims=TargetClaimsOutput(visual_selection_status='incomplete'), cache_dir=tmp_path)
    gaps = {item['request']['ref_id']: item['request']['priority'] for item in result.unresolved
            if item['reason'] == 'missing_caption_location'}
    assert gaps == {'fig:4': 'high'}


@pytest.mark.parametrize('uncertainty', ['none', 'None.', 'N/A', ''])
def test_honest_no_uncertainty_values_still_corroborate(uncertainty):
    parsed = parse_observations(json.dumps({'observations': [observation(uncertainty=uncertainty)]}),
                                selected(), {'results': QUOTE})
    assert parsed[0]['corroborated']


def test_paraphrased_observation_requires_review_even_with_verbatim_quote():
    para = 'Condition A reached 90% accuracy, above condition B at 80%.'
    parsed = parse_observations(json.dumps({'observations': [observation(observation=para)]}),
                                selected(), {'results': QUOTE})
    assert not parsed[0]['corroborated']


def test_supplementary_figure_quote_does_not_corroborate_main_figure():
    quote = 'The measured accuracy was 90% for condition A and 80% for condition B (Supplementary Fig. 2).'
    parsed = parse_observations(json.dumps({'observations': [observation(observation=quote, source_quote=quote)]}),
                                selected(), {'results': quote})
    assert not parsed[0]['corroborated']


def test_prompt_offers_prose_candidates_per_reference(tmp_path, monkeypatch):
    pdf = tmp_path/'paper.pdf'; pdf.write_bytes(b'source')
    monkeypatch.setattr(llm, 'resolve_openai_endpoint', lambda: SimpleNamespace(url='http://localhost:1234/v1'))
    monkeypatch.setattr(model_config, 'for_phase', lambda *a: model_config.ModelConfig('openai-compatible', 'gpt-6.1-sol', 0, 1200))
    page = 'Table 1. Counts\n' + 'x' * 3000 + '\nFigure 2. Measured result'
    def prepare(pdf, entries, requests, result, cache, deadline):
        captions = {'table:1': 'Table 1. Counts', 'fig:2': 'Figure 2. Measured result'}
        result.selected = [{'request': req.to_dict(), 'page': 1, 'caption': captions[req.ref_id]}
                           for req in requests]
        return [ImageInput(b'png', 900, 1200, 'Page 1')]
    monkeypatch.setattr(visuals, 'prepare_images', prepare)
    prompts = []
    def call(**kw):
        prompts.append(json.loads(kw['prompt']))
        return llm.LLMResponse('{"observations":[]}', 'gpt-6.1-sol', 0, 1, 1)
    monkeypatch.setattr(llm, 'call', call)
    visuals.inspect_visuals(pdf_path=pdf, pages=[page], sections={'results': QUOTE},
        inventory=[entry('table:1', main=False), entry('fig:2', main=False)],
        target_claims=TargetClaimsOutput(visual_selection_status='complete',
            visual_requests=[request('table:1'), request('fig:2')]), cache_dir=tmp_path/'cache')
    context = prompts[0]['source_context']
    assert set(context) == {'table:1', 'fig:2'}
    assert 'Table 1. Counts' in context['table:1'] and 'Figure 2. Measured' in context['fig:2']
    assert prompts[0]['prose_candidates']['fig:2'] == [QUOTE]


def test_incomplete_selection_sends_no_image_request(tmp_path, monkeypatch):
    monkeypatch.setattr(visuals, 'prepare_images', lambda *a: pytest.fail('no render expected'))
    monkeypatch.setattr(llm, 'call', lambda **kw: pytest.fail('no request expected'))
    pdf = tmp_path/'paper.pdf'; pdf.write_bytes(b'source')
    result = visuals.inspect_visuals(pdf_path=pdf, pages=[QUOTE], sections={'results': QUOTE},
        inventory=[entry()], target_claims=TargetClaimsOutput(visual_selection_status='off'),
        cache_dir=tmp_path)
    assert result.request_count == 0 and result.status == 'needs_review'
    assert {item['reason'] for item in result.unresolved} == {'selection_incomplete'}


def test_provider_failure_propagates_instead_of_becoming_a_review_veto(tmp_path, monkeypatch):
    pdf = tmp_path/'paper.pdf'; pdf.write_bytes(b'source')
    monkeypatch.setattr(llm, 'resolve_openai_endpoint', lambda: SimpleNamespace(url='http://localhost:1234/v1'))
    monkeypatch.setattr(model_config, 'for_phase', lambda *a: model_config.ModelConfig('openai-compatible', 'gpt-6.1-sol', 0, 1200))
    def prepare(pdf, entries, requests, result, cache, deadline):
        result.selected = selected()
        return [ImageInput(b'png', 900, 1200, 'Page 1')]
    monkeypatch.setattr(visuals, 'prepare_images', prepare)
    def call(**kw):
        raise llm.ProviderUnavailable('Visual inspection HTTP 429; no retry was sent.')
    monkeypatch.setattr(llm, 'call', call)
    with pytest.raises(llm.ProviderUnavailable):
        visuals.inspect_visuals(pdf_path=pdf, pages=[QUOTE], sections={}, inventory=[entry()],
            target_claims=TargetClaimsOutput(visual_selection_status='complete'), cache_dir=tmp_path/'c')


def test_rejected_effort_is_omitted_not_sent_as_null(image_call, monkeypatch):
    _, sent = image_call
    cfg = model_config.ModelConfig('openai-compatible', 'gpt-6.1-sol', 0, 1200, reasoning_effort='low',
                                   image_token_bound=32768)
    monkeypatch.setattr(model_config, 'for_phase', lambda *a: cfg)
    monkeypatch.setitem(llm._EFFORT_REJECTED, ('http://localhost:1234/v1', 'gpt-6.1-sol'), {'low'})
    llm.call(phase='visual_inspect', prompt='context', images=[ImageInput(b'png', 900, 1200, 'Page 1')])
    assert 'reasoning_effort' not in sent[0]


def test_image_payload_matches_text_transport_temperature_rule(image_call, monkeypatch):
    _, sent = image_call
    cfg = model_config.ModelConfig('openai-compatible', 'gpt-6.1-sol', 0, 1200, reasoning_effort='none',
                                   image_token_bound=32768)
    monkeypatch.setattr(model_config, 'for_phase', lambda *a: cfg)
    llm.call(phase='visual_inspect', prompt='context', images=[ImageInput(b'png', 900, 1200, 'Page 1')])
    assert sent[0]['temperature'] == 0 and sent[0]['reasoning_effort'] == 'none'


def test_preflight_refuses_unbounded_image_model_under_a_budget(monkeypatch):
    from researchwiki.agents import llm_images
    monkeypatch.setattr(model_config, 'for_phase',
                        lambda *a: model_config.ModelConfig('openai-compatible', 'gpt-6-luna', 0, 1200))
    monkeypatch.setattr(llm, 'missing_provider_credentials', lambda **kw: [])
    llm_images.preflight_visual_provider()  # no budget: no bound needed
    for budget in (dict(max_tokens=200000), dict(max_cost_usd=1.0)):
        with pytest.raises(model_config.ModelConfigUnavailable, match='image_token_bound'):
            llm_images.preflight_visual_provider(**budget)
    monkeypatch.setattr(model_config, 'for_phase', lambda *a: model_config.ModelConfig(
        'openai-compatible', 'gpt-6-luna', 0, 1200, image_token_bound=8000))
    llm_images.preflight_visual_provider(max_tokens=200000)


def test_quoted_image_token_bound_is_an_int(tmp_path, monkeypatch):
    cfg = tmp_path/'models.yaml'
    cfg.write_text('roles:\n  judge: {provider: openai-compatible, model: m, image_token_bound: "8000"}\n'
                   'phases:\n  visual_inspect: {role: judge, image_token_bound: "9000"}\n'
                   'base_url: http://localhost:1234/v1\n')
    monkeypatch.setenv('RW_MODELS_CONFIG', str(cfg))
    model_config.clear_caches()
    try:
        phase = model_config.for_phase('visual_inspect')
        role = model_config.for_phase('target_claims')
    finally:
        monkeypatch.delenv('RW_MODELS_CONFIG')
        model_config.clear_caches()
    assert phase.image_token_bound == 9000 and isinstance(phase.image_token_bound, int)
    assert role.image_token_bound == 8000 and isinstance(role.image_token_bound, int)


def test_visual_phase_duration_counts_in_timing():
    from researchwiki.tasks.insights_timing import TIMED_ROLES
    assert 'visual_inspect' in TIMED_ROLES


def test_benchmark_validation_does_not_depend_on_assert(tmp_path):
    import subprocess, sys
    suite = Path(__file__).resolve().parents[1] / 'benchmark-fixtures' / 'visual-inspection'
    source = (suite / 'benchmark.py').read_text()
    assert '\n    assert ' not in source.split('def validate():', 1)[1].split('def prepare(', 1)[0]
    out = subprocess.run([sys.executable, '-I', '-O', str(suite / 'benchmark.py'), 'validate'],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)['source_hashes_verified'] is True


def test_small_panel_figure_page_resolves_for_inspection():
    """A labelled page at ~5% artwork (a single small panel) is inspectable."""
    class SmallPanel(FakePage):
        def get_objects(self, **kw):
            yield SimpleNamespace(type=visuals.pypdfium2.raw.FPDF_PAGEOBJ_PATH,
                                  get_bounds=lambda: (0, 0, 160, 157))  # 5.2% of 612x792
    assert visuals.resolve_page(FakeDoc([SmallPanel()]), entry()) == 1
    assert visuals.resolve_page(FakeDoc([FakePage(False)]), entry()) is None
