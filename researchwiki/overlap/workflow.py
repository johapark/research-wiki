"""Stage source-grounded decisions outside the corpus lock, then commit once."""
from __future__ import annotations

from ..errors import EnvironmentFailure
from ..pdf.citation import CitationVerifier, target_identity
from ..paths import papers_dir
from .judging import JudgeSession, CROSS_LINK_VERDICTS
from . import receipts


def citation_proofs(a, b, verifier):
    proofs = []
    for source, target in ((a, b), (b, a)):
        own_doi = source.str_field('doi', '').strip().lower()
        expected = target_identity(target)
        if own_doi and own_doi == expected['doi']:
            proofs.append({'source_stem': source.stem, 'target_stem': target.stem,
                           'verified': False, 'reason': 'source and target DOI identify the same document'})
        else:
            proofs.append(verifier.verify(papers_dir() / f'{source.stem}.pdf',
                                           source.stem, expected))
    return proofs


def already_linked(a, b):
    import re
    def has(source, target):
        pattern = r'\[\[(?:[^\]|#]*/)?' + re.escape(target.stem) + r'[\]|#]'
        return re.search(pattern, source.path.read_text(encoding='utf-8')) is not None
    return has(a, b) or has(b, a)


def evaluate(stem, candidates, pages, *, compared, conn, sim_threshold, dry_run,
             judge_fn=None, session=None, run_id=None, exact=False):
    from ..tasks.claim_overlap import claims_fingerprint
    session = session or JudgeSession(dry_run=dry_run, judge_fn=judge_fn)
    run_id = run_id or receipts.new_run_id()
    involved = {stem: pages[stem]}
    for cand in candidates:
        if cand.existing_stem in pages:
            involved[cand.existing_stem] = pages[cand.existing_stem]
    # Hash inputs before judging so changes during review are detectable; with
    # no candidates nothing is staged, so there is nothing to pin.
    inputs = receipts.snapshot_inputs(involved.values(), conn) if candidates else {}
    result = {'stem': stem, 'run_id': run_id, 'n_candidates': len(candidates),
              'applied': [], 'edge_only': [], 'coincidence': [], 'skipped': [],
              'judge_failed': [], 'dry_run': dry_run, 'judged_pairs': 0,
              'reused_judgments': 0, 'committed_link_pairs': 0}
    links, edges = [], []
    verifier = getattr(session, 'citation_verifier', None)
    if verifier is None:
        verifier = CitationVerifier()
        session.citation_verifier = verifier
    interruption = None
    for cand in candidates:
        target = pages.get(cand.existing_stem)
        if target is None:
            # Stale claim rows for a hand-deleted page: nothing to judge or
            # link, and retrying cannot help until `db rebuild`. Skipping (as
            # before receipts) keeps coverage completable.
            result['skipped'].append({'existing': cand.existing_stem, 'cosine': cand.cosine,
                                      'reason': 'missing wiki page'})
            continue
        record = {'existing': target.key, 'cosine': cand.cosine}
        if already_linked(pages[stem], target):
            result['skipped'].append(dict(record, reason='already-linked'))
            continue
        a = dict(cand.new_claim, paper_stem=stem)
        b = dict(cand.existing_claim, paper_stem=target.stem)
        before_calls = session.calls
        try:
            a, b, verdict = session.judge(a, b)
        except EnvironmentFailure as exc:
            result['judged_pairs'] += session.calls - before_calls
            interruption = exc
            result['judge_failed'].append(dict(record, reason=str(exc)))
            break
        result['judged_pairs'] += session.calls - before_calls
        if verdict is None:
            result['judge_failed'].append(dict(record, reason='invalid verdict/direction or unparseable reply'))
            continue
        result['reused_judgments'] += int(verdict['reused'])
        record.update(rationale=verdict['rationale'], direction=verdict['direction'],
                      verdict=verdict['verdict'], cache_key=verdict['cache_key'],
                      reused=verdict['reused'], model_metadata=verdict.get('_metadata', {}),
                      claims=[a, b])
        relation = verdict['verdict']
        if relation == 'none':
            result['coincidence'].append(record)
            continue
        record['relation'] = relation
        src, tgt = (b, a) if verdict['direction'] == 'b_to_a' else (a, b)
        edges.append({'src': src, 'tgt': tgt, 'relation': relation,
                      'rationale': verdict['rationale'], 'cosine': cand.cosine,
                      'judge_model': verdict.get('_metadata', {}).get('model') or ''})
        proofs = citation_proofs(pages[stem], target, verifier) if relation in CROSS_LINK_VERDICTS else []
        record['citation_evidence'] = proofs
        proven = any(p['verified'] for p in proofs)
        linked = relation in CROSS_LINK_VERDICTS and proven
        if not linked and relation in CROSS_LINK_VERDICTS:
            record['reason'] = 'no explicit citation verified in either PDF'
        (result['applied'] if linked else result['edge_only']).append(record)
        if linked:
            insertions = [{'path': str(source.path.resolve()), 'target_key': dest.key,
                           'bullet': f'- [[{dest.key}]] — {receipts.NOTE}'}
                          for source, dest in ((pages[stem], target), (target, pages[stem]))]
            links.append({'stems': [stem, target.stem], 'insertions': insertions,
                          'citation_evidence': proofs})
    if not dry_run:
        cov = None if exact else {
            'complete': not result['judge_failed'], 'fingerprint': claims_fingerprint(compared),
            'n_claims': len(compared), 'n_candidates': len(candidates),
            'n_judged': sum(len(result[k]) for k in ('applied', 'edge_only', 'coincidence')),
            'sim_threshold': sim_threshold,
        }
        if not links and not edges:
            # Nothing touches a page or the graph: record coverage directly,
            # without a receipt, mutation journal or second round of PDF hashing.
            # Same guard as receipt reconciliation: never close a stem whose
            # stored claims differ from the set actually compared.
            from ..tasks.claim_overlap import _claims_for_stem, record_run
            if (cov and cov['complete']
                    and claims_fingerprint(_claims_for_stem(conn, stem)) == cov['fingerprint']):
                # `direct:` carries a nanosecond stamp so an older receipt's
                # replay (e.g. a later revert) cannot supersede this record.
                record_run(conn, stem, fingerprint=cov['fingerprint'], n_claims=cov['n_claims'],
                           n_candidates=cov['n_candidates'], n_judged=cov['n_judged'],
                           n_confirmed=0, sim_threshold=sim_threshold,
                           source=receipts.direct_source(run_id))
            result['status'] = 'incomplete' if result['judge_failed'] else 'committed'
            if interruption:
                interruption.claim_overlap_result = result
                raise interruption
            return result
        path, doc = receipts.stage(result, run_id=run_id, stem=stem, inputs=inputs,
                                   links=links, edges=edges, coverage=cov)
        result['receipt'] = str(path)
        try:
            receipts.commit(path, doc, conn)
            _insertion_counts(result, doc, exact=exact)
            result['status'] = 'incomplete' if result['judge_failed'] else 'committed'
        except Exception as exc:
            durable = receipts.load(path)
            if durable['status'] == 'committed':
                _insertion_counts(result, durable, exact=exact)
            result['status'] = ('maintenance-pending' if durable['status'] == 'committed' else 'incomplete')
            result['maintenance_error'] = f'{type(exc).__name__}: {exc}'
            if durable['status'] != 'committed':
                result['judge_failed'].append({'reason': result['maintenance_error']})
    else:
        result['status'] = 'preview'
    if interruption:
        interruption.claim_overlap_result = result
        raise interruption
    return result


def _insertion_counts(result, doc, *, exact):
    for record, link in zip(result['applied'], doc['links']):
        a, b = (item['owned'] for item in link['insertions'])
        record['wrote'] = {'new_page': a, 'existing_page': b}
        if exact:
            record['wrote'].update(a_page=a, b_page=b)
    result['committed_link_pairs'] = sum(link.get('active') and link.get('committed_pair', False)
                                         for link in doc['links'])
