"""Durable overlap outcomes, atomic link application, and ownership-aware revert."""
from __future__ import annotations

from contextlib import contextmanager

import hashlib
import json
import re
import time
import uuid
from pathlib import Path

from ..backlinks import append_related_paper
from ..fsatomic import write_json_atomic, write_text_atomic
from ..mutation import corpus_lock, mutation, recover_pending, pending_journals
from ..paths import papers_dir, wiki_root, wiki_dir
from ..pdf.citation import VERSION, file_hash, target_identity
from ..wiki import read_page

NOTE = 'claim-grounded match (auto-added; claim-overlap)'
RECEIPT_VERSION = 1


def new_run_id():
    return time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:12]


def root():
    return wiki_root() / '.ingest' / 'claim-overlap'


def direct_source(run_id):
    """Coverage source for a stem that committed nothing and so has no receipt."""
    return f'direct:{run_id}:{time.time_ns()}'


def run_path(run_id):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', run_id):
        raise ValueError('invalid run ID')
    return root() / run_id


def load(path):
    from .receipt_schema import validate
    doc = json.loads(path.read_text(encoding='utf-8'))
    try:
        validate(doc, RECEIPT_VERSION)
    except ValueError as exc:
        raise ValueError(f'invalid claim-overlap receipt {path}: {exc}') from exc
    return doc


def _log_unreadable(path, exc):
    from ..log import log
    log(f'claim-overlap: skipped unreadable receipt {path}: {exc}', tag='claim-overlap')


def try_load(path):
    """`load` for corpus walks: an unreadable or foreign file is skipped, not fatal."""
    try:
        return load(path)
    except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
        _log_unreadable(path, exc)
        return None


def _page_signature(page):
    from ..tasks.claim_overlap import claims_from_page, claims_fingerprint
    return claims_fingerprint(claims_from_page(page.stem, page.path))


def snapshot_inputs(pages, conn):
    from ..tasks.claim_overlap import _claims_for_stem, claims_fingerprint
    return {p.stem: {'path': str(p.path.resolve()), 'page_claims': _page_signature(p),
                     'db_claims': claims_fingerprint(_claims_for_stem(conn, p.stem)),
                     'pdf_sha256': file_hash(papers_dir() / f'{p.stem}.pdf'),
                     'identity': target_identity(p)} for p in pages}


def _require_fresh(inputs, conn, *, citation_inputs=True):
    from ..tasks.claim_overlap import _claims_for_stem, claims_fingerprint
    for stem, expected in inputs.items():
        page = read_page(Path(expected['path']))
        if page is None or _page_signature(page) != expected['page_claims']:
            raise ValueError(f'page claims changed during review: {stem}')
        if claims_fingerprint(_claims_for_stem(conn, stem)) != expected['db_claims']:
            raise ValueError(f'database claims changed during review: {stem}')
        if citation_inputs and (
                file_hash(papers_dir() / f'{stem}.pdf') != expected['pdf_sha256']
                or target_identity(page) != expected['identity']):
            raise ValueError(f'citation inputs changed during review: {stem}; retry review')


def _safe_page(raw):
    path = Path(raw).resolve()
    if wiki_dir().resolve() not in path.parents or path.suffix != '.md':
        raise ValueError(f'receipt page lies outside wiki: {path}')
    return path


def stage(result, *, run_id, stem, inputs, links, edges, coverage):
    path = run_path(run_id) / f'{stem}-{uuid.uuid4().hex[:8]}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {'schema_version': RECEIPT_VERSION, 'run_id': run_id, 'stem': stem,
           'created_at': time.time_ns(), 'status': 'staged', 'inputs': inputs,
           'verifier_version': VERSION, 'result': result, 'links': links,
           'edges': edges, 'coverage': coverage, 'bookkeeping_pending': True}
    write_json_atomic(path, doc)
    return path, doc


def commit(path, doc, conn):
    """Receipt and actual insertion ownership commit with the pages."""
    paths = [path] + [_safe_page(i['path']) for link in doc['links'] for i in link['insertions']]
    with mutation(paths, operation='claim-overlap',
                  details={'repair_projections': True, 'receipt': str(path)}) as snap:
        _require_fresh(doc['inputs'], conn)
        if doc['verifier_version'] != VERSION:
            raise ValueError('citation verifier changed; retry review')
        for link in doc['links']:
            proven = any(
                p.get('verified') and p.get('verifier_version') == VERSION
                and p.get('source_pdf_sha256')
                and p['source_pdf_sha256'] == doc['inputs'][p['source_stem']]['pdf_sha256']
                and p['expected_identity'] == doc['inputs'][p['target_stem']]['identity']
                and p.get('page') and p.get('excerpt') and p.get('match_span')
                for p in link['citation_evidence'])
            if not proven:
                raise ValueError('link receipt has no current entry-scoped citation proof')
            link['event_id'] = link.get('event_id') or uuid.uuid4().hex
            link['active'] = True
            for insertion in link['insertions']:
                page = _safe_page(insertion['path'])
                before = page.read_text(encoding='utf-8')
                insertion['owned'] = append_related_paper(
                    _safe_page(insertion['path']), insertion['target_key'], note=NOTE, sync_db=False)
                if insertion['owned']:
                    insertion['before_text'] = before
                    insertion['after_sha256'] = file_hash(page)
            link['committed_pair'] = any(i['owned'] for i in link['insertions'])
        doc['status'] = 'committed'
        write_json_atomic(path, doc)
        snap.mark_committed()
        snap.reconcile()
        reconcile_receipt(path, conn=conn)
    return doc


def _edge_write(spec):
    from ..tasks.claim_overlap import _persist_typed_edge
    _persist_typed_edge(spec['src']['paper_stem'], spec['src'],
                        spec['tgt']['paper_stem'], spec['tgt'], spec['relation'],
                        spec['rationale'], spec['cosine'], judge_model=spec.get('judge_model', ''))


def reconcile_receipt(path, *, conn=None):
    """Idempotent projection replay. Never supersede a newer run or claim set."""
    from ..tasks.claim_overlap import _claims_for_stem, claims_fingerprint, record_run, _default_conn
    doc = load(path)
    if doc['status'] not in {'committed', 'reverted'}:
        return
    close = conn is None
    c = conn or _default_conn()
    try:
        if doc.get('edges_pending', True):
            for edge in doc['edges']:
                _edge_write(edge)
            doc['edges_pending'] = False
            write_json_atomic(path, doc)
        cov = doc.get('coverage')
        if cov and cov['complete']:
            fp = claims_fingerprint(_claims_for_stem(c, doc['stem']))
            current = c.execute('SELECT ran_at, source FROM claim_overlap_runs WHERE paper_stem=?',
                                (doc['stem'],)).fetchone()
            source = f"receipt:{doc['run_id']}:{path.name}"
            # Nanosecond event IDs order receipts even within one second;
            # legacy rows use seconds and always precede these records.
            newer = current and current[0] > doc['created_at'] // 1_000_000_000
            if current and current[1].startswith('receipt:') and current[1] != source:
                _, other_run, other_file = current[1].split(':', 2)
                other = try_load(run_path(other_run) / other_file)
                newer = newer or bool(other and other.get('created_at', 0) > doc['created_at'])
            elif current and current[1].startswith('direct:'):
                newer = newer or int(current[1].rsplit(':', 1)[1]) > doc['created_at']
            if fp == cov['fingerprint'] and not newer:
                confirmed = sum(link.get('active', False) for link in doc['links'])
                record_run(c, doc['stem'], fingerprint=fp, n_claims=cov['n_claims'],
                           n_candidates=cov['n_candidates'], n_judged=cov['n_judged'],
                           n_confirmed=confirmed, sim_threshold=cov['sim_threshold'],
                           ran_at=doc['created_at'] // 1_000_000_000, source=source)
        doc['bookkeeping_pending'] = False
        write_json_atomic(path, doc)
    finally:
        if close:
            c.close()


def _blocks_overlap(journal):
    """A retained journal blocks overlap only if it could roll back our pages.

    Committed journals awaiting projection repair (e.g. a promote whose index
    update failed) never restore page bytes, so they cannot undo or be undone
    by an overlap write. Active journals, and any claim-overlap journal, can.
    """
    return (journal.get('status') != 'committed'
            or str(journal.get('operation', '')).startswith('claim-overlap'))


def recover(*, conn=None):
    """Drain journals, then finish receipt bookkeeping. Call once per invocation."""
    notes = recover_pending()
    with corpus_lock():
        remaining = [j for j in pending_journals() if _blocks_overlap(j)]
        if remaining:
            from ..errors import EnvironmentFailure
            raise EnvironmentFailure(f"claim-overlap recovery incomplete: {len(remaining)} retained mutation journal(s)")
        # Only receipts still owing bookkeeping need parsing; completed ones are
        # skipped by a cheap substring check before any JSON decode.
        for path in sorted(root().glob('*/*.json')):
            try:
                raw = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError) as exc:
                _log_unreadable(path, exc)
                continue
            if not re.search(r'"bookkeeping_pending"\s*:\s*true\b', raw):
                continue
            doc = try_load(path)
            if doc is None or doc.get('record_type') == 'run':
                continue
            if doc.get('bookkeeping_pending') and doc.get('status') in {'committed', 'reverted'}:
                reconcile_receipt(path, conn=conn)
    return notes


def revert(run_id, *, stem=None, link=None, apply=False, conn=None):
    """Preview by default; only exact, owned bullets can be removed."""
    if apply:
        recover(conn=conn)
    directory = run_path(run_id)
    if not directory.is_dir():
        raise ValueError(f'unknown claim-overlap run: {run_id}')
    reports = []
    with corpus_lock():
        docs = [(p, d) for p in directory.glob('*.json') if (d := try_load(p)) is not None]
        for path, doc in sorted(docs, key=lambda item: item[1].get('created_at', 0), reverse=True):
            if doc.get('record_type') == 'run':
                continue
            if stem and doc['stem'] != stem:
                continue
            if doc['status'] not in {'committed', 'reverted'}:
                continue
            selected = [item for item in reversed(doc['links']) if item.get('active')
                        and (not link or sorted(item['stems']) == sorted(link))]
            if not selected:
                continue
            touched = [path] + [_safe_page(i['path']) for l in selected for i in l['insertions']]
            context = (mutation(touched, operation='claim-overlap-revert',
                                details={'repair_projections': True, 'receipt': str(path)})
                       if apply else _preview())
            with context as snap:
                for item in selected:
                    unresolved = False
                    for ins in item['insertions']:
                        if not ins.get('owned') or ins.get('reverted'):
                            continue
                        page = _safe_page(ins['path'])
                        body = page.read_text(encoding='utf-8') if page.exists() else ''
                        bullet = ins['bullet']
                        # Exact line and original marker. Ambiguous duplicates
                        # might include a later manual copy; leave them alone.
                        lines = body.splitlines(keepends=True)
                        matching = [j for j, line in enumerate(lines) if line.rstrip('\r\n') == bullet]
                        removable = len(matching) == 1 and 'auto-added; claim-overlap' in bullet
                        reason = 'remove' if removable else 'edited, absent or ambiguous bullet'
                        reports.append({'stem': doc['stem'], 'event_id': item['event_id'],
                                        'path': str(page), 'outcome': reason})
                        if removable and apply:
                            if (ins.get('before_text') is not None
                                    and hashlib.sha256(body.encode()).hexdigest() == ins.get('after_sha256')):
                                restored = ins['before_text']
                            else:
                                del lines[matching[0]]
                                restored = ''.join(lines)
                            write_text_atomic(page, restored)
                            ins['reverted'] = True
                        elif not removable:
                            unresolved = True
                    if apply and not unresolved:
                        item['active'] = False
                if apply:
                    doc['status'] = 'reverted'
                    doc['bookkeeping_pending'] = True
                    write_json_atomic(path, doc)
                    snap.mark_committed()
                    snap.reconcile()
                    reconcile_receipt(path, conn=conn)
    return {'run_id': run_id, 'dry_run': not apply, 'reverted': reports}


# Revert needs the transaction handle explicitly; kept as a tiny context adapter
# so preview never allocates a journal or writes a receipt.

@contextmanager
def _preview():
    yield None
