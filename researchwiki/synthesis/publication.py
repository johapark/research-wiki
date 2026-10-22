"""Explicit journalled publication; scope and application guard checked at commit."""
from pathlib import Path
import re

from ..answers.sources import snapshot_scope
from ..errors import EnvironmentFailure
from ..fsatomic import file_sha256, write_text_atomic
from ..log import append_log_md
from ..mutation import corpus_lock, mutation, recover_pending, pending_journals
from ..paths import index_path, log_path, wiki_dir
from ..wiki import find_stem_collision
from .evidence import sources_for
from .gates import page_gates
from .receipt import SynthesisReceipt


def _guard(path, expected):
    if path is None and expected is None:
        return
    if path is None or not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('publication guard requires a path and SHA256')
    path = Path(path).resolve()
    if Path.cwd().resolve() not in path.parents:
        raise ValueError('publication guard must belong to the selected library')
    try:
        actual = file_sha256(path)
    except OSError as exc:
        raise EnvironmentFailure('publication guard is unavailable; resolve the selection again') from exc
    if actual != expected:
        raise ValueError('application selection changed before publication; reload and review the output')


def _catalog(receipt, slug):
    from ..agents.promote import _append_index_entry
    path = index_path()
    text = path.read_text(encoding='utf-8') if path.exists() else '---\ntype: meta\n---\n\n# Index\n'
    if not re.search(r'^## synthesis\s*$', text, re.M):
        write_text_atomic(path, text.rstrip() + '\n\n## synthesis\n\n')
    if not _append_index_entry(slug, 'synthesis', receipt.title, receipt.created_at[:4], None,
                               receipt.draft.hook.text, short_name=receipt.title):
        raise EnvironmentFailure('synthesis catalog could not be updated')


def publish(receipt, scope, slug, *, guard_path=None, guard_sha256=None):
    receipt = SynthesisReceipt.from_dict(receipt.to_dict())
    if receipt.status != 'verified':
        raise ValueError('only a verified synthesis may be published')
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', slug or ''):
        raise ValueError('publication requires a canonical bare slug')
    scope.check_root()
    if receipt.scope.fingerprint != scope.fingerprint:
        raise ValueError('saved synthesis no longer matches the current selection')
    text = receipt.markdown()
    out = wiki_dir() / 'synthesis' / f'{slug}.md'
    with corpus_lock():
        recover_pending()
        _guard(guard_path, guard_sha256)
        snapshot_scope(scope)
        if receipt.sources != sources_for(scope):
            raise ValueError('synthesis citation paths no longer match the canonical sources')
        gates = page_gates(text, receipt.question, scope)
        if not gates.passed:
            raise ValueError('synthesis page gates failed on publication; review or regenerate')
        excluded = {d.stem for d in receipt.draft.coverage if d.decision == 'exclude'}
        if not set(gates.coverage_omissions).issubset(excluded):
            raise ValueError('publication has an unreviewed coverage omission')
        if out.exists():
            if out.read_text(encoding='utf-8') != text:
                raise ValueError('synthesis page already exists with different content; choose another slug')
            pending = any(j.get('details', {}).get('stem') == slug for j in pending_journals())
            return {'publication': 'already_published', 'path': str(out),
                    'projections': 'maintenance_pending' if pending else 'current'}
        if find_stem_collision(slug) is not None:
            raise ValueError('publication slug already belongs to another wiki page')
        try:
            with mutation([out, index_path(), log_path()], operation='synthesis',
                          details={'stem': slug, 'repair_projections': True,
                                   'source_fingerprint': scope.fingerprint, 'receipt_digest': receipt.digest}) as snap:
                # Recheck after snapshot setup and before any canonical write.
                _guard(guard_path, guard_sha256)
                snapshot_scope(scope)
                if receipt.sources != sources_for(scope):
                    raise ValueError('synthesis citation paths changed before publication')
                write_text_atomic(out, text)
                _catalog(receipt, slug)
                append_log_md('synthesis', f'{receipt.title} → synthesis/{slug}',
                              f'Grounding {gates.grounding_total} units; fidelity {gates.fidelity_graded} graded. '
                              f'PDF/claim entailment and theme/coverage review passed. Receipt {receipt.digest}.')
                snap.mark_committed()
        except OSError as exc:
            raise EnvironmentFailure('cannot publish synthesis; journal preserves the prior canonical state') from exc
        try:
            snap.reconcile()
        except Exception as exc:
            return {'publication': 'published', 'path': str(out), 'projections': 'maintenance_pending',
                    'warning': str(exc), 'journal': str(snap.journal_path)}
    return {'publication': 'published', 'path': str(out), 'projections': 'current'}
