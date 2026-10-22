"""Generic opt-in evidence selection at the engine CLI boundary.

The engine reads a versioned paper selection, independent of project storage or
commercial applications. Scope failures never fall back to library retrieval.
"""
from pathlib import Path
import json
import sys

from ..errors import EnvironmentFailure
from ..evidence import EvidenceScope
from ..fsatomic import file_sha256


def add_scope_argument(parser):
    parser.add_argument('--scope-file', metavar='JSON', help='Version 1 evidence selection; empty selections return no evidence.')


def read_scope_file(path):
    try:
        data = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
    except OSError as exc:
        raise EnvironmentFailure(f'cannot read evidence scope: {path}') from exc
    scope = EvidenceScope.from_dict(data)
    scope.check_root()
    for stem, page_hash, pdf_hash in scope.source_revisions:
        pages = list((Path.cwd() / 'wiki').glob(f'*/{stem}.md'))
        pdf = Path.cwd() / 'papers' / f'{stem}.pdf'
        try:
            current = len(pages) == 1 and file_sha256(pages[0]) == page_hash and file_sha256(pdf) == pdf_hash
        except OSError as exc:
            raise EnvironmentFailure(f'evidence source is unavailable: {stem}') from exc
        if not current:
            raise ValueError(f'evidence scope is stale for {stem}; rebuild the selection')
    return scope


def load_scope(args, parser):
    args.evidence_scope = None
    if args.scope_file is None:
        return
    try:
        args.evidence_scope = read_scope_file(args.scope_file)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    scope = args.evidence_scope
    print(f'Evidence scope: {len(scope.stems)} papers; fingerprint {scope.fingerprint}', file=sys.stderr)


def scope_options(args):
    scope = getattr(args, 'evidence_scope', None)
    return {'scope': scope} if scope is not None else {}


def scoped_payload(args, payload):
    scope = getattr(args, 'evidence_scope', None)
    if scope is None:
        return payload
    return {'scope': {**scope.to_dict(), 'fingerprint': scope.fingerprint}, 'results': payload}
