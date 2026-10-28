"""Run opt-in real-model synthesis evaluation on bundled CC-BY PDFs.

This spends model calls using the checkout's active provider configuration.
Expectations stay out of prompts. Review receipts against the full PDFs using
benchmark-fixtures/generated-content/README.md; pipeline verification alone is
not an accuracy score. The real user library is never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from researchwiki import __version__  # noqa: E402
from researchwiki.__main__ import _load_dotenv  # noqa: E402
from researchwiki.agents import llm, model_config  # noqa: E402
from researchwiki.db.rebuild import rebuild  # noqa: E402
from researchwiki.evidence import EvidenceScope  # noqa: E402
from researchwiki.index.incremental import rebuild_page_indexes  # noqa: E402
from researchwiki.pdf.text import extract_pdf_page_texts  # noqa: E402
from researchwiki.synthesis import compose  # noqa: E402
from researchwiki.synthesis.render import render  # noqa: E402


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def prepare(output, spec):
    """Build minimal retrieval records with exact PDF quotations, not summaries.

    No existing private wiki page, state DB, or cached grade is copied. Claims
    are extracted by normal db rebuild but left ungraded, so the run exercises
    PDF-backed fallback rather than claiming coverage of graded-claim retrieval.
    """
    library = output / 'library'
    for name in ('papers', 'wiki/other', 'wiki/synthesis', 'config', 'texts'):
        (library / name).mkdir(parents=True, exist_ok=True)
    (library / 'wiki/index.md').write_text('---\ntype: meta\n---\n', encoding='utf-8')
    (library / 'wiki/log.md').write_text('---\ntype: meta\n---\n', encoding='utf-8')
    hashes = {}
    for alias, source in spec['papers'].items():
        stem = source['stem']
        pdf = ROOT / 'benchmark-fixtures/pdfs' / f'{stem}.pdf'
        shutil.copy2(pdf, library / 'papers' / pdf.name)
        pages = extract_pdf_page_texts(pdf, max_pages=80)
        page = ' '.join(pages[source['page'] - 1].split())
        start = page.index(source['start'])
        end = page.index(source['end'], start) if source['end'] else len(page)
        quote = page[start:end].strip()
        metadata = yaml.safe_load((ROOT / 'benchmark-fixtures' / f'{stem}.yaml').read_text())
        header = yaml.safe_dump({'title': metadata['title'], 'type': 'paper', 'category': ['other']})
        text = f'---\n{header}---\n\n## Summary\n\n{quote}\n'
        (library / 'wiki/other' / f'{stem}.md').write_text(text, encoding='utf-8')
        (library / 'texts' / f'{stem}.txt').write_text('\n\n'.join(
            f'=== PDF PAGE {i} ===\n{p}' for i, p in enumerate(pages, 1)), encoding='utf-8')
        hashes[alias] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    return library, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--case', action='append', help='Run only selected case IDs')
    args = parser.parse_args()
    os.chdir(ROOT)
    _load_dotenv()
    # Pin an absolute model configuration before changing to the disposable
    # library. Provider environment variables retain ordinary precedence.
    config = model_config.config_path()
    if config.exists():
        os.environ['RW_MODELS_CONFIG'] = str(config.resolve())
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = output / 'manifest.json'
    if existing.exists() and json.loads(existing.read_text()).get('results'):
        parser.error('output already contains model results; use a new directory to preserve the run')
    fixture = ROOT / 'benchmark-fixtures/generated-content/cases.yaml'
    spec = yaml.safe_load(fixture.read_text())
    selected = [c for c in spec['cases'] if not args.case or c['id'] in args.case]
    if any(c['operation'] != 'compose' for c in selected):
        parser.error('this evaluator supports synthesis cases only')
    if args.case and set(args.case) - {c['id'] for c in selected}:
        parser.error('unknown case ID')
    library, hashes = prepare(output, spec)
    os.environ['RESEARCHWIKI_DB_PATH'] = str(library / 'state.db')
    os.environ['RESEARCHWIKI_LOCK_DIR'] = str(library / 'locks')
    os.chdir(library)
    rebuild()
    rebuild_page_indexes(semantic=False)
    manifest = {'schema_version': 1, 'framework_version': __version__,
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / 'researchwiki').rglob('*.py'))},
                'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
                'pdf_sha256': hashes, 'routing': {
                    p: {'provider': model_config.for_phase(p).provider, 'model': model_config.for_phase(p).model}
                    for p in ('synthesis_author', 'synthesis_support')},
                'results': []}
    write_json(output / 'manifest.json', manifest)
    if args.prepare_only:
        print(f'Prepared isolated real-PDF library: {library}', flush=True)
        return 0
    original_call = llm.call
    calls = []

    def record_call(**kwargs):
        response = original_call(**kwargs)
        calls.append({'phase': kwargs['phase'], 'system': kwargs.get('system'),
                      'prompt': kwargs['prompt'], 'schema': kwargs.get('schema'),
                      'response': response.text, 'usage': llm.response_usage(response)})
        write_json(output / f"{case['id']}.calls.json", calls)
        return response

    llm.call = record_call
    for case in selected:
        calls = []
        print(f"Running {case['id']}...", flush=True)
        scope = EvidenceScope(stems=frozenset(spec['papers'][p]['stem'] for p in case['papers']),
                              library_root=str(library))
        started = time.monotonic()
        result = {'id': case['id'], 'operation': case['operation'], 'expectation': case['expectation']}
        try:
            receipt = compose(
                case['title'], case['question'], scope)
            path = output / f"{case['id']}.json"
            write_json(path, receipt.to_dict())
            result.update(status=receipt.status, reason=receipt.reason,
                          receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if receipt.draft is not None:
                (output / f"{case['id']}.md").write_text(render(receipt), encoding='utf-8')
        except Exception as exc:
            # Keep failures in the denominator and the diagnostic record; never
            # relabel an operational failure as scientifically correct abstention.
            result.update(status='operational_error', reason=f'{type(exc).__name__}: {exc}')
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        manifest['results'].append(result)
        write_json(output / 'manifest.json', manifest)
        print(f"  {result['status']}: {result['reason']}", flush=True)
    return int(any(r['status'] == 'operational_error' for r in manifest['results']))


if __name__ == '__main__':
    raise SystemExit(main())
