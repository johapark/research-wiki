"""Disposable archive acceptance, including a minimal installed-wheel environment.

This exercises record primitives with synthetic opaque content. It is not a paper
ingestion, provider fixture, or claim-quality evaluation.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-installed', action='store_true')
    args = parser.parse_args()
    import researchwiki
    from researchwiki.records import (
        Artifact, AttemptFinished, AttemptSpec, CapturePolicy, CheckpointStale,
        Producer, RecordCorrupt, RecordStore, SourceRef, extraction_requirements,
    )
    from researchwiki.records.jsonio import canonical_bytes
    if args.expect_installed:
        assert Path(researchwiki.__file__).is_relative_to(Path(sys.prefix)), researchwiki.__file__
        assert 'site-packages' in Path(researchwiki.__file__).parts
    with tempfile.TemporaryDirectory(prefix='researchwiki-records-acceptance-') as tmp:
        root = Path(tmp)
        original = root / 'input.pdf'
        original.write_bytes(b'%PDF-synthetic opaque archive source; no PDF parsing is claimed')
        store = RecordStore.create(root / 'archive-v1')
        pdf = store.put_file(original)
        original.write_bytes(b'%PDF-different source version')
        changed = store.put_file(original)
        assert pdf.sha256 != changed.sha256
        text = store.put_blob(b'Synthetic source text.\n\nSecond block.',
                              media_type='text/plain', encoding='utf-8')
        source = SourceRef(document=pdf, role='original', representation=text)
        stamp = datetime.now(timezone.utc).isoformat()
        spec = AttemptSpec(attempt_id=str(uuid.uuid4()), policy=CapturePolicy(mode='required'),
                           producer=Producer(name='synthetic-archive-acceptance', version='1'),
                           requirements=extraction_requirements(), created_at=stamp)
        store.start_attempt(spec)
        raw_response = b'```json\n{"claims":[' + b'full unparsed content ' * 100 + b'\n'
        payloads = {
            'source.original': pdf,
            'source.representation': text,
            'target.request': store.put_json({'synthetic_request': True}),
            'target.response': store.put_blob(raw_response),
            'target.result': store.put_json({'synthetic_fixture': 'not real phase output'}),
            'target.trace': store.put_json({'synthetic_trace': 'malformed response retained'}),
        }
        previous = ()
        for requirement in spec.requirements:
            value = Artifact(artifact_id=uuid.uuid4().hex, attempt_id=spec.attempt_id,
                             kind=requirement.kind, payload=payloads[requirement.slot],
                             producer=spec.producer, created_at=stamp, origin='synthetic',
                             sources=(source,), parents=previous)
            store.record_artifact(requirement.slot, value)
            previous = (value.artifact_id,)
        terminal = AttemptFinished(execution='finished', outcome='draft_ready',
                                   publication='not_published', projections='not_applicable')
        finished = store.finish_attempt(spec.attempt_id, terminal)
        assert finished.capture_state == 'complete'
        assert store.verify(spec.attempt_id).event_count == 8
        assert store.read_blob(payloads['target.response']) == raw_response
        checkpoint = store.root / 'attempts' / spec.attempt_id / 'manifest.json'
        checkpoint.unlink()
        assert store.verify(spec.attempt_id).checkpoint == 'missing'
        try:
            store.read_attempt(spec.attempt_id)
        except CheckpointStale:
            pass
        else:
            raise AssertionError('missing checkpoint was silently accepted')
        assert store.repair_checkpoint(spec.attempt_id) == finished
        destination = root / 'relocated-v1'
        shutil.move(store.root, destination)
        reopened = RecordStore(destination)
        assert reopened.read_attempt(spec.attempt_id) == finished
        assert reopened.read_blob(pdf) != original.read_bytes()
        assert reopened.read_blob(changed) == original.read_bytes()
        checkpoint = destination / 'attempts' / spec.attempt_id / 'manifest.json'
        checkpoint.write_bytes(canonical_bytes({**finished.to_dict(), 'schema_version': 99}))
        before = checkpoint.read_bytes()
        try:
            reopened.repair_checkpoint(spec.attempt_id)
        except RecordCorrupt:
            pass
        else:
            raise AssertionError('future-version checkpoint was replaced')
        assert checkpoint.read_bytes() == before
        assert not any(p.suffix == '.db' for p in root.rglob('*'))
    print(json.dumps({'status': 'passed', 'scope': 'synthetic archive primitives',
                      'provider_calls': 0, 'corpus_databases': 0}))


if __name__ == '__main__':
    main()
