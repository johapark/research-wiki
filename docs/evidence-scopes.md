# Explicit evidence selections

The Community engine accepts a versioned paper selection independently of any
application's project database. Existing operations retain whole-library behavior
when no scope is supplied. An empty selection allows no evidence.

```python
import json
from pathlib import Path
from researchwiki.evidence import EvidenceScope

scope = EvidenceScope({"smith-2026-example-paper-title"}, library_root=str(Path.cwd()))
Path("selection.json").write_text(json.dumps(scope.to_dict()), encoding="utf-8")
```

```bash
researchwiki search "validation setting" --scope-file selection.json --json
researchwiki claims "validation setting" --scope-file selection.json --include-context --json
researchwiki pdf-search PAPER_STEM "validation setting" --scope-file selection.json --json
researchwiki synthesize --title "Method comparison" --scope-file selection.json --topic-seed "validation setting"
researchwiki export --scope-file selection.json --format bibtex --out references.bib
```

Run from the selected library checkout root. A scoped JSON retrieval response has
`scope` and `results`; unscoped JSON shapes stay unchanged. Bibliography streams
retain their format, while JSON reports include the envelope. Keyword, semantic,
hybrid, See-Also, and optional model reranking retain the selection. Filters apply
before ranking/top-K and before model context construction. Semantic fallbacks
never broaden the selection. PDF lookup rejects outside stems before opening or
indexing the source. Cross-paper and commentary pages are excluded from scoped
page retrieval because their full dependency closure is not established.

A version 1 scope contains these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1` |
| `stems` | Selected canonical bare paper stems |
| `library_root` | Optional local execution root; rejects use in another library |
| `library_id` | Optional stable application-supplied library ID |
| `selection_id`, `selection_revision` | Optional application-supplied selection identity/revision |
| `source_revisions` | Optional complete list of `[stem, page_sha256, pdf_sha256]` triples |
| `fingerprint` | Optional integrity fingerprint; included in emitted result packets |

Serialize with `EvidenceScope.to_dict()` so all required fields are present.
IDs, when supplied, are 32 lowercase hexadecimal characters. The fingerprint
excludes the local mount path to support portable snapshots. When hashes are
supplied, the CLI verifies current source bytes before retrieval. Unsupported,
malformed, stale, or unavailable scope files fail without falling back to the
library. Scope JSON is an integration input, not a secret or authorization token.

Synthesis defaults to the selection's papers. Explicit `--papers` can only narrow
that set; topic-seed claims use the same narrowed selection. Its evidence directory
records `scope.json`. The author must still apply the ordinary grounding/fidelity
gates. Scoped OKF export is refused because exported prose/links may depend on
outside papers. Bibliography export is supported, including empty scopes.

The local read-only MCP `search` and `claims` tools accept an optional `scope_file`
path and re-read it for each request. They preserve their existing result lists.
An application manages its own project/reference decisions, creates these scope
files, and revalidates its membership/revision before reusing outputs. The engine
neither stores projects nor knows their commercial tier. Source selection is not
Team authorization, remote sharing, or a proof that arbitrary prose has no
outside dependencies. Existing paper text can mention other papers.

Exit codes retain each operation's existing contract: retrieval no-hits is 1,
malformed/stale scope input is 1, and unavailable scope/source storage is 2.
Application consumers should pin the engine revision until the selected Python
helpers gain a stable API; see [the compatibility contract](../COMPATIBILITY.md).
