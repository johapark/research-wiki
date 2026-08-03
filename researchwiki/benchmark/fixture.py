"""Fixture loader for the eval harness — content + retrieval flavors.

Two fixture kinds, selected by file location and `fixture_type:`:

  - **Content** (the original): `benchmark-fixtures/{stem}.yaml`. Declares what
    a thorough wiki page for `papers/{stem}.pdf` should capture — headline
    claims, capabilities, limitations, related papers. No `fixture_type:`
    key (treated as default).

  - **Retrieval**: `benchmark-fixtures/retrieval/{kind}/{slug}.yaml` with
    `fixture_type: claims | pages`. Declares a query and the
    (paper_stem, section, position) anchors that should rank in top-K.

`load_fixture(stem)` returns whichever shape the YAML declares; callers
dispatch on the returned type. `find_fixtures()` recurses into the
retrieval/ subtree so listings include both kinds (path-shaped stems
like `retrieval/claims/foo`).

See `benchmark-fixtures/README.md` and `benchmark-fixtures/retrieval/README.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

import yaml

from ..paths import wiki_root


Importance = Literal["critical", "high", "normal"]
FixtureType = Literal["content", "claims", "pages"]


def _fixtures_dir() -> Path:
    return wiki_root() / "benchmark-fixtures"


@dataclass
class Relation:
    """Optional structured form of a comparison claim. When present, the scorer
    can mechanically check that the page pairs `ratio` with `comparator`,
    catching comparator-drift failures (e.g., "20× vs RCSB" when the source
    says "20× vs pyScoMotif")."""
    subject: str
    ratio: str
    comparator: str


@dataclass
class FixtureItem:
    """One graded item in a fixture — claim, capability, limitation, or
    related-paper entry. The same shape applies to all four; only the section
    name and the optional `relation` field differ in practice."""
    id: str
    importance: Importance
    verbalization: str
    location: str | None = None
    relation: Relation | None = None
    # Related-papers-only field; ignored elsewhere.
    link: str | None = None
    rationale: str | None = None


@dataclass
class ContentFixture:
    paper_stem: str
    paper_type: str
    title: str
    notes: str | None
    headline_claims: list[FixtureItem]
    capabilities: list[FixtureItem]
    limitations: list[FixtureItem]
    related_papers: list[FixtureItem]
    # YYYY-MM-DD when the paper was published. The scorer warns when this
    # precedes the LLM training cutoff — a high score on a contaminated
    # fixture may reflect memorized training data rather than pipeline
    # quality (per SEI's test-set-contamination warning).
    published_at: str | None = None
    source_path: Path = field(default_factory=lambda: Path())

    def all_items(self) -> list[tuple[str, FixtureItem]]:
        """Return [(axis_name, item), ...] across all four axes — convenient
        for the scorer's per-axis loop."""
        out: list[tuple[str, FixtureItem]] = []
        for it in self.headline_claims:
            out.append(("headline_claims", it))
        for it in self.capabilities:
            out.append(("capabilities", it))
        for it in self.limitations:
            out.append(("limitations", it))
        for it in self.related_papers:
            out.append(("related_papers", it))
        return out


def _parse_item(d: dict) -> FixtureItem:
    rel_d = d.get("relation")
    rel = (
        Relation(subject=rel_d["subject"], ratio=rel_d["ratio"], comparator=rel_d["comparator"])
        if rel_d
        else None
    )
    importance = d.get("importance", "normal")
    if importance not in ("critical", "high", "normal"):
        raise ValueError(
            f"fixture item {d.get('id')!r}: importance must be "
            f"'critical' | 'high' | 'normal', got {importance!r}"
        )
    # `id` is required for verbalization items (the scorer keys regression
    # diffs by it). For related_papers entries the link is itself a stable
    # identifier, so we synthesize the id from the wikilink target if not
    # explicitly given — saves the fixture author from typing it twice.
    item_id = d.get("id") or d.get("link")
    if not item_id:
        raise ValueError(
            f"fixture item missing both 'id' and 'link': {d!r}"
        )
    return FixtureItem(
        id=item_id,
        importance=importance,
        verbalization=(d.get("verbalization") or "").strip(),
        location=d.get("location"),
        relation=rel,
        link=d.get("link"),
        rationale=d.get("rationale"),
    )


# ── Retrieval-fixture dataclasses ────────────────────────────────────


@dataclass(frozen=True)
class ExpectedClaim:
    """An anchor in the claims DB, keyed on (paper_stem, section, position) —
    matches the claims-table schema exactly. Used by claim-level retrieval
    fixtures."""
    paper_stem: str
    section: str        # e.g. "key_contributions", "methodology", "results"
    position: int       # 0-indexed within section
    importance: Importance
    rationale: str = ""

    def key(self) -> tuple[str, str, int]:
        return (self.paper_stem, self.section, self.position)


@dataclass(frozen=True)
class ExpectedPage:
    """A page-level anchor. Stem-only; may include category prefix
    (e.g. `synthesis/foo`). Used by page-level retrieval fixtures.

    `expected_rank` is optional: when set, the scorer flags a hit at any
    other rank as a `rank_violation` (recalled but not in the right place).
    Use sparingly — only when one paper is the unambiguous answer."""
    paper_stem: str
    importance: Importance
    rationale: str = ""
    expected_rank: int | None = None


@dataclass(frozen=True)
class NegativeAnchor:
    """A paper-stem that should NOT appear in top-K. Diagnoses
    token-overlap-without-semantics failures (e.g., `cgt/luo-2024-CRISPR-BERT`
    surfacing on a "neural retrieval BERT" query)."""
    paper_stem: str
    rationale: str = ""


@dataclass
class RetrievalFixture:
    fixture_id: str
    fixture_type: Literal["claims", "pages"]
    query: str
    k: int
    expected_claims: list[ExpectedClaim]
    expected_pages: list[ExpectedPage]
    must_not_appear: list[NegativeAnchor]
    notes: str | None = None
    source_path: Path = field(default_factory=lambda: Path())


# Union return type for `load_fixture` dispatch.
Fixture = Union[ContentFixture, RetrievalFixture]


def _parse_expected_claim(d: dict, fixture_id: str) -> ExpectedClaim:
    for required in ("paper_stem", "section", "position", "importance"):
        if required not in d:
            raise ValueError(
                f"{fixture_id}: expected_claim missing {required!r}: {d!r}"
            )
    importance = d["importance"]
    if importance not in ("critical", "high", "normal"):
        raise ValueError(
            f"{fixture_id}: importance must be critical|high|normal, got {importance!r}"
        )
    return ExpectedClaim(
        paper_stem=str(d["paper_stem"]),
        section=str(d["section"]),
        position=int(d["position"]),
        importance=importance,
        rationale=(d.get("rationale") or "").strip(),
    )


def _parse_expected_page(d: dict, fixture_id: str) -> ExpectedPage:
    for required in ("paper_stem", "importance"):
        if required not in d:
            raise ValueError(
                f"{fixture_id}: expected_page missing {required!r}: {d!r}"
            )
    importance = d["importance"]
    if importance not in ("critical", "high", "normal"):
        raise ValueError(
            f"{fixture_id}: importance must be critical|high|normal, got {importance!r}"
        )
    rank = d.get("expected_rank")
    return ExpectedPage(
        paper_stem=str(d["paper_stem"]),
        importance=importance,
        rationale=(d.get("rationale") or "").strip(),
        expected_rank=int(rank) if rank is not None else None,
    )


def _parse_negative_anchor(d: dict, fixture_id: str) -> NegativeAnchor:
    if "paper_stem" not in d:
        raise ValueError(
            f"{fixture_id}: must_not_appear entry missing 'paper_stem': {d!r}"
        )
    return NegativeAnchor(
        paper_stem=str(d["paper_stem"]),
        rationale=(d.get("rationale") or "").strip(),
    )


def _parse_retrieval_fixture(raw: dict, path: Path) -> RetrievalFixture:
    fixture_id = raw.get("fixture_id") or path.stem
    ftype = raw.get("fixture_type")
    if ftype not in ("claims", "pages"):
        raise ValueError(
            f"{path}: retrieval fixture must declare fixture_type: 'claims' "
            f"or 'pages'; got {ftype!r}"
        )
    query = (raw.get("query") or "").strip()
    if not query:
        raise ValueError(f"{path}: retrieval fixture requires non-empty 'query'")
    k = int(raw.get("k", 10))
    expected_claims = [
        _parse_expected_claim(d, fixture_id)
        for d in (raw.get("expected_claims") or [])
    ]
    expected_pages = [
        _parse_expected_page(d, fixture_id)
        for d in (raw.get("expected_pages") or [])
    ]
    must_not_appear = [
        _parse_negative_anchor(d, fixture_id)
        for d in (raw.get("must_not_appear") or [])
    ]

    # Cross-checks: claims fixture must have expected_claims, not expected_pages
    if ftype == "claims":
        if not expected_claims:
            raise ValueError(
                f"{path}: claims fixture requires non-empty 'expected_claims'"
            )
        if expected_pages:
            raise ValueError(
                f"{path}: claims fixture should not declare 'expected_pages'; "
                f"use a pages fixture for that"
            )
    else:  # pages
        if not expected_pages:
            raise ValueError(
                f"{path}: pages fixture requires non-empty 'expected_pages'"
            )
        if expected_claims:
            raise ValueError(
                f"{path}: pages fixture should not declare 'expected_claims'; "
                f"use a claims fixture for that"
            )

    return RetrievalFixture(
        fixture_id=fixture_id,
        fixture_type=ftype,
        query=query,
        k=k,
        expected_claims=expected_claims,
        expected_pages=expected_pages,
        must_not_appear=must_not_appear,
        notes=(raw.get("notes") or "").strip() or None,
        source_path=path,
    )


# ── Loader + finder (dispatch by fixture_type) ───────────────────────


def load_fixture(stem: str) -> Fixture:
    """Load a fixture YAML. Returns ContentFixture or RetrievalFixture
    based on the file's `fixture_type:` field.

    `stem` may include path components for retrieval fixtures, e.g.
    `retrieval/claims/foo` resolves to `benchmark-fixtures/retrieval/claims/foo.yaml`.
    Raises FileNotFoundError if missing, ValueError on schema violations.
    """
    path = _fixtures_dir() / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture not found: {path}. Available: {find_fixtures()}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")

    ftype = raw.get("fixture_type")
    if ftype in ("claims", "pages"):
        return _parse_retrieval_fixture(raw, path)

    # Default: content-coverage fixture (no fixture_type or 'content').
    if ftype not in (None, "content"):
        raise ValueError(
            f"{path}: unknown fixture_type {ftype!r}; expected 'content', "
            f"'claims', or 'pages'"
        )
    required = ("paper_stem", "paper_type", "title")
    missing = [k for k in required if not raw.get(k)]
    if missing:
        raise ValueError(f"{path}: missing required field(s): {missing}")
    published_at = raw.get("published_at")
    if published_at is not None:
        published_at = str(published_at).strip() or None
    return ContentFixture(
        paper_stem=raw["paper_stem"],
        paper_type=raw["paper_type"],
        title=raw["title"],
        notes=(raw.get("notes") or "").strip() or None,
        headline_claims=[_parse_item(d) for d in (raw.get("headline_claims") or [])],
        capabilities=[_parse_item(d) for d in (raw.get("capabilities") or [])],
        limitations=[_parse_item(d) for d in (raw.get("limitations") or [])],
        related_papers=[_parse_item(d) for d in (raw.get("related_papers") or [])],
        published_at=published_at,
        source_path=path,
    )


def find_fixtures() -> list[str]:
    """Return all fixture stems present in `benchmark-fixtures/`, recursively.

    Stems are path-shaped relative to `benchmark-fixtures/`:
      - `kim-2026-...` for content fixtures at the root
      - `retrieval/claims/foo` for retrieval fixtures in subdirs

    Pass any of these directly to `load_fixture(stem)`.
    """
    d = _fixtures_dir()
    if not d.exists():
        return []
    return sorted(
        str(p.relative_to(d).with_suffix(""))
        for p in d.rglob("*.yaml")
    )
