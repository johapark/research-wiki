"""Corpus-availability checks for retrieval fixtures."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from ..db import get_connection
from .fixture import RetrievalFixture


@dataclass(frozen=True)
class FixtureAvailability:
    missing_expected: list[str]
    inactive_negatives: list[str]

    @property
    def available(self) -> bool:
        return not self.missing_expected


def validate_fixture_anchors(fixture: RetrievalFixture) -> FixtureAvailability:
    """Resolve expected anchors before absence can be scored as a miss."""
    if fixture.fixture_type == "claims":
        conn = get_connection()
        try:
            claim_keys = {
                (str(row[0]), str(row[1]), int(row[2]))
                for row in conn.execute(
                    "SELECT paper_stem, section, position FROM claims "
                    "WHERE is_cross_ref = 0"
                )
            }
        finally:
            conn.close()
        present_stems = {key[0] for key in claim_keys}
        missing = [
            f"{expected.paper_stem}§{expected.section}#{expected.position}"
            for expected in fixture.expected_claims
            if expected.key() not in claim_keys
        ]
    else:
        from ..wiki import read_pages

        keys = {page.key for page in read_pages()}
        present_stems = {key.split("/", 1)[-1] for key in keys}
        missing = [
            expected.paper_stem for expected in fixture.expected_pages
            if expected.paper_stem not in keys
            and expected.paper_stem.split("/", 1)[-1] not in present_stems
        ]
    inactive = [
        anchor.paper_stem for anchor in fixture.must_not_appear
        if anchor.paper_stem.split("/", 1)[-1] not in present_stems
    ]
    return FixtureAvailability(missing, inactive)


def preflight_or_report(fixture: RetrievalFixture, *, as_json: bool) -> bool:
    """Print actionable availability diagnostics; return whether scoring may run."""
    availability = validate_fixture_anchors(fixture)
    if not availability.available:
        if as_json:
            print(json.dumps({
                "fixture_id": fixture.fixture_id,
                "status": "unavailable",
                "missing_expected": availability.missing_expected,
                "inactive_negatives": availability.inactive_negatives,
            }, indent=2))
        else:
            print(
                f"retrieval fixture unavailable: {fixture.fixture_id}: expected "
                "anchor(s) absent from the current corpus: "
                + ", ".join(availability.missing_expected),
                file=sys.stderr,
            )
        return False
    if availability.inactive_negatives:
        print(
            "warning: inactive must_not anchor(s), absent from corpus: "
            + ", ".join(availability.inactive_negatives),
            file=sys.stderr,
        )
    return True
