"""Guarded XML parsing for the two whitelisted APIs that answer in XML.

PubMed efetch and the arXiv Atom feed are the only non-JSON responses any
provider reads. The stdlib parser is safe here: Python's expat (≥ 2.4.1) caps
entity amplification, and ElementTree never resolves external entities. The
remaining check is belt-and-braces and cheap — neither API ever declares an
entity, so a body that does is not a response we understand. A `<!DOCTYPE`
line alone is fine: efetch always sends one.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from ._http import StructuredProviderUnavailable


def parse_xml(text: str, *, provider: str) -> ET.Element:
    """Parse a response body, raising `StructuredProviderUnavailable` on garbage.

    A malformed body after HTTP 200 is a provider fault (a truncated transfer,
    a maintenance page), not a bad query, so it reports as an environment
    failure like a JSON parse error does in `curl_json`.
    """
    if "<!ENTITY" in text[:4096]:
        raise StructuredProviderUnavailable(f"{provider} returned XML declaring entities; refusing to parse")
    try:
        return ET.fromstring(text)
    except ET.ParseError as exc:
        raise StructuredProviderUnavailable(f"{provider} returned malformed XML ({exc})") from exc
