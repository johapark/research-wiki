"""Backfill missing YAML fields on paper pages.

✅ Use when: lint reports `missing_keywords` or `missing_doi` and you want to
   mass-populate the fields without re-ingesting.
❌ Don't use: as part of a normal ingest — the ingest pipeline already
   produces both fields at reconcile / promote time.

Targets (each has its own flags — run `backfill <target> --help`):

  researchwiki backfill keywords   # LLM-generated keywords
  researchwiki backfill doi        # DOI lookup via Semantic Scholar, Crossref fallback

Both accept `--dry-run` and `--limit N`; `keywords` also has `--batch-size N`
(default 10) and `--reindex`. See per-target `--help` for the full list.

Exit code: 0 always (per-page failures are reported inline; a failed lookup
leaves the page unchanged).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

from ..agents.phases import KeywordsOutput, propose_keywords_batch, render_keywords_yaml
from ..fsatomic import write_text_atomic
from ..stems import first_author_surname
from ..wiki import read_page, read_pages


# ---------- shared helpers ----------


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dry-run", action="store_true",
                   help="Compute proposals but don't write the files.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many candidate pages.")


def _normalise_author(s: str) -> str:
    """Lowercase, ASCII-fold, strip non-alnum. For author-match sanity checks."""
    if not s:
        return ""
    ascii_form = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_form.lower())


def _insert_after_key(page_path: Path, new_line: str, after_key: str) -> None:
    """Insert `new_line` immediately after the first frontmatter line whose
    key is `after_key`. No-op with a warning if the anchor is missing."""
    text = page_path.read_text()
    if not text.startswith("---\n"):
        print(f"          → no frontmatter; skipped {page_path.name}", file=sys.stderr)
        return
    end = text.find("\n---\n", 4)
    if end < 0:
        print(f"          → unterminated frontmatter; skipped {page_path.name}", file=sys.stderr)
        return
    fm_lines = text[4:end].split("\n")
    for idx, line in enumerate(fm_lines):
        if line.startswith(f"{after_key}:"):
            fm_lines.insert(idx + 1, new_line)
            write_text_atomic(page_path, f"---\n" + "\n".join(fm_lines) + f"\n---\n" + text[end + 5:])
            return
    print(f"          → no `{after_key}:` anchor; skipped {page_path.name}", file=sys.stderr)


def _replace_or_insert(page_path: Path, key: str, value: str, after_key: str = "year") -> None:
    """Replace an existing `key: ...` line (any value) or insert after `after_key:`."""
    text = page_path.read_text()
    new_line = f"{key}: {value}"
    if re.search(rf"^{key}:\s*.*$", text, re.MULTILINE):
        text = re.sub(rf"^{key}:\s*.*$", new_line, text, count=1, flags=re.MULTILINE)
        write_text_atomic(page_path, text)
        return
    _insert_after_key(page_path, new_line, after_key)


# ---------- KEYWORDS ----------


def _find_keyword_candidates() -> list[Path]:
    """Paper pages with fewer than 3 keyword items."""
    out: list[Path] = []
    for p in read_pages():
        if p.path.parent.name in ("synthesis", "references", "concepts"):
            continue
        if p.fm.get("type", "paper") != "paper":
            continue
        if len(p.list_field("keywords")) >= 3:
            continue
        out.append(p.path)
    return sorted(out)


def _insert_keywords_line(page_path: Path, keywords_line: str) -> None:
    """Insert `keywords: [...]` immediately before the existing `tags:` line
    (schema convention). Fallback: end of frontmatter."""
    text = page_path.read_text()
    if not text.startswith("---\n"):
        print(f"          → no frontmatter; skipped {page_path.name}", file=sys.stderr)
        return
    end = text.find("\n---\n", 4)
    if end < 0:
        print(f"          → unterminated frontmatter; skipped {page_path.name}", file=sys.stderr)
        return
    fm_lines = text[4:end].split("\n")
    insert_at = len(fm_lines)
    for idx, line in enumerate(fm_lines):
        if line.startswith("tags:"):
            insert_at = idx
            break
    fm_lines.insert(insert_at, keywords_line)
    write_text_atomic(page_path, f"---\n" + "\n".join(fm_lines) + f"\n---\n" + text[end + 5:])


def _run_keywords(args: argparse.Namespace) -> int:
    if args.batch_size < 1:
        print("--batch-size must be ≥1", file=sys.stderr)
        return 1

    candidates = _find_keyword_candidates()
    if args.limit:
        candidates = candidates[:args.limit]

    if not candidates:
        print("No paper pages missing keywords. Nothing to backfill.")
        return 0

    n_batches = (len(candidates) + args.batch_size - 1) // args.batch_size
    print(f"Backfilling keywords for {len(candidates)} page(s) "
          f"in {n_batches} batch(es) of up to {args.batch_size}"
          f"{' (dry run)' if args.dry_run else ''}.")
    print()

    written = skipped_low_quality = failed = seen = 0
    t0 = time.time()

    for chunk_start in range(0, len(candidates), args.batch_size):
        chunk = candidates[chunk_start:chunk_start + args.batch_size]
        batch_no = chunk_start // args.batch_size + 1

        items: list[dict] = []
        page_lookup: dict[str, tuple[Path, str]] = {}
        for page_path in chunk:
            seen += 1
            page = read_page(page_path)
            if page is None:
                print(f"[{seen}/{len(candidates)}] {page_path.name} — unreadable; skip")
                failed += 1
                continue
            title = page.str_field("title", "(no title)")
            items.append({
                "key": page.key,
                "title": title,
                "year": page.str_field("year"),
                "body": page.body,
            })
            page_lookup[page.key] = (page_path, title)

        if not items:
            continue

        print(f"[batch {batch_no}/{n_batches}] {len(items)} page(s) → 1 LLM call")
        try:
            results = propose_keywords_batch(items=items)
        except Exception as e:
            print(f"  batch call failed: {e}")
            failed += len(items)
            continue

        for item in items:
            key = item["key"]
            page_path, title = page_lookup[key]
            kw_out = results.get(key, KeywordsOutput(model="(missing)"))
            print(f"    {key} — {title[:60]}")
            if kw_out.model == "(missing)":
                print(f"          → not in batch response; skip")
                failed += 1
                continue
            if kw_out.model == "(failed)":
                print(f"          → call failed")
                failed += 1
                continue
            rendered = render_keywords_yaml(kw_out.keywords)
            if rendered is None:
                print(f"          → only {len(kw_out.keywords)} kept (need ≥5); skip")
                skipped_low_quality += 1
                continue
            print(f"          → {kw_out.keywords}")
            if not args.dry_run:
                _insert_keywords_line(page_path, rendered)
                written += 1

    dt = time.time() - t0
    print()
    print(f"Done in {dt:.1f}s ({n_batches} batch(es), batch size {args.batch_size}).")
    print(f"  Wrote keywords:   {written}")
    print(f"  Low-quality skip: {skipped_low_quality}")
    print(f"  Failed:           {failed}")
    if args.dry_run and not failed:
        print(f"  (dry run — re-run without --dry-run to commit)")

    if args.reindex and written:
        print()
        print("Rebuilding indexes...")
        from . import reindex
        reindex.main([])

    return 0


# ---------- DOI ----------


def _find_doi_candidates() -> list[Path]:
    """Paper pages with no DOI (or DOI is placeholder text)."""
    out: list[Path] = []
    for p in read_pages():
        if p.path.parent.name in ("synthesis", "references", "concepts"):
            continue
        if p.fm.get("type", "paper") != "paper":
            continue
        doi = (p.fm.get("doi") or "").strip().lower()
        if doi and doi not in ("todo", "none", "null"):
            continue
        out.append(p.path)
    return sorted(out)


def _lookup_s2(title: str, first_author: str, wiki_year: int | None) -> tuple[str, str] | None:
    """Return (doi, arxiv_id) if S2 hit passes sanity checks, else None."""
    from ..providers.semantic_scholar import SemanticScholarProvider
    try:
        art = SemanticScholarProvider().search_by_title(title)
    except Exception:
        return None
    if not art:
        return None
    if not _sanity_ok(getattr(art, "year", None),
                      [a.get("name", "") if isinstance(a, dict) else str(a)
                       for a in (getattr(art, "authors", None) or [])],
                      getattr(art, "title", "") or "",
                      first_author, wiki_year, title):
        return None
    doi = (getattr(art, "doi", None) or "").strip()
    ext = getattr(art, "external_ids", None) or {}
    arxiv = ext.get("ArXiv") if isinstance(ext, dict) else None
    # Prefer the canonical arXiv-DOI namespace over publisher aggregators
    # (ResearchGate DOIs, ACM proceedings DOIs for conference-later-journal
    # papers, etc.) — 10.48550/arXiv.X routes to the deposited preprint.
    if arxiv:
        return (f"10.48550/arXiv.{arxiv}", arxiv)
    if doi:
        return (doi, None)
    return None


def _lookup_crossref(title: str, first_author: str, wiki_year: int | None) -> tuple[str, str] | None:
    """Crossref title-search fallback. Returns (doi, None) on sanity-passing hit."""
    import json, subprocess, urllib.parse
    params = {"query.title": title[:200], "rows": "5"}
    if first_author:
        params["query.author"] = first_author
    url = f"https://api.crossref.org/works?{urllib.parse.urlencode(params)}"
    ua = "researchwiki/0.1 (mailto:noreply@example.com)"
    try:
        proc = subprocess.run(["curl", "-sS", "-A", ua, url],
                              capture_output=True, text=True, timeout=30)
        data = json.loads(proc.stdout)
    except Exception:
        return None
    for item in (data.get("message", {}) or {}).get("items", []):
        doi = (item.get("DOI") or "").strip()
        if not doi:
            continue
        year = None
        for k in ("published-print", "published-online", "issued"):
            dp = (item.get(k) or {}).get("date-parts")
            if dp and dp[0] and dp[0][0]:
                year = dp[0][0]
                break
        authors = [f"{a.get('given','')} {a.get('family','')}"
                   for a in (item.get("author") or [])]
        cand_title = " ".join(item.get("title") or [])
        if _sanity_ok(year, authors, cand_title, first_author, wiki_year, title):
            return (doi, None)
    return None


_STOPWORDS = frozenset({"a","an","the","of","for","with","and","or","in","on","at","to",
                        "from","by","as","across","over","all","that","this","these","those","is"})


def _title_tokens(s: str) -> set[str]:
    """Lowercase content-word tokens for Jaccard-style title-match sanity."""
    if not s:
        return set()
    ascii_form = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    toks = re.findall(r"[a-z0-9]+", ascii_form.lower())
    return {t for t in toks if len(t) > 2 and t not in _STOPWORDS}


def _sanity_ok(cand_year: int | None, cand_authors: list[str], cand_title: str,
               first_author: str, wiki_year: int | None, wiki_title: str) -> bool:
    """Guard against wrong matches. Reject unless ALL:
      - wiki first-author's last name appears (normalised) in a result author
      - candidate year is within ±1 of the wiki year (or year is unknown)
      - Jaccard(title tokens) ≥ 0.5, i.e. at least half the content words overlap
        — catches "Liao 2025" false positives where the surname is common
        and Crossref returns an unrelated paper with a similar-sounding title
    """
    fa = _normalise_author(first_author)
    if not fa:
        return False
    if not any(fa in _normalise_author(a) for a in cand_authors):
        return False
    if wiki_year is not None and cand_year is not None:
        if abs(int(cand_year) - int(wiki_year)) > 1:
            return False
    wt, ct = _title_tokens(wiki_title), _title_tokens(cand_title)
    if wt and ct:
        overlap = len(wt & ct) / max(1, min(len(wt), len(ct)))
        if overlap < 0.5:
            return False
    return True


def _run_dois(args: argparse.Namespace) -> int:
    candidates = _find_doi_candidates()
    if args.limit:
        candidates = candidates[:args.limit]

    if not candidates:
        print("No paper pages missing DOI. Nothing to backfill.")
        return 0

    print(f"Backfilling DOIs for {len(candidates)} page(s)"
          f"{' (dry run)' if args.dry_run else ''}.")
    print()

    written = skipped = failed = 0
    t0 = time.time()

    for i, page_path in enumerate(candidates, 1):
        page = read_page(page_path)
        if page is None:
            print(f"[{i}/{len(candidates)}] {page_path.name} — unreadable; skip")
            failed += 1
            continue

        title = page.str_field("title", "")
        authors = page.str_field("authors", "")
        first_author = first_author_surname([authors.split(",")[0]]) if authors else ""
        year_str = page.str_field("year", "")
        try:
            year = int(year_str) if year_str else None
        except ValueError:
            year = None

        print(f"[{i}/{len(candidates)}] {page.key}")
        print(f"          title: {title[:70]}")
        print(f"          author: {first_author or '(none)'}  year: {year or '(none)'}")

        if not title:
            print(f"          → no title in YAML; skip")
            failed += 1
            continue

        hit = _lookup_s2(title, first_author, year)
        source = "s2"
        if not hit:
            hit = _lookup_crossref(title, first_author, year)
            source = "crossref"
        if not hit:
            print(f"          → no confident match (S2 + Crossref); skip")
            skipped += 1
            continue

        doi, arxiv = hit
        print(f"          → doi={doi}{'  arxiv=' + arxiv if arxiv else ''}  ({source})")

        if args.dry_run:
            continue

        _replace_or_insert(page_path, "doi", doi, after_key="year")
        if arxiv:
            _replace_or_insert(page_path, "arxiv_id", f'"{arxiv}"', after_key="doi")
        written += 1

    dt = time.time() - t0
    print()
    print(f"Done in {dt:.1f}s.")
    print(f"  Wrote DOIs:      {written}")
    print(f"  Skipped (no match): {skipped}")
    print(f"  Failed:          {failed}")
    if args.dry_run and (written == 0):
        print(f"  (dry run — re-run without --dry-run to commit)")

    if args.reindex and written:
        print()
        print("Rebuilding indexes...")
        from . import reindex
        reindex.main([])

    return 0


# ---------- CLI ----------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="researchwiki backfill",
        description="Backfill missing YAML fields on paper pages.",
    )
    subs = parser.add_subparsers(dest="target", required=True, metavar="TARGET")

    kw = subs.add_parser("keywords",
                         help="LLM-generated keywords for pages predating the field.")
    _add_common_flags(kw)
    kw.add_argument("--batch-size", type=int, default=10,
                    help="Pages per LLM call (default 10). 1 = legacy per-page.")
    kw.add_argument("--reindex", action="store_true",
                    help="After backfill, run `researchwiki reindex`.")
    kw.set_defaults(func=_run_keywords)

    doi = subs.add_parser("doi",
                          help="DOI lookup via Semantic Scholar → Crossref fallback.")
    _add_common_flags(doi)
    doi.add_argument("--reindex", action="store_true",
                     help="After backfill, run `researchwiki reindex`.")
    doi.set_defaults(func=_run_dois)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)
