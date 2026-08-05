"""Backfill missing YAML fields on existing pages.

✅ Use when: lint reports `missing_keywords` / `missing_doi` / `missing_hook`,
   or pages arrived from another framework without this framework's fields, and
   you want to mass-populate without re-ingesting.
❌ Don't use: as part of a normal ingest — the pipeline already produces these
   at reconcile / author / promote time.

Targets (each has its own flags — run `backfill <target> --help`):

  researchwiki backfill keywords   # LLM-generated keywords (batched)
  researchwiki backfill doi        # DOI lookup via Semantic Scholar, Crossref fallback
  researchwiki backfill hook       # LLM-generated catalog gloss (+ short_name)

All accept `--dry-run` and `--limit N`; `keywords` adds `--batch-size N`
(default 10), `hook` adds `-w/--workers N`, and all three take `--reindex`.

Two derivation styles live here deliberately. `keywords` and `hook` read the
page body — prose that is already PDF-grounded, so no provenance field is needed
under Rule 1. `doi` cannot work that way: an LLM asked for a DOI invents a
plausible one, so it goes to S2/Crossref and every hit clears `_sanity_ok`
(author, year, title overlap) before it is written. When adding a target, decide
which style it is; if the value isn't legible in the page, it needs a lookup and
a sanity check, not a prompt.

Exit code: 0 always (per-page failures are reported inline; a failed lookup or
call leaves the page unchanged).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from ..agents.phases import (
    MIN_KEYWORDS,
    KeywordsOutput,
    propose_keywords_batch,
    render_keywords_yaml,
)
from ..fsatomic import write_text_atomic
from ..stems import first_author_surname
from .. import metadata_sanity
from ..providers.crossref import verify_doi_via_crossref
from ..wiki import read_page, read_pages


# ---------- shared helpers ----------


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dry-run", action="store_true",
                   help="Compute proposals but don't write the files.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many candidate pages.")


# Author/title/year matching lives in `researchwiki.metadata_sanity` so this
# module, `agents.phases.reconcile` and `--verify` all apply one rule. Aliased
# rather than re-implemented; see that module's docstring for why reconcile
# needed it.
_normalise_author = metadata_sanity.normalise_author


def _insert_after_key(page_path: Path, new_line: str, after_key: str) -> None:
    """Insert `new_line` immediately after the first frontmatter line whose
    key is `after_key`. No-op with a warning if the anchor is missing."""
    text = page_path.read_text(encoding="utf-8")
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
    text = page_path.read_text(encoding="utf-8")
    new_line = f"{key}: {value}"
    if re.search(rf"^{key}:\s*.*$", text, re.MULTILINE):
        text = re.sub(rf"^{key}:\s*.*$", new_line, text, count=1, flags=re.MULTILINE)
        write_text_atomic(page_path, text)
        return
    _insert_after_key(page_path, new_line, after_key)


# ---------- KEYWORDS ----------


def _find_keyword_candidates() -> list[Path]:
    """Paper pages with fewer than `MIN_KEYWORDS` keyword items.

    Was `>= 3` while `render_keywords_yaml` refuses to write below 5, which
    meant a page holding 3-4 keywords was never even *selected* here — and lint
    didn't flag it either. Three thresholds, one dead zone, nothing able to move
    a page out of it. All four sites now read `MIN_KEYWORDS`.
    """
    out: list[Path] = []
    for p in read_pages():
        if p.path.parent.name in ("synthesis", "references", "concepts"):
            continue
        if p.fm.get("type", "paper") != "paper":
            continue
        if len(p.list_field("keywords")) >= MIN_KEYWORDS:
            continue
        out.append(p.path)
    return sorted(out)


def _insert_keywords_line(page_path: Path, keywords_line: str) -> None:
    """Insert `keywords: [...]` immediately before the existing `tags:` line
    (schema convention). Fallback: end of frontmatter."""
    text = page_path.read_text(encoding="utf-8")
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
                print(f"          → only {len(kw_out.keywords)} kept "
                      f"(need ≥{MIN_KEYWORDS}); skip")
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


# ---------- HOOK (+ short_name) ----------


# A page needs some prose for a gloss to summarise. Below this the model would be
# inventing rather than reading, so we skip and say so.
_MIN_BODY_CHARS = 200


def _find_hook_candidates() -> list[Path]:
    """Catalog pages with no `hook:`, straight from lint's own check.

    Delegated rather than re-filtered locally (the way `_find_keyword_candidates`
    does) because "needs a hook" is broader than "is a paper": synthesis, idea,
    concept and reference pages all get an `index.md` bullet and so all need the
    field. Sharing the predicate means `backfill hook` and `lint`'s
    `missing_hook` can't drift apart about which pages have a gap.
    """
    from .lint.walk import all_pages, page_key
    from .lint.yaml_checks import find_missing_hook

    pages = all_pages()
    pages_fm: dict[Path, dict] = {}
    for p in pages:
        pg = read_page(p)
        pages_fm[p] = pg.fm if pg else {}
    wanted = set(find_missing_hook(pages, pages_fm))
    return sorted(p for p in pages if page_key(p) in wanted)


def _insert_hook_lines(page_path: Path, hook: str, short_name: str) -> bool:
    """Write `hook:` (and `short_name:` when supplied) into the frontmatter.

    Anchors after `short_name:` if present, else `title:` — both single-line
    scalars on every page, so the insert never lands inside a block scalar (e.g.
    concept pages' `concept_thesis: |`). Returns True when the file was written.
    """
    from ..agents.promote import _yaml_dq

    text = page_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        print(f"          → no frontmatter; skipped {page_path.name}", file=sys.stderr)
        return False
    end = text.find("\n---\n", 4)
    if end < 0:
        print(f"          → unterminated frontmatter; skipped {page_path.name}", file=sys.stderr)
        return False

    fm_lines = [ln for ln in text[4:end].split("\n")
                if not re.match(r"^hook\s*:", ln)]
    new_lines = [f"hook: {_yaml_dq(hook)}"]
    if short_name:
        new_lines.append(f"short_name: {short_name}")

    anchor = None
    for key in ("short_name", "title"):
        for idx, line in enumerate(fm_lines):
            if line.startswith(f"{key}:"):
                anchor = idx
                break
        if anchor is not None:
            break
    if anchor is None:
        print(f"          → no `short_name:`/`title:` anchor; skipped {page_path.name}",
              file=sys.stderr)
        return False

    fm_lines[anchor + 1:anchor + 1] = new_lines
    write_text_atomic(page_path, "---\n" + "\n".join(fm_lines) + "\n---\n" + text[end + 5:])
    return True


def _propose_one_hook(page_path: Path, want_short: bool) -> dict:
    """One LLM call for one page. Returns a plain dict so it can cross threads."""
    from ..agents.phases import propose_short_name

    page = read_page(page_path)
    if page is None:
        return {"path": page_path, "key": page_path.name, "status": "failed",
                "reason": "unreadable / no frontmatter"}
    key = page.key
    if len((page.body or "").strip()) < _MIN_BODY_CHARS:
        return {"path": page_path, "key": key, "status": "skipped",
                "reason": f"body under {_MIN_BODY_CHARS} chars — nothing to gloss"}

    out = propose_short_name(
        metadata={"title": page.str_field("title"), "year": page.str_field("year"),
                  "authors": page.str_field("authors")},
        draft_text=page.body or "",
    )
    if out.model == "(failed)":
        return {"path": page_path, "key": key, "status": "failed",
                "reason": "LLM call failed"}
    has_short = bool(page.str_field("short_name").strip())
    short = out.name if (want_short and out.name != "TODO" and not has_short) else ""
    if not out.hook:
        return {"path": page_path, "key": key, "status": "skipped",
                "reason": "no usable HOOK in the response"}
    return {"path": page_path, "key": key, "status": "ok",
            "hook": out.hook, "short_name": short,
            "title": page.str_field("title", "(no title)")}


def _run_hooks(args: argparse.Namespace) -> int:
    if args.workers < 1:
        print("--workers must be ≥1", file=sys.stderr)
        return 1

    candidates = _find_hook_candidates()
    if args.limit:
        candidates = candidates[:args.limit]

    if not candidates:
        print("No pages missing a hook. Nothing to backfill.")
        return 0

    print(f"Backfilling hooks for {len(candidates)} page(s) "
          f"with {args.workers} worker(s)"
          f"{' (dry run)' if args.dry_run else ''}.")
    print()

    written = skipped = failed = 0
    t0 = time.time()

    # The Anthropic SDK is sync, so threads give IO concurrency. Each call is
    # independent and only its own page is written, so there's nothing to race.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda p: _propose_one_hook(p, not args.no_short_name), candidates))

    for i, res in enumerate(results, 1):
        print(f"[{i}/{len(candidates)}] {res['key']}")
        if res["status"] != "ok":
            print(f"          → {res['reason']}")
            if res["status"] == "failed":
                failed += 1
            else:
                skipped += 1
            continue
        print(f"          → hook ({len(res['hook'])}): {res['hook'][:150]}"
              f"{'…' if len(res['hook']) > 150 else ''}")
        if res["short_name"]:
            print(f"          → short_name: {res['short_name']}")
        if args.dry_run:
            continue
        if _insert_hook_lines(res["path"], res["hook"], res["short_name"]):
            written += 1
        else:
            skipped += 1

    dt = time.time() - t0
    print()
    print(f"Done in {dt:.1f}s.")
    print(f"  Wrote hooks:     {written}")
    print(f"  Skipped:         {skipped}")
    print(f"  Failed:          {failed}")
    if args.dry_run and written == 0:
        print(f"  (dry run — re-run without --dry-run to commit)")
    if not args.dry_run and written:
        print()
        print("Frontmatter changed — run `researchwiki db rebuild` "
              "(and `reindex`, or pass --reindex next time).")

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
        if (doi and doi not in ("todo", "none", "null")
                and not metadata_sanity.is_placeholder_doi(doi)):
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


_title_tokens = metadata_sanity.title_tokens
_sanity_ok = metadata_sanity.sanity_ok


# Words dropped when building a venue acronym: `International Conference on
# Machine Learning` must reduce to `icml`, not `icoml`.
_VENUE_SKIP_WORDS = frozenset({"of", "on", "the", "and", "in", "for", "a", "an"})


def _venue_agrees(page_venue: str, provider_venue: str) -> bool:
    """Do two venue strings plausibly name the same venue?

    Three passes, because the same venue is written three different ways across
    a corpus and a naive comparison is dominated by noise rather than by findings:

      1. **Substring** after normalisation — the easy majority.
      2. **Abbreviation**, token-wise with prefix tolerance: every word of the
         shorter side must prefix the matching word of the longer one, so
         `Nat Biotechnol` agrees with `Nature Biotechnology`. This is how
         provider records for journals are usually abbreviated.
      3. **Acronym**, tested as a subsequence: the initials of the provider's
         significant words (`icml`, `nips`) against the page's letters, which
         tolerates the expanded forms people write by hand (`NeurIPS`).

    Loose on purpose: this backs an advisory report, where a false "agrees" costs
    a missed nudge and a false "disagrees" costs a reviewer's attention. Measured
    on the corpus, pass 1 alone reported same-venue pairs like `NeurIPS 2025` vs
    `Neural Information Processing Systems`.
    """
    a = re.sub(r"[^a-z]", "", page_venue.lower())
    b = re.sub(r"[^a-z]", "", provider_venue.lower())
    if not a or not b:
        return True
    if a in b or b in a:
        return True

    wa = [w for w in re.findall(r"[a-z]+", page_venue.lower()) if w not in _VENUE_SKIP_WORDS]
    wb = [w for w in re.findall(r"[a-z]+", provider_venue.lower()) if w not in _VENUE_SKIP_WORDS]
    if wa and wb and len(wa) == len(wb):
        if all(x.startswith(y) or y.startswith(x) for x, y in zip(wa, wb)):
            return True

    for words, other in ((wb, a), (wa, b)):
        acronym = "".join(w[0] for w in words)
        if len(acronym) >= 3:
            it = iter(other)
            if all(ch in it for ch in acronym):
                return True
    return False


def _pages_with_doi() -> list[Path]:
    """Paper pages that already carry a DOI — the `--verify` work list.

    Exact complement of `_find_doi_candidates`, which selects the *missing*-DOI
    pages. A page whose DOI is wrong is invisible to every existing check: `lint`
    only reports `missing_doi`, and a DOI that resolves looks identical to one
    that belongs.
    """
    out: list[Path] = []
    for p in read_pages():
        if p.path.parent.name in ("synthesis", "references", "concepts"):
            continue
        if p.fm.get("type", "paper") != "paper":
            continue
        doi = (p.fm.get("doi") or "").strip().lower()
        if not doi or doi in ("todo", "none", "null"):
            continue
        if metadata_sanity.is_placeholder_doi(doi):
            continue          # reported by `_find_doi_candidates` as missing
        out.append(p.path)
    return sorted(out)


def _run_doi_verify(args: argparse.Namespace) -> int:
    """Check that each page's recorded DOI actually resolves to that paper.

    Read-only by design — it reports and never rewrites. A DOI mismatch has more
    than one possible cause (wrong DOI scavenged at ingest, a page retitled after
    the fact, a preprint/journal pair sharing one page) and picking among them is
    a reviewer's call, not something to guess at scale.

    Motivated by `reconcile`'s URL-DOI hunt having adopted candidates on
    resolvability alone until 2026-08-05: it validated that a DOI *exists*, not
    that it is this paper's. On the DeepSpCas9 ingest it offered the Adam
    optimizer's arXiv DOI, scavenged from the reference list, and only an accident
    of Crossref coverage stopped it being written. Any page ingested before that
    fix may carry a silently wrong DOI, and this is how you find out.

    Verdicts: `ok` (provider record matches), `MISMATCH` (resolves to a different
    paper — `metadata_sanity` says why), `unresolved` (provider has no record;
    not evidence of a wrong DOI, commonly a fresh deposit), `skipped` (page lacks
    the title/author needed to judge).
    """
    pages = _pages_with_doi()
    if args.limit:
        pages = pages[:args.limit]
    if not pages:
        print("No paper pages with a DOI. Nothing to verify.")
        return 0

    from ..providers.semantic_scholar import SemanticScholarProvider
    provider = SemanticScholarProvider()

    print(f"Verifying {len(pages)} recorded DOI(s) against Semantic Scholar.")
    print()
    ok = mismatch = unresolved = skipped = malformed = 0
    bad: list[tuple[str, str, str]] = []
    year_drift: list[tuple[str, str, str]] = []
    bad_authors: list[tuple[str, str, str]] = []
    provider_conflict: list[tuple[str, str, str]] = []
    venue_drift: list[tuple[str, str, str]] = []
    t0 = time.time()

    for i, page_path in enumerate(pages, 1):
        page = read_page(page_path)
        if page is None:
            skipped += 1
            continue
        doi = page.str_field("doi", "").strip()
        title = page.str_field("title", "")
        authors = page.str_field("authors", "")
        first_author = first_author_surname([authors.split(",")[0]]) if authors else ""
        year_str = page.str_field("year", "")
        year = int(year_str) if year_str.isdigit() else None
        key = f"{page_path.parent.name}/{page_path.stem}"

        if not title or not first_author:
            skipped += 1
            continue
        try:
            art = provider.get_by_doi(doi)
        except Exception:
            art = None
        if art is None:
            unresolved += 1
            continue

        cand_authors = [a.get("name", "") if isinstance(a, dict) else str(a)
                        for a in (getattr(art, "authors", None) or [])]
        why = metadata_sanity.reject_reason(
            getattr(art, "year", None), cand_authors,
            getattr(art, "title", "") or "", first_author, year, title,
        )
        if why and why.startswith("year "):
            # Author AND title matched, so it is the same paper; only the year
            # disagrees. S2 frequently reports the preprint year against a page
            # recording the journal year — four such pairs on the first sweep,
            # off by two. That is a year problem (`lint`'s `stem_year_drift`
            # territory), not a wrong DOI, so it must not read as one.
            ok += 1
            year_drift.append((key, doi, why))
        elif why and "authors field is malformed" in why:
            # The page can't be judged, and the defect is the page's, not the DOI's.
            malformed += 1
            bad_authors.append((key, doi, why))
        elif why:
            # Second opinion before accusing the page. One provider's DOI→record
            # mapping is not authoritative: on the first sweep, S2 resolved
            # `10.18653/v1/2022.findings-naacl.6` to a different NAACL Findings
            # paper, while Crossref — the registration agency's own data —
            # returned MultiVerS, the paper the page actually is. Reporting on S2
            # alone would have sent a reviewer to correct a correct DOI.
            cr = verify_doi_via_crossref(doi)
            cr_why = None
            if cr:
                cr_why = metadata_sanity.reject_reason(
                    cr.get("year"), list(cr.get("authors") or []),
                    cr.get("title") or "", first_author, year, title,
                )
            if cr and not cr_why:
                ok += 1
                provider_conflict.append((key, doi, why))
            else:
                mismatch += 1
                bad.append((key, doi, why))
                print(f"[{i}/{len(pages)}] MISMATCH {key}")
                print(f"          doi={doi}")
                print(f"          s2: {why}")
                if cr_why:
                    print(f"          crossref agrees it's wrong: {cr_why}")
                else:
                    print(f"          crossref has no record either")
        else:
            ok += 1
            # Same provider record, so the venue comparison is free here. A
            # purely local check cannot make it (see `lint`'s `venue_suspect`).
            page_venue = page.str_field("venue", "").strip()
            prov_venue = (getattr(art, "venue", None) or "").strip()
            if page_venue and prov_venue:
                if not _venue_agrees(page_venue, prov_venue):
                    venue_drift.append((key, page_venue, prov_venue))

    print()
    print(f"Done in {time.time() - t0:.1f}s.")
    print(f"  matches:        {ok}")
    print(f"  MISMATCHES:     {mismatch}")
    print(f"  year-only drift:{len(year_drift)}  (same paper, provider year differs)")
    print(f"  provider conflict: {len(provider_conflict)}  "
          f"(S2 disagreed, Crossref confirmed the page — DOI is fine)")
    print(f"  unjudgeable:    {malformed}  (page's `authors:` field is malformed)")
    print(f"  unresolved:     {unresolved}  (provider has no record — not a verdict)")
    print(f"  skipped:        {skipped}  (no title/author to judge against)")
    if bad:
        print()
        print("Wrong DOIs to review:")
        for key, doi, why in bad:
            print(f"  - {key}\n      {doi} — {why}")
    if provider_conflict:
        print()
        print(f"S2 and Crossref disagree on {len(provider_conflict)} DOI(s). Crossref "
              f"matched the page, so the DOI is kept; S2's record is the odd one out:")
        for key, doi, why in provider_conflict[:15]:
            print(f"  - {key}: {doi} — s2 said {why}")
    if year_drift:
        print()
        print(f"Year disagrees with the provider on {len(year_drift)} page(s) "
              f"(advisory — author and title matched, so the DOI is right):")
        for key, doi, why in year_drift[:25]:
            print(f"  - {key}: {why}")
    if bad_authors:
        print()
        print(f"Malformed `authors:` on {len(bad_authors)} page(s) — the DOI could not "
              f"be judged. Fix the field (`Chuai et al.` → the real author list):")
        for key, doi, why in bad_authors:
            print(f"  - {key}: {why}")
    if venue_drift:
        print()
        print(f"Venue disagrees with the provider on {len(venue_drift)} page(s) "
              f"(advisory; the provider is usually right):")
        for key, pv, sv in venue_drift[:25]:
            print(f"  - {key}: page={pv!r} provider={sv!r}")
    # A mismatch is a finding, not a tool failure: exit 1 so a scripted caller
    # notices, matching the exit-code contract's "no-result-where-expected" slot.
    return 1 if bad else 0


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
        description="Backfill missing YAML fields on existing pages.",
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
    doi.add_argument("--verify", action="store_true",
                     help="Instead of filling missing DOIs, check the DOIs already "
                          "recorded: does each resolve to that paper? Read-only. "
                          "Exits 1 if any mismatch is found.")
    doi.set_defaults(func=lambda a: _run_doi_verify(a) if a.verify else _run_dois(a))

    hook = subs.add_parser("hook",
                           help="LLM-generated catalog gloss (+ short_name) from page prose.")
    _add_common_flags(hook)
    hook.add_argument("-w", "--workers", type=int, default=4,
                      help="Concurrent LLM calls (default 4). Lower for rate-limited providers.")
    hook.add_argument("--no-short-name", action="store_true",
                      help="Write only hook:, never add a missing short_name:.")
    hook.add_argument("--reindex", action="store_true",
                      help="After backfill, run `researchwiki reindex`.")
    hook.set_defaults(func=_run_hooks)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)
