# Whitelist-API lookups — retraction-check, preprint-check, orcid-lookup

Three CLI wrappers around PubMed / bioRxiv / ORCID. Use as needed — none is part of routine ingest. All access is mediated through `researchwiki` (Rule 1 whitelist); never raw `WebFetch`/`WebSearch`.

## retraction-check — PubMed retraction lookup

```bash
researchwiki retraction-check --doi 10.1126/science.1112286
researchwiki retraction-check --all
researchwiki retraction-check --all --json
```

Reads only NLM-canonical `pubtype` (`Retracted Publication`, `Retraction of Publication`); never fetches retraction reasons (prose, behind Rule 1). Record: `{doi, pmid, retracted, is_retraction_notice, pubtypes, pubdate, source: "pubmed", fetched_at}`. Cached to `.web-cache/pubmed_esummary__{PMID}.json`. Preprints not in PubMed: `pmid: null`.

When flagging a retracted paper: record `retracted_at`, `retracted_source: pubmed`, `retracted_fetched`.

## preprint-check — bioRxiv preprint ↔ journal pairing

```bash
researchwiki preprint-check --doi 10.1101/2021.11.18.469088
researchwiki preprint-check --all
researchwiki preprint-check --all --json
```

Key signal: `published_doi` + `published_in_wiki`. Two workflows:

1. **Preprint → journal update**: update YAML `doi:`/`title:`/venue, **keep stem** — preserves back-links.
2. **Duplicate prevention**: `--all` warns if `published_doi` is already a separate page.

Record `doi_source: biorxiv-journal-detection`, `doi_fetched: YYYY-MM-DD`. Recognized prefixes: `10.1101/`, `10.64898/`, `10.31219/`, `10.20944/`.

## orcid-lookup — author disambiguation

```bash
researchwiki orcid-lookup --orcid 0000-0002-3707-9889
researchwiki orcid-lookup --name "John Doench"
researchwiki orcid-lookup --given John --family Doench --json
```

Re-exposes: `given_names`, `family_name`, `credit_name`, `other_names`, `latest_affiliation` = `{organization, role, department, start_year, end_year}`.

Two workflows:

1. **Identity confirmation**: confirm "J. Doench" / "John G. Doench" are the same person. Record `first_author_orcid:` + fetch date.
2. **Affiliation verification**: pick employment where `start_year` ≤ paper year ≤ `end_year`. Record `first_author_affiliation_source: orcid`, `first_author_affiliation_fetched:`.

Limits: not everyone has a public ORCID; many have empty employment. Common names return many candidates — disambiguate against the paper's DOI.
