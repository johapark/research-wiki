# Security Policy

## Supported versions

This project is pre-1.0 (`0.1.0`, Development Status :: 3 - Alpha). Only the
latest `main` receives fixes; there are no backports.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's private vulnerability reporting: the repository's **Security** tab →
**Report a vulnerability**. That opens a private advisory visible only to the
maintainer.

Include what you'd expect: affected version/commit, reproduction steps, and the
impact you think it has. As a solo-maintained alpha project there is no formal
SLA — expect a best-effort response, and please allow time before any public
disclosure.

## What this project touches

Knowing the actual surface makes reports more useful.

### Credentials

LLM provider keys are read from the environment, typically via a local `.env`:

- `OPENAI_API_KEY` — OpenAI and any OpenAI-compatible backend (LM Studio, vLLM,
  Upstage, Gemini's OpenAI shim, …)
- `ANTHROPIC_API_KEY` — Anthropic and Anthropic-compatible third parties

`.env` and named `.env.*` profiles are gitignored and no key is committed.
**Never commit one**, restrict each credential file with `chmod 600`, and
prefer `git add <path>` over `git add -A` so a stray local file can't be swept
in. Select a named profile explicitly with
`researchwiki --env-file .env.NAME COMMAND`; this avoids accidentally loading
the root profile for a different provider.

⚠️ **`RW_LLM_BASE_URL` and a model config's `base_url:` redirect where your key
is sent.** The environment variable wins when both are present, and the key
travels as a Bearer token to whatever host is selected. Treat both like
credential settings: only use endpoints you trust. The same caution applies to
`ANTHROPIC_BASE_URL` when using Anthropic-compatible third parties. Both env
overrides and YAML endpoints require HTTPS remotely; plain HTTP is allowed only
on loopback, and credential-bearing or structurally unsafe URLs are rejected
before preflight or transmission. A missing explicit `RW_MODELS_CONFIG`, or any
selected model config that exists but is unreadable, malformed, incomplete, or
has an ambiguous endpoint, fails closed rather than silently switching to
another provider. The built-in OpenAI route is used only when the implicit
`config/models.yaml` is absent.

### What leaves your machine

Ingesting a paper is **not** a local-only operation:

- **Paper text goes to your configured LLM provider.** Extracted sections and
  claims are sent as prompt content to whichever model `RW_MODELS_CONFIG`
  selects. If you're working with confidential or unpublished manuscripts,
  choose a local backend (LM Studio / vLLM / llama.cpp) or a provider whose data
  policy you accept.
- **Structural metadata goes to whitelisted APIs**: Semantic Scholar, Crossref,
  PubMed, bioRxiv/medRxiv, ORCID — DOIs, titles, and identifiers, cached under
  `.s2-cache/`, `.crossref-cache/`, `.web-cache/`.

Your wiki content itself (`wiki/`, `papers/`, `inbox/`) is gitignored and never
uploaded by this project.

### PDFs are untrusted input

Ingested PDFs are attacker-influenceable content that ends up inside LLM prompts,
so **prompt injection via a crafted PDF is a real class of risk**. A malicious
document could try to steer the authoring model into writing false content or
emitting attacker-chosen text into a wiki page.

The pipeline's grading and grounding gates, sandbox fallback
(`.agent-output/`), and human review of promotion all reduce the blast radius,
but **none of them is a security boundary**. Ingest PDFs you obtained from
sources you trust, and review promoted pages.

### MCP server

`researchwiki mcp-serve` is **read-only by design** — search, claims, and
check-grounding only. It refuses ingest/synthesize/evolve/lint-fix; write
operations stay on the CLI where they're reviewable.

It speaks **stdio with no authentication**, which is intentional for local
clients (Claude Desktop, IDE integrations). It is **not** hardened for remote or
multi-tenant exposure — don't put it behind a network transport without
designing auth first. It serves whatever is in your local wiki to whatever client
you configure.

### Local data

- The state DB lives at
  `~/.local/share/researchwiki/repos/<repo>-<hash>/state.db` — scoped to the
  checkout's absolute path, so moving the checkout opens a fresh, empty DB.
  Pin it with `RESEARCHWIKI_DB_PATH` to opt out of path-keying (an old flat
  `~/.local/share/researchwiki/state.db` is seeded from once, non-destructively,
  for repos that predate the scoping). It's derived and rebuildable via
  `researchwiki db rebuild` — **except** `ingest_iterations`, which holds
  non-reconstructable cost/telemetry history.
- `researchwiki db query` is enforced read-only (`mode=ro` plus
  `PRAGMA query_only = ON`), so ad-hoc SQL cannot mutate the DB.

## Out of scope

These are quality or operational issues, not vulnerabilities:

- The LLM writing inaccurate, incomplete, or overstated wiki content. That's what
  `check-grounding`, `grade synthesis`, and the promotion gates exist to catch —
  please file a normal issue.
- Provider rate limits, quota errors, or transient HTTP failures.
- Cost overruns from model configuration.
