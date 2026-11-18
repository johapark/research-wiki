# Standalone dashboard — agent-managed lifecycle

Trigger: the user asks to open, refresh, manage, or stop the browser dashboard,
or an agent session starts/resumes with `.researchwiki/dashboard.json` present.

The dashboard reads canonical Markdown and source-file availability. It never
authors research prose, modifies wiki pages, spends model calls, or rebuilds
indexes. Browser bookmarks and appearance stay in local browser storage.

## Open or reuse

1. From the repository root, run `researchwiki dashboard --status --json` through
   the chat agent's execution tool. In a network-restricted session, use the
   approved permission path for local HTTP access; a denied health probe is not
   proof the server stopped.
2. If `running: true`, reuse the returned URL. Do not start a duplicate server.
3. Otherwise, run `researchwiki dashboard --serve --session <chat-session-label>`
   as one long-lived execution-tool process. Retain its tool session ID so the
   operating agent can monitor it across turns. Wait for the URL, then verify
   it with `--status --json` before opening it with the browser skill.
4. Keep the result open for the user and report the URL. Do not stop a dashboard
   the user is viewing merely because the agent's current reply ends.

On session start/resume, reconnect once using the same health/reuse procedure.
If an interrupted execution left a stale receipt, restart the server. An explicit
stop removes the receipt, so a later session does not undo the user's shutdown.

The running server updates automatically after ingests or Markdown changes;
the browser polls every 30 seconds and also refreshes on returning to the tab.
The Refresh button fetches immediately. Ingestion does not need an extra hook.

## Session ownership

`--session` records a human-readable owner label in the per-repository runtime
receipt under `.researchwiki/dashboard.json`. It does not invent a chat-host
session-end event. `--status --json` reports URL, PID, owner PID, and session
without exposing the control token.

If the chat host supplies a **session-lifetime** owning process, pass
`--owner-pid <pid>` as well: the server shuts down when that process exits.
Never use the PID of a temporary shell, one tool invocation, or a subprocess
that ends before the chat session. Without a reliable lifecycle hook, leave
the server available until the user requests shutdown; do not claim automatic
shutdown at the exact end of a chat.

## Stop or restart

`researchwiki dashboard --stop` verifies repository and server identity, then
sends an authenticated local shutdown request. It never kills an arbitrary PID.
For a code upgrade or changed server configuration, stop the existing server,
wait for the execution session to exit, and start the replacement. A stale
runtime receipt is not an active process; startup replaces it after health
verification. Servers bind only to `127.0.0.1` and serve only wiki Markdown and
PDFs under the content directories.

## Offline export

`researchwiki dashboard` writes `output/dashboard.html`. `--out PATH.html`
chooses another location outside the canonical content trees. The entire library
and reader are embedded, with no CDN or build step; source-PDF and Markdown links
remain relative to the generated file. Regenerate after changes. `--vault NAME`
sets the optional Obsidian link target when the vault name differs from the repo.
