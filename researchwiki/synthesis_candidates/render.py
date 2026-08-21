"""Proposal-markdown rendering — writes one `.ingest/synthesis-candidates/{slug}.md`
per candidate.

The rendered proposal is the artifact a human reviews. Two shapes:

  - `verdict=='new'`: recommend creating a fresh synthesis; includes a
    copy-paste `researchwiki synthesize` command with the in_scope
    members (or all members when judging is off).
  - `verdict=='extend'`: recommend augmenting the nearest synthesis;
    lists which cluster members are already covered vs. missing,
    grouped by scope-fit verdict when the judge ran.

`render_proposal(candidate)` is the sole public entry point.
"""

from __future__ import annotations

import shlex

from .detect import Candidate, DEFAULT_EXTEND


def _verdict_rationale(c: Candidate, key: str) -> str:
    if not c.judged:
        return ""
    for v in c.member_verdicts:
        if v.key == key:
            return v.rationale
    return ""


def render_proposal(c: Candidate) -> str:
    """Render the proposal markdown, branching on verdict.

    For "new": emphasizes the gap and provides a copy-paste `synthesize`
    command listing all members. With B4.5 judging on, the command lists
    only in_scope members; tangential/out_of_scope members are surfaced
    separately so the human can override.

    For "extend": emphasizes the existing synthesis at `nearest_synthesis`,
    shows which members are already in it (no-op) and which are missing
    (the actual diff). With judging on, the missing members are split into
    in_scope (recommended for the table), tangential (recommended for
    tensions section), and out_of_scope (recommended to skip).
    """
    lines = [
        "---",
        f"verdict: {c.verdict}",
        f"members: {len(c.members)}",
        f"density: {c.density:.2f}",
        f"slug: {c.slug}",
        f"judged: {c.judged}",
        "edge_signals:",
        f"  wikilink: {c.edge_signal_counts.get('wikilink', 0)}",
        f"  semantic: {c.edge_signal_counts.get('semantic', 0)}",
        f"  keyword:  {c.edge_signal_counts.get('keyword', 0)}",
    ]
    if c.nearest_synthesis:
        lines.append(f"nearest_synthesis: {c.nearest_synthesis}")
        lines.append(f"nearest_synthesis_overlap: {c.nearest_synthesis_overlap:.2f}")
    if c.judged and c.verdict == "new" and c.judge_topic:
        # Quote the topic to keep YAML strict: titles often contain colons.
        lines.append(f"judge_topic: {c.judge_topic!r}")
    lines.extend(["---", ""])

    if c.verdict == "extend":
        _render_extend_body(c, lines)
    else:
        _render_new_body(c, lines)

    # Common to both: edge breakdown + reject hint.
    lines.extend([
        "",
        "## Edges (top 10 by total weight)",
        "",
    ])
    sorted_edges = sorted(c.edges, key=lambda e: -e.total)[:10]
    for e in sorted_edges:
        lines.append(f"- [[{e.a}]] ↔ [[{e.b}]]  total={e.total:.2f}  ({', '.join(e.signals)})")
    lines.extend([
        "",
        "## To reject this candidate",
        "",
        f"`rm .ingest/synthesis-candidates/{c.slug}.md` (or `rm -rf` the whole dir).",
    ])
    return "\n".join(lines) + "\n"


def _render_new_body(c: Candidate, lines: list[str]) -> None:
    """Body for verdict=='new' — recommend creating a fresh synthesis."""
    lines.extend([
        f"# NEW — synthesis candidate ({len(c.members)} papers)",
        "",
    ])
    if c.judged and c.judge_topic:
        lines.extend([
            f"**LLM-proposed topic:** *{c.judge_topic}*",
            "",
        ])
    if c.nearest_synthesis:
        lines.append(
            f"No existing synthesis substantially covers this cluster. The "
            f"closest match is [[{c.nearest_synthesis}]] at "
            f"{c.nearest_synthesis_overlap:.0%} overlap, below the "
            f"{int(DEFAULT_EXTEND * 100)}% \"extend\" threshold."
        )
    else:
        lines.append("No existing synthesis substantially covers this cluster.")
    lines.append("")

    # When judging is on, group members by verdict so the human's eye lands on
    # the in-scope ones first. When judging is off, fall back to a flat list.
    if c.judged:
        groups: dict[str, list[str]] = {"in_scope": [], "tangential": [], "out_of_scope": []}
        for k in c.members:
            for v in c.member_verdicts:
                if v.key == k:
                    groups.setdefault(v.verdict, []).append(k)
                    break
            else:
                groups.setdefault("unjudged", []).append(k)
        for label, header in [
            ("in_scope", "## Members — in_scope (recommended primary members)"),
            ("tangential", "## Members — tangential (consider as related-work refs)"),
            ("out_of_scope", "## Members — out_of_scope (cluster noise; skip)"),
            ("unjudged", "## Members — unjudged"),
        ]:
            if not groups.get(label):
                continue
            lines.extend([header, ""])
            for k in sorted(groups[label]):
                title = c.titles.get(k, "")[:110]
                rationale = _verdict_rationale(c, k)
                lines.append(f"- [[{k}]] — {title}")
                if rationale:
                    lines.append(f"  - *judge:* {rationale}")
            lines.append("")
    else:
        lines.extend(["## Members", ""])
        for k in c.members:
            title = c.titles.get(k, "")[:120]
            lines.append(f"- [[{k}]] — {title}")
        lines.append("")

    lines.extend([
        "## Common keywords (≥ 2 members across the *whole* cluster)",
        "",
        ", ".join(c.common_keywords[:15]) if c.common_keywords else "_(none — keyword overlap is below floor)_",
        "",
        "## To accept this candidate",
        "",
    ])
    if c.judged and c.judge_topic:
        accept_title = c.judge_topic
    else:
        accept_title = "<your title>"

    # The synthesize command lists in_scope members when judging ran;
    # otherwise all cluster members.
    accept_members = (
        [k for k in c.members
         if any(v.key == k and v.verdict == "in_scope" for v in c.member_verdicts)]
        if c.judged else c.members
    )
    if not accept_members:
        accept_members = c.members  # judge rejected everything; let human decide

    quoted_title = shlex.quote(accept_title)
    lines.extend([
        f"1. Confirm the topic title (LLM proposed: *{accept_title}*).",
        "2. Run:",
        "",
        "   ```",
        "   researchwiki synthesize \\",
        f"       --title {quoted_title} \\",
        f"       --topic-seed {quoted_title} \\",
        "       --papers \\",
    ])
    for i, key in enumerate(accept_members):
        # Keep the category prefix. It disambiguates duplicate stems and lets
        # `synthesize` infer the correct content category from the members.
        continuation = " \\" if i < len(accept_members) - 1 else ""
        lines.append(f"           {shlex.quote(key)}{continuation}")
    lines.append("   ```")
    lines.append("")
    lines.append("3. Edit the scaffolded synthesis page; reference each member "
                 "with a one-line relationship description. For tangential members, "
                 "consider a *Related work* footnote rather than a primary entry.")


def _render_extend_body(c: Candidate, lines: list[str]) -> None:
    """Body for verdict=='extend' — recommend augmenting the nearest synthesis."""
    missing_set = set(c.members_missing_from_nearest)
    already_covered = [k for k in c.members if k not in missing_set]
    lines.extend([
        f"# EXTEND — augment [[{c.nearest_synthesis}]] "
        f"({c.nearest_synthesis_overlap:.0%} of cluster already covered)",
        "",
        f"This cluster has {len(c.members)} members. The existing synthesis "
        f"[[{c.nearest_synthesis}]] already references "
        f"{len(already_covered)} of them — but is missing "
        f"{len(c.members_missing_from_nearest)} members that the cluster "
        f"detector judged thematically related. Recommend editing the existing "
        f"page to add the missing members rather than creating a new synthesis.",
        "",
        "## Already in the synthesis",
        "",
    ])
    if already_covered:
        for k in sorted(already_covered):
            title = c.titles.get(k, "")
            lines.append(f"- [[{k}]] — {title[:120]}")
    else:
        lines.append("_(none yet)_")
    lines.append("")

    # When judging is on, split missing members by verdict so the user knows
    # at a glance which ones the LLM thinks belong in the main structure vs.
    # in tensions vs. should be skipped.
    if c.judged and c.members_missing_from_nearest:
        groups: dict[str, list[str]] = {"in_scope": [], "tangential": [], "out_of_scope": []}
        for k in c.members_missing_from_nearest:
            for v in c.member_verdicts:
                if v.key == k:
                    groups.setdefault(v.verdict, []).append(k)
                    break
            else:
                groups.setdefault("unjudged", []).append(k)
        for label, header in [
            ("in_scope", "## Missing — in_scope (recommended for the main synthesis structure)"),
            ("tangential", "## Missing — tangential (recommended for *tensions* or *open questions* section)"),
            ("out_of_scope", "## Missing — out_of_scope (recommended to SKIP)"),
            ("unjudged", "## Missing — unjudged"),
        ]:
            if not groups.get(label):
                continue
            lines.extend([header, ""])
            for k in sorted(groups[label]):
                title = c.titles.get(k, "")[:110]
                rationale = _verdict_rationale(c, k)
                lines.append(f"- [[{k}]] — {title}")
                if rationale:
                    lines.append(f"  - *judge:* {rationale}")
            lines.append("")
    elif c.members_missing_from_nearest:
        lines.extend([
            "## Missing — add these to the synthesis",
            "",
        ])
        for k in sorted(c.members_missing_from_nearest):
            title = c.titles.get(k, "")
            lines.append(f"- [[{k}]] — {title[:120]}")
        lines.append("")
    else:
        lines.append("_(no missing members — verdict mis-classified, this should be 'covered')_")
        lines.append("")

    lines.extend([
        "## Common keywords (≥ 2 members across the *whole* cluster)",
        "",
        ", ".join(c.common_keywords[:15]) if c.common_keywords else "_(none — keyword overlap is below floor)_",
        "",
        "## To accept this candidate",
        "",
        f"Open `wiki/{c.nearest_synthesis}.md`. For *in_scope* missing members, "
        f"add an entry under the appropriate section with a one-line relationship "
        f"description, cite it in the body, and add a matching footnote under "
        f"`## References` when the page uses footnotes. "
        f"For *tangential* members, add a bullet to the *Tensions* / *Open questions* "
        f"section instead. Skip *out_of_scope* members. Update the page's "
        f"`generated_at:` field once done.",
    ])
