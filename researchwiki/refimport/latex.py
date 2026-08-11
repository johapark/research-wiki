"""LaTeX-escape removal for BibTeX values.

Defensive, not central. A real 532-item ReadCube BibTeX export contains **zero**
backslashes, `$` or braces in its titles — it writes raw Unicode instead. Zotero,
JabRef and hand-maintained `.bib` files do escape, so a title arriving as
`Gr{\\"u}newald` must not reach `derive_stem` with its markup intact: the
`[^a-z0-9-]` pass would delete the braces and backslash and leave `grunewald`
spelled `gr"unewald` → `grunewald` by luck, or `{CRISPR}` spelled `crispr` by
luck, and neither is luck worth relying on.

What this module does *not* do is Unicode normalization. Dashes, accented
letters and NBSP are `stems.strip_diacritics`' job, because stem and slug
derivation must agree and that is where they meet. Everything here is strictly
about LaTeX markup.
"""

from __future__ import annotations

import re

#: `\'e` / `\'{e}` / `{\'e}` — an accent command applied to one letter. The
#: letter is kept and the accent dropped; `strip_diacritics` would fold the
#: precomposed character to the same ASCII base anyway, so resolving to the
#: bare letter here loses nothing and avoids a 200-row accent table.
#:
#: The class holds **only** accent commands that cannot be confused with
#: something more common. Three exclusions, each from a real failure:
#:
#:   - `a`, `o` — `\aa` and `\o` are *letter* commands. Admitting their initials
#:     made `H{\aa}kan` parse as accent-`\a` applied to `a`, giving `Hakan`.
#:   - `t` — the tie accent, dropped entirely. It collides with the whole
#:     `\text…` family (`\textendash`, `\textbf`), which is orders of magnitude
#:     more common in bibliographic data, and its real form `\t{oo}` spans two
#:     letters so this single-letter pattern would mangle it regardless.
_ACCENT_CMD = re.compile(r"\\[`'^\"~=.uvHcdbrk]\s*\{?\s*([A-Za-z])\s*\}?")

#: Standalone letter commands: `\o` (ø), `\l` (ł), `\ss` (ß), `\ae`, `\aa`, `\O`…
#: Mapped to the Unicode letter so `strip_diacritics`' `_TRANSLITERATE` table —
#: the one place this project decides how `ł` romanizes — makes the final call.
_LETTER_CMDS = {
    r"\ss": "ß", r"\ae": "æ", r"\AE": "Æ", r"\oe": "œ", r"\OE": "Œ",
    r"\aa": "å", r"\AA": "Å", r"\o": "ø", r"\O": "Ø",
    r"\l": "ł", r"\L": "Ł", r"\dh": "ð", r"\DH": "Ð",
    r"\th": "þ", r"\TH": "Þ", r"\i": "ı", r"\j": "j",
}

#: Text commands that mean punctuation.
_TEXT_CMDS = {
    r"\textendash": "-", r"\textemdash": "-", r"\texthyphen": "-",
    r"\textquoteright": "'", r"\textquoteleft": "'",
    r"\textquotedblright": '"', r"\textquotedblleft": '"',
    r"\ldots": "...", r"\dots": "...",
    r"\textregistered": "", r"\texttrademark": "", r"\copyright": "",
    r"\textdegree": "", r"\textpm": "", r"\textmu": "u",
}

#: Escaped literals — `\&` is a real `&`, not a command.
_ESCAPED_LITERALS = re.compile(r"\\([&%$#_{}])")

#: Greek and a few math letters, spelled out. A title reading `$\alpha$-synuclein`
#: should stem as `alpha-synuclein`, which is how the word is said and searched
#: for; deleting the command yields `-synuclein`.
_MATH_LETTERS = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "zeta": "zeta", "eta": "eta", "theta": "theta",
    "iota": "iota", "kappa": "kappa", "lambda": "lambda", "mu": "mu",
    "nu": "nu", "xi": "xi", "pi": "pi", "rho": "rho", "sigma": "sigma",
    "tau": "tau", "upsilon": "upsilon", "phi": "phi", "chi": "chi",
    "psi": "psi", "omega": "omega",
    "Alpha": "Alpha", "Beta": "Beta", "Gamma": "Gamma", "Delta": "Delta",
    "Sigma": "Sigma", "Omega": "Omega", "Phi": "Phi", "Psi": "Psi",
    "times": "x", "pm": "", "approx": "", "sim": "", "to": "",
}
_MATH_CMD = re.compile(r"\\([A-Za-z]+)")


def _strip_math(text: str) -> str:
    """Replace `$…$` spans with their spelled-out content.

    Math mode is where a title hides `$\\alpha$` and `$10^{-9}$`. Superscript and
    subscript markers are dropped rather than rendered — `10^{-9}` becomes
    `10-9`, which is what a reader typing the title would produce.
    """
    def render(m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace("^", "").replace("_", "")
        inner = _MATH_CMD.sub(lambda c: _MATH_LETTERS.get(c.group(1), c.group(1)), inner)
        return inner.replace("{", "").replace("}", "")

    return re.sub(r"\$([^$]*)\$", render, text)


def delatex(text: str | None) -> str:
    """BibTeX field value → plain text.

    Order matters and is not arbitrary. Every step below is placed where it is
    because putting it later broke a real case:

      1. Escaped literals (`\\&` → `&`), so step 7's brace-stripping never has
         to distinguish a literal `\\{` from a group delimiter.
      2. Math spans, before any command substitution, because `$\\alpha$` has to
         be read as a unit.
      3. **Named letter/text commands, before accents.** `\\textendash` starts
         with `\\t`, which is also the tie-accent command — accents first turned
         `long\\textendash read` into `longextendash read`. Longest name first,
         so `\\AA` is not consumed by `\\A`, and each match gobbles trailing
         whitespace the way LaTeX does for a control word (`\\textendash read`
         is one token pair, `–read`, not `– read`).
      4. Accent commands, which by now cannot collide with a named command.
      5. Unknown commands → their bare name plus a space, *before* braces go, so
         `\\textbf{Important}` reads `textbf Important` rather than welding into
         `textbfImportant`. Keeping the name is wrong-but-visible; deleting the
         command can silently drop a word.
      6. `~` → space (a non-breaking space in LaTeX, not a character).
      7. Braces last: whatever survives is grouping or brace-protection
         (`{CRISPR}`), both of which vanish leaving their content.

    Returns `""` for None/empty rather than raising — a missing title is a
    triage decision (`thin-metadata`), not a parse error.
    """
    if not text:
        return ""
    s = _ESCAPED_LITERALS.sub(r"\1", text)
    s = _strip_math(s)
    for cmd in sorted({**_LETTER_CMDS, **_TEXT_CMDS}, key=len, reverse=True):
        repl = _LETTER_CMDS.get(cmd, _TEXT_CMDS.get(cmd, ""))
        # `(?![A-Za-z])` keeps `\o` out of `\oe` and `\omega`; `\s*` reproduces
        # LaTeX's rule that a control word swallows the spaces after it.
        s = re.sub(re.escape(cmd) + r"(?![A-Za-z])\s*", repl, s)
    s = _ACCENT_CMD.sub(r"\1", s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1 ", s)
    s = s.replace("~", " ")
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()
