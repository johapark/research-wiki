"""Research Wiki — LLM-assisted personal research paper wiki framework.

Rule 1 applies: every wiki claim must be grounded in a PDF we have.
Semantic Scholar is the one structural-metadata provider supported today;
OpenAlex is planned. See CLAUDE.md for the full contract.
"""

# The project's single source of truth for the version. `pyproject.toml` reads it
# from here via [tool.setuptools.dynamic]; `--version` reads it directly. Bump it
# as part of a release commit, never on its own — see CONTRIBUTING.md § Releasing,
# whose invariants `tests/test_version.py` enforces.
__version__ = "0.4.1"
