"""Evaluation harnesses for the framework's own behaviour.

Distinct from `benchmark/`, which scores *model output* against fixture papers.
This package tests the framework's contracts — starting with whether CLAUDE.md's
trigger-gated prompt pointers actually fire (`triggers`).
"""
