---
title: Release Validation
description: Reproducibility and release checks for the MicrobleedNet research package
author: MicrobleedNet contributors
ms.date: 2026-08-01
ms.topic: reference
---

## Automated checks

The locked environment was synchronized with `uv sync --frozen --all-groups`.
The following checks pass:

* `uv run python -m compileall -q src`
* `uv run pytest -q`, 32 tests passed
* `uv build`, producing a source archive and wheel
* `uv run microbleednet --help`
* `uv run microbleednet preprocess --help`
* `uv run microbleednet evaluate --help`

The following required checks do not pass yet:

* `uv run ruff check src tests` reports 207 existing findings, including import
  ordering, unused imports, and line length violations.
* `uv run pyright src tests` reports 101 existing errors across the typed source
  and test surface.

These findings are release blockers. They are not suppressed by changing the
lint or type-check configuration.

## Research checks

The synthetic workflow is covered by `tests/test_end_to_end.py` and passes in
the locked environment. A clean-environment wheel install, legally usable real
T2*-GRE or SWI subject, source-space visual overlay review, and labeled cohort
run were not performed because this repository does not contain suitable data,
checkpoints, or an external FSL/CUDA research environment.

Before a research release, archive the following with each run:

* The exact wheel or source revision and `uv.lock`
* Configuration files and preprocessing parameters
* Dataset access statement, subject split, and annotation provenance
* Checkpoint files and their training logs
* Per-subject prediction, component, matching, and aggregate reports
* Source-space overlay review notes

No performance claim should be published until those checks are completed on a
legally usable, documented cohort.
