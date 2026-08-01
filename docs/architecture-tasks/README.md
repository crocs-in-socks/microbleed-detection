---
title: Architecture Tasks
description: Ordered, independently-shippable slices that move the repo toward ARCHITECTURE.md
status: working
---

# Architecture tasks

Each file in this directory is **one slice of work** that can be implemented and
verified on its own, in its own PR. They exist because migrating the whole
repository to [`ARCHITECTURE.md`](../../ARCHITECTURE.md) in a single pass cannot
be done reliably — the invariants touch ~55 files and the test suite is
deliberately thin, so a big-bang refactor can silently break scientific
behavior.

## How to run a slice

Ask for a slice by number. When implementing one, I will:

1. Read the files named in the slice and confirm the current state matches its
   preconditions. If it doesn't, I stop and report rather than guessing.
2. Make the smallest change that satisfies the slice.
3. Verify with the slice's commands **and** read the diff for accidental
   behavior changes.
4. Report what changed, what I checked, and anything that surprised me.

## Rules that apply to every slice

- **Never change scientific behavior** — preprocessing math, model architecture,
  loss equations, thresholds, patch sizes, or label semantics — unless the slice
  explicitly says so. If a change forces one, stop and report.
- Keep changes inside the named files. If a slice needs a file it doesn't list,
  stop and report instead of expanding scope.
- Every slice must leave `ruff`, `pyright`, `compileall`, and `pytest` green
  (or no worse than it found them, if a slice says so explicitly).

## Suggested order

Lower numbers are lower risk and have fewer dependencies. This is a recommended
order, not a hard requirement; each slice states its own dependencies.

| #  | Slice                                             | Risk   | Depends on | Status |
| -- | ------------------------------------------------- | ------ | ---------- | ------ |
| 01 | Rename `predict` → `infer`                        | low    | —          | done   |
| 10 | Connect typed config models to CLI + pipelines    | high   | —          | done (preprocess/evaluate/infer; train carved to 11) |
| 02 | Document config keys in CLI help                  | low    | 10         | done   |
| 03 | Logging: stdlib in core/pipelines, Rich in cli    | medium | —          | done   |
| 04 | Error-handling boundary                           | medium | 03         | done   |
| 05 | Remove constants drift (aug + distillation)       | medium | 10         | todo (blocked on 11) |
| 06 | Dissolve remaining constants files into models    | medium | 05         | todo   |
| 07 | Isolate optional / heavy dependencies             | low    | —          | done   |
| 08 | Replace loose dicts at layer boundaries           | high   | 10         | todo   |
| 09 | Provenance as a first-class output                | medium | —          | partial (typed record done; emission wiring → 11) |
| 11 | Typed config for the `train` command              | high   | 10         | todo   |

Slices 01, 03, 07 are the safest starting points. **Slice 10 is the
foundational refactor**: it connects the orphaned `config.py` models to the
running code and unblocks the help-rendering half of 02, plus 05 and 08. It is
high risk, so it is split one command per invocation (see the slice file). Do
slice 10 before the slices that depend on it, but after the low-risk 01/03/07 if
you want a stable surface first. **Slice 11** finishes the one command slice 10
left untyped (`train`), which is a larger typed-records refactor of an untested
loop, so it is its own slice.
