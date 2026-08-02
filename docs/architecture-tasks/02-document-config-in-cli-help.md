# Slice 02 — Document config keys in CLI help

**Goal:** every configuration field a user can set carries a description, unit,
and default on its Pydantic model, and each command's `--help` surfaces those so
a researcher can configure any workflow without reading source.

**Risk:** low (additive). **Depends on:** nothing (do after 01 if both are queued,
to avoid churn on renamed files).

## Files in scope

- `src/microbleednet/config.py` (add `Field(..., description=...)` metadata)
- `src/microbleednet/cli/*.py` (help text / epilog wiring for each command)
- `tests/test_end_to_end.py` may gain a light assertion that help renders

## Confirm first

- Read `config.py` and list every user-facing field, its meaning, unit, paper
  default, and allowed range. Cross-check defaults against the paper values
  recorded in `IMPLEMENTATION_PLAN.md` / `PROJECT_COMPLETION_PLAN.md`.
- Check how Typer help is currently produced so descriptions can be derived from
  the model rather than duplicated in hand-written strings.

## Steps

1. Add `Field` `description` (and `ge`/`le` bounds where a range applies) to each
   config field. Put the unit in the description (e.g. "voxels", "mm^3").
2. Ensure command help renders the relevant fields — either by generating an
   options/keys summary from the model, or by referencing the model in the
   command epilog. Do not hand-copy values that already live on the model.
3. Replace any placeholder help text ("add more detail", etc.).

## Must not change

- Field names, defaults, or validation rules. This slice only adds descriptions
  and, where a real bound exists in the paper, explicit `ge`/`le` constraints
  that the current values already satisfy. If any existing value would fail a
  bound you add, stop and report — that's a real inconsistency, not a doc task.

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src tests
uv run --all-groups pytest -q
uv run microbleednet preprocess --help
uv run microbleednet train --help
uv run microbleednet infer --help
uv run microbleednet evaluate --help
```

Read each `--help` output and confirm keys, units, and defaults appear.

## Done when

- Every settable key is documented from its model.
- No placeholder help text remains; all gates green.
