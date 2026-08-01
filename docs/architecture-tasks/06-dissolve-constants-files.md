# Slice 06 — Dissolve remaining constants files into models

**Goal:** the remaining standalone constants (`constants.py`,
`core/constants.py`, `pipelines/constants.py`) are dissolved into the datamodels
they belong to, per `ARCHITECTURE.md`. A separate constants file survives only
for a value that genuinely belongs to no model.

**Risk:** medium (many small references). **Depends on:** 05 (drift removed first).

## Files in scope

- `src/microbleednet/constants.py`
- `src/microbleednet/core/constants.py`
- `src/microbleednet/pipelines/constants.py`
- Their importers across `core/`, `pipelines/`, `cli/`
- Relevant model files: `config.py`, `manifests.py`, `records.py`, and the
  module a structural constant belongs to

## Confirm first

- Enumerate every remaining constant in the three files and classify each:
  - **Scientific/tunable** → config-model field default.
  - **Structural, not user-tunable** (26-connectivity, `schema_version`,
    `manifest_type` literals, checkpoint filenames, dir/suffix names) →
    class-level constant on the owning model/module.
  - **Genuinely model-less** → keep, and note why in the PR.
- List all importers so nothing is left dangling.

## Steps

1. Move each constant to its classified home, one group per commit if large.
2. Update importers to read from the new location.
3. Delete the now-empty constants files.
4. Prefer namespacing structural constants on the model (e.g.
   `Manifest.SCHEMA_VERSION`) rather than a floating module global.

## Must not change

- The constant values, or any behavior depending on them.
- Manifest schema versions or literals (moving where they're defined is fine;
  changing them is not).

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src tests
uv run python -m compileall -q src
uv run --all-groups pytest -q
```

Grep to prove the deleted modules are no longer imported anywhere.

## Done when

- Constants live on the models that own them; stray constants files removed.
- Values and behavior unchanged; all gates green.
