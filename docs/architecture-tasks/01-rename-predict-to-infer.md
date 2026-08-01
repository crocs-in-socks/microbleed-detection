# Slice 01 — Rename `predict` → `infer`

**Goal:** the inference command, module, pipeline, and config are all named
`infer`, consistent with `ARCHITECTURE.md`. Pure rename; no behavior change.

**Risk:** low. **Depends on:** nothing.

## Files in scope

- `src/microbleednet/cli/predict.py` → `src/microbleednet/cli/infer.py`
- `src/microbleednet/pipelines/predict.py` → `src/microbleednet/pipelines/infer.py`
- `src/microbleednet/cli/entrypoint.py` (command registration)
- `src/microbleednet/config.py` (`PredictCommandConfig` → `InferCommandConfig`)
- `src/microbleednet/cli/config.py` only if it names the config class
- `tests/` files that import or invoke the predict command
- Any manifest type literal that says `predict` (e.g. `PredictionManifest`)

## Confirm first

- Grep the tree for `predict`, `Predict`, `prediction` to find every reference,
  including the CLI command string, manifest `manifest_type` literals, docstrings,
  and README/config examples.
- Decide with the requester whether the **output manifest** and on-disk artifact
  names also rename, or only the command/module/config. Default: rename the
  command, module, pipeline, and config; **keep `PredictionManifest` /
  `manifest_type="prediction"` as-is** to avoid a schema-version bump, unless
  asked otherwise. Note the decision in the PR.

## Steps

1. `git mv` the two module files to their new names.
2. Rename the Typer command from `predict` to `infer` and update the callback /
   function name to match the naming rule (`infer_command`).
3. Rename `PredictCommandConfig` → `InferCommandConfig` and update all imports.
4. Update `entrypoint.py` registration and help wiring.
5. Update tests and any config/doc examples to the new names.
6. Update the README command list if it names `predict`.

## Must not change

- The inference algorithm, thresholds, batching, or output contents.
- The manifest schema/version (unless the requester opted into renaming it).

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src tests
uv run python -m compileall -q src
uv run --all-groups pytest -q
uv run microbleednet --help
uv run microbleednet infer --help
```

Also grep to prove no stray `predict` command reference remains.

## Done when

- `microbleednet infer` runs the former `predict` flow; `predict` is gone.
- All gates green; diff is a rename only.
