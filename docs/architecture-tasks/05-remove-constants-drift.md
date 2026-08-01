# Slice 05 — Remove constants drift (augmentation + distillation)

**Goal:** the paper parameters that are currently defined **twice** — once as
validated config fields and once as an unvalidated mirror in `core/constants.py`
— have a single source of truth (the config model). The mirror is removed and
core functions receive the values as arguments.

**Risk:** medium (touches core call sites). **Depends on:** nothing.

## Known drift (confirm before editing)

- Augmentation ranges (translation offset, noise variance, blur sigma):
  `config.AugmentationConfig` vs `core/constants.py`.
- Student distillation hyperparameters (`temperature`, `alpha`, `beta`):
  `config.StudentConfig` vs `core/constants.py`.

## Files in scope

- `src/microbleednet/core/constants.py` (remove the mirrored values only)
- `src/microbleednet/core/transforms/augmentations.py` (accept values as args)
- `src/microbleednet/core/common/losses.py` and/or `tasks.py` (distillation args)
- The pipeline modules that call the above and hold the config
  (`pipelines/train.py`, `pipelines/utils.py`, others as found)
- `tests/test_losses.py`, `tests/test_patches.py` if they read the constants

## Confirm first

- Open `core/constants.py` and record the exact current values.
- Confirm the config-model defaults are **numerically identical** to the mirror.
  If they differ, stop and report — that is the drift bug itself and the
  requester must choose the correct value before proceeding.
- Find every import of the mirrored constants.

## Steps

1. Change the augmentation/distillation core functions to take the values as
   explicit parameters (with no default, or defaulting to the paper value only if
   that keeps a public API stable — prefer explicit).
2. Pass the values from the pipeline, sourced from the validated config model.
3. Delete the mirrored entries from `core/constants.py` (leave unrelated
   constants alone; full dissolution is slice 06).
4. Update tests to source values from the config model.

## Must not change

- The numerical values themselves, or the math that uses them. This is a
  single-source-of-truth move, not a retune.

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src tests
uv run python -m compileall -q src
uv run --all-groups pytest tests/test_losses.py tests/test_patches.py -q
uv run --all-groups pytest -q
```

Grep to prove the mirrored names are gone and unimported.

## Done when

- Augmentation and distillation params come only from config.
- Values unchanged; all gates green.
