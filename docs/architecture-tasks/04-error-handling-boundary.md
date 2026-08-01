# Slice 04 — Error-handling boundary

**Goal:** errors are raised where detected and formatted only in the cli. core
raises precise precondition errors; pipelines re-raise after writing a `failed`
manifest; cli catches expected domain errors, formats them, and returns nonzero.
No swallowed exceptions; no `complete` manifest from a `finally` block.

**Risk:** medium. **Depends on:** 03 (so error reporting uses logging, not print).

## Files in scope

- `src/microbleednet/cli/errors.py` and each `src/microbleednet/cli/*.py` command
- Pipeline modules that write manifests in `finally` or catch broadly
  (`pipelines/preprocess.py`, `pipelines/train.py`, others as found)
- `tests/test_end_to_end.py` for failure-path assertions

## Confirm first

- Read `cli/errors.py` to see the current domain-error mapping.
- Grep pipelines for `except Exception`, bare `except`, and `finally:` blocks
  that write manifests; list each and classify it as "must re-raise" or "fine".
- Confirm which exceptions the cli currently catches.

## Steps

1. In pipelines, ensure any broad catch exists **only** to write a `failed`
   manifest, and that it `raise`s the original exception afterward (chained).
   Move `complete`-manifest writes out of `finally` to the success path.
2. In core, leave/strengthen precise precondition errors; remove any broad catches
   that hide them.
3. In cli, catch the expected domain error types, format via `errors.py`, and set
   a nonzero exit code. Do **not** catch unexpected exceptions — let them surface.
4. Add/extend a test that injects a mid-run failure and asserts: no `complete`
   manifest is written, a `failed` manifest with a nonempty error exists, and the
   command exits nonzero.

## Must not change

- Which operations succeed on the happy path, or their outputs.
- Error *types* raised by core, unless a slice-approved precision improvement.

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src tests
uv run --all-groups pytest -q
```

Grep to confirm no manifest `complete` write remains in a `finally` block.

## Done when

- Failures surface with a nonzero exit and a `failed` manifest; no swallowing.
- Happy-path behavior and outputs unchanged; all gates green.
