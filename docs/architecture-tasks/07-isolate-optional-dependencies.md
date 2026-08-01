# Slice 07 — Isolate optional / heavy dependencies

**Goal:** optional and external dependencies (FSL BET, plotting) never break
`import microbleednet` or the core acceptance tests. Each is behind a capability
check or lazy import and declared in an appropriate dependency group.

**Risk:** low. **Depends on:** nothing.

## Files in scope

- `src/microbleednet/core/transforms/basic.py` (FSL BET invocation) and any
  module shelling out to external tools
- `src/microbleednet/core/evaluation/froc.py` / `pipelines/evaluate.py`
  (plotting import)
- `pyproject.toml` (optional-dependency group declaration)

## Confirm first

- Grep for top-level imports of plotting libraries (e.g. `matplotlib`) and for
  the FSL BET call site.
- Check `pyproject.toml` for existing optional-dependency groups.
- Confirm the core acceptance tests do not already depend on these being present.

## Steps

1. Move any plotting import to inside the function that produces the plot; when
   absent, log a clear "plot skipped, install extras" message and continue
   producing the non-plot outputs.
2. Ensure FSL BET is invoked only from the step that needs it, behind an explicit
   availability check that raises an actionable error when `bet` is missing — not
   a hard import or implicit failure.
3. Declare optional packages in a dedicated `pyproject.toml` group (e.g. `plots`).

## Must not change

- The outputs produced when the optional dependency **is** present.
- CUDA/AMP/compilation defaults (they must already be opt-in; if not, that's a
  separate slice — report it, don't fix it here).

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src tests
uv run --all-groups pytest -q
uv run python -c "import microbleednet; print('import ok')"
```

If feasible, confirm `import microbleednet` and the core tests pass in an
environment without the optional plotting extra installed.

## Done when

- Missing optional deps degrade one feature with a clear message.
- Import and core tests succeed without them; all gates green.
