# Slice 03 — Logging: stdlib in core/pipelines, Rich in cli

**Goal:** core and pipelines emit logs via the standard library `logging` module
only; the cli configures logging once and installs a Rich handler for
presentation. No `print` for user-facing output in core/pipelines.

**Risk:** medium (cross-cutting but mostly additive/substitution).
**Depends on:** nothing.

## Files in scope

- All modules under `src/microbleednet/core/` and `src/microbleednet/pipelines/`
  that currently `print` or need log output
- `src/microbleednet/cli/entrypoint.py` (one-time logging configuration)
- `pyproject.toml` if `rich` must be added as a dependency
- `tests/` only if a test asserts on printed output that now logs

## Confirm first

- Grep for `print(` across `core/` and `pipelines/` to enumerate every call.
- Check whether `rich` is already a dependency.
- Confirm there is no existing logging configuration that would conflict.

## Steps

1. In each core/pipeline module that emits output, add
   `logger = logging.getLogger(__name__)` and replace `print(...)` with the
   appropriate `logger.info/debug/warning(...)`. Preserve the message content and
   the information conveyed; only change the transport.
2. Do **not** add handlers, `basicConfig`, or level setting inside core/pipelines.
3. In `entrypoint.py`, configure logging once (level from a `--verbose`/`--quiet`
   flag or default) and attach `rich.logging.RichHandler`.
4. Keep any genuinely user-facing summary lines (final artifact paths, counts) in
   the cli, rendered with Rich — not in the pipeline.

## Must not change

- What information is reported, or control flow. A log call must not alter
  execution (no side effects inside log arguments).
- No Rich objects imported or constructed in core/pipelines.

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src tests
uv run --all-groups pytest -q
uv run microbleednet --help
```

- Grep to prove zero `print(` remain in `core/` and `pipelines/`.
- Run a tiny command (e.g. `index-data --help` or the end-to-end fixture flow)
  and confirm logs render through Rich.

## Done when

- core/pipelines log via `logging`; cli owns Rich presentation.
- No `print` in core/pipelines; all gates green.
