# Slice 09 — Provenance as a first-class output

**Goal:** every command that writes model artifacts emits a validated provenance
record for that run, and `capture_provenance` returns a typed model rather than a
loose dict.

**Risk:** medium. **Depends on:** nothing (overlaps 08 for the dict→model part;
if 08 already typed `capture_provenance`, this slice only wires the emission).

## Files in scope

- `src/microbleednet/provenance.py`
- The write-producing pipelines (`pipelines/preprocess.py`, `pipelines/train.py`,
  `pipelines/infer.py`, `pipelines/evaluate.py`)
- `tests/test_end_to_end.py`

## Confirm first

- Read `provenance.py`: what `capture_provenance` collects and returns, and
  whether `write_provenance` / `write_split_manifest` already exist.
- Check each write-producing pipeline for whether it currently emits provenance.
- Confirm the split-manifest write-once behavior exists (refuses conflicting
  overwrite).

## Steps

1. If not already typed (or done in slice 08), make `capture_provenance` return an
   immutable Pydantic model capturing config, seed, dependency/PyTorch/CUDA
   versions, device, and source revision.
2. Ensure each write-producing pipeline calls it and writes the provenance
   atomically into the run directory alongside its outputs.
3. Confirm subject splits are written once and never silently regenerated.
4. Extend the end-to-end test to assert a provenance record exists for a run and
   contains the config and seed.

## Must not change

- The scientific outputs of any command.
- Seeding behavior or split contents.

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src tests
uv run --all-groups pytest -q
```

## Done when

- Each artifact-writing command emits a validated provenance record.
- A run can be traced to its config, seed, and revision; all gates green.
