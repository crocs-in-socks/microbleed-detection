# Slice 08 — Replace loose dicts at layer boundaries

**Goal:** no raw dictionary or ad-hoc `TypedDict` crosses a layer boundary. The
known offenders become typed records/models.

**Risk:** high (changes signatures across layers). **Depends on:** nothing, but
do it after the lower-risk slices so the surface is otherwise stable.

> This slice is the most likely to need splitting. Prefer one offender per PR.
> If any single offender is large, implement just that one and leave the rest for
> a follow-up run.

## Known offenders (confirm before editing)

- `pipelines/utils.py`: `SubjectPayload` TypedDict; `collect_patches(...,
  patcher_parameters: dict[str, object])`.
- `pipelines/train.py`: `_model_metadata(config) -> dict[str, object]`.
- `pipelines/evaluate.py`, `pipelines/infer.py`: result `TypedDict`s carrying
  `list[dict[str, Any]]`.
- `manifests.PreprocessedDatasetManifest.preprocess_parameters: dict[str, object]`.
- `provenance.capture_provenance(...) -> dict[str, Any]`.

## Files in scope (pick ONE offender per invocation)

Name the single offender and its direct call sites when you ask for this slice.
Touch `records.py` / `manifests.py` to add the replacing model, plus the
producer and consumer of that value.

## Confirm first

- Read the offender and trace every producer and consumer.
- Decide the replacing type: frozen dataclass (in-memory, may hold arrays) vs
  immutable Pydantic model (crosses a JSON boundary).
- Check whether an existing record (`SubjectRecord`, `CandidateComponent`, etc.)
  already fits before creating a new one.

## Steps

1. Define or reuse the typed model.
2. Change the producer to return it and the consumer to accept it.
3. Remove the TypedDict / `dict[str, object]` annotation and any `.get()` on
   required fields.
4. Update tests to construct the typed value.

## Must not change

- The data carried or computed — only its representation.
- Manifest schema version (adding a typed subfield that serializes identically is
  fine; changing serialized shape is a schema change — stop and report if so).

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright <the touched files>
uv run --all-groups pytest -q
```

Grep to confirm the specific offender's dict annotation is gone.

## Done when

- The chosen offender is a typed model end to end; no `.get()` for its required
  fields; all gates green.
