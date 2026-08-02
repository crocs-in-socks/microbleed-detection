# Slice 10 — Connect the typed config models to the CLI and pipelines

**Goal:** make the Pydantic models in `config.py` the single source of truth for
every config-driven command. The CLI parses a config file into a typed model,
validates it there, and passes the model (or its typed sub-models) into the
pipeline. `load_config(path) -> dict` + `require_keys` + `path_value` and the
flat-key loose-dict flow are retired for these commands.

**Risk:** high (changes the CLI↔pipeline contract for four commands). **Depends
on:** nothing. **Blocks:** the help-rendering half of slice 02, and makes slices
05 and 08 tractable.

> **This slice MUST be split — one command per invocation.** Do NOT convert all
> four commands at once. Order: `preprocess` → `infer` → `evaluate` → `train`
> (train last; it is the largest and threads the most parameters). Name the
> single command when you ask for this slice.

## Why this slice exists (current reality)

- `config.py` defines immutable, validated, paper-defaulted models
  (`DataConfig`, `PreprocessingConfig`, `DetectorConfig`, `TeacherConfig`,
  `StudentConfig`, `TrainerConfig`, `PostprocessingConfig`, `EvaluationConfig`)
  and composite run configs (`DetectorRunConfig`, `TeacherRunConfig`,
  `StudentRunConfig`, `InferenceConfig`). **Nothing imports them.** They are
  orphaned.
- `cli/entrypoint.py` is the entire CLI. Every command calls
  `_config(path, required)` → `load_config` (JSON/TOML → `dict`) +
  `require_keys`, then indexes **flat** keys: `dataset_dir`,
  `preprocessor_parameters`, `experiment_dir`, `datasplit_parameters`,
  `detector_parameters`, `discriminator_teacher_parameters`,
  `discriminator_student_parameters`, `volume_path`, `output_dir`,
  `model_parameters`, `detector_checkpoint`, `student_checkpoint`,
  `preprocess_parameters`, `subjects`, `metadata`, `device`,
  `detector_threshold`, `student_threshold`, `patch_batch_size`.
- Every pipeline `execute(...)` consumes loose dicts / `**kwargs`:
  - `preprocess.execute(dataset_dir: Path, preprocessor_parameters: dict)`
  - `train.execute(dataset_dir, experiment_dir, datasplit_parameters,
    detector_parameters, discriminator_teacher_parameters,
    discriminator_student_parameters)` and the `_stage_parameters(parameters:
    dict, ...) -> dict` helpers that mutate nested dict keys in place.
  - `evaluate.execute(subjects, output_dir, metadata)`
  - `infer.execute(*args, **kwargs)` → `predict_volume(...)`
- **Two shapes disagree.** The models are nested (`data`, `preprocessing`,
  `detector`, `trainer`, ...) and match `configs/*.paper.json`. The CLI's flat
  runtime keys (`preprocessor_parameters`, `model_parameters`, checkpoints,
  thresholds, I/O paths) are a different vocabulary. This slice must reconcile
  them, not pretend they already match.

## The reconciliation decision (read before coding)

The config file carries two kinds of information that the current flat schema
mixes together:

1. **Scientific / model parameters** — belong in the nested `config.py` models
   (paper defaults, validators, units documented per field).
2. **Runtime wiring** — input volume path, output dir, dataset dir, experiment
   dir, checkpoint paths, device, batch size, per-subject prediction/reference
   paths. These are *invocation* inputs, not paper science.

Do **not** force runtime wiring into the science models. Instead, for the
command in scope, define one **command config model** in `config.py` (immutable,
`extra="forbid"`) that composes:

- the relevant science sub-model(s) from `config.py`, and
- typed runtime fields (`Path`, `torch.device` as a validated string, etc.).

Example target for `preprocess` (illustrative — confirm field names against the
pipeline before writing):

```python
class PreprocessCommandConfig(FrozenConfig):
    dataset_dir: Path
    preprocessing: PreprocessingConfig = PreprocessingConfig()
```

The CLI then does `PreprocessCommandConfig.model_validate(raw)` where `raw =
load_config(path)` (kept only as the JSON/TOML **reader**, no longer the
contract). Validation, missing-key errors, and range checks now come from
Pydantic, so `require_keys` and per-field `path_value` calls for this command
are deleted.

> If reconciling a command reveals that its config file shape must change,
> update the matching `configs/*.json` example in the same invocation and say so
> in the report. Changing an on-disk config *schema* is allowed here (it is the
> point of the slice) but must be called out explicitly — it is the one place
> this slice is permitted to change a persisted shape.

## Files in scope (for the ONE command chosen)

- `src/microbleednet/config.py` — add the command config model.
- `src/microbleednet/cli/entrypoint.py` — parse into the model; drop
  `_config`/`require_keys`/`path_value` usage for this command.
- `src/microbleednet/pipelines/<command>.py` — change `execute(...)` to accept
  the typed model (or its typed sub-models); stop indexing dict keys.
- `configs/<command>*.json` — only if the file shape changes; update the example.
- The command's test in `tests/` — construct the new config shape.

Do **not** touch `cli/config.py`'s `load_config` reader beyond what the command
needs, and do **not** remove `require_keys`/`path_value` until the **last**
command stops using them (a later invocation removes the now-dead helpers).

## Confirm first

- Read the pipeline `execute(...)` and every helper it calls; list exactly which
  dict keys are read and whether any are mutated in place (train mutates
  `stage_parameters[...]`). A frozen model cannot be mutated — the pipeline must
  build its runtime scratch from the model, not on it.
- Map each current flat key to either a science sub-model field or a runtime
  field. Write the mapping in the report before editing.
- Confirm whether an existing composite (`InferenceConfig`, `DetectorRunConfig`,
  ...) already fits; prefer composing it over duplicating fields.

## Steps

1. Add the command config model to `config.py` (compose science sub-models +
   runtime fields; keep `extra="forbid"`, frozen).
2. In `entrypoint.py`, replace `_config(...)` + flat indexing with
   `Model.model_validate(load_config(path))`. Map Pydantic `ValidationError` to
   `typer.BadParameter` so CLI UX is unchanged (a clean error, not a traceback).
3. Change the pipeline `execute(...)` signature to accept the typed model. Inside
   the pipeline, read attributes (`config.preprocessing.extract_brain`) instead
   of `dict["..."]`. Build any mutable scratch dicts locally from the model.
4. Update the command's test to write/construct the new config shape.
5. Keep `--dry-run` behavior identical: validate (now via the model) and return
   without writing.

## Must not change

- **Scientific behavior** — the values that reach the core layer must be numerically
  identical for an equivalent config. This slice moves *where* values are
  validated and *how* they are typed, not what they are. Paper defaults in the
  models must equal the values the flat dicts previously supplied.
- The command's CLI surface (option names, `--dry-run`, exit codes, the
  human-facing success/error messages) beyond turning tracebacks into
  `BadParameter`.
- Other commands' code paths. One command per invocation.

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src/microbleednet/config.py src/microbleednet/cli/entrypoint.py src/microbleednet/pipelines/<command>.py
uv run --all-groups pytest -q
```

- Confirm the chosen command no longer calls `require_keys` or indexes flat dict
  keys (grep `entrypoint.py`).
- Confirm the pipeline `execute` signature names the typed model.
- Run the command's own test and its `--help` (help must still exit 0).

## Done when

- The chosen command parses its config into a typed `config.py` model, the
  pipeline consumes that model, no loose dict for that command crosses the
  CLI↔pipeline boundary, `--dry-run` and success messages are unchanged, and all
  gates are green.
- Report states the flat-key → typed-field mapping used and whether any
  `configs/*.json` example shape changed.

## Follow-up (final invocation only)

Once all four commands are converted, a closing pass removes the now-dead
`require_keys` / `path_value` / `_config` helpers and narrows `load_config` to a
pure JSON/TOML reader returning the raw object for `model_validate`. Only do this
when grep shows zero remaining callers.
