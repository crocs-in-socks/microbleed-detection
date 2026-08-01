# Slice 11 — Typed config for the train command

**Goal:** the `train` command parses its config into a typed model and the train
pipeline consumes typed configs end to end, the same way `preprocess`,
`evaluate`, and `infer` already do (slice 10). This is carved out of slice 10
because train's internals are a deep web of nested loose dicts with in-place
mutation and **no test coverage of its `execute()` path** — it is a typed-records
refactor (slice-08 scale), not a mechanical config-parse swap.

**Risk:** high (untested runtime path, many nested dicts, mutation). **Depends
on:** slice 10 (preprocess/evaluate/infer already typed; naming already
standardized to `input_channels`/`output_classes`).

> Consider splitting: (a) CLI parse + `execute()` signature, (b) `_stage_parameters`
> and the per-stage builders, (c) the loader/sampler/trainer parameter blocks.
> Land (a) first so the boundary is typed even if the internals stay dicts for a
> follow-up.

## Current reality (confirm before editing)

- `cli/entrypoint.py::train_command` still uses `_config(config, (...))` +
  `path_value`, reading flat keys: `dataset_dir`, `experiment_dir`,
  `datasplit_parameters`, `detector_parameters`,
  `discriminator_teacher_parameters`, `discriminator_student_parameters`.
- `pipelines/train.py::execute(dataset_dir, experiment_dir, datasplit_parameters,
  detector_parameters, discriminator_teacher_parameters,
  discriminator_student_parameters)` — every parameter is a `dict`.
- Internally, `_stage_parameters(parameters, experiment_dir, stage) -> dict`
  deep-copies and `setdefault`s **nested blocks**: `model_parameters`,
  `trainer_parameters`, `patcher_parameters`, `dataset_parameters`,
  `sampler_parameters`, `dataloader_parameters`, `task_parameters`,
  `fit_parameters`. `train_detector/teacher/student` then **mutate** those blocks
  in place (`stage_parameters["model_parameters"]["input_channels"] = 2`,
  `patcher_parameters["model"] = detector`, etc.) and inject live objects
  (`torch.device`, the detector `nn.Module`, the teacher) into the dicts.
- These block shapes do **not** match the existing `config.py` models.
  `TrainerConfig` (learning_rate, batch_size, ...) is the *paper* trainer config;
  the pipeline's `trainer_parameters` is a *runtime* block (device,
  optimizer_parameters, scheduler_parameters, checkpoint_dir) consumed by
  `Trainer.__init__`. They are different vocabularies — reconcile, don't conflate.

## The core tension (read before coding)

A frozen Pydantic config cannot carry live objects (`nn.Module`, `torch.device`)
and cannot be mutated. Train currently threads models/devices *through* the
parameter dicts. The typed design must separate:

1. **Config** (immutable, from disk): model hyperparameters
   (`DetectorConfig`/`TeacherConfig`/`StudentConfig`), trainer hyperparameters,
   patcher/sampler settings, data split. Use / extend the `config.py` models.
2. **Runtime wiring** (built in the pipeline, not from disk): the constructed
   models, the resolved `torch.device`, checkpoint dirs derived from
   `experiment_dir`. Pass these as explicit function arguments or a small
   in-memory dataclass — never inside the frozen config.

So `_stage_parameters` should stop being a mutable dict scratchpad. Replace it
with: (a) the typed stage config (immutable), and (b) a frozen dataclass of
derived runtime paths/objects the stage needs.

## Files in scope

- `src/microbleednet/config.py` — add `TrainCommandConfig` (composes the three
  stage configs + `dataset_dir`, `experiment_dir`, split settings) and any
  runtime trainer/patcher sub-models needed.
- `src/microbleednet/cli/entrypoint.py` — parse into the typed model via the
  existing `_parse_config`; drop `_config`/`path_value` for train. **This is the
  last consumer of `_config`/`require_keys`/`path_value`** — once train stops
  using them, remove the now-dead helpers from `entrypoint.py` and narrow
  `cli/config.py::load_config` to a pure reader (grep to confirm zero callers
  first, per slice 10's follow-up note).
- `src/microbleednet/pipelines/train.py` — retype `execute` and the stage
  builders; stop mutating parameter dicts.
- `src/microbleednet/pipelines/utils.py` — `patch_subject_target_centered`
  takes a `model`/`device`/`threshold`; keep those as explicit runtime args.
- `configs/detector.paper.json`, `configs/teacher.paper.json`,
  `configs/student.paper.json` — reconcile to the new shape and update in the
  same change; call out the schema change in the report.
- `tests/` — **add a dry-run test** for `train` (parity with the preprocess
  dry-run test) so the CLI parse path has coverage even though the full training
  loop stays untested. If feasible, a tiny smoke test of `_stage_parameters`'
  typed replacement.

## Must not change

- **Scientific behavior**: the hardcoded paper wiring the pipeline injects today
  must be preserved exactly — `input_channels = 2` for every stage, teacher/student
  `patch_size = 24`, `train_augmentation_factor = 5` for teacher and student,
  detector `initial_channels` default 64, the detector-threshold constant used by
  the patchers, distillation setup. Move where these live (into typed defaults),
  not their values.
- Checkpoint file layout and manifest schema/`status` transitions
  (`running`/`complete`/`failed`) in `_run_stage`.
- The order detector → teacher → student and the checkpoint hand-off between them.

## Steps

1. Add `TrainCommandConfig` (+ any runtime sub-models) to `config.py`.
2. Retype `execute()` to accept it; parse it in `train_command`.
3. Replace `_stage_parameters`'s dict scratchpad with typed config + a runtime
   dataclass; update `train_detector/teacher/student` to read attributes and pass
   runtime objects explicitly instead of stuffing them into dicts.
4. Reconcile the `configs/*.paper.json` shapes; update the examples.
5. Add the `train` dry-run test; keep `--dry-run` writing nothing.
6. Remove dead `_config`/`require_keys`/`path_value`; narrow `load_config`.

## Verify

```bash
uv run --all-groups ruff check src tests
uv run --all-groups pyright src/microbleednet/config.py src/microbleednet/cli/entrypoint.py src/microbleednet/pipelines/train.py src/microbleednet/pipelines/utils.py
uv run --all-groups pytest -q
uv run microbleednet train --help
```

- Grep `train.py` for `["` dict indexing on parameter blocks — should be gone for
  config values (runtime scratch may remain as a typed dataclass).
- Grep the repo for `_config(` / `require_keys(` / `path_value(` — zero callers.

## Done when

- `train` parses its config into a typed model, the pipeline consumes typed
  configs, no config value crosses the boundary as a loose dict, scientific wiring
  is numerically identical, the dead loose-dict helpers are gone, a train dry-run
  test exists, and all gates are green.
- Report states the flat→typed mapping and the `configs/*.json` schema changes.
