# Architecture

MicrobleedNet is a research workflow for cerebral microbleed detection in NIfTI brain MRI. It implements a three-stage teacher/student deep-learning pipeline — detector → discriminator-teacher → discriminator-student — behind a command-line front end.

## Layering

The package is organized into four layers with a strict dependency direction: `cli → orchestration → core → root`. Root modules depend only on each other.

| Layer | Location | Responsibility |
|---|---|---|
| CLI | `cli/` | Typer front end: parse config, validate, dry-run, delegate to a pipe. |
| Orchestration | `orchestration/` | Turn a validated config into a completed command: read/write manifests, coordinate `core`. No ML math. |
| Core | `core/` | Internal domain logic: models, losses, training loops, transforms, evaluation. |
| Root | Package root | No modules; layer imports flow through `core`. |

## Validation boundary

The CLI is the boundary for user-provided configuration. It parses TOML through
the Pydantic config models and checks command preconditions before delegating to
an orchestration pipe. Orchestration and core receive trusted configuration
values and do not repeat scalar configuration validation.

Core remains reusable within the application and by another interface that
adopts the same configuration contract. It validates runtime data and internal
invariants where they become observable, such as loaded-volume geometry,
tensor shape and dtype, manifest consistency, and checkpoint compatibility.
Those conditions cannot be established by validating a CLI configuration alone.

## Root modules

These sit below everything else and define the shared contracts.


`core/datamodels.py` owns the `FrozenModel` base, the core configs it consumes (the model-architecture contracts `ModelArchitecture`/`ClassifierArchitecture`, plus `AugmentationConfig` and `TrainerConfig`), the `Modality` type alias, and the records it produces. Preprocessing takes the modality directly; the fixed pipeline performs canonical reorientation, BET, modality-dependent N4 correction, normalization, tight cropping, and vessel inpainting. The orchestration configs that compose them (the `DetectorConfig`/`TeacherConfig`/`StudentConfig` family, `PostprocessingConfig`, `DataSplitConfig`, and the top-level per-command configs) live in `orchestration/configs.py`, where descriptions carry the paper's defaults and validators enforce invariants (e.g. student `alpha + beta == 1`).

`core` owns the datamodels it consumes and produces so it is self-sufficient — see `core/datamodels.py` below. It is an internal library, not an alternative user-input boundary.

## CLI

The CLI module is organized so that `entrypoint.py` is a pure controller and every command is generated from a single declarative table rather than hand-written.

- **`cli/entrypoint.py`** — Builds the root Typer app, calls `commands.register_commands(app)` to attach every command, owns the logging callback, and exposes `main()`. It holds no command logic of its own.
- **`cli/commands.py`** — The command table and its registration. Every command is the same skeleton — parse a config, check preconditions, honor `--dry-run`, defer-import one pipe, run it, report — so each is described by a `CommandSpec` (name, help, config model, pipe module, precondition callable, and dry-run/success message builders) and built by `build_command`. `register_commands` builds all of these plus the `describe` meta-command, which renders each command's config keys from its model (via `config_fields`). The pipe is imported by module name inside the generated command body, so `--help` and `describe` never import torch.
- **`cli/utils.py`** — Everything the commands share: TOML config loading and validation (`parse_config`), the `require_file`/`require_dir` precondition checks, config-key introspection (`config_fields`) for `describe`, the `describe` hint shown in each command's `--help` epilog, and output helpers.

Each command's config model is owned by `orchestration/configs.py` and imported into `commands.py`; the CLI defines no config models of its own.

Each command reads a single `.toml` config file (`--config`). Run `microbleednet describe <command>` for that command's configuration keys, descriptions, and defaults — derived from the config models so the reference never drifts from the code.

| Command | Pipe | Purpose |
|---|---|---|
| `index-data` | `orchestration.pipes.index_data` | Match volumes to masks by a `{subject_id}` pattern; write (or accumulate into) the raw dataset manifest. |
| `preprocess` | `orchestration.pipes.preprocess` | Preprocess every indexed subject; write the preprocessed manifest. |
| `train` | `orchestration.pipes.train` | Train detector, teacher, and student sequentially. |
| `infer` | `orchestration.pipes.infer` | Run detection + student discrimination on one volume. |
| `evaluate` | `orchestration.pipes.evaluate` | Infer over an indexed dataset, then score predictions against its reference masks. |
| `describe` | — | Print a command's configuration keys, descriptions, and defaults. |

## Orchestration

The orchestration module separates the pipes — one runnable command each — from the code they share.

- **`orchestration/pipes/`** — One module per pipe (`index_data.py`, `preprocess.py`, `train.py`, `infer.py`, `evaluate.py`), each exposing an `execute()` entry point that the matching CLI command calls.
- **`orchestration/layouts.py`** — The on-disk directory layouts as filesystem contracts (not user-tunable config): `DatasetLayout` (the dataset directory's manifest paths, preprocessed dirs, and filename suffixes) and `ExperimentLayout` (the experiment directory's per-stage checkpoint/patch/manifest paths). Each keeps its path knowledge in one place instead of scattering string literals across the pipes; the pipes use the default instances.
- **`orchestration/configs.py`** — The top-level configs the pipes consume verbatim (each `execute()` takes its config whole): `IndexDataConfig`, `PreprocessConfig`, `TrainConfig`, `InferConfig`, and `EvaluateConfig`, plus the `SUBJECT_ID_PLACEHOLDER` constant. The CLI imports these back so the dependency direction stays `cli → orchestration`; loading a config from a TOML file is a CLI concern (`cli/utils.py`).
- **`orchestration/atomic_io.py`** — Atomic JSON and text persistence (`read_json`, `write_json_atomic`, `write_text_atomic`) for manifests, provenance, and command reports. It writes a temporary sibling file, flushes and syncs it, then replaces the target so a crash does not expose a truncated artifact.
- **`orchestration/manifests.py`** — The manifest *models* (`Manifest` envelope plus `RawDatasetManifest`, `PreprocessedDatasetManifest`, `SplitManifest`, `TrainingStageManifest`) together with the read/write helpers (`read_manifest`, `write_manifest`, `timestamp`) that enforce their durability contract: reads reject unversioned or incomplete manifests.
- **`orchestration/records.py`** — `PredictionSummary`, the frozen dataclass the infer pipe returns to its caller (artifact paths + candidate count). Owned here because only the orchestration layer consumes it, mirroring how `core` owns the records it produces in `core/datamodels.py`.
- **`orchestration/provenance.py`** — Reproducibility: RNG seeding (`seed_everything`, `seed_worker`) and capture of config, seed, dependency versions, git revision, and device into `provenance.json`. Lives here rather than at the root because only the pipes consume it (train/infer/evaluate) and it imports torch — keeping it out of the root preserves the guarantee that the CLI-facing root/`configs`/`layouts` modules stay torch-free.
- **`orchestration/utils.py`** — Helpers shared across pipes (e.g. patch collection, model teardown).

## Data flow

1. **Index** — Raw NIfTI volumes and optional masks are matched by subject ID into a `RawDatasetManifest`. Re-running against an existing dataset directory *accumulates*: each run appends its directory pair as a new `RawSource` and merges its subjects into the manifest (preserving the original `created_at`). Subject IDs stay globally unique — a collision with an already-indexed subject is a hard error — and an optional `source_id` namespaces a source's IDs as `{source_id}_{subject_id}` to keep multiple sources distinct.
2. **Preprocess** — Each subject is reoriented, brain-extracted (FSL BET), bias-corrected with N4 for T2*-GRE/SWI, normalized, inverted for T2*-GRE/SWI, tight-cropped, and vessel-inpainted, producing a `PreprocessResult` and a `PreprocessedDatasetManifest`. The `CropTransform` is retained so predictions can later be restored to source space.
3. **Train** — A train/test split is persisted as a `SplitManifest` alongside `provenance.json`. Volumes are turned into per-subject memory-mapped `.npy` patch stacks and records, then fed to the trainer. Three stages run in order, each writing a `TrainingStageManifest`. Every forward pass consumes 2 channels: the image plus its FRST response.
4. **Infer** — One source volume is preprocessed; the detector produces a probability map that is thresholded into candidate components; the student scores each candidate; accepted candidates form a mask that is restored to source space and written as prediction/probability NIfTIs plus per-candidate JSON/CSV records.
5. **Evaluate** — Each indexed subject with a reference mask is inferred (checkpoints resolved from the training `experiment_dir` or passed explicitly), and the source-space predictions are component-matched to the raw manifest's reference masks (Hungarian assignment on voxel overlap), aggregated into cluster TPR / precision / FP-per-subject, and written to `evaluation.json` and CSVs, with an optional FROC threshold sweep.

## Model

All three models share a 3D U-Net-style `FeatureExtractor`. The input is 2 channels (image + FRST); the paper's initial channel width is 64.

- **Detector** (stage 1) — `FeatureExtractor` + `Segmentor`. Produces the voxel-wise probability map that is thresholded into candidate components.
- **Discriminator teacher** (stage 2) — Adds a `Classifier` head to the segmentor. Its shared trunk is initialized from the trained detector; the auxiliary segmentation branch supervises teacher training.
- **Discriminator student** (stage 3) — `FeatureExtractor` + `Classifier` only. Trained by distilling from the frozen teacher: loss = α·supervised cross-entropy + β·temperature-scaled KL. This lighter model is used at inference to accept or reject detector candidates.

At inference only the detector and student are loaded: the detector proposes candidates and the student discriminates them. The classifier head requires a 24³ patch, so teacher and student patch at 24 while the detector patches at 48.

## Core subpackages

- **`core/datamodels.py`** — The datamodels `core` owns so it is self-sufficient: the modality type alias, the configs it consumes (`AugmentationConfig`, `TrainerConfig`), and the records it produces from image data (`PreprocessResult` and the `ImageGeometry`/`CropTransform` it carries, plus the `FloatArray`/`IntArray` aliases). The orchestration layer imports these contracts when composing its per-command configs.
- **`core/io.py`** — NIfTI, NumPy array, and Torch checkpoint I/O, including atomic writes for patch stacks and checkpoints.
- **`core/common/`** — `layers.py` (U-Net blocks), `models.py` (the three models), `losses.py` (Dice, detector, teacher, distillation, student losses), `tasks.py` (binds a model to its loss for the trainer).
- **`core/engines/`** — `processor.py` (stateless image-space preprocess / infer / restore / candidate labeling), `trainers.py` (`Trainer`: Adam, LR schedule, AMP, early stopping, versioned checkpoints, resume), `evaluators.py` (validation-loss loop).
- **`core/dataloading/`** — `patchers.py` (materialize checksummed `.npz` patches), `datasets.py` (patch datasets with on-the-fly augmentation), `samplers.py` (`EqualBatchSampler` for class-balanced batches).
- **`core/evaluation/`** — `matching.py` (component matching via Hungarian assignment), `metrics.py` (aggregate cluster metrics), `froc.py` (threshold sweep).
- **`core/postprocessing/`** — `components.py` (label mask into components), `filters.py` (reject candidates by volume, eccentricity, boundary distance).
- **`core/transforms/`** — `volume_ops.py` (reorient, BET, N4, normalize, crop), `frst.py` (Fast Radial Symmetry Transform + the FRST channel), `patch.py` (numpy patch extraction), `augmentations.py` (translate/noise/blur), `inpaint_vessels.py` (vessel masking + inpainting).
- **`core/utils.py`** — Model helpers: `unwrap_model` and `initialize_teacher_from_detector`.
