---
title: MicrobleedNet Architecture
description: The intended (target) architecture every change must converge toward
audience: contributors and reviewers
status: normative
---

# MicrobleedNet Architecture

This document describes the **desired state** of the repository — the shape the
code must converge toward, not necessarily what exists today. It is normative:
when code and this document disagree, the code is wrong unless this document is
changed first through review.

The scope is a researcher-facing implementation of the detector / discriminator
teacher / discriminator student workflow from Sundaresan et al. (2023). It is
research software, not a medical device.

## Guiding principles

These principles override any convenience. When a change violates one, the
change is wrong.

1. **Minimalism.** Write the least code that satisfies the requirement. More
   code means more surface for bugs and confusion. Delete before you add. There
   is exactly one supported way to do each thing.
2. **DRY.** A fact, rule, or computation is expressed in exactly one place. No
   copy-pasted logic, no value represented twice (e.g. never both `label` and
   `has_microbleed`).
3. **Composition over inheritance and frameworks.** Build behavior by combining
   small, single-purpose functions and data models. Do not introduce generic
   repositories, service locators, event buses, plugin systems, or speculative
   base classes.
4. **Typed boundaries, never loose dicts.** Every value that crosses a function,
   layer, or persistence boundary is a typed model or frozen record. Raw
   dictionaries are not passed between layers.
5. **Validate at the edges.** Anything read from or written to disk (config,
   NIfTI, manifests, checkpoints) is validated against a schema. Invalid data
   fails loudly with an actionable message; it never enters computation.
6. **Configurable, with paper defaults.** Every scientific parameter is
   overridable by the user and defaults to the value from the reference paper.
   Defaults live in the datamodels, not in scattered constant files.
7. **Reproducible and auditable.** Every run records its configuration, seed,
   environment, and provenance. Every reported number traces back to a stored
   per-subject record.

## The three layers

Dependencies point in **one direction only**. A violation of this direction is
an architectural defect.

```text
        cli  ─────────────►  pipelines  ─────────────►  core
         │                        │                       │
         │                        ▼                       ▼
         └──────────────►  shared contracts (root package modules)
```

### core — computation and business logic

`src/microbleednet/core/`

- Owns all processing and scientific logic: transforms, models, losses,
  datasets, samplers, training/eval engines, post-processing, evaluation math.
- Is **completely independent** of the pipeline and cli layers. `core` must
  never import from `pipelines` or `cli`.
- Functions are reusable in isolation. They validate their own local
  preconditions (array rank, matching shapes, positive patch size, finite
  affines) — these protect the reusable API and are *not* redundant with dataset
  validation.
- Does not orchestrate workflows, decide artifact locations, or re-scan
  unrelated subjects. It operates on the data it is handed.

### pipelines — orchestration

`src/microbleednet/pipelines/`

- Owns workflow order, artifact lifecycle, and stage prerequisites. Composes
  `core` functions and classes into the five commands.
- Reads and writes manifests; enforces "publish complete last"; handles atomic
  writes and failure records.
- May import `core` (including constructing `core` models) and shared contracts.
  Must **never** import from `cli`.
- Each command exposes one entry point named `execute`.

### cli — user interface

`src/microbleednet/cli/`

- Owns argument parsing, input validation, help text / documentation of what
  each option does, user-facing error messages, and concise result summaries.
- A cli module does exactly: parse arguments → load one typed command config →
  call one pipeline `execute` → format the result. It contains **no** NumPy,
  nibabel, torch, manifest-payload, or scientific logic.
- Each command lives in its own module (`cli/<command>.py`); `entrypoint.py`
  only builds the app and mounts subcommands.
- Converts expected domain errors into concise messages with nonzero exit codes.
  It does not catch unexpected exceptions.
- **The cli is the documentation surface.** Every command's `--help` must
  describe each accepted configuration key, its meaning, its unit, its default
  (the paper value), and its allowed range. A researcher should be able to
  discover and configure any workflow from `--help` alone, without reading
  source. The config models are the single source of truth for these values;
  help text is derived from their field descriptions rather than duplicated.

### shared contracts — root package modules

`src/microbleednet/config.py`, `records.py`, `manifests.py`, `provenance.py`

- Own the datamodels used by more than one layer: validated configuration,
  in-memory numerical records, persisted manifests, run provenance.
- Depend on nothing above them.

## The five commands

| Command      | cli module            | pipeline                | Produces                                   |
| ------------ | --------------------- | ----------------------- | ------------------------------------------ |
| `index-data` | `cli/index_data.py`   | `pipelines/index_data`  | Raw dataset manifest (subject pairing)     |
| `preprocess` | `cli/preprocess.py`   | `pipelines/preprocess`  | Preprocessed volumes + manifest            |
| `train`      | `cli/train.py`        | `pipelines/train`       | Detector, teacher, student checkpoints     |
| `evaluate`   | `cli/evaluate.py`     | `pipelines/evaluate`    | Lesion-level metrics, FROC, reports        |
| `infer`      | `cli/infer.py`        | `pipelines/infer`       | Source-space masks, prob maps, components  |

Every command supports `--dry-run`: it runs the same config parsing and
prerequisite validation as a real run but writes no model artifacts.

Dataset content/geometry validation (expensive per-subject NIfTI checks) is out
of scope for this MVP. Basic structural validation still happens at the edges —
indexing rejects mispaired/duplicate subjects, and core functions validate their
own preconditions (shape, affine, spacing) when they load an image. A dedicated
`validate-data` command may be reintroduced later without changing this
architecture.

## Data models and contracts

Never pass loose dictionaries between layers. Use the right kind of model for
the job:

- **Immutable Pydantic models** — for anything parsed from or serialized to
  JSON: configuration and manifests. Forbid unknown fields; make instances
  immutable (frozen).
- **Frozen dataclasses** — for in-memory numerical records that carry NumPy
  arrays (they should not be forced through JSON serialization).
- `Path` internally; serialize to string only at the JSON boundary.
- No `.get()` for required fields, no `Any` to dodge a type error at a boundary,
  no unchecked `cast()` or `# type: ignore` to complete work.

Representative records: `SubjectRecord`, `ImageGeometry`, `CropTransform`,
`PreprocessResult`, `PatchRecord`, `CandidateComponent`. Representative configs:
per-stage scientific settings composed into `PreprocessCommandConfig`,
`TrainCommandConfig`, `InferCommandConfig`, `EvaluateCommandConfig`.

Each config field carries a description, unit, and default; the cli help text is
generated from these so documentation cannot drift from the model.

### Where constants live — no standalone constants file

If a value can be expressed readably and maintainably inside a datamodel, it
does not get a separate constants module.

- **Scientific / paper defaults** (patch sizes 48 & 24, class weights, learning
  schedule, thresholds, augmentation factors, distillation `T`/`α`/`β`, volume
  and eccentricity thresholds) are **default field values on the config
  models**. That makes them typed, validated, documented, and overridable in
  one place.
- **Structural constants** that are not user-tunable (26-connectivity, manifest
  `schema_version`, `manifest_type` literals) live as class-level constants on
  the datamodel they belong to.

The current `constants.py`, `core/constants.py`, and `pipelines/constants.py`
are to be dissolved into the relevant models. A separate constants file is only
justified if a value genuinely belongs to no model — which should be rare.

**Concrete drift to eliminate.** Today several paper parameters are defined
*twice* — once as validated config fields and once as an unvalidated mirror in
`core/constants.py` — with no cross-check, so the two can silently disagree:

- Augmentation ranges (translation offset, noise variance, blur sigma) exist in
  both `AugmentationConfig` and `core/constants.py`.
- Student distillation hyperparameters (`temperature`, `alpha`, `beta`) exist in
  both `StudentConfig` and `core/constants.py`.

The config model is the single source of truth; the mirror must be removed. Core
functions receive these values as arguments from the pipeline, not by importing
a constants module.

## Persistence contracts

### Manifests

Every persisted manifest carries: `schema_version`, `manifest_type`, `status`
(`running` | `complete` | `failed`), `created_at`, `updated_at`, and a typed
payload. A `failed` manifest carries a nonempty error string. A consumer may
only read a `complete` manifest. Unversioned manifests are rejected with an
actionable message — never silently inferred as legacy.

### Atomic, publish-last writes

There is **one** low-level atomic JSON writer and **one** atomic text writer.
Write to a temporary file in the target directory, flush and fsync, then
replace. Data files (NIfTI, CSV) are written first; the `complete` manifest that
advertises them is written **last**, and never from a `finally` block. A failed
stage never leaves a `complete` manifest and never destroys a previous valid
output.

### Validation freshness

A validation report is only trusted if its inputs are unchanged, checked via the
source manifest digest plus file size and mtime — without reloading voxel data.

## Error handling

Errors are raised where they are detected and formatted where they are shown.

- **core** raises precise precondition errors (`ValueError`, `TypeError`, custom
  domain exceptions) when its local invariants are violated. It does not log
  user-facing messages, catch broadly, or decide exit codes.
- **pipelines** raise domain/workflow errors (missing prerequisite, incomplete
  manifest, failed stage). A pipeline catches `Exception` only where it must
  write a `failed` manifest, then **re-raises the original exception** —
  chained, never swallowed.
- **cli** is the only layer that catches expected domain errors, converts them
  into concise messages, and returns a nonzero exit code. It **does not catch
  unexpected exceptions** — those propagate so the failure is visible with a
  traceback.

Do not suppress exceptions at layer boundaries. Do not use a bare `except:` or
catch `Exception` to keep going.

## The four kinds of validation

These are deliberately distinct and **must not be merged**. When something looks
"redundant," it usually belongs to a different kind below.

1. **Configuration validation** — parse external config into immutable Pydantic
   models exactly once, at the command boundary.
2. **Dataset validation** — expensive NIfTI content/geometry checks. Out of
   scope as a standalone command for this MVP (see the commands section), but
   the *category* remains: when reintroduced it is a pipeline that reads each
   subject once and writes a report.
3. **Artifact-loading validation** — whenever a manifest or checkpoint crosses a
   persistence boundary, validate its schema version, status, and required
   fields before use.
4. **Core-contract validation** — local preconditions that make a public core
   function safe to call in isolation (array rank, matching shapes, finite
   affines, positive patch size). These are **not** redundant with dataset
   validation; they protect the reusable API. Core code must not re-scan
   unrelated subjects or reopen files a caller already loaded.

## Logging

Logging is layered the same way dependencies are, so that presentation lives
only in the cli.

- **core and pipelines** use the standard library `logging` module via a
  module-level `logger = logging.getLogger(__name__)`. They emit plain,
  structured log records (events, counts, timings, per-subject outcomes) and
  **never** import a presentation library, print directly, or format for a
  terminal. They do not configure handlers or levels — a library must not hijack
  the root logger.
- **cli** owns presentation. It configures logging once at startup and installs
  a [Rich](https://rich.readthedocs.io/) handler (`rich.logging.RichHandler`) so
  the log records emitted by core and pipelines are rendered for the user with
  color, level, and formatting. Progress bars, spinners, and tables are a cli
  concern built on Rich; core and pipelines expose progress as ordinary log
  events or lightweight callbacks, not Rich objects.

This keeps core and pipelines importable and testable without a terminal, lets
Rich be swapped or silenced entirely from one place, and ensures library code
never dictates how output looks.

## Reproducibility and provenance

Provenance is a **first-class output**, not an afterthought.

- Every command that writes model artifacts emits a provenance record for that
  run. A stored result you cannot trace back to its config, seed, and code
  revision is a defect.
- One experiment seed; derive worker seeds from it. Seed Python, NumPy, PyTorch,
  CUDA, samplers, and DataLoader workers.
- Persist exact subject splits as immutable manifests: write-once, never
  regenerated implicitly, and refuse to overwrite an existing split with
  different contents.
- Capture config, seed, dependency/PyTorch/CUDA versions, device, and source
  revision into a provenance record per run.
- CPU float32 is the correctness path. AMP and compilation are explicit,
  opt-in optimizations with parity expectations — never silent defaults that
  affect correctness tests.
- Do not claim bitwise GPU determinism when an op is nondeterministic.

## Naming and imports

- `snake_case` for functions, variables, modules, and JSON field names;
  `PascalCase` for classes and Pydantic models.
- Command adapters end in `_command`; pipeline entry points are named `execute`;
  private helpers start with `_` and are never imported across modules.
- `*_path` for files, `*_dir` for directories, `*_image` for NIfTI images,
  `*_array` for NumPy arrays, `*_manifest` for parsed models, `*_payload` only
  before parsing.
- Absolute `microbleednet.*` imports. Import modules for namespaced calls,
  classes directly for types. Group imports: stdlib, third-party, local.
- Avoid a bare `utils` name imported across packages; prefer `core_utils`,
  `pipeline_utils`.

## Testing standard

Testing protects scientific correctness and prevents silent corruption; it is
not a coverage exercise. The suite is a small set of high-value acceptance
checks (spatial integrity, patch geometry, losses, teacher transfer,
checkpoints, metrics, end-to-end), run on CPU float32. Add a new test file only
when a real defect cannot be reproduced in the existing ones.

## Optional and heavy dependencies

Some capabilities depend on tools that are not always present. These are
isolated so the core test suite and ordinary use run without them.

- **External tools** (FSL BET for skull stripping) are invoked only from the
  transform/pipeline step that needs them, behind an explicit capability check
  with an actionable error when absent. They are never a hard import at module
  load.
- **Optional Python packages** (e.g. plotting for FROC curves) are imported
  lazily inside the function that uses them, or guarded behind an
  optional-dependency boundary, and declared in a dedicated dependency group.
- **CUDA / AMP / compilation** are opt-in optimizations, never required for
  correctness. CPU float32 must work with none of them installed or enabled.

A missing optional dependency degrades one feature with a clear message; it must
never break `import microbleednet` or the core acceptance tests.

## Anti-patterns — what NOT to do

These are the specific over-engineering and correctness traps this project
rejects. Introducing any of them is grounds to reject a change.

- No generic repository, service locator, event bus, or plugin system.
- No speculative base classes or inheritance where composition of small
  functions works. Prefer a function over a class; prefer a class over a
  hierarchy.
- No `Any`, unchecked `cast()`, or `# type: ignore` to get past a boundary type
  error.
- No `.get()` for a required manifest or config field; required means required.
- No loose dictionaries passed between layers.
- No representing the same fact twice (e.g. `label` and `has_microbleed`, or a
  config value mirrored in a constants file).
- No `print` or Rich objects in core/pipelines; no direct file writes bypassing
  the atomic writers.
- No catching `Exception` to keep going; no reporting a stage `complete` from a
  `finally` block.
- No mutating a caller-owned config, or stuffing runtime objects (models,
  devices, paths) into config models.
- No adding code "in case we need it later." Add it in the change that needs it.

## Definition of done for any change

- Dependency direction is respected (cli → pipelines → core).
- No loose dict crosses a boundary; new data is a typed model.
- Anything read/written to disk is validated and written atomically.
- New scientific parameters are config fields with paper defaults, not constants.
- New config keys are documented in the command's `--help`.
- core/pipelines log via `logging` only; presentation stays in the cli.
- Errors are raised at detection, formatted only in the cli.
- No duplicated logic; the simplest existing abstraction is reused.
- No anti-pattern from the list above is introduced.
- `ruff`, `pyright`, `compileall`, and `pytest` all pass.
