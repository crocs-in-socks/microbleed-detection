---
title: Architecture Alignment Audit
description: What's left to align with ARCHITECTURE.md, what to remove, and duplication to simplify
status: working
---

# Architecture alignment audit

This is a point-in-time reading of the codebase against the normative
[`ARCHITECTURE.md`](../../ARCHITECTURE.md), taken after slices 01–11. It exists to
answer three questions the user asked directly:

1. **What is left to do** to align the repo with the architecture?
2. **What should be removed** because it is no longer necessary?
3. **What redundancies / duplication** can be simplified?

Every claim below was verified first-hand (grep + file reads) at the time of
writing; file:line references are included so each item can be re-checked before
it is acted on. Nothing here changes code — it is a work-list. Each item is
scoped so it could become its own architecture slice (same rules as the other
slices: never change scientific behavior, keep the quality gates green).

The items are ordered by severity within each section. Severity reflects
distance from the normative contract and blast radius, not effort.

---

## Part 1 — What's left to align

### A1. No `manifests.py`; every manifest is a loose dict (highest severity)

ARCHITECTURE.md (line 108) requires a shared-contract module
`src/microbleednet/manifests.py` with a typed Pydantic manifest model. It does
not exist — `src/microbleednet/` holds only `config.py`, `provenance.py`,
`records.py`, `__init__.py`. A repo-wide grep for `schema_version` and
`manifest_type` returns **zero hits**.

The contract (ARCHITECTURE.md lines 191–195) says every manifest carries
`schema_version`, `manifest_type`, `status` (running|complete|failed),
`created_at`, `updated_at`, plus a typed payload. Actual state:

| Manifest | Written at | Has status? | Missing |
| --- | --- | --- | --- |
| raw | `pipelines/index_data.py:69-84` | no | schema_version, manifest_type, status, updated_at (uses `created_on`, not `created_at`) |
| preprocessed | `pipelines/preprocess.py:63-68` | no | same four; also re-serialized inside the per-subject loop |
| train stage | `pipelines/train.py:109-111` | **yes** (running/complete/failed) | schema_version, manifest_type, created_at, updated_at |
| infer components | `pipelines/infer.py:132` | no | not a manifest envelope at all |
| evaluate report | `pipelines/evaluate.py:31-32` | no | not a manifest envelope at all |

Consumers also read required fields with `.get()` — `train.py:378`
(`.get("subjects", [])`), `preprocess.py:20` — which ARCHITECTURE.md names as an
anti-pattern (lines 195, 342), and read with bare `open()`+`json.load` with no
status/version validation (`train.py:376-377`, `preprocess.py:18-19`), violating
"artifact-loading validation" (lines 241–243) and "a consumer may only read a
`complete` manifest."

**To align:** introduce `manifests.py` with a typed base model carrying the five
envelope fields + a typed payload; migrate the five writers to it; add a
`read_manifest()` that validates version/type and refuses non-`complete`
manifests. This is large and touches scientific-adjacent I/O, so it should be
split (one manifest per invocation), starting with the raw/preprocessed pair.

### A2. Only one atomic writer, and it is bypassed

`core/utils.write_json_atomic` (`core/utils.py:31-41`) is the single correct
atomic JSON writer (temp-in-dir → dump → flush → fsync → `os.replace`). But:

- `core/dataloading/patchers.py:88-92` hand-rolls a **second** atomic JSON writer
  for the patch manifest, and it omits `flush`/`fsync`/`sort_keys`.
  `patchers.py:63-67` hand-rolls the same temp+replace dance for `.npz`.
- `core/evaluation/froc.py:23` writes JSON via `write_text(json.dumps(...))` —
  **non-atomic**, bypassing the writer entirely.
- The mandated **atomic text writer** (ARCHITECTURE.md line 199) does not exist.

**To align:** route patchers and froc through `write_json_atomic`; add an atomic
text writer alongside it; keep both in the shared contract (see A3).

### A3. Dependency-direction violation: shared contract reaches up into core

`provenance.py:13` does `from microbleednet.core import utils` to reuse
`write_json_atomic`. `provenance.py` is a shared-contract root module; core sits
*above* it in the dependency order, so this is an inverted edge (ARCHITECTURE.md
one-directional deps).

**To align:** move `write_json_atomic` (and the new atomic text writer from A2)
into a shared-contract module — either a small `io`/`atomic.py` root module or
`manifests.py` — that both `core` and `provenance` may import downward. Removes
the inverted edge and gives A2 a single home.

### A4. Trainer ignores its typed config and hardcodes scientific hyperparameters (high severity)

`core/engines/trainers.py` does not honor `TrainerConfig`:

- `trainers.py:71` overwrites `optimizer_parameters["lr"] = 1e-3`, discarding
  `TrainerConfig.learning_rate`.
- `trainers.py:73-80` build a `LambdaLR` with hardcoded decay `0.1 **
  (completed_epochs // 2)` floored at `1e-6 / lr`, ignoring
  `TrainerConfig.learning_rate_factor`, `.learning_rate_period`, and
  `.minimum_learning_rate`.
- `trainers.py:102` caps epochs at `min(n_epochs, 100)` — a silent ceiling on the
  configured `max_epochs`.
- `compile_model` defaults `True` (`_DEFAULT_COMPILE_MODEL`).

This is the single source-of-truth principle broken in the place it matters most
scientifically: the config the user passes is silently disregarded. **This
changes scientific behavior to fix, so it needs an explicit slice** that says so,
with a test asserting the configured LR/scheduler/epochs reach the optimizer.

### A5. `BaseTask.training_step` signature mismatch + inconsistent batch keys

The batch dict keys are inconsistent across the codebase: some sites use
`x`/`y`, others `volume`/`mask`/`label`. `BaseTask.training_step`'s signature
does not line up with what the trainer passes. This is a latent
correctness/typing gap.

**To align:** define a typed batch record (frozen dataclass carrying the arrays,
per the records convention) and a single `training_step(self, batch: Batch)`
signature; migrate datasets/tasks/trainer to it. Pairs naturally with B-series
loose-dict work below.

### A6. Split manifests are never persisted; the helper that would is dead & wrong-shaped

ARCHITECTURE.md (lines 282–283) requires subject splits persisted as write-once
immutable manifests. `train.py:381-386` does a 2-way `train_test_split`
(train/validation only) and never writes it — the split regenerates implicitly
every run from `datasplit.random_state`. Meanwhile
`provenance.write_split_manifest` (`provenance.py:132-153`) exists but has **no
caller**, and requires exactly `{train, validation, tuning, test}` — a 4-way
schema **incompatible** with the pipeline's 2-way split. It cannot be wired in
as-is.

**To align:** either persist the actual 2-way split via a manifest and reshape
`write_split_manifest` to match, or delete the helper (see C-series) and write a
purpose-built split manifest. Also unmet: DataLoader worker seeding — `train.py`
sets no `worker_init_fn` (see A7).

### A7. Reproducibility gap: DataLoader workers unseeded

`provenance.seed_worker` (`provenance.py:57`) exists but has no caller; the
DataLoaders at `train.py:179-184` set no `worker_init_fn`. ARCHITECTURE.md line
280 requires worker seeds derived from the experiment seed.

**To align:** pass `worker_init_fn=seed_worker` (or inline equivalent) to the
DataLoaders. Small, self-contained.

### A8. FROC is a declared `evaluate` output but is never produced

ARCHITECTURE.md line 121 lists FROC among `evaluate` outputs, and
`EvaluationConfig` carries detector/discriminator threshold sweeps. But
`evaluate.py` produces `evaluation.json` + CSVs + a bar chart and **never calls
FROC**. `froc.write_froc` has no caller; `froc.sweep_thresholds` is exercised
only by `tests/test_metrics.py`.

**To align (decision needed):** either wire FROC into `evaluate` (compute the
sweep, write it atomically per A2) — making the module and the threshold config
live — or, if FROC is out of scope for this project, remove the module and the
threshold fields (see C-series). These are mutually exclusive; pick one.

### A9. Postprocessing is declared in config but never runs

`core/postprocessing/` (`components.py`, `filters.py`) and `PostprocessingConfig`
(minimum_volume_mm3, eccentricity, boundary distance — all paper values) are
**never imported by any pipeline**. `infer` emits raw student-accepted masks with
no morphological filtering. `PostprocessingConfig` is referenced only as a field
on the dead `*RunConfig`/`InferenceConfig` models (config.py:270, 288 — see C6),
so config, package, and feature are dead together.

**To align (decision needed):** either wire postprocessing into `infer`
(consume `PostprocessingConfig`, filter candidates before writing the mask) — a
scientific-behavior change needing an explicit slice — or remove the package and
config if the project intentionally omits it. Pick one; do not leave it wired but
inert.

### A10. Loose dicts at layer boundaries (slice 08 remainder), ranked

Slice 08 converted the infer result to `PredictionSummary`; these boundaries
remain loose (ARCHITECTURE.md principle 4). Ranked by boundary-crossing severity:

- **B1** `patchers.py:73-84` — patch records built as dicts (crosses into
  datasets/tasks). `PatchRecord` already exists (records.py:94) and is unused.
- **B2** `infer.py:48-59` — `_candidate_records` builds candidate dicts.
  `CandidateComponent` exists (records.py:140) and is unused.
- **B3** `pipelines/utils.py` — subject / `patcher_parameters` dicts.
- **B4** `train.py` — subjects passed as dicts; `.get("subjects", [])`.
- **B5** `datasets.py` `load_patch` — `.get()` on patch dicts.
- **B6** `evaluate.execute` → `dict[str, Any]` return (CLI reads
  `result['subjects']` at entrypoint.py:202).
- **B7** `matching.match_components` / `aggregate_metrics` → `dict[str, Any]`.
- **B8** `provenance.py:21,91-92` — `configuration: dict[str, Any]` /
  `BaseModel | dict[str, Any]` boundary uses `Any`.

**To align:** continue slice 08 one boundary per invocation, preferring the ones
whose typed record already exists (B1, B2) since those are pure adoption.

### A11. Cross-module private import + magic patch size

`infer.py:21` imports `_extract_fixed_patch` from `transforms.patch` — a private
helper imported across modules (ARCHITECTURE.md line 296 forbids this). The patch
size 24 is open-coded there (`center - 12`, size `24`) and re-expressed as
`_CANDIDATE_PATCH_SIZE = 24` in `train.py:46` and in `config.py` field defaults.

**To align:** promote a public patch-extraction entry point; source the patch
size from one place (the config field default).

### A12. CLI command modules not split out

4 of 5 config-driven commands (preprocess/train/infer/evaluate) are inline
`@app.command` handlers in `entrypoint.py:112-202`; only `index_data` is a
separate `cli/<command>.py` module. ARCHITECTURE.md's CLI layout implies
per-command modules. Also: the `validate_data` and index-data `validate_parameters`
adapters lack the `_command` suffix the other handlers use.

**To align:** extract per-command modules (low risk, mechanical) and normalize
the adapter naming. Lowest-severity alignment item.

---

## Part 2 — What to remove

Everything here was confirmed to have **no caller** in `src/` or `tests/` at
audit time. Confirm again immediately before deleting.

- **C1. `validate-data` command** (`entrypoint.py:92-109`). Out-of-scope
  (indexing/validation is `index-data`'s job); hand-rolls manifest reading with
  `.get()` and loose dicts, contradicting A1. Note: `test_end_to_end.py:17`
  currently asserts its `--help` works, so remove the test reference too.
- **C2. `provenance.write_split_manifest`** (`provenance.py:132`). Dead and
  schema-incompatible with the actual split (A6).
- **C3. `provenance.seed_worker`** (`provenance.py:57`). Dead — *unless* A7 wires
  it in. Resolve A7 first; only remove if A7 is declined.
- **C4. `core/utils.load_teacher_for_student`** (`core/utils.py:84-85`). Dead thin
  wrapper; student loading calls `load_model_weights` directly (`train.py:337`).
- **C5. `pipelines/index_data.remove_overlap`** (`index_data.py:120-128`). Dead.
- **C6. Dead config hierarchy** — `DetectorRunConfig`, `TeacherRunConfig`,
  `StudentRunConfig`, `InferenceConfig` (config.py:264-289). None referenced
  outside config.py; the CLI uses the `*CommandConfig` set instead. These are the
  sole referents of `PostprocessingConfig`/`EvaluationConfig` in the run path —
  removing them (or resolving A8/A9) unblocks removing the orphaned packages.
- **C7. Unused typed records** — `SubjectRecord`, `PatchRecord`,
  `CandidateComponent` (records.py:23/94/140). **Prefer adopting over deleting**
  (B1/B2/B4 need exactly these). Delete only what remains unused after the
  loose-dict work.
- **C8. `core/evaluation/froc.py`** — remove only if A8 is decided as
  "out of scope"; otherwise wire it in.
- **C9. `core/postprocessing/`** (`components.py`, `filters.py`) — remove only if
  A9 is decided as "out of scope"; otherwise wire it in.
- **C10. Stale `.pyc` remnants** — `src/microbleednet/__pycache__/constants.cpython-*.pyc`
  and `core/__pycache__/constants.cpython-*.pyc` (source modules already deleted
  in slices 05/06). Harmless but confusing; clear on next clean.
- **C11. Dead `EvaluationConfig` threshold fields** — remove only if A8 is
  declined (the sweeps have no consumer without FROC).

Not found (were named in an earlier plan but do not exist): `IMPLEMENTATION_PLAN.md`,
`PROJECT_COMPLETION_PLAN.md`, `CONSISTENCY_IMPLEMENTATION_PLAN.md` — nothing to
remove.

---

## Part 3 — Redundancies & duplication to simplify

- **D1. Forbidden double-representation** — `patchers.py:79-80` stores both
  `"label": int(has_microbleed)` **and** `"has_microbleed": has_microbleed`.
  ARCHITECTURE.md names this exact anti-pattern (lines 28-29, 344: "never both
  `label` and `has_microbleed`"). Keep one; derive the other.
- **D2. FRST channel-concat repeated 5×** — the `x = torch.cat((x, frst.apply(x)),
  dim=1)` pattern appears at `tasks.py:23-24`, `tasks.py:44-45`, `tasks.py:68-69`,
  `processor.py:114-115`, and inline at `infer.py:70`. Extract one
  `prepend_frst_channel(x)` helper. (`train.py:40-42`'s `_STAGE_INPUT_CHANNELS=2`
  comment documents the same fact a sixth time.)
- **D3. Softmax → positive-class probability → numpy repeated 3×** —
  `pipelines/utils.py:53-54`, `infer.py:100`, `infer.py:72`. One
  `positive_class_probability(logits)` helper. (Loss math in `losses.py` is
  legitimately distinct — leave it.)
- **D4. Candidate generation duplicated** — `pipelines/utils.py:52-60`
  (`patch_subject_target_centered`) and `infer.py:99-110` both do infer → softmax
  → threshold → `label(connectivity=3)` → `regionprops` → per-region mean prob.
  Two independent implementations of one operation; unify.
- **D5. Atomic-write dance duplicated** — see A2 (patchers ×2, froc). Folds into
  the A2/A3 fix.
- **D6. Patch size 24 expressed in several places** — see A11.

---

## Suggested sequencing

The alignment items have a natural dependency order:

1. **A3 first** (relocate atomic writers to the shared contract) — it is a
   precondition for doing A2 and A1 cleanly, and removes the inverted edge.
2. **A2** (route all writes through the single atomic writer; add the text
   writer) — folds in D5.
3. **A1** (typed `manifests.py` + validating reader) — the largest item; split
   per manifest.
4. **Decisions A8 (FROC) and A9 (postprocessing)** — each resolves to either a
   wiring slice or a removal (C8/C9/C11). Make these calls before touching the
   evaluate/infer surfaces so the loose-dict work (B6/B2) targets the final shape.
5. **A4** (trainer config-hardcoding) — highest scientific severity; explicit
   behavior-change slice with a guarding test.
6. **A10 (B1/B2 first)**, **A5**, **A6/A7**, then the low-risk **A11/A12** and the
   remaining removals (C-series) as their blockers clear.

The duplication fixes (D2/D3/D4) are safe to land opportunistically alongside the
slice that touches each site, since they are pure extractions with no behavior
change.
