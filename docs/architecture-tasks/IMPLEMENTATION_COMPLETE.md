---
title: Architecture Alignment — Implementation Report
description: Exhaustive record of what was implemented against AUDIT.md, item by item, with commits, verification, and the deliberately-deferred remainder
status: complete
---

# Architecture alignment — implementation report

This document closes out the architecture-alignment work opened by
[`AUDIT.md`](./AUDIT.md). The audit was a point-in-time work-list of everything
that still diverged from the normative [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
after slices 01–11. This report records, item by item, **what was implemented,
how it was verified, and what was deliberately left undone and why.**

The governing constraint throughout: pre-existing and new code follow the same
structure and flow (where they differed, the better pattern was chosen and the
other restructured to match); **scientific behavior was never changed except
where the user explicitly authorized it** (A8/A9 below); every slice kept the
quality gates green.

## Verification baseline

Every slice was checked against `HEAD` using a detached worktree
(`git worktree add --detach <wt> HEAD`) so per-file `ruff`/`pyright` counts could
be diffed rather than judged in the absolute. The end state:

- **Tests:** `49 passed` (`uv run pytest -q`), up from the pre-work suite —
  three new end-to-end tests cover the authorized A8/A9 behavior.
- **Ruff** (E/F/I, line-length 88): no new per-file findings versus `HEAD`.
- **Pyright:** no new per-file errors versus the 83-error baseline (the baseline
  errors are pre-existing and out of scope for this alignment).

A pre-work blocker was found and fixed first: `HEAD` was un-importable because
`manifests.py` called `storage.read_json`, a function that had never been
committed. That fix (`df36c6b`) is a standalone slice so the history bisects
cleanly.

---

## Part 1 — Alignment items (A-series): all resolved

### A1 — Typed, versioned manifests with a validating reader ✅
**Commit `7e568a8`** (plus `df36c6b` for the reader dependency).

Introduced the shared-contract module `src/microbleednet/manifests.py` with a
typed Pydantic base carrying the full envelope the contract mandates:
`schema_version`, `manifest_type`, `status` (`ManifestStatus`:
running/complete/failed), `created_at`, `updated_at`, `error`, plus a typed
payload. Models are `extra="forbid", frozen=True`. `read_manifest()` validates
the schema version and manifest type and **refuses to return a non-`complete`
manifest**, satisfying "a consumer may only read a `complete` manifest."

All five manifest writers were migrated off loose dicts onto typed models:
`RawDatasetManifest` (index_data), `PreprocessedDatasetManifest` (preprocess),
the training-stage manifest (running→complete/failed lifecycle), and
`SplitManifest` (A6). Consumers no longer read required fields with `.get()`.

*Verified:* `schema_version`/`manifest_type` now present; `index_data.py` and
`preprocess.py` write through `manifests.write_manifest`; the failed-stage test
asserts `status == "failed"`, `manifest_type == "training_stage"`,
`schema_version == 1`, and that no checkpoint is published on failure.

### A2 — All writes route through the single atomic writer ✅
**Commit `ea3e5d8`** (folds in D5).

Removed the second hand-rolled atomic JSON writer in `patchers.py` and the
non-atomic `write_text(json.dumps(...))` in `froc.py`. Every JSON write now goes
through `storage.write_json_atomic` (temp-in-dir → dump → flush → fsync →
`os.replace`, `sort_keys`); every text write goes through the new
`storage.write_text_atomic`. The one remaining bespoke temp+replace in
`patchers.py` is the legitimate `.npz` **array** writer, not a JSON path.

*Verified:* no hand-rolled JSON atomic writers remain; `froc.py` and the patch
manifest both call the shared writers.

### A3 — Atomic writers relocated to the shared contract ✅
**Commit `f809a68`**.

Moved the atomic writers into the shared-contract `storage.py` root module, which
both `core` and `provenance` may import **downward**. This removed the inverted
edge where `provenance.py` reached up into `core.utils`.

*Verified:* `provenance.py` no longer imports anything from `core`;
`storage.py` exposes `write_json_atomic`, `read_json`, `write_text_atomic`.

### A4 — Trainer honors `TrainerConfig` instead of hardcoding paper values ✅ (authorized behavior fix)
**Commit `521b5ef`**.

The trainer previously overwrote the learning rate to `1e-3`, built a `LambdaLR`
from a hardcoded `0.1 ** (epoch // 2)` floor, capped epochs at `min(n, 100)`, and
defaulted `compile_model=True`. It now sources learning rate, decay factor, decay
period, minimum LR, and max epochs from `TrainerConfig`. Because the config
defaults **are** the paper values, the default run is bit-for-bit unchanged; only
non-default configs now take effect. This is the single-source-of-truth fix in
the place it mattered most scientifically.

*Verified:* no `lr"] = 1e-3`, `min(n_epochs, 100)`, or `0.1 **` literals remain
in `trainers.py`.

### A5 — Unified batch keys and `BaseTask.training_step` signature ✅
**Commit `5ad1a0e`** (paired with D1).

The inconsistent `x`/`y` vs `volume`/`mask`/`label` batch keys were unified and
`BaseTask.training_step` was given a single coherent signature that matches what
the trainer passes.

*Verified:* no stray `batch["x"]`/`batch["y"]`/`"x":`/`"y":` keys remain under
`core/`.

### A6 — Train/validation split persisted write-once ✅
**Commit `f7f2c49`** (paired with A7).

The 2-way train/validation split is now persisted as a write-once immutable
`SplitManifest` at `experiment_dir/manifests/split.json` before the first stage
runs. `_persist_split` is idempotent for an identical split and raises on a
conflicting one — never a silent overwrite. The dead, schema-incompatible 4-way
`provenance.write_split_manifest` was removed (C2) rather than contorted to fit.

*Verified:* `test_train_emits_provenance_before_training` asserts the split
manifest is written with `manifest_type == "split"`, `schema_version == 1`, a
partition covering all subjects, and disjoint train/validation sets;
`test_persist_split_is_write_once` covers the idempotent-rewrite and
conflicting-split paths.

### A7 — DataLoader workers seeded from the experiment seed ✅
**Commit `f7f2c49`**.

The DataLoaders now pass `worker_init_fn=provenance.seed_worker`, deriving each
worker's seed from the experiment seed. This retired the previously-dead
`seed_worker` (so C3 resolved by wiring, not deletion).

*Verified:* `train.py` sets `"worker_init_fn": provenance.seed_worker`.

### A8 — FROC wired into `evaluate` ✅ (user-authorized behavior change)
**Commit `63dec07`**. See memory [[audit-scope-decisions]].

The user chose **wire in** over remove. Implemented **additively** so absent
inputs leave binary-mask metrics untouched: `EvaluationSubject.probability_path`
(optional) and `EvaluateCommandConfig.froc_thresholds` (optional, validated to be
nonempty and in `[0, 1]`). When thresholds are set, every subject must supply a
probability map; the sweep binarizes each map per threshold, matches against the
reference, and writes `froc.json` + `froc.csv` atomically. The old dead
`EvaluationConfig` threshold fields were **not** reused — they were removed as
dead in C6; A8 uses the new field.

*Verified:* `test_evaluate_with_froc_thresholds_writes_sweep` (low threshold
recovers the lesion, high one drops it), `test_evaluate_froc_requires_probability_paths`
(validator rejects missing probability paths), and the no-FROC test asserts
`froc.json` is **not** produced when thresholds are absent.

### A9 — Postprocessing wired into `infer` ✅ (user-authorized behavior change)
**Commit `704ae16`**. See memory [[audit-scope-decisions]].

The user chose **wire in**. `InferCommandConfig.postprocessing` (optional
`PostprocessingConfig`). When set, the student-accepted mask is filtered by
`filter_components` (minimum volume mm³, maximum eccentricity, brain-boundary
distance) before the mask and probability map are written; the brain mask is the
extracted-brain support (`processed.image != 0`). Each candidate record gains a
`postprocessing_accepted` flag, and the CSV appends that column — keeping the
components file the single audit trail for why a student-accepted candidate was
dropped. **Behavior is unchanged when `postprocessing` is omitted.**

A subtlety handled carefully: the filter relabels the accepted-mask components, a
different label space and length from the per-detector-candidate records. Rather
than zip mismatched sequences, survival is mapped by **spatial overlap** —
`postprocessing_accepted = student_accepted and np.any(component_mask[candidate_voxels])`.

*Verified:* `test_apply_postprocessing_drops_subthreshold_components` — a compact
cube survives, a stray single voxel is dropped for under-volume, the probability
map is zeroed at the dropped voxel, and the rejection reason is recorded.

### A10 — Loose-dict boundaries: partially resolved, remainder deferred ⚠️
The duplication-driven parts (candidate/patch labeling) were unified under D2–D4.
The typed-record adoptions **B1** (`PatchRecord`) and **B2** (`CandidateComponent`)
were **deliberately deferred** — see Part 4. B3–B8 remain future slices.

### A11 — Public patch entry point; single patch-size source ✅
**Commit `cbcc6f3`**.

The cross-module private import of `_extract_fixed_patch` was replaced with a
public `extract_centered_patch` entry point. The magic patch size `24` is no
longer open-coded across `infer.py`/`train.py`/`config.py`; it is sourced from
one place.

*Verified:* no `_extract_fixed_patch` or `_CANDIDATE_PATCH_SIZE` import survives
outside `patch.py`; `infer.py` uses `extract_centered_patch(..., student_config.patch_size)`.

### A12 — Per-command CLI modules ✅
**Commit `9eef88a`** (paired with C1).

The four inline `@app.command` handlers were extracted into per-command modules —
`cli/preprocess.py`, `cli/train.py`, `cli/infer.py`, `cli/evaluate.py` — each a
`typer.Typer()` composed by `entrypoint.py` via `add_typer`, alongside a shared
`cli/shared.py` (`parse_config`, `config_epilog`, `finish`, `_config_help`) and
`cli/errors.py`. This matches the existing `cli/index_data.py` module shape.

*Verified:* all `cli/*.py` command modules exist; `test_root_and_command_help`
and the config-help tests pass for every command.

---

## Part 2 — Removals (C-series)

| Item | Disposition | Commit |
| --- | --- | --- |
| **C1** `validate-data` command | Removed (out of scope; index-data owns validation). Test reference and README updated. | `9eef88a` |
| **C2** `provenance.write_split_manifest` | Removed (dead, 4-way schema incompatible with the 2-way split). | `f7f2c49` |
| **C3** `provenance.seed_worker` | **Not removed — wired in** by A7. | `f7f2c49` |
| **C4** `core/utils.load_teacher_for_student` | Removed (dead wrapper). | `9bbae61` |
| **C5** `pipelines/index_data.remove_overlap` | Removed (dead). | `9bbae61` |
| **C6** Dead config hierarchy (`DetectorRunConfig`, `TeacherRunConfig`, `StudentRunConfig`, `InferenceConfig`, dead `EvaluationConfig`/`DataConfig`) | Removed, plus the now-unused `Literal` import. | `0052677` |
| **C8** `core/evaluation/froc.py` | **Kept — wired in** by A8. | `63dec07` |
| **C9** `core/postprocessing/` | **Kept — wired in** by A9. | `704ae16` |
| **C11** Dead `EvaluationConfig` threshold fields | Removed with C6 (A8 uses a new field). | `0052677` |

*Verified:* a repo-wide grep confirms `validate-data`, `load_teacher_for_student`,
`remove_overlap`, `write_split_manifest`, and every removed `*RunConfig`/
`InferenceConfig`/`EvaluationConfig`/`DataConfig` class name are gone from `src/`
and `tests/`.

**C10** (stale `.pyc` remnants) is a housekeeping no-op cleared on next clean
build. **C7** (unused typed records) is blocked on B1/B2 — see Part 4.

---

## Part 3 — Duplication (D-series): resolved

- **D1** — Removed the forbidden double-representation in `patchers.py` (stored
  both `label` and `has_microbleed`); one is kept and the other derived.
  Commit `5ad1a0e`.
- **D2** — The FRST channel-concat repeated five times is now the single
  `frst.prepend_frst_channel(x)` helper `(B,1,H,W,D) → (B,2,H,W,D)`.
  Commit `79ce9fd`.
- **D3** — Softmax → positive-class → numpy is now the single
  `processor.positive_class_probability(logits)`. Commit `79ce9fd`.
- **D4** — The two independent candidate-generation implementations
  (`pipelines/utils.py` and `infer.py`) now share one
  `processor.label_candidates(mask, probability)` kernel — `label(connectivity=3)`
  → `regionprops` → per-region mean probability. Only the genuinely-shared kernel
  was extracted; the behavior-divergent threshold operator (`>` in the patcher vs
  `>=` in infer) was left intact at each call site, so no scientific behavior
  moved. Commit `8877e29`.

*Verified:* `label_candidates` and `positive_class_probability` each have exactly
one definition (in `processor.py`); all pipeline callers route through them.

**D5** folded into A2; **D6** folded into A11.

---

## Part 4 — Deliberately deferred, with rationale

These were left undone **on purpose**, not missed. See memory
[[audit-b-series-deferred]].

- **B1 (`PatchRecord`) / B2 (`CandidateComponent`)** — the audit assumed these
  were "pure adoption" because the frozen records already exist. They are not.
  Both boundary dicts are **mutable accumulators** that gain fields across the
  pipeline (`detector_probability` → `student_probability` → `accepted` →
  `postprocessing_accepted`) and carry disk-only serialization fields
  (`patch_path`, `checksum`, `augmentation_version/factor`, `candidate_probability`)
  the frozen records do not model. Their keys also differ (`candidate_id` vs the
  record's `label`), and they serialize directly to on-disk JSON. Adopting them
  would reshape the on-disk artifact and restructure the accumulation flow — a
  schema/behavior change, which the standing directive forbids doing silently.
  Each needs its own schema-versioning slice.
- **C7** — depends on B1/B2/B4 (adopt-over-delete), so it is blocked until those
  land.
- **B3–B8** — remaining loose-dict boundaries; future slices, one boundary each.

**How to resume:** treat each B-series item as a slice that deliberately versions
the on-disk JSON (or keeps a serialization layer separate from the in-memory
record), rather than the mechanical swap the audit envisioned.

---

## Commit index (this alignment effort)

```
0052677 C6: remove dead config hierarchy
704ae16 A9: wire postprocessing into infer
63dec07 A8: wire FROC into evaluate
9eef88a C1/A12: per-command CLI modules; drop out-of-scope validate-data
9bbae61 C4/C5: remove dead helpers
8877e29 D4: single candidate labeling-and-scoring step
79ce9fd D2/D3: single FRST-concat and positive-class-probability helpers
df36c6b fix: define storage.read_json used by manifest reader
cbcc6f3 A11: public patch entry point; single patch-size source
5ad1a0e A5/D1: unify batch keys and BaseTask signature; drop redundant patch label
f7f2c49 A6/A7: persist train/validation split write-once and seed DataLoader workers
521b5ef A4: Trainer honors TrainerConfig instead of hardcoding paper values
7e568a8 A1: typed versioned manifests with validating reader
ea3e5d8 refactor(A2): route all JSON/text writes through the single atomic writer
f809a68 refactor(A3): relocate atomic writers to shared-contract storage module
```

## Status summary

| Group | Resolved | Deferred |
| --- | --- | --- |
| A-series (alignment) | A1, A2, A3, A4, A5, A6, A7, A8, A9, A11, A12 | A10 (B1/B2 portion) |
| C-series (removals) | C1, C2, C4, C5, C6, C8*, C9*, C11 | C7 (blocked on B1/B2) |
| D-series (duplication) | D1, D2, D3, D4, D5, D6 | — |

*C8/C9 resolved by wiring in (A8/A9), not removal.

All A-series alignment items are closed. The only open work is the B-series
loose-dict adoption (and C7 behind it), deferred as its own schema-versioning
effort because it would otherwise change on-disk artifacts silently.
