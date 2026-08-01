---
title: MicrobleedNet Research Implementation
description: Research-only implementation of a three-stage cerebral microbleed detection workflow
author: MicrobleedNet contributors
ms.date: 2026-08-01
ms.topic: overview
---

## Purpose

MicrobleedNet is a research implementation of the detector, discriminator
teacher, and discriminator student workflow described by Sundaresan et al.
(2023). It accepts T2*-GRE or SWI NIfTI data and produces source-space
component masks, probability maps, and auditable evaluation records.

> [!WARNING]
> This package is research software. It is not a medical device, has no
> clinical validation claim, and must not be used for diagnosis or treatment.

The reference paper is [MicrobleedNet](https://doi.org/10.3389/fninf.2023.1204186).
The authors' software is available at the
[public repository](https://github.com/v-sundaresan/microbleed-detection).

## Scope

Supported inputs are one T2*-GRE or SWI volume per subject in NIfTI format.
Training and lesion-level evaluation require masks. QSM, cross-validation,
clinical deployment, and redistribution of restricted cohorts are out of
scope.

Canonical orientation means axis reorientation to a standard orientation. It
does not perform MNI registration or spatial normalization to a template.

## Installation

Use Python 3.13 and the locked project environment:

```powershell
uv sync --frozen --all-groups
```

Preprocessing requires FSL BET on `PATH` and SimpleITK. CUDA is optional. CPU
execution is the correctness path and uses float32 inference and training.

## Workflow

Index volumes and masks with a pattern containing exactly one
`{subject_id}` placeholder:

```powershell
uv run microbleednet index-data `
	--input-dir data/raw/volumes `
	--label-dir data/raw/masks `
	--dataset-dir data/dataset `
	--volume-pattern "{subject_id}.nii.gz" `
	--mask-pattern "{subject_id}.nii.gz"
```

Validate the indexed subjects and preprocess them with a JSON or TOML
configuration:

```powershell
uv run microbleednet validate-data --dataset-dir data/dataset
uv run microbleednet preprocess --config path/to/preprocess.json --dry-run
uv run microbleednet preprocess --config path/to/preprocess.json
```

Training, prediction, and evaluation require explicit configuration files:

```powershell
uv run microbleednet train --config path/to/train.json --dry-run
uv run microbleednet predict --config path/to/predict.json --dry-run
uv run microbleednet evaluate --config path/to/evaluate.json --dry-run
```

Run `uv run microbleednet COMMAND --help` for the required keys and path
contracts. The files under `configs/` contain paper parameter fragments; add
dataset and experiment paths in workflow-specific JSON or TOML files. Dry
runs validate configuration and input paths without creating model outputs.

## Preprocessing and geometry

The preprocessing pipeline reorients the image, optionally extracts the brain
with FSL BET, applies SimpleITK N4 bias correction, normalizes and inverts
intensities, removes vessel-like structures, and tight-crops the volume. The
crop and orientation transform are retained so prediction arrays can be
restored to the source shape and affine.

N4 is an intentional implementation difference from the paper's FAST
preprocessing step. See [implementation differences](docs/implementation-differences.md).

## Outputs

Prediction writes a source-space binary NIfTI mask, a source-space probability
NIfTI, and JSON/CSV component tables. Component records retain candidate IDs,
centroids, detector probabilities, student probabilities, and acceptance
decisions.

Evaluation writes per-subject matches, aggregate TP/FN/FP counts, derived
cluster metrics, and optional plots. Every aggregate count is derived from the
stored component matches.

## Research traceability

The paper-to-code mapping is maintained in
[paper traceability](docs/paper-traceability.md). The model card documents
intended use, known failure modes, and data limitations. Do not infer a
performance claim from the included tests: the repository does not bundle a
restricted clinical cohort or released checkpoint.

## Troubleshooting

* FSL errors: verify that `bet` is installed and available on `PATH`.
* CUDA errors: rerun on CPU with `"device": "cpu"` and disable compilation.
* Memory errors: reduce the configured batch size and use `num_workers: 0`.
* Invalid geometry: verify that image and mask shapes and affines match before
	indexing or preprocessing.
* Empty detections: prediction writes valid empty masks and component tables.

## Validation

```powershell
uv run ruff check src tests
uv run pyright src tests
uv run python -m compileall -q src
uv run pytest -q
uv build
```
