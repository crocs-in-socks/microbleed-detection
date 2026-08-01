---
title: Implementation Differences
description: Documented differences between the MicrobleedNet paper and this research implementation
author: MicrobleedNet contributors
ms.date: 2026-08-01
ms.topic: reference
---

## Purpose

This log records deliberate or necessary choices that differ from the paper or
from ambiguous historical implementation behavior. It is part of experiment
provenance and should be read with the [paper traceability matrix](paper-traceability.md).

## Preprocessing

The paper describes FAST preprocessing. This repository uses SimpleITK N4 bias
field correction instead. N4 is deterministic for a fixed input and settings,
but it is not numerically equivalent to FAST.

Canonical orientation is interpreted as axis reorientation to a standard
orientation. It is not MNI registration and does not align a subject to a
population template.

## Candidate and boundary filters

The implementation records boundary distance in voxel units using an unscaled
Euclidean distance transform. The paper describes a millimeter criterion. Users
must record voxel spacing and the configured voxel threshold; the two criteria
must not be treated as interchangeable.

The eccentricity filter uses an explicit 2D projection because the selected
scikit-image region property is not a supported 3D eccentricity measure. The
projection with the largest foreground area is evaluated, and the chosen rule
is fixed rather than inferred at runtime.

## Component matching

Evaluation uses SciPy maximum-weight one-to-one component matching with voxel
overlap as the weight. Assignments with zero overlap are discarded. This
prevents one reference or prediction from inflating the true-positive count
when components fragment or merge.

## Numerical and runtime differences

The implementation targets Python 3.13 and modern PyTorch and SciPy releases.
CPU correctness paths use float32 without autocast. CUDA paths may use float16
autocast when explicitly enabled. Small numerical differences from historical
Python, PyTorch, CUDA, and scikit-image versions are expected and must be
recorded with the environment lockfile.

## Data and split differences

No restricted cohort or released checkpoint is bundled. Exact evaluation data,
subject split, and checkpoint provenance must be supplied by the researcher.
The package does not implement cross-validation.
