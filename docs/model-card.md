---
title: MicrobleedNet Model Card
description: Intended use, limitations, and known risks for the MicrobleedNet research implementation
author: MicrobleedNet contributors
ms.date: 2026-08-01
ms.topic: concept
---

## Model summary

MicrobleedNet is a three-stage research workflow for candidate detection and
candidate classification of cerebral microbleeds in T2*-GRE or SWI MRI. The
stages are a 3D detector, a segmentation/classification teacher, and a
knowledge-distilled classification student.

The repository does not include a validated clinical checkpoint or a
redistributable training cohort. No performance claim is made here.

## Intended use

The model is intended for research experiments, reproducibility checks, and
method development on legally usable datasets with compatible annotations. It
is not intended for diagnosis, triage, treatment decisions, or autonomous
clinical use.

## Data and populations

The supported modality contracts are T2*-GRE and SWI NIfTI volumes. The
available repository does not define a representative population statement
because no training cohort is distributed with the package. Users must record
cohort demographics, scanner vendors, acquisition protocols, preprocessing
settings, subject split, and annotation protocol with every experiment.

## Known failure modes

* Scanner, field strength, sequence, and reconstruction differences can change
  intensity and artifact distributions.
* Motion, susceptibility artifacts, vessels, sulci, and incomplete brain masks
  can produce false candidates or missed lesions.
* Very small, boundary-touching, or partially annotated lesions are sensitive
  to patch padding and annotation conventions.
* Missing or mismatched image and mask geometry invalidates training and
  evaluation.
* The output depends on detector and discriminator thresholds and should not be
  compared across runs without recording them.

## Limitations

QSM is unsupported. Cross-validation and external validation are outside the
current workflow. Results depend on scanner, acquisition, pathology,
preprocessing, annotation conventions, and the exact train/test split.

## Evaluation cautions

Evaluation uses one-to-one maximum-overlap component matching. This is an
explicit disambiguation for fragmented references and merged predictions; it
is not a claim that the matching rule is the only clinically meaningful choice.
Threshold tuning and final evaluation must use separate data and provenance.

## Safety and reporting

Report the dataset access status, subject split, checkpoint identifiers,
configuration files, software version, device, thresholds, and per-subject
component matches. Do not publish restricted images, masks, or derived data
without authorization.
