---
title: Paper Traceability
description: Mapping between the MicrobleedNet paper and this implementation
author: MicrobleedNet contributors
ms.date: 2026-08-01
ms.topic: reference
---

## Paper Traceability

## Phase 5 model geometry

The detector preserves the current U-Net-like path and maps two-channel
`48 x 48 x 48` inputs to two-channel outputs of the same spatial size.

The discriminator classifier uses the repository's current channel schedule.
Its 1x1 projection uses explicit zero padding, so a `24 x 24 x 24` input is
reduced by the two pooling layers to one spatial voxel. The flattened feature
count is therefore the projected channel count, not the historical hard-coded
1,024 features. The classifier asserts this geometry at runtime.

This is an implementation difference from the historical 1,024-feature
assumption. No adaptive pooling or silent dense-layer reshaping was introduced.