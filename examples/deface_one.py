#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Minimal example: deface one NIfTI volume with QA renders."""

import sys

import radveil

src, dst = sys.argv[1], sys.argv[2]
report = radveil.deface_auto(src, dst, mode="adaptive", qa_dir="qa", mask_path=dst.replace(".nii", "_mask.nii"))
print(f"defaced {report['voxels_changed']} voxels "
      f"({report['views_detected']} views, "
      f"residual detections: {report['residual_detections']})")
