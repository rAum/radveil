#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Batch example: deface a directory of NIfTI volumes concurrently."""

import sys

import radveil


def main():
    pattern, out_dir = sys.argv[1], sys.argv[2]
    reports = radveil.deface_batch(pattern, output_dir=out_dir, mode="adaptive", workers=4)
    ok = sum(1 for r in reports if "error" not in r)
    print(f"{ok}/{len(reports)} defaced")
    for r in reports:
        if "error" in r:
            print("NEEDS REVIEW:", r["input"], "->", r["error"])


if __name__ == "__main__":
    main()
