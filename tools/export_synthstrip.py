#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Convert SynthStrip PyTorch weights into RadVeil's `.rvss` format.

Usage:
    python tools/export_synthstrip.py synthstrip.1.pt \
        crates/radveil-core/models/synthstrip.1.rvss [--golden tests-golden.bin]

SynthStrip (Hoopes et al., NeuroImage 2022) weights are distributed by the
FreeSurfer team under the MIT license or CC BY 4.0 (licensee's choice):
https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/ . RadVeil uses them
under MIT; see crates/radveil-core/models/LICENSE-synthstrip.

Format (little endian):
    b"RVS1", u32 n_layers,
    per layer (encoder, decoder, remaining; forward order):
        u32 cin, u32 cout,
        f16 weight[27][cin][cout]   tap = (kd*3 + kh)*3 + kw
        f32 bias[cout]

`--golden` also writes a numerical reference for the Rust test: a seeded
random 64^3 input and the model's output, both float32 (needs torch).
"""

from __future__ import annotations

import argparse
import struct

import numpy as np
import torch

ORDER = (
    [f"encoder.{l}.{c}" for l in range(6) for c in range(2)]
    + [f"decoder.{l}.{c}" for l in range(6) for c in range(2)]
    + [f"remaining.{i}" for i in range(3)]
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("out")
    ap.add_argument("--golden")
    args = ap.parse_args()

    sd = torch.load(args.pt, map_location="cpu", weights_only=False)["model_state_dict"]
    assert len(sd) == 2 * len(ORDER), sorted(sd)
    with open(args.out, "wb") as f:
        f.write(b"RVS1")
        f.write(struct.pack("<I", len(ORDER)))
        for name in ORDER:
            w = sd[f"{name}.conv.weight"].numpy()  # (cout, cin, 3, 3, 3)
            b = sd[f"{name}.conv.bias"].numpy()
            cout, cin = w.shape[:2]
            f.write(struct.pack("<II", cin, cout))
            # -> [kd][kh][kw][cin][cout]
            f.write(np.ascontiguousarray(w.transpose(2, 3, 4, 1, 0)).astype("<f2").tobytes())
            f.write(b.astype("<f4").tobytes())
    print(f"wrote {args.out}")

    if args.golden:
        model = build_model()
        # Match the Rust engine: weights rounded to fp16.
        model.load_state_dict({k: v.half().float() for k, v in sd.items()})
        model.eval()
        rng = np.random.default_rng(0)
        x = rng.random((64, 64, 64), dtype=np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(x)[None, None]).squeeze().numpy()
        with open(args.golden, "wb") as f:
            f.write(x.astype("<f4").tobytes())
            f.write(y.astype("<f4").tobytes())
        print(f"wrote {args.golden} (output range {y.min():.3f}..{y.max():.3f})")


def build_model():
    """The SynthStrip U-Net (mri_synthstrip's StripModel, default config)."""
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, cin, cout, act=True):
            super().__init__()
            self.conv = nn.Conv3d(cin, cout, 3, 1, 1)
            self.act = nn.LeakyReLU(0.2) if act else None

        def forward(self, x):
            x = self.conv(x)
            return self.act(x) if self.act is not None else x

    enc = [(1, 16), (16, 16), (16, 32), (32, 32), (32, 64), (64, 64)] + [(64, 64)] * 6
    dec = [(64, 64)] * 2 + [(128, 64), (64, 64)] * 4 + [(96, 32), (32, 32)]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.ModuleList(
                nn.ModuleList([Block(*enc[2 * l]), Block(*enc[2 * l + 1])]) for l in range(6)
            )
            self.decoder = nn.ModuleList(
                nn.ModuleList([Block(*dec[2 * l]), Block(*dec[2 * l + 1])]) for l in range(6)
            )
            self.remaining = nn.ModuleList([Block(48, 16), Block(16, 16), Block(16, 1, act=False)])

        def forward(self, x):
            hist = [x]
            for convs in self.encoder:
                for c in convs:
                    x = c(x)
                hist.append(x)
                x = nn.functional.max_pool3d(x, 2)
            for convs in self.decoder:
                for c in convs:
                    x = c(x)
                x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                x = torch.cat([x, hist.pop()], dim=1)
            for c in self.remaining:
                x = c(x)
            return x

    return Net()


if __name__ == "__main__":
    main()
