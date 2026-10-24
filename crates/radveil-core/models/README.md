# Embedded models

## SynthStrip (brain extraction)

`synthstrip.1.rvss` holds the SynthStrip U-Net (Hoopes et al., NeuroImage
2022) that runs `deskull`'s default method: 27 3×3×3 convolutions, 2.57M
parameters, float16 weights and float32 biases, 5.1 MB. It is converted
from the official `synthstrip.1.pt` by `tools/export_synthstrip.py` and
runs on RadVeil's own engine (`src/synthstrip.rs`: shifted-GEMM
convolutions on `matrixmultiply`, no PyTorch). Its output matches PyTorch
to < 2e-3 mm on the golden test input
(`tests/data/synthstrip_golden.bin`, a seeded random 64³ input and the
PyTorch output, written by `tools/export_synthstrip.py --golden`; no imaging
data). License: MIT, as offered by the
SynthStrip authors (see `LICENSE-synthstrip`).

## RadVeil brain (`.rvb`, brain extraction)

`radveil deskull --method rvbrain --model m.rvb` runs RadVeil's own, smaller
brain-extraction network from a `.rvb` file. **No weights ship yet**: the file
is passed on the command line (or `$RADVEIL_RVBRAIN_MODEL`) until a model
earns being embedded. To embed one, drop it at `rvbrain.rvb` and set
`EMBEDDED` in `src/rvbrain.rs` to
`Some(include_bytes!("../models/rvbrain.rvb"))`; `models/*.rvb` is already in
the crate's `include` list. Mind the budget: the crate is 7.7 MB (compressed)
of the 10 MB crates.io limit.

Unlike `.rvss` above, a `.rvb` carries its own architecture — grid spacing,
level count, tail length, activation slope, mask threshold and every layer's
shape — so one engine runs any model in the family and a mismatched file is
rejected rather than read as garbage. The format is documented in full at the
top of `src/rvbrain.rs`, and the engine is pinned by
`rvbrain::tests::matches_numpy_reference`, which runs a tiny model with seeded random
weights (`tests/data/rvbrain_golden.rvb`, 15,658 parameters; not a brain
extractor) on a seeded random input and compares against the output of an
independent numpy implementation (`tests/data/rvbrain_golden.bin`). Neither
test file contains imaging data or trained weights.

A model trained by this project and distilled from SynthStrip's predictions
(which the SynthStrip MIT license permits) would be covered by this
repository's license; the training data used must be recorded alongside it
before any such model ships.

# Embedded face models

RadVeil's built-in face detector runs two networks from Google's
[MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
on RadVeil's own inference engine (`src/nn.rs`, pure Rust with SIMD
kernels). No ML runtime, MediaPipe framework, Bazel, TensorFlow or native
library is needed to build or run RadVeil.

| File | Network | Input | Outputs |
|---|---|---|---|
| `blazeface_short_range.rvnn` | BlazeFace (short range) face detector | 128×128×3, RGB in [-1, 1] | 896 SSD anchors × (box + 6 keypoints), scores |
| `face_mesh_v2.rvnn` | Face Mesh v2 landmark model | 256×256×3, RGB in [0, 1] | 478 × (x, y, z) in crop pixels, face-presence logit |

`.rvnn` is a flat op list (convolution, depthwise convolution, residual
add, max-pool, channel padding, output-head concat, logistic) with padding
resolved, ReLU/PReLU fused into the producing op, and the float16 weights
as published. Pre- and post-processing (SSD anchor decoding, weighted
NMS, the eye-aligned 1.5× ROI crop, the 478 → 68-point index map) are in
`src/detect.rs`.

## License

The model weights are © Google LLC, released under the **Apache License
2.0** (both model cards state "LICENSED UNDER Apache License, Version
2.0"); the license text is in `LICENSE` next to this file, attribution in
the repository `NOTICE`, and the model cards are linked from Google's
[Face Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).

**Modifications (Apache-2.0 §4(b)):** these files are not the original
TFLite models. RadVeil re-encoded them into `.rvnn` (a lossless change of
container: same operations, same float16 weights). The pre- and
post-processing that MediaPipe's framework performs around the models
was reimplemented independently in `src/detect.rs`.

**Intended use.** The model cards list human-life-critical decisions and
facial recognition/identification as out of scope. RadVeil uses the
detector only to *locate and remove* faces from rendered head scans, and
never stores, compares or identifies faces.

## Provenance / how to regenerate

`tools/convert_mediapipe_models.py` downloads the float16
`face_landmarker.task` bundle and runs `tools/export_rvnn.py --verify` on
its TFLite files. The exporter refuses unsupported ops and checks its
NumPy reference interpreter against TensorFlow Lite's interpreter.

- source: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
  (sha256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`)
- `face_detector.tflite` (sha256 `b4578f35…0152f`) → `blazeface_short_range.rvnn`
  (sha256 `5480f4075c2f3af79122fadd2288bf74e106e95b46873668951f1182796fb0f3`)
- `face_landmarks_detector.tflite` (sha256 `c7d54204…d43d5`) → `face_mesh_v2.rvnn`
  (sha256 `60f97aff1fd8d511b9c3d27c0b4e56ef371d059a06b341aed3991e58297c8c04`)

The export is deterministic. `tests/data/nn_golden.bin` holds TensorFlow
Lite's outputs on fixed inputs; the engine must match them to 1e-4
relative (`cargo test`).

## Performance (Apple M-series, one core)

| | BlazeFace | Face Mesh v2 | model load |
|---|---|---|---|
| RadVeil engine | 1.2 ms | 4.0 ms | 4 ms |
| tract 0.23 (previous) | 4.4 ms | 17.5 ms | 130 ms |
| ONNX Runtime 1.30 (reference) | 1.0 ms | 3.2 ms | — |

`cargo run --release --example nn_bench -p radveil-core` (`PROFILE=1` for
a per-op breakdown).

## Evaluated and rejected

- **Face Mesh v1** (192 px, 4× faster): landmarks agree with v2 to 0.05·d,
  but its face-presence output is ~0 even on real faces. Without that
  gate, verification kept re-detecting blurred heads (2.4 vs 0.6
  escalation rounds per subject) and one defaced output was accepted.
- **tract** (NNEF): correct, but 4× slower, 130 ms model load, and 13 MB
  of library code.
