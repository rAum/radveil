# Legal & data-governance notes

*This is engineering documentation, not legal advice.*

RadVeil is Copyright 2026 Piotr Gródek and the RadVeil contributors, and
is licensed under the Apache License, Version 2.0. It is a free,
open-source project developed in the author's spare time. It is not
affiliated with, sponsored or endorsed by, and does not represent any
current or former employer of the author.

## Licensing

| Component | License | Notes |
|---|---|---|
| RadVeil (all crates + Python package) | Apache-2.0 | |
| Rust dependencies (85 crates, incl. the web demo's wasm-bindgen) and the Rust standard library | MIT / Apache-2.0 / BSD / ISC / Zlib and similar, no copyleft | full list and texts in `THIRD_PARTY_LICENSES.txt` |
| Built-in face detector models (MediaPipe BlazeFace + Face Mesh v2, embedded) | Apache-2.0 (Google, per both model cards) | permissive, commercial-friendly; converted format, see `crates/radveil-core/models/` |
| SynthStrip brain-extraction weights (embedded in default builds) | MIT (SynthStrip authors; also offered as CC BY 4.0) | permissive, commercial-friendly; notice in `THIRD_PARTY_LICENSES.txt` and `crates/radveil-core/models/LICENSE-synthstrip` |
| Website fonts (`site/fonts/`: Archivo, Source Serif 4, IBM Plex Mono) | SIL Open Font License 1.1 | used by the project website only, not part of any package; licenses and sources in `site/fonts/README.md` |
| dlib (runtime, optional; includes its HOG face-box detector) | Boost Software License 1.0 | permissive, commercial-friendly; training-data note below |
| dlib 5-point landmark model (optional; RadVeil's dlib default) | CC0-1.0 (dlib model repository) | permissive, commercial-friendly; downloaded by `fetch_predictor()`, checksums pinned |
| **dlib 68-point landmark models (optional, never selected implicitly)** | **research use** | see below |

The default detector (`radveil auto`, `radveil.deface_auto`,
`radveil_deface_auto`) uses only the embedded MediaPipe models, so a
default installation contains no research-only component. Provenance and
conversion of the models: `crates/radveil-core/models/README.md`.

### Commercial use: summary

A default installation (wheel, crate, CLI or C library) contains only
Apache-2.0 and permissively licensed components, all of which permit
commercial use, modification and closed-source redistribution. What
redistributors must do:

- **Keep the notices.** Ship `LICENSE`, `NOTICE` and
  `THIRD_PARTY_LICENSES.txt` with any binary you distribute (the Python
  wheel, the crates and the release archives of the CLI and C library
  already contain them). If you modify RadVeil
  files, mark them as changed (Apache-2.0 §4).
- **Do not add dlib's 68-point models** to a commercial product unless you
  have a license for their training data (see below). The 5-point default
  needs no such license.
- **Model-card scope.** Google's model cards place human-life-critical
  decisions and facial recognition/identification out of scope (usage
  guidance, not license conditions). RadVeil locates faces only to remove
  them; it never identifies, compares or stores faces. It is a
  de-identification aid and not a medical device. Clinical or regulated
  deployments need their own validation.
- **Names.** Apache-2.0 grants no trademark rights: describe the models
  as "MediaPipe models by Google", but don't brand your product with
  Google's or MediaPipe's names. The same goes for SynthStrip and dlib.
  RadVeil itself names them only to identify the components it uses; it
  is not a product of, or endorsed by, any of their owners.
- **SynthStrip.** Keep the MIT notice (it is in
  `THIRD_PARTY_LICENSES.txt`). Its authors ask that published results
  cite Hoopes et al., NeuroImage 260 (2022) 119474.

`tools/third_party_licenses.py --check` runs in CI and fails when a
dependency is added outside the license allow-list or the notices file is
out of date.

### The optional dlib detector

Only relevant with `detector="dlib"` or `fallback_detector="dlib"`. The
default detector is the built-in MediaPipe one; dlib is a second,
independent option (its own models, its own code), so RadVeil does not
depend on a single vendor's detector. dlib is not a dependency:
`pip install radveil` does not install it, nothing imports it unless you
ask for that detector, and the wheel, crates, CLI and C library contain
no dlib code or model.

- **Landmark model (default): `shape_predictor_5_face_landmarks.dat`.**
  The dlib model repository publishes it under **CC0-1.0**, without the
  iBUG restriction of the 68-point models; its publisher trained it on
  7,198 internet images with independently created annotations.
  `radveil.fetch_predictor()` downloads it and verifies pinned SHA-256
  checksums (archive and model); it is not bundled. That is a permissive
  model grant, not a warranty that every underlying photograph's rights
  were cleared; the publisher gives no image-by-image rights ledger.
  The model measures four eye corners and the nose base; RadVeil infers a
  mouth centre (0.35 inter-eye distances below the nose) and uses no jaw
  points. Reports say so (`landmark_model_points: 5`,
  `mouth_landmark_inferred: true`).
- **Face-box detector: dlib's built-in HOG detector**
  (`get_frontal_face_detector`). Its weights ship inside dlib under the
  Boost license; dlib's source says it was trained on images from Labeled
  Faces in the Wild, a public dataset of news photographs without an
  explicit license. Same provenance boundary as above.
- **68-point models (`shape_predictor_68_face_landmarks.dat`, GTX
  variant): not distributed and never selected implicitly.** They were
  trained on iBUG 300-W, whose license **excludes commercial use** per
  dlib's model documentation. RadVeil uses one only when you pass its
  path, and then warns once. Commercial users need permission from
  Imperial College London (iBUG).

Organizations that need a fully auditable training-data lineage should
plug in their own detector (`detector=callable`; `Volume.deface` and the
C API also accept 68 world-space points from any source) and check that
detector's model license themselves.

## Privacy regulation context

RadVeil reduces the facial re-identification risk of shared head imaging
(relevant to HIPAA de-identification and GDPR pseudonymization
practices). It is one layer of a de-identification process, **not** the
whole process:

- **Header metadata is deliberately untouched.** Patient name, IDs,
  dates, and site information in NIfTI descriptions/extensions or
  associated files (JSON sidecars, DICOM) must be scrubbed by your
  existing pipeline.
- Verify outputs per your governance: the built-in check (re-running the
  face detector on renders of the result) demonstrates resistance to
  that class of attack, not to every possible one. The exported binary
  mask supports auditing exactly which voxels changed.
- The pipeline **refuses loudly** when it cannot find a plausible face;
  treat refusals as items for human review, never as silently processed.

## Not a medical device

RadVeil is a research/data-management tool. It performs no diagnostic,
therapeutic, or clinical decision function and is not certified as a
medical device (no FDA/MDR clearance). Defaced volumes remain suitable
for research analysis, but any clinical use of processed data is the
operator's responsibility.

Integrators who build RadVeil into a regulated product can use the
per-release CycloneDX SBOMs and the vulnerability-handling process in
[sbom.md](sbom.md) for their SOUP/OTS documentation.

## Evaluation data

The repository contains **no imaging data**. The only rendered face is in
the website images (`site/img/`): 3D Slicer's public "MRHead" sample, which
its donor, the person pictured, gave to the 3D Slicer project "to be used
without any restrictions" (Slicer's `SampleData` module).
`tools/make_site_images.py` regenerates those images from the
checksum-verified download. Tests use synthetic phantoms generated at
runtime; the binary fixtures in `crates/radveil-core/tests/data/` are
seeded random inputs, a random-weight test model and the matching
reference outputs. The brain-safety and re-detection figures in the
documentation and on the website come from the public SynthStrip
evaluation set (`experiments/face_brain_safety.py`); the ear figures were
measured on public IXI scans (`experiments/ears/`) and one clinical scan
that is not distributed: the author's private scan. Publishing figures that show a recognizable
(pre-defacing) face requires the subject's consent.
