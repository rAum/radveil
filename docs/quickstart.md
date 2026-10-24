# Quickstart

## Install

```bash
pip install radveil
```

RadVeil ships as a prebuilt wheel for Linux (x86-64/aarch64), macOS
(x86-64/arm64) and Windows, Python ≥ 3.9. The face detector is built in
(MediaPipe BlazeFace + Face Mesh, Apache-2.0, run in Rust): no model
download, no compiler, no extra dependency.

Without Python, the CLI does the same in one step:

```bash
cargo install radveil-cli
radveil auto t1.nii.gz t1_defaced.nii.gz --mode adaptive --report report.json
```

**Optional: dlib.** A second, independent detector is available as
`detector="dlib"`, or as a fallback for scans the default detector rejects
(`fallback_detector="dlib"`): `pip install radveil[dlib]` (compiles from
source with CMake) plus `radveil.fetch_predictor()` for its default model,
dlib's 5-point predictor (~9 MB, CC0-1.0, checksum-verified; see
[legal.md](legal.md)). Any other 68-point landmark
detector can be plugged in as a callable (`detector=f(rgb,
try_rotations)`, see [parameters.md](parameters.md)).

## Deface one volume

```python
import radveil

report = radveil.deface_auto(
    "t1.nii.gz", "t1_defaced.nii.gz",
    mode="adaptive",                    # blur | adaptive | cut | block
    qa_dir="qa",                        # optional before/after renders
)
print(report)
# {'views_detected': 3, 'residual_detections': 0, 'voxels_changed': ...,
#  'eye_dist_mm': 63.1, 'verify_rounds': 0, ...}
```

That's the whole pipeline: render → detect → lift to 3D → anonymize →
**verify by re-running the detector** → save. Every parameter has an
auto default scaled from the detected face size ([parameters.md](parameters.md)).

What the guarantees mean:

- `residual_detections: 0` — the face detector no longer finds a face in
  renders of the output (escalating automatically until it doesn't).
- Every voxel outside the reported mask is **bit-exact** identical to
  the input; datatype, `scl_slope/inter` scaling and header are preserved.
- A `RuntimeError` means no plausible face was found — the tool refuses
  loudly instead of guessing (e.g. on already-defaced or non-head data).

## Deface a cohort

```python
import radveil

reports = radveil.deface_batch("cohort/*.nii.gz", output_dir="defaced/", mode="adaptive")
refused = [r for r in reports if "error" in r]   # no validated face: review these
```

Volumes run concurrently on threads using all cores. See
[deployment.md](deployment.md) for throughput, memory sizing and running
RadVeil as an on-the-fly service (`Volume.from_bytes` / `to_bytes`).

## CLI (no Python needed for rendering/masking)

```bash
cargo install radveil-cli               # or build from source
radveil auto t1.nii.gz out.nii.gz --region face+ears --report report.json
radveil deskull t1.nii.gz --mask-out brain.nii.gz   # brain extraction
radveil info t1.nii.gz                  # shape, spacing, LPS/RAS, affine
radveil render t1.nii.gz -o renders/
radveil deface t1.nii.gz out.nii.gz --mode adaptive --landmarks lm.json
```

`radveil auto` runs the whole pipeline with the built-in detector, as
`deface_auto` does in Python. `render` + `deface` are the step-by-step
route for landmarks from your own detector — see [examples.md](examples.md).

## Verify a result yourself

```python
vol = radveil.Volume("t1_defaced.nii.gz")
view = vol.render(yaw_deg=0)
radveil.detect_landmarks(view["rgb"])   # -> None on properly defaced data
```
