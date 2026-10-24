# Examples

Runnable scripts live in [`examples/`](../examples/).

## Step-by-step pipeline (full control)

```python
import radveil

vol = radveil.Volume("t1.nii.gz")
print(vol.orientation, vol.shape, vol.spacing)     # e.g. LPS (256, 256, 160)

views = [vol.render(yaw_deg=y) for y in (-20, 0, 20)]
lms = [radveil.detect_landmarks(v["rgb"]) for v in views]   # (68,2) or None
pts3d = radveil.lift_landmarks(views, lms)                  # 68 x (x,y,z) mm
report = vol.deface(pts3d, mode="cut", return_mask=True)
mask = report["mask"]                                       # audit trail
vol.save("out.nii.gz")
```

Each `view` dict carries `rgb` (H,W,3 uint8), `pos` (H,W,3 float32 world
position of the surface hit), `hit` and `boundary` masks — everything
needed to lift any 2D annotation into scanner space.

## In-memory (no files): arrays + affine

```python
import numpy as np, nibabel as nib, radveil

img = nib.load("t1.nii.gz")
vol = radveil.Volume.from_array(
    np.asanyarray(img.dataobj).astype(np.float32), img.affine
)
report = radveil.deface_volume(vol, mode="adaptive")
defaced = vol.data()                                # (nx, ny, nz) float32
```

This is the integration point for DICOM series (read with pydicom/SimpleITK, hand over the array and its
voxel-to-world affine in RAS+ mm, as nibabel uses), or any custom pipeline.

## Cross-series propagation (detect once, deface the whole session)

Landmarks live in world coordinates, so a face found on the 3D T1 can
deface co-registered 2D series where detection is harder:

```python
t1 = radveil.Volume("t1.nii.gz")
report = radveil.deface_volume(t1, mode="adaptive")
pts3d = report["landmarks_3d"]

flair = radveil.Volume("flair.nii.gz")
flair.deface(pts3d, mode="adaptive")
flair.save("flair_defaced.nii.gz")
t1.save("t1_defaced.nii.gz")
```

## CT

Nothing special — Hounsfield data is detected automatically (air floor
below −500 HU switches the surface threshold to −400 HU, and `cut` fills
with air rather than water-density 0):

```python
radveil.deface_auto("head_ct.nii.gz", "head_ct_defaced.nii.gz", mode="cut")
```

## Region only (process it yourself)

An escape hatch for manual or custom preprocessing: `anonymization_mask`
runs detection, validation and the safety refusals, and returns the exact
region `deface_auto` would rewrite, without touching the volume. It is
API-only (Python, Rust `radveil_core::anonymization_mask`, C
`radveil_anonymization_mask`); the CLI always anonymizes.

```python
rep = radveil.anonymization_mask("t1.nii.gz", mask_path="t1_face_region.nii.gz",
                                 brain_check=True)
mask = rep["mask"]          # uint8, the volume's voxel grid, 1 = would be rewritten
print(rep["voxels_changed"], rep["brain"])
```

Nothing is verified in this mode: the mask says where RadVeil would
anonymize, not that your processing of it defeats re-identification.
Check the result yourself, e.g. render it and run `radveil.detect_landmarks`
on the views, or run `deface_auto` on your output and expect a refusal.

## Batch with tuning

```python
reports = radveil.deface_batch(
    "cohort/*.nii.gz", output_dir="defaced/",
    mode="blur",                   # workers default: min(cpu_count, 32, n)
    blur_sigma_mm=12.0,            # explicit override of the auto sigma
    skin_depth_mm=4.0,             # shallower cut below the skin
    max_verify_rounds=5,
)
for r in reports:
    status = "REFUSED: " + r["error"] if "error" in r else "ok"
    print(r["input"], status)
```

Throughput, memory sizing and the service setup:
[deployment.md](deployment.md).

## C

```c
#include "radveil.h"
RadveilVolume *vol = radveil_volume_load("t1.nii.gz");
/* one call with the built-in detector: radveil_deface_auto(...), or
   render a view; run YOUR detector on rgb; lift via pos; then: */
radveil_deface(vol, points_68x3, "adaptive", NULL, NULL, &report);
radveil_volume_save(vol, "out.nii.gz", 1);
radveil_volume_free(vol);
```

Full example: [crates/radveil-capi/examples-c/smoke.c](../crates/radveil-capi/examples-c/smoke.c).
