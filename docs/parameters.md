# Parameters & tuning

Design rule: **every geometric parameter has an auto default scaled from
the detected inter-eye distance `d`** (~60-65 mm adult). You should
rarely need to set anything; when you do, explicit values are in
millimeters and override the auto scaling.

## Rendering / detection

| Parameter | Default | Notes |
|---|---|---|
| `size` | 512 | render resolution = detector input. 512 is the validated attacker strength; don't lower it to save time on the verify pass |
| `yaws`, `pitch` | (-20, 0, 20), 0 | ±20° detects best; a 6-direction sweep with image rotations engages automatically when these fail |
| `threshold_frac` | 0.14 | surface threshold as a fraction of the p99 intensity (MRI). Raise for noisy backgrounds, lower for dim skin |
| `threshold` | auto | absolute override; CT (p1 < −500) auto-selects −400 HU |
| `detector` | `"mediapipe"` | built-in MediaPipe detector; `"dlib"` (optional install) or a callable `f(rgb, try_rotations) -> None \| (68,2) array \| (array, score)`. Whatever the detector, verification also attacks the output with the built-in one |
| `fallback_detector` | None | same values: used when `detector` yields no validated face (the volume is untouched by then). Measured on the SynthStrip scans MediaPipe refused: `"dlib"` processed 64 of 381, including real faces MediaPipe missed on IXI PD/T2 and QIN T1, and also 37 already-defaced FSM scans |
| `fallback_brain_guard` | True | a fallback run uses the brain guard (those scans are the unusual ones: without it, 6 of the 64 above touched the brain; with it, none) |
| `detector_upsample` | 1 | dlib only: pyramid upsampling; 0 is tried first automatically |
| `predictor_path` | 5-point default | dlib only: an explicit path or `RADVEIL_SHAPE_PREDICTOR` wins; otherwise only `shape_predictor_5_face_landmarks.dat` (CC0, `radveil.fetch_predictor()`) is looked for. 68-point models are never picked up implicitly and warn about their license |

Detections are validated before anything is modified: views whose lifted
3D landmarks agree (median distance ≤ 0.45·d over brows/nose/eyes/mouth)
form a consensus group, and a group of ≥ 2 views is accepted. A face seen
in a single view must be re-found within 0.25·d from ≥ 4 of 7 renders
around its implied face axis (0°, ±20°, ±35° yaw, ±20° pitch).
Verification counts a re-detection only if it lies within 0.75·d of the
anonymized face.

## Region of interest (face frame units)

| Parameter | Auto | Meaning |
|---|---|---|
| `width_factor` | 1.05 | ROI half-width as a multiple of `d` |
| `above_eyes_mm` | 0.60·d | margin above the eye line (covers brows/forehead edge) |
| `below_chin_mm` | 0.43·d | margin below the chin; chin position itself is clamped to ≤ 2d below the eyes |
| `skin_depth_mm` | 0.07·d (~4.5 mm) | **brain-safety bound**: the face ROI's back wall follows the *measured* skin, this far below it (nose and lips are cut down to the surrounding skin level). A column whose first tissue was not seen through in-scan air is a field-of-view cut and is not touched; if most eye/cheek columns are like that, `deface_auto` refuses. Measured on the SynthStrip set (201 scans processed, expert brain masks): 4 touch the brain, ≤ 0.06 ml each, all already-defaced FSM scans; each extra mm of depth costs about 1 mm of margin |
| `depth_behind_mm` | 0.21·d | hull floor: the skin-following wall never reaches further behind the eye-skin surface than this |
| `eye_surface_anchor` | True | anchor the depth bounds to the *measured* eye-skin surface instead of the raw landmark plane. Thick-slice scans lift the eye landmarks a few mm deep, which would push `depth_behind_mm` into the frontal lobe; the measured bias is reported as `eye_bias_mm` and is ~0 on well-resolved scans |
| `eye_bias_mm` | measured | explicit bias override (mm). Pass a previously reported value to pin the geometry across repeated calls — verification escalations do this automatically, since the bias cannot be re-measured on an already-anonymized volume |
| `depth_front_mm` | 1.55·d | forward extent (past the nose tip) |
| `level_face` | off | measure the face skin along the level head axes (eye axis + scanner superior, the ear frame) instead of the face frame, whose forward axis tilts ~25–30° down. Along the tilted axis the chin hides the lower lip and the neck hides the chin, so the skin-deep wall leaves them in place: on 38 SynthStrip scans the skin was removed at 41% of lower-lip and 12% of chin landmarks, with `level_face` at 78% and 70% (`experiments/face_part_coverage.py`). SynthStrip set (582 scans, expert brain masks): 213 processed instead of 201 (12 scans whose face was refused as "not seen through air" are now handled), 0 faces left, brain mask touched on 7 scans, at most 0.05 ml: the same 4 already-defaced FSM scans as without it, 2 more FSM scans (4 and 7 voxels) and one 6 mm-slice QIN scan (2 voxels of bone-level intensity at the mask's edge); median clearance 6.4 mm; with `brain_guard` 0 scans touched (CLI `--level-face`, C `RADVEIL_LEVEL_FACE`) |
| `spare_air` | off | leave background air untouched. A region voxel is rewritten only if it is tissue before or after (air a blur leaves as air, or a cut sets to background, keeps its value), and nothing is written beyond a smooth shell ~1.6σ around the original tissue. If a shell-bounded block still leaves a detectable face, verification fills the whole region as without the flag. On 37 SynthStrip scans: median changed volume −49% (blur) / −60% (cut), tissue changed and residual faces unchanged; the report's `air_voxels_spared` counts first-round voxels left as they were (CLI `--spare-air`, C `RADVEIL_SPARE_AIR`) |

## Ear regions (ear frame units)

The pinna is a stable biometric identifier; `region="ears"` /
`"face+ears"` anonymize it with the same modes as the face. No landmark
detector emits ear points, so ears are located from **surface evidence**
in an *ear frame*: the eye axis from the face frame, `up` = the scanner's
superior axis made orthogonal to it, `forward` anterior. (The face frame's
own `up` runs mouth → eyes on the skin and tilts by 30-40° when the mouth
lands on a truncated field of view, which moved the pinna by more than an
ear height; a volume whose scanner axes do not put the head upright —
more than 60° off — keeps the face frame.)

Inside a search window each (v,w) column is ray-marched laterally to
build a surface height map; a morphological opening of that map gives
the scalp baseline. Protrusion blobs are segmented at several levels
(3-10 mm, so a flattened pinna merged with scalp bulges at the lowest
level still separates higher up) and gated by pinna plausibility: away
from the window border and the head edge, pinna-sized, and centred within
3 SD of the measured pinna position. Skin speckle on the cheek or a
jaw-angle ridge is rejected rather than taken as the ear. The hull is
the chosen blob's padded box, grown to at least a pinna-sized box around
its centre. With no plausible blob the fallback box (which spans every
measured pinna) is used and the report's per-ear `detected` flag is
false; when detected, `protrusion_mm` records the evidence strength.

The medial cut wall, per column: under the detected pinna, the scalp
interpolated from the blob's rim minus `ear_guard_mm` (severs the pinna
at its root); on other protrusions the surface minus `ear_guard_mm`; on
flat scalp the surface minus 0.75·`ear_guard_mm`. It never goes deeper
than the local median scalp minus 0.15·d, nor medial of `ear_inner_mm`,
nor medial of the first dark run below the scalp — the skull's outer
table or the air of the ear canal — so a mis-estimated scalp (thin or
dark scalp on T2/PD) cannot carry the cut into the brain.

**Calibration.** Pinnae hand-marked on true lateral renders of 53 IXI T1
subjects (106 ears) and a clinical T1 (not distributed), lifted to 3D through the
render's position map: pinna centre `v` −0.39 ± 0.19·d, `w` −1.50 ±
0.12·d; pinna surface (2nd-98th percentile) `v` −1.37..0.66·d, `w`
−2.19..−0.82·d. The clinical scan lies inside the IXI spread in this frame.

**Measured** (block mode, [IXI T1](https://brain-development.org/ixi-dataset/) + the clinical author's head T1, 108 ears): 107 detected, every
detected blob centred on the marked pinna, ≥95% of the marked pinna
surface inside the ROI for all 108 (median 99.0%); 0 voxels of the
SynthStrip expert brain masks modified on the 50 [XI T1 that have them
(clearance median 4.8 mm, 5th percentile 2.6 mm). Across the other 185
processable SynthStrip MRI scans (T1/T2/PD, adult and infant) 8 touch
the expert mask, at most 55 voxels (0.055 ml) — 5 are [IXI T2](https://brain-development.org/ixi-dataset/), where the
scalp beside the ear is thin and dim, and one is an FSM scan whose prior
defacing left the brain at the surface. With `brain_guard` on, 7 of the
8 drop to 0 (clearance ≥ 1.6 mm); the infant T1 drops from 22 to 6 voxels,
where SynthStrip's own brain mask, which the guard protects, misses part of
the expert mask (`experiments/ears/guard_ears.py`).

| Parameter | Auto | Meaning |
|---|---|---|
| `ear_inner_mm` | 0.75·d | absolute medial floor — the cut never reaches medial of this |
| `ear_guard_mm` | 0.05·d | cut depth below the scalp under the pinna (and below the surface on other protrusions; 0.75× on flat scalp) |
| `ear_outer_mm` | 2.20·d | lateral bound (past any protruding pinna) |
| `ear_down_mm` | 0.35·d | fallback box centre, below the eye line (ear-frame v axis) |
| `ear_back_mm` | 1.50·d | fallback box centre, behind the eye plane (ear-frame w axis) |
| `ear_height_mm` | 2.30·d | fallback box extent along v |
| `ear_depth_mm` | 1.60·d | fallback box extent along w |

The search window always covers the calibrated range (`v` −1.7..1.0·d,
`w` −2.5..−0.55·d) and grows to include the fallback box when that is
moved. Per-ear `v_mm`/`w_mm`/`u_mm` in the report are in the ear frame.
`RADVEIL_EAR_DEBUG=<prefix>` dumps each ear's height/baseline maps,
candidate blobs and cut wall as JSON.

## Modes & region

| Parameter | Auto | Notes |
|---|---|---|
| `mode` | — | `blur`, `adaptive` (recommended), `cut`, `block` |
| `region` | `face` | `face`, `ears`, `face+ears`. Verification is face-detector based, so it is skipped for `ears` |
| `ear_mode` | same as `mode` | separate mode for the ear regions in the same pass, e.g. `mode="adaptive", ear_mode="cut"` (CLI `--ear-mode`, C `radveil_deface_regions`). `cut` or `block` is recommended for ears: blur leaves 10–16% of the pinna surface in place |
| `blur_sigma_mm` | 0.17·d | blur/adaptive kernel scale |
| `adaptive_factor` | 2.5 | heavy-kernel multiple where facial lines are strong |

## Verification

| Parameter | Default | Notes |
|---|---|---|
| `verify` | True | re-render + re-detect the anonymized volume |
| `max_verify_rounds` | 3 | blur escalation rounds (σ ×1.6 each) |
| `verify_effort` | `balanced` | extra views verification attacks the output from, centred on the best detection view: `fast` none (detection views only), `balanced` a 3×3 grid (yaw −25/0/+25 × pitch −20/0/+20), `thorough` 5×5 (yaw −45…+45 × pitch −30…+30). On 60 SynthStrip scans attacked from 20 angles none of the levels uses, a face was still found on 8 (`fast`), 4 (`balanced`) and 4 (`thorough`) scans, at a median 0.7 / 1.1 / 1.8 s per scan (CLI `--verify-effort`, C `verify` = `RADVEIL_VERIFY_FAST` / `_ON` / `_THOROUGH`) |
| `partial_face` | off | faces cut by the field of view (an axial slab ending at the nose): if the upright views validate no face, also detect at ±15° pitch; build the frame from eyes and nose when the mouth landmarks lie on the cut; measure the skin like `level_face`; verify on every detection yaw × pitch −30…+30, ignoring "faces" on the cut's cross-section. SynthStrip set: 75 of 147 such scans anonymized, 0 faces left, no faceless scan accepted, but 95 instead of 49 already-defaced scans are processed again (1–2 brain-mask voxels on 2 of them) (CLI `--partial-face`, C `RADVEIL_PARTIAL_FACE`) |
| `auto_threshold` | off | noisy, dim scans (e.g. infant T1s, whose background noise renders as a solid shell at 0.14 × p99): when no face is found, retry at 0.3 and 0.45 × p99. Ignored with an explicit threshold or on CT. SynthStrip set: all 8 refused infant scans processed, no faceless scan accepted, but 21 already-defaced scans re-processed (4 touch 3–31 brain-mask voxels). Infant faces are small, so prefer `cut` or `block` (CLI `--auto-threshold`, C `RADVEIL_AUTO_THRESHOLD`) |
| `verify_pitches` | (0, −15) | camera pitches re-attacked at every detected yaw; the −15° view catches orbit structure a cut or heavy blur exposes to a downward-tilted attacker |
| `escalate_to_block` | True | final opaque fill if detection persists (cut/blur can expose skull orbits that re-detect) |

## I/O & runtime

| Parameter | Default | Notes |
|---|---|---|
| `gzip_level` | 1 | 0-9. Level 1 ≈ 5× faster than 6, ~3% larger |
| `workers` (`deface_batch`) | min(cpu_count, 32, n) | threads in one process (`executor="processes"` for worker processes; rayon threads divided automatically) |
| `RAYON_NUM_THREADS` | all cores | per-process CPU parallelism |

## Brain safety check

| Parameter | Default | Notes |
|---|---|---|
| `brain_check` | off | extract the brain from the input and count modified brain voxels; CLI exit status 3 when > 0 |
| `brain_guard` | off | extract the brain from the input and never rewrite it or anything within `brain_guard_mm` (3 mm) of it; the report's `protected_voxels_spared` counts what it kept. SynthStrip set: 0 scans touched, minimum clearance 3.6 mm. Costs ~3 s and ~3 GB per scan (CLI `--brain-guard`, C flag `RADVEIL_BRAIN_GUARD`) |

## Brain extraction (deskull)

| Parameter | Default | Notes |
|---|---|---|
| `method` | synthstrip | `synthstrip` (network, any contrast/CT, mask includes CSF), `rvbrain` (RadVeil's own smaller network, needs `model`), or `classical` (model-free); the rows below except `modality`/`fill` are classical-only |
| `model` | – | rvbrain: the `.rvb` weights (CLI also reads `$RADVEIL_RVBRAIN_MODEL`); no trained weights ship yet. CLI only (`--model`); the Python `deskull` runs `synthstrip` and `classical` |
| `cubic` | on | rvbrain, CLI only: resample the predicted distance with Catmull-Rom cubic rather than trilinear — the field is smooth, so this recovers boundary detail the coarser grid gives up (`--no-cubic` to compare) |
| `modality` | auto | `mri` / `ct`; auto = CT when the 1st percentile < −500 HU |
| `frac` | 0.5 | MRI fractional intensity threshold (BET `-f`); smaller = larger brain |
| `face` | off | locate the face and use it as a prior (MRI); falls back to plain when no face is found or the result is < 200 ml |
| `robust` | on | re-centre on the extracted brain and re-run (≤3 passes) |
| `fill` | 0 / air | value outside the brain in the stripped volume |
| `low_memory` | off | synthstrip: run the half-resolution decoder level in slabs too — ~0.4 GB less, ~3% slower, identical output |
| `ct_window` | −15..100 HU | soft tissue considered brain/CSF |
| `ct_bone_hu` | 250 | bone threshold; tissue with bone in ≥5 of 6 axis directions is intracranial |
| `ct_erode_mm` | 3 | erosion that separates the brain from extracranial tissue |

`radveil conform` exposes the same grid construction on its own
(`--voxel-mm`, `--multiple`, `--min`, `--max`, or `--model` to take them from
a `.rvb`): the volume reoriented to LIA, resampled isotropically, cropped to
the head, padded to the network's block size and intensity normalised.

Face prior geometry lives in the eyes+nose frame (the mouth is not used,
because it often lies on the field-of-view cut plane). The brain floor table
in `deskull_floor.rs` is generated by `experiments/deskull_calibrate.py`
from ground-truth masks of 58 IXI T1/T2/PD scans with a detected face. Each
node is the 15th percentile of the subjects' floors, minus a 0.1·d margin.

## Tuning recipes

- **More aggressive anonymization**: `mode="block"`, or raise
  `blur_sigma_mm` / `width_factor`.
- **Preserve maximal anatomy**: `mode="blur"` with default margins keeps
  gross facial morphology; check the QA renders.
- **Tight FOV (face cut below nose)**: nothing to do — boundary-flagged
  jaw landmarks are rejected and the chin clamp bounds the ROI.
- **Unusual head sizes (pediatric, phantom)**: nothing to do — auto
  scaling follows the measured eye distance.
- **Face not detected**: inspect `Volume(...).render()["rgb"]` visually;
  adjust `threshold_frac` if the surface render looks wrong (noise shell
  or missing skin), or supply landmarks manually via `Volume.deface`.
