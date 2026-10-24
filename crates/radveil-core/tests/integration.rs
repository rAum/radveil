// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests on a synthetic head phantom (no patient data needed).
#![allow(clippy::needless_range_loop, clippy::field_reassign_with_default)]

use ndarray::Array3;

use nifti::NiftiHeader;
use radveil_core::{
    apply, deface, face_frame_from_landmarks, plan, plan_mask, render_view, DefaceMode,
    DefaceParams, Region, RenderParams, Volume,
};

/// Build a synthetic "head": an ellipsoid with nose- and ear-like bumps,
/// stored LPS-ordered (like a typical scanner export) with scl scaling,
/// int16.
fn make_phantom(dir: &std::path::Path) -> std::path::PathBuf {
    make_phantom_with(dir, Ears::Pinnae)
}

/// What sits on the sides of the phantom head.
#[derive(Clone, Copy, PartialEq)]
enum Ears {
    /// Pinna-like bumps at the measured pinna position.
    Pinnae,
    /// No pinnae.
    None,
    /// No pinnae, but a speckle-like bump low on the cheek (the kind of
    /// facial noise that must not be taken for a pinna).
    CheekDecoy,
}

fn make_phantom_with(dir: &std::path::Path, ears: Ears) -> std::path::PathBuf {
    let n = 96usize;
    let mut data = Array3::<i16>::zeros((n, n, n));
    let c = (n / 2) as f64;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // LPS voxel axes: +i = Left, +j = Posterior, +k = Superior
                let x = i as f64 - c; // left
                let y = j as f64 - c; // posterior
                let z = k as f64 - c; // superior
                let head = (x / 30.0).powi(2) + (y / 35.0).powi(2) + (z / 38.0).powi(2);
                let mut v = if head <= 1.0 { 800.0 } else { 0.0 };
                // nose: bump on the anterior side (y negative in LPS)
                let nose =
                    (x / 6.0).powi(2) + ((y + 40.0) / 10.0).powi(2) + ((z + 5.0) / 8.0).powi(2);
                if nose <= 1.0 {
                    v = 800.0;
                }
                // ears: protruding bumps just off the lateral surface
                // (small air gap, like a real pinna standing off the head),
                // centered at world (±66, -4, -8): with the landmark set
                // used in the tests that is the measured pinna position,
                // 0.39 inter-eye distances below and 1.5 behind the eyes
                let ear = ((x.abs() - 33.0) / 2.5).powi(2)
                    + ((y - 2.0) / 6.0).powi(2)
                    + ((z + 4.0) / 7.0).powi(2);
                if ears == Ears::Pinnae && ear <= 1.0 {
                    v = 800.0;
                }
                // decoy: a smaller bump on the lower cheek, world
                // (±62, +26, -38) — 1.1 below and 0.8 behind the eyes
                let decoy = ((x.abs() - 31.0) / 2.5).powi(2)
                    + ((y + 13.0) / 4.0).powi(2)
                    + ((z + 19.0) / 4.0).powi(2);
                if ears == Ears::CheekDecoy && decoy <= 1.0 {
                    v = 800.0;
                }
                data[[i, j, k]] = v as i16;
            }
        }
    }
    let mut h = NiftiHeader::default();
    h.pixdim = [1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0];
    h.scl_slope = 0.5;
    h.scl_inter = 10.0;
    h.sform_code = 1;
    // LPS storage: voxel +i -> world -x (RAS Left), +j -> world -y, +k -> +z
    h.srow_x = [-2.0, 0.0, 0.0, 96.0];
    h.srow_y = [0.0, -2.0, 0.0, 96.0];
    h.srow_z = [0.0, 0.0, 2.0, -96.0];
    let path = dir.join(match ears {
        Ears::Pinnae => "phantom.nii",
        Ears::None => "phantom_noears.nii",
        Ears::CheekDecoy => "phantom_decoy.nii",
    });
    // Write through Volume::save so scl_slope/inter survive (plain nifti-rs
    // write resets them), storing scaled values in memory.
    let affine = [
        [-2.0, 0.0, 0.0, 96.0],
        [0.0, -2.0, 0.0, 96.0],
        [0.0, 0.0, 2.0, -96.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let scaled = data.mapv(|r| r as f32 * 0.5 + 10.0);
    let vol = Volume {
        accel: Default::default(),
        data: scaled,
        affine,
        world_to_voxel: radveil_core::affine::invert(&affine),
        spacing: [2.0, 2.0, 2.0],
        header: h,
        datatype: nifti::NiftiType::Int16,
        scl_slope: 0.5,
        scl_inter: 10.0,
    };
    vol.save(&path).unwrap();
    path
}

#[test]
fn phantom_pipeline() {
    let dir = std::env::temp_dir().join(format!("radveil_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);

    let v = Volume::load(&path).unwrap();
    assert_eq!(v.orientation(), "LPS");
    assert_eq!(v.shape(), [96, 96, 96]);
    // scl applied: 800 * 0.5 + 10 = 410
    let mx = v.percentile(99.0);
    assert!((mx - 410.0).abs() < 1.0, "p99 = {mx}");

    // Render frontal view: nose bump must be the closest surface near center.
    let view = render_view(&v, 0.0, 0.0, &RenderParams::default());
    let n_hits = view.hit.iter().filter(|&&h| h).count();
    assert!(n_hits > 10_000, "hits = {n_hits}");
    let center = view.world_at(256.0, 256.0, 8).unwrap();
    // The frontal surface near the middle should be on the anterior side.
    assert!(center[1] > 20.0, "front surface at {center:?}");

    // Hand-placed 3D landmarks on the phantom face (world RAS mm):
    // eyes above/left-right of the nose bump, mouth below.
    let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
    for i in 36..42 {
        lm[i] = Some([-22.0, 62.0, 10.0]); // right eye (world -x is patient right here)
    }
    for i in 42..48 {
        lm[i] = Some([22.0, 62.0, 10.0]); // left eye
    }
    lm[30] = Some([0.0, 80.0, -10.0]); // nose tip (bump apex ~ y=+80)
    for i in 48..68 {
        lm[i] = Some([0.0, 40.0, -35.0]); // mouth
    }
    lm[8] = Some([0.0, 50.0, -55.0]); // chin

    let frame = face_frame_from_landmarks(&lm).unwrap();
    assert!((frame.eye_dist - 44.0).abs() < 1.0);
    // forward must point anterior (+y)
    assert!(frame.forward[1] > 0.8, "forward = {:?}", frame.forward);

    // Cut mode: verify mask confinement and bit-exactness outside.
    let mut vc = Volume::load(&path).unwrap();
    let params = DefaceParams {
        mode: DefaceMode::Cut,
        ..Default::default()
    };
    let (report, mask) = deface(&mut vc, &frame, &params);
    assert!(report.voxels_changed > 1000);
    assert!(report.voxels_changed < report.voxels_total / 4);

    let orig = Volume::load(&path).unwrap();
    let mut outside_diff = 0usize;
    let mut inside_nonzero = 0usize;
    for ((idx, &m), (&a, &b)) in mask
        .indexed_iter()
        .zip(orig.data.iter().zip(vc.data.iter()))
    {
        let _ = idx;
        if m == 0 {
            if a != b {
                outside_diff += 1;
            }
        } else if b != 10.0 {
            // cut writes raw 0 => scaled value is scl_inter = 10
            // (in-memory it is set to scaled 0.0 before unscaling on save)
            inside_nonzero += 1;
        }
    }
    assert_eq!(outside_diff, 0, "voxels outside the mask were modified");
    let _ = inside_nonzero;

    // Save and confirm raw roundtrip: dtype/scaling preserved, voxels outside
    // the mask bit-identical to the original file.
    let out = dir.join("phantom_defaced.nii.gz");
    vc.save(&out).unwrap();
    let reread = Volume::load(&out).unwrap();
    assert_eq!(reread.datatype, orig.datatype);
    assert_eq!(reread.scl_slope, orig.scl_slope);
    assert_eq!(reread.scl_inter, orig.scl_inter);
    for ((&m, &a), &b) in mask.iter().zip(orig.data.iter()).zip(reread.data.iter()) {
        if m == 0 {
            assert_eq!(a, b, "outside-mask voxel changed after save/load");
        }
    }

    // Nose bump must be gone: frontal surface near image center receded.
    let after = render_view(&reread, 0.0, 0.0, &RenderParams::default());
    if let Some(c2) = after.world_at(256.0, 256.0, 8) {
        assert!(
            c2[1] < center[1] - 5.0,
            "face surface did not recede: {} -> {}",
            center[1],
            c2[1]
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn blur_and_block_modes() {
    let dir = std::env::temp_dir().join(format!("radveil_test_bb_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);

    let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
    for i in 36..42 {
        lm[i] = Some([-22.0, 62.0, 10.0]);
    }
    for i in 42..48 {
        lm[i] = Some([22.0, 62.0, 10.0]);
    }
    lm[30] = Some([0.0, 80.0, -10.0]);
    for i in 48..68 {
        lm[i] = Some([0.0, 40.0, -35.0]);
    }
    let frame = face_frame_from_landmarks(&lm).unwrap();

    for mode in [DefaceMode::Blur, DefaceMode::Block, DefaceMode::Adaptive] {
        let mut v = Volume::load(&path).unwrap();
        let orig = v.data.clone();
        let (report, mask) = deface(
            &mut v,
            &frame,
            &DefaceParams {
                mode,
                ..Default::default()
            },
        );
        assert!(report.voxels_changed > 1000);
        for ((&m, &a), &b) in mask.iter().zip(orig.iter()).zip(v.data.iter()) {
            if m == 0 {
                assert_eq!(a, b);
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn ear_regions() {
    let dir = std::env::temp_dir().join(format!("radveil_test_ear_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);

    let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
    for i in 36..42 {
        lm[i] = Some([-22.0, 62.0, 10.0]);
    }
    for i in 42..48 {
        lm[i] = Some([22.0, 62.0, 10.0]);
    }
    lm[30] = Some([0.0, 80.0, -10.0]);
    for i in 48..68 {
        lm[i] = Some([0.0, 40.0, -35.0]);
    }
    lm[8] = Some([0.0, 50.0, -55.0]);
    let frame = face_frame_from_landmarks(&lm).unwrap();

    // Voxel indices (LPS storage, world = [-2i+96, -2j+96, 2k-96]):
    let ear_a = [15usize, 50, 44]; // world (+66, -4, -8) — ear bump center
    let ear_b = [81usize, 50, 44]; // world (-66, -4, -8) — the other ear
    let nose = [48usize, 8, 43]; // world (0, 80, -10) — nose tip

    // Ears only, cut: both ear bumps rewritten, the face untouched.
    let orig = Volume::load(&path).unwrap();
    let mut ve = Volume::load(&path).unwrap();
    let (rep_ears, mask_ears) = deface(
        &mut ve,
        &frame,
        &DefaceParams {
            mode: DefaceMode::Cut,
            region: Region::Ears,
            ..Default::default()
        },
    );
    assert_eq!(rep_ears.region, Region::Ears);
    let labels: Vec<&str> = rep_ears.rois.iter().map(|r| r.label.as_str()).collect();
    assert_eq!(labels, ["ear_left", "ear_right"]);
    for r in &rep_ears.rois {
        assert!(r.voxels > 100, "{}: only {} voxels", r.label, r.voxels);
        assert_eq!(
            r.detected,
            Some(true),
            "{}: pinna not detected from surface evidence",
            r.label
        );
        let p = r.protrusion_mm.expect("protrusion reported when detected");
        assert!(p > 5.0, "{}: protrusion only {p:.1} mm", r.label);
    }
    assert_eq!(
        rep_ears.voxels_changed,
        rep_ears.rois.iter().map(|r| r.voxels).sum::<usize>()
    );
    for p in [ear_a, ear_b] {
        assert_eq!(mask_ears[p], 1, "ear voxel {p:?} not in mask");
        assert_eq!(ve.data[p], 0.0, "ear voxel {p:?} not cut");
    }
    assert_eq!(mask_ears[nose], 0, "nose voxel masked by ears region");
    assert_eq!(
        ve.data[nose], orig.data[nose],
        "nose modified by ears region"
    );
    for ((&m, &a), &b) in mask_ears.iter().zip(orig.data.iter()).zip(ve.data.iter()) {
        if m == 0 {
            assert_eq!(a, b, "voxel outside the ear mask was modified");
        }
    }

    // Face + ears: face ROI plus both ear ROIs, strictly more than either.
    let mut vb = Volume::load(&path).unwrap();
    let (rep_both, mask_both) = deface(
        &mut vb,
        &frame,
        &DefaceParams {
            mode: DefaceMode::Cut,
            region: Region::FaceAndEars,
            ..Default::default()
        },
    );
    let labels: Vec<&str> = rep_both.rois.iter().map(|r| r.label.as_str()).collect();
    assert_eq!(labels, ["face", "ear_left", "ear_right"]);
    assert_eq!(rep_both.rois[0].detected, None);
    assert!(rep_both.voxels_changed > rep_ears.voxels_changed);
    assert_eq!(mask_both[nose], 1);
    assert_eq!(mask_both[ear_a], 1);
    assert_eq!(vb.data[nose], 0.0);

    // Blur mode over ears: mask-confined, ear voxels actually altered.
    let mut vblur = Volume::load(&path).unwrap();
    let (_, mask_blur) = deface(
        &mut vblur,
        &frame,
        &DefaceParams {
            mode: DefaceMode::Blur,
            region: Region::Ears,
            ..Default::default()
        },
    );
    let mut inside_altered = 0usize;
    for ((&m, &a), &b) in mask_blur
        .iter()
        .zip(orig.data.iter())
        .zip(vblur.data.iter())
    {
        if m == 0 {
            assert_eq!(a, b);
        } else if a != b {
            inside_altered += 1;
        }
    }
    assert!(
        inside_altered > 100,
        "blur changed only {inside_altered} ear voxels"
    );

    // A head without pinnae: no protrusion evidence in the search window,
    // so the ROI must fall back to the prior and be flagged as undetected.
    // With only a cheek-level decoy bump the same holds: a protrusion far
    // from the measured pinna position is not taken for the ear.
    for kind in [Ears::None, Ears::CheekDecoy] {
        let mut vf = Volume::load(make_phantom_with(&dir, kind)).unwrap();
        let (rep_fb, _) = deface(
            &mut vf,
            &frame,
            &DefaceParams {
                mode: DefaceMode::Cut,
                region: Region::Ears,
                ..Default::default()
            },
        );
        for r in &rep_fb.rois {
            assert_eq!(r.detected, Some(false), "{}: expected fallback", r.label);
            assert!(r.protrusion_mm.is_none());
            assert!(
                r.voxels > 100,
                "{}: fallback ROI only {} voxels",
                r.label,
                r.voxels
            );
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Thick-slice scans lift the eye landmarks a few mm inside the head; the
/// eye-surface anchor must measure that bias and pull the face ROI's
/// posterior wall back out, sparing the tissue behind the true skin.
#[test]
fn eye_surface_anchor_corrects_deep_landmarks() {
    let dir = std::env::temp_dir().join(format!("radveil_test_anchor_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);

    let base: Vec<(std::ops::RangeInclusive<usize>, [f64; 3])> = vec![
        (36..=41, [-22.0, 62.0, 10.0]),
        (42..=47, [22.0, 62.0, 10.0]),
        (30..=30, [0.0, 80.0, -10.0]),
        (48..=67, [0.0, 40.0, -35.0]),
        (8..=8, [0.0, 50.0, -55.0]),
    ];
    let build = |shift_mm: f64, fwd: [f64; 3]| {
        let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
        for (range, p) in &base {
            for i in range.clone() {
                lm[i] = Some([
                    p[0] - shift_mm * fwd[0],
                    p[1] - shift_mm * fwd[1],
                    p[2] - shift_mm * fwd[2],
                ]);
            }
        }
        lm
    };

    let frame0 = face_frame_from_landmarks(&build(0.0, [0.0; 3])).unwrap();
    let mut v0 = Volume::load(&path).unwrap();
    let (rep0, _) = deface(
        &mut v0,
        &frame0,
        &DefaceParams {
            mode: DefaceMode::Cut,
            ..Default::default()
        },
    );

    // Same anatomy, landmarks displaced 6 mm posteriorly along `forward`.
    let lm_deep = build(6.0, frame0.forward);
    let frame1 = face_frame_from_landmarks(&lm_deep).unwrap();
    let mut v1 = Volume::load(&path).unwrap();
    let (rep1, mask_anchored) = deface(
        &mut v1,
        &frame1,
        &DefaceParams {
            mode: DefaceMode::Cut,
            ..Default::default()
        },
    );
    let gain = rep1.eye_bias_mm - rep0.eye_bias_mm;
    assert!(
        (gain - 6.0).abs() < 2.5,
        "bias should track the 6 mm landmark displacement: {:.1} -> {:.1}",
        rep0.eye_bias_mm,
        rep1.eye_bias_mm
    );

    // Opting out reverts to raw landmark-plane anchoring.
    let mut v2 = Volume::load(&path).unwrap();
    let (rep2, mask_raw) = deface(
        &mut v2,
        &frame1,
        &DefaceParams {
            mode: DefaceMode::Cut,
            eye_surface_anchor: Some(false),
            ..Default::default()
        },
    );
    assert_eq!(rep2.eye_bias_mm, 0.0);

    // An explicit bias pins the geometry (used by verification escalations,
    // which must not re-measure on an anonymized volume).
    let mut v3 = Volume::load(&path).unwrap();
    let (rep3, _) = deface(
        &mut v3,
        &frame1,
        &DefaceParams {
            mode: DefaceMode::Cut,
            eye_bias_mm: Some(rep1.eye_bias_mm),
            ..Default::default()
        },
    );
    assert!((rep3.eye_bias_mm - rep1.eye_bias_mm).abs() < 1e-9);

    // A voxel 5 mm behind the (displaced) eye plane lies 11 mm below the
    // true skin. The skin-following wall spares it either way; the anchor
    // now only positions the ROI hull.
    let p = radveil_core::affine::add(
        frame1.origin,
        radveil_core::affine::scale(frame1.forward, -5.0),
    );
    let vx = radveil_core::affine::apply(&v1.world_to_voxel, p);
    let idx = [
        vx[0].round() as usize,
        vx[1].round() as usize,
        vx[2].round() as usize,
    ];
    assert_eq!(
        mask_raw[idx], 0,
        "tissue below the skin depth must be spared"
    );
    assert_eq!(
        mask_anchored[idx], 0,
        "anchored wall must spare tissue behind the true skin"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// CT phantom in Hounsfield units: air -1000, soft tissue +40, skull +700.
/// Verifies the CT-aware auto threshold (skin surface, not bone) and that
/// cut mode fills with air rather than water-density zero.
#[test]
fn ct_phantom_pipeline() {
    let dir = std::env::temp_dir().join(format!("radveil_test_ct_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let n = 96usize;
    let c = (n / 2) as f64;
    let mut data = ndarray::Array3::<f32>::zeros((n, n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let x = i as f64 - c;
                let y = j as f64 - c;
                let z = k as f64 - c;
                let head = (x / 30.0).powi(2) + (y / 35.0).powi(2) + (z / 38.0).powi(2);
                let skull = (x / 26.0).powi(2) + (y / 31.0).powi(2) + (z / 34.0).powi(2);
                let nose =
                    (x / 6.0).powi(2) + ((y + 40.0) / 10.0).powi(2) + ((z + 5.0) / 8.0).powi(2);
                let mut v = -1000.0; // air
                if head <= 1.0 || nose <= 1.0 {
                    v = 40.0; // soft tissue
                }
                if (0.85..=1.0).contains(&skull) {
                    v = 700.0; // bone shell
                }
                data[[i, j, k]] = v;
            }
        }
    }
    let affine = [
        [-2.0, 0.0, 0.0, 96.0],
        [0.0, -2.0, 0.0, 96.0],
        [0.0, 0.0, 2.0, -96.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut h = NiftiHeader::default();
    h.pixdim = [1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0];
    h.sform_code = 1;
    h.srow_x = [-2.0, 0.0, 0.0, 96.0];
    h.srow_y = [0.0, -2.0, 0.0, 96.0];
    h.srow_z = [0.0, 0.0, 2.0, -96.0];
    let vol = Volume {
        accel: Default::default(),
        data,
        affine,
        world_to_voxel: radveil_core::affine::invert(&affine),
        spacing: [2.0, 2.0, 2.0],
        header: h,
        datatype: nifti::NiftiType::Int16,
        scl_slope: 1.0,
        scl_inter: 0.0,
    };
    let path = dir.join("ct_phantom.nii.gz");
    vol.save(&path).unwrap();

    let v = Volume::load(&path).unwrap();
    let view = render_view(&v, 0.0, 0.0, &RenderParams::default());
    // Auto threshold must pick the CT air/skin boundary.
    assert!(
        (view.threshold - -400.0).abs() < 1.0,
        "CT threshold = {}",
        view.threshold
    );
    // The frontal surface near the image center is the nose bump apex
    // (skin, y ~ +80mm), not the skull (y ~ +55mm).
    let p = view.world_at(256.0, 256.0, 8).unwrap();
    assert!(p[1] > 70.0, "hit skin at {p:?}");

    // Deface with the same hand-placed landmarks as the MRI phantom.
    let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
    for i in 36..42 {
        lm[i] = Some([-22.0, 62.0, 10.0]);
    }
    for i in 42..48 {
        lm[i] = Some([22.0, 62.0, 10.0]);
    }
    lm[30] = Some([0.0, 80.0, -10.0]);
    for i in 48..68 {
        lm[i] = Some([0.0, 40.0, -35.0]);
    }
    let frame = face_frame_from_landmarks(&lm).unwrap();
    let mut vc = Volume::load(&path).unwrap();
    let (report, mask) = deface(
        &mut vc,
        &frame,
        &DefaceParams {
            mode: DefaceMode::Cut,
            ..Default::default()
        },
    );
    assert!(report.voxels_changed > 1000);
    // Cut must fill with air (~-1000 HU), never 0 HU (water).
    let mut filled = Vec::new();
    for (&m, &val) in mask.iter().zip(vc.data.iter()) {
        if m == 1 {
            filled.push(val);
        }
    }
    let mean: f32 = filled.iter().sum::<f32>() / filled.len() as f32;
    assert!(mean < -900.0, "cut fill mean = {mean} HU");
    // Nose gone on re-render.
    let after = render_view(&vc, 0.0, 0.0, &RenderParams::default());
    let p2 = after.world_at(256.0, 256.0, 8).unwrap();
    assert!(
        p2[1] < p[1] - 5.0,
        "nose not removed: {} -> {}",
        p[1],
        p2[1]
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The built-in detector must not find a validated face on the phantom
/// (a nose bump is not a face), and a refusal must leave the volume
/// untouched.
#[cfg(feature = "detector")]
#[test]
fn deface_auto_refuses_faceless_phantom() {
    let dir = std::env::temp_dir().join(format!("radveil_test_auto_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);
    let mut vol = Volume::load(&path).unwrap();
    let before = vol.data.clone();
    let mut det = radveil_core::MediaPipeDetector::default();
    let res = radveil_core::deface_auto(&mut vol, &mut det, &radveil_core::AutoParams::default());
    assert!(res.is_err());
    assert_eq!(vol.data, before);
    std::fs::remove_dir_all(&dir).ok();
}

/// A detector that returns fixed per-yaw landmarks drives the pipeline
/// through shape consensus, anonymization and verification.
#[test]
fn deface_auto_with_custom_detector() {
    use radveil_core::pipeline::Detection;
    struct Known(Vec<(Vec<u8>, Vec<[f64; 2]>)>);
    impl radveil_core::FaceDetector for Known {
        fn detect(
            &mut self,
            rgb: &[u8],
            _w: usize,
            _h: usize,
            _rot: bool,
        ) -> Result<Option<Detection>, String> {
            Ok(self
                .0
                .iter()
                .find(|(img, _)| img == rgb)
                .map(|(_, lm)| Detection {
                    landmarks: lm.clone(),
                    score: 0.9,
                }))
        }
    }

    let dir = std::env::temp_dir().join(format!("radveil_test_auto2_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);
    let mut vol = Volume::load(&path).unwrap();
    // Phantom landmarks in world space (same set as the other tests).
    let mut lm3: Vec<Option<[f64; 3]>> = vec![None; 68];
    for i in 36..42 {
        lm3[i] = Some([-22.0, 62.0, 10.0]);
    }
    for i in 42..48 {
        lm3[i] = Some([22.0, 62.0, 10.0]);
    }
    lm3[30] = Some([0.0, 80.0, -10.0]);
    for i in 48..68 {
        lm3[i] = Some([0.0, 40.0, -35.0]);
    }
    let params = radveil_core::AutoParams {
        yaws: vec![-15.0, 0.0, 15.0],
        size: 256,
        ..Default::default()
    };
    let rp = RenderParams {
        size: 256,
        threshold_frac: params.threshold_frac,
        ..Default::default()
    };
    let mut known = Vec::new();
    for &y in &params.yaws {
        let view = render_view(&vol, y, 0.0, &rp);
        let n = view.camera.width;
        let pts: Vec<[f64; 2]> = lm3
            .iter()
            .map(|p| match p {
                None => [-1000.0, -1000.0],
                Some(p) => {
                    let mut best = (f64::INFINITY, [0.0, 0.0]);
                    for i in 0..n * n {
                        if !view.hit[i] {
                            continue;
                        }
                        let q = &view.pos[i * 3..i * 3 + 3];
                        let d = (0..3).map(|k| (q[k] as f64 - p[k]).powi(2)).sum::<f64>();
                        if d < best.0 {
                            best = (d, [(i % n) as f64, (i / n) as f64]);
                        }
                    }
                    best.1
                }
            })
            .collect();
        known.push((view.rgb.clone(), pts));
    }
    let rep = radveil_core::deface_auto(&mut vol, &mut Known(known), &params).unwrap();
    assert_eq!(rep.views_detected, 3);
    assert!(!rep.orientation_sweep);
    assert_eq!(rep.verify_rounds, Some(0));
    assert_eq!(rep.residual_detections, Some(0));
    assert!(rep.deface.voxels_changed > 1000);
    std::fs::remove_dir_all(&dir).ok();
}

/// One pass can use different modes for the face and the ears: the face
/// region is blurred (values stay in range, vary) while the ear regions are
/// cut to background, and everything outside the union mask is unchanged.
#[test]
fn per_region_modes() {
    let dir = std::env::temp_dir().join(format!("radveil_mixed_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);
    let orig = Volume::load(&path).unwrap();
    let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
    for i in 36..42 {
        lm[i] = Some([-22.0, 62.0, 10.0]);
    }
    for i in 42..48 {
        lm[i] = Some([22.0, 62.0, 10.0]);
    }
    lm[30] = Some([0.0, 80.0, -10.0]);
    for i in 48..68 {
        lm[i] = Some([0.0, 40.0, -35.0]);
    }
    lm[8] = Some([0.0, 50.0, -55.0]);
    let frame = face_frame_from_landmarks(&lm).unwrap();

    let mut v = Volume::load(&path).unwrap();
    let (rep, mask) = deface(
        &mut v,
        &frame,
        &DefaceParams {
            mode: DefaceMode::Blur,
            ear_mode: Some(DefaceMode::Cut),
            region: Region::FaceAndEars,
            ..Default::default()
        },
    );
    assert_eq!(rep.mode, DefaceMode::Blur);
    assert_eq!(rep.ear_mode, DefaceMode::Cut);
    assert!(mask.iter().all(|&m| m <= 1), "exported mask must be binary");

    // Face-only and ears-only masks from separate runs identify the regions.
    let region_mask = |region| {
        let mut w = Volume::load(&path).unwrap();
        deface(
            &mut w,
            &frame,
            &DefaceParams {
                mode: DefaceMode::Block,
                region,
                ..Default::default()
            },
        )
        .1
    };
    let face = region_mask(Region::Face);
    let ears = region_mask(Region::Ears);

    let fill = orig.percentile(1.0).min(0.0);
    let mut ear_voxels = 0usize;
    let mut face_changed = 0usize;
    for ((idx, &m), (&a, &b)) in mask.indexed_iter().zip(orig.data.iter().zip(v.data.iter())) {
        if m == 0 {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "voxel {idx:?} outside mask changed"
            );
        } else if ears[idx] == 1 && face[idx] == 0 {
            ear_voxels += 1;
            assert_eq!(b, fill, "ear voxel {idx:?} not cut");
        } else if face[idx] == 1 && (a - b).abs() > 1e-3 {
            assert_ne!(b, fill);
            face_changed += 1;
        }
    }
    assert!(ear_voxels > 100, "only {ear_voxels} ear voxels");
    assert!(
        face_changed > 100,
        "blur changed only {face_changed} face voxels"
    );

    // ear_mode = None keeps the single-mode behaviour.
    let mut u = Volume::load(&path).unwrap();
    let (rep_same, _) = deface(
        &mut u,
        &frame,
        &DefaceParams {
            mode: DefaceMode::Cut,
            region: Region::FaceAndEars,
            ..Default::default()
        },
    );
    assert_eq!(rep_same.ear_mode, DefaceMode::Cut);
    std::fs::remove_dir_all(&dir).ok();
}

/// The filesystem-free I/O used by the WebAssembly build: `from_bytes` +
/// `encode_like` must produce exactly what `load` + `save` produce, from
/// both plain and gzipped input.
#[test]
fn byte_io_matches_file_io() {
    let dir = std::env::temp_dir().join(format!("radveil_test_bytes_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);
    let raw = std::fs::read(&path).unwrap();
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    std::io::Write::write_all(&mut gz, &raw).unwrap();
    let gz = gz.finish().unwrap();

    let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
    for i in 36..42 {
        lm[i] = Some([-22.0, 62.0, 10.0]);
    }
    for i in 42..48 {
        lm[i] = Some([22.0, 62.0, 10.0]);
    }
    lm[30] = Some([0.0, 80.0, -10.0]);
    for i in 48..68 {
        lm[i] = Some([0.0, 40.0, -35.0]);
    }
    let frame = face_frame_from_landmarks(&lm).unwrap();
    let params = DefaceParams::default();

    let mut file_vol = Volume::load(&path).unwrap();
    deface(&mut file_vol, &frame, &params);
    let out = dir.join("out.nii");
    file_vol.save(&out).unwrap();
    let expected = std::fs::read(&out).unwrap();

    for input in [&raw, &gz] {
        let mut v = Volume::from_bytes(input).unwrap();
        assert_eq!(v.data, Volume::load(&path).unwrap().data);
        deface(&mut v, &frame, &params);
        assert_eq!(v.encode_like(input).unwrap(), expected);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// with the face, it reaches the frontal lobe at brow height on real heads.
#[test]
fn face_roi_stays_skin_deep() {
    let dir = std::env::temp_dir().join(format!("radveil_test_skin_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);

    // Upright, and with the mouth pushed forward so `up` leans back ~20°;
    // in the face frame and in the level head frame.
    for (mouth_y, level_face) in [(40.0, false), (62.0, false), (40.0, true), (62.0, true)] {
        let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
        for (range, p) in [
            (36..=41, [-22.0, 62.0, 10.0]),
            (42..=47, [22.0, 62.0, 10.0]),
            (30..=30, [0.0, 80.0, -10.0]),
            (48..=67, [0.0, mouth_y, -35.0]),
            (8..=8, [0.0, 50.0, -55.0]),
        ] {
            for i in range {
                lm[i] = Some(p);
            }
        }
        let frame = face_frame_from_landmarks(&lm).unwrap();
        let mut v = Volume::load(&path).unwrap();
        let (rep, mask) = deface(
            &mut v,
            &frame,
            &DefaceParams {
                mode: DefaceMode::Cut,
                level_face,
                ..Default::default()
            },
        );
        assert!(rep.voxels_changed > 0);
        // Phantom head: ellipsoid with semi-axes 30/35/38 voxels (2 mm).
        let shrink = (rep.resolved.skin_depth_mm + 4.0) / 2.0;
        let c = 48.0;
        let mut deep = 0;
        for ((i, j, k), &m) in mask.indexed_iter() {
            let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
            let inner = (x / (30.0 - shrink)).powi(2)
                + (y / (35.0 - shrink)).powi(2)
                + (z / (38.0 - shrink)).powi(2);
            if m == 1 && inner <= 1.0 {
                deep += 1;
            }
        }
        assert_eq!(
            deep, 0,
            "mouth_y {mouth_y}, level_face {level_face}: {deep} voxels changed deeper than \
             skin_depth below the surface"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// rewrites, and leaves the volume untouched.
#[test]
fn plan_mask_matches_apply_and_modifies_nothing() {
    let dir = std::env::temp_dir().join(format!("radveil_test_planmask_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);
    let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
    for (range, p) in [
        (36..=41, [-22.0, 62.0, 10.0]),
        (42..=47, [22.0, 62.0, 10.0]),
        (30..=30, [0.0, 80.0, -10.0]),
        (48..=67, [0.0, 40.0, -35.0]),
        (8..=8, [0.0, 50.0, -55.0]),
    ] {
        for i in range {
            lm[i] = Some(p);
        }
    }
    let frame = face_frame_from_landmarks(&lm).unwrap();
    let params = DefaceParams {
        mode: DefaceMode::Cut,
        region: Region::FaceAndEars,
        ..Default::default()
    };

    let v = Volume::load(&path).unwrap();
    let p = plan(&v, &frame, &params);
    let (rep, mask) = plan_mask(&v, &frame, &p, params.mode);
    assert_eq!(v.data, Volume::load(&path).unwrap().data, "volume modified");
    assert_eq!(rep.voxels_changed, mask.iter().filter(|&&m| m == 1).count());

    let mut w = Volume::load(&path).unwrap();
    let (rep2, mask2) = apply(&mut w, &frame, &p, params.mode, None);
    assert_eq!(mask, mask2);
    assert_eq!(rep.voxels_changed, rep2.voxels_changed);
    assert_eq!(rep.protected_voxels_spared, None);

    // A protected set (the brain guard) is never rewritten: protect the
    // front half of the region and it drops out of the mask, counted.
    let mut protect = ndarray::Array3::<u8>::zeros(mask.raw_dim());
    let marked: Vec<_> = mask
        .indexed_iter()
        .filter(|(_, &m)| m == 1)
        .map(|(i, _)| i)
        .collect();
    for &i in &marked[..marked.len() / 2] {
        protect[i] = 1;
    }
    let guarded = p.clone().with_protected(protect.clone());
    let mut g = Volume::load(&path).unwrap();
    let before = g.data.clone();
    let (rep3, mask3) = apply(&mut g, &frame, &guarded, params.mode, None);
    assert_eq!(rep3.protected_voxels_spared, Some(marked.len() / 2));
    assert_eq!(rep3.voxels_changed, rep.voxels_changed - marked.len() / 2);
    for (idx, &pr) in protect.indexed_iter() {
        if pr == 1 {
            assert_eq!(mask3[idx], 0);
            assert_eq!(g.data[idx], before[idx], "protected voxel rewritten");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// `spare_air` writes a subset of the default region, never far air, and
/// gives every voxel it writes the same value as the default. (Face only:
/// with ears too, the ear blur reads the face pass's output where the
/// regions meet, which legitimately differs.)
#[test]
fn spare_air_leaves_far_air_untouched() {
    let dir = std::env::temp_dir().join(format!("radveil_test_spareair_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = make_phantom(&dir);
    let mut lm: Vec<Option<[f64; 3]>> = vec![None; 68];
    for (range, p) in [
        (36..=41, [-22.0, 62.0, 10.0]),
        (42..=47, [22.0, 62.0, 10.0]),
        (30..=30, [0.0, 80.0, -10.0]),
        (48..=67, [0.0, 40.0, -35.0]),
        (8..=8, [0.0, 50.0, -55.0]),
    ] {
        for i in range {
            lm[i] = Some(p);
        }
    }
    let frame = face_frame_from_landmarks(&lm).unwrap();
    let orig = Volume::load(&path).unwrap();
    let thr = 0.14 * orig.percentile(99.0);
    for mode in [DefaceMode::Cut, DefaceMode::Blur, DefaceMode::Block] {
        let run = |spare_air: bool| {
            let mut v = Volume::load(&path).unwrap();
            let params = DefaceParams {
                mode,
                region: Region::Face,
                spare_air,
                ..Default::default()
            };
            let p = plan(&v, &frame, &params);
            let (rep, mask) = apply(&mut v, &frame, &p, mode, None);
            (v, rep, mask)
        };
        let (off, rep_off, mask_off) = run(false);
        let (on, rep_on, mask_on) = run(true);
        assert!(
            rep_on.voxels_changed < rep_off.voxels_changed,
            "{mode:?}: nothing spared"
        );
        assert_eq!(
            rep_on.air_voxels_spared,
            Some(rep_off.voxels_changed - rep_on.voxels_changed)
        );
        assert_eq!(rep_off.air_voxels_spared, None);
        for (idx, &m) in mask_on.indexed_iter() {
            if m == 1 {
                assert_eq!(
                    mask_off[idx], 1,
                    "{mode:?}: wrote outside the default region"
                );
                assert_eq!(
                    on.data[idx], off.data[idx],
                    "{mode:?}: written value differs"
                );
            } else {
                assert_eq!(
                    on.data[idx], orig.data[idx],
                    "{mode:?}: unmasked voxel changed"
                );
                if mask_off[idx] == 1 && mode != DefaceMode::Block {
                    // Skipped by the air test (not the shell): air that
                    // stays air.
                    assert!(
                        orig.data[idx] < thr || off.data[idx] == orig.data[idx],
                        "{mode:?}: tissue skipped"
                    );
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
