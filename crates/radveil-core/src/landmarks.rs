// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! 2D->3D landmark lifting and face coordinate frame.
//!
//! Landmarks follow the standard 68-point facial-landmark layout:
//! 0-16 jaw, 17-26 brows, 27-35 nose (30 = tip), 36-41 right eye (image
//! left), 42-47 left eye, 48-67 mouth.

use crate::affine::{add, cross, dot, normalize, scale, sub, Vec3};
use crate::render::RenderView;
use serde::{Deserialize, Serialize};

/// 2D landmarks detected on one rendered view, in pixel coordinates
/// (x right, y down) of that view.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ViewLandmarks {
    pub yaw_deg: f64,
    pub pitch_deg: f64,
    pub points: Vec<[f64; 2]>,
}

/// Lift one view's 2D landmarks into world space via its position map.
///
/// Points are assumed to follow 68-point indexing: jaw-line landmarks
/// (0-16) are rejected when they fall on the volume boundary — with a
/// tight field of view the chin/jaw often lands on the scan cut plane,
/// which is not anatomy. Central features (eyes/nose/mouth) keep boundary
/// hits, since dropping them entirely would lose the face frame; the
/// defacer's anthropometric clamps bound their influence instead.
pub fn unproject(view: &RenderView, points_2d: &[[f64; 2]]) -> Vec<Option<Vec3>> {
    points_2d
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i <= 16 {
                view.world_at_anatomical(p[0], p[1], 4)
            } else {
                view.world_at(p[0], p[1], 4)
            }
        })
        .collect()
}

/// Combine per-view 3D landmark sets into one (coordinate-wise median).
pub fn combine_views(per_view: &[Vec<Option<Vec3>>]) -> Vec<Option<Vec3>> {
    if per_view.is_empty() {
        return Vec::new();
    }
    let n = per_view.iter().map(|v| v.len()).max().unwrap_or(0);
    (0..n)
        .map(|i| {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut zs = Vec::new();
            for v in per_view {
                if let Some(Some(p)) = v.get(i) {
                    xs.push(p[0]);
                    ys.push(p[1]);
                    zs.push(p[2]);
                }
            }
            if xs.is_empty() {
                None
            } else {
                Some([median(&mut xs), median(&mut ys), median(&mut zs)])
            }
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn mean_of(landmarks: &[Option<Vec3>], idx: std::ops::RangeInclusive<usize>) -> Option<Vec3> {
    let mut acc = [0.0; 3];
    let mut n = 0.0;
    for i in idx {
        if let Some(Some(p)) = landmarks.get(i) {
            acc = add(acc, *p);
            n += 1.0;
        }
    }
    if n == 0.0 {
        None
    } else {
        Some(scale(acc, 1.0 / n))
    }
}

/// Orthonormal face-aligned frame in world space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FaceFrame {
    /// Midpoint between the eyes (on the skin surface) — frame origin.
    pub origin: Vec3,
    /// Unit axis across the face (eye to eye).
    pub right: Vec3,
    /// Unit axis up the face (mouth to forehead).
    pub up: Vec3,
    /// Unit axis out of the face (posterior to anterior).
    pub forward: Vec3,
    /// Inter-eye-center distance in mm (scale reference).
    pub eye_dist: f64,
    /// Key points, world mm.
    pub eye_l: Vec3,
    pub eye_r: Vec3,
    pub nose_tip: Vec3,
    pub mouth: Vec3,
    pub chin: Option<Vec3>,
    /// The lifted mouth was implausible (dim lips lift onto tissue behind
    /// the face on T2/PD) and `mouth` was re-placed on the nose-derived
    /// axis; the deface widens its protrusion zone to absorb the larger
    /// frame uncertainty.
    #[serde(default)]
    pub mouth_reconstructed: bool,
}

/// Face `up` derived from the nose axis alone: the line from the upper
/// dorsum (landmarks 27-28) to the tip/nostril cluster (30-35), projected
/// perpendicular to `right` and tilted by [`NOSE_LEAN_DEG`]. The nose is
/// the most reliably lifted feature — bright, protruding skin that renders
/// in every contrast — unlike the lips, which are dim on T2/PD and lift
/// onto tissue far behind the face. Returns None when the nose points are
/// missing, the axis is too short, or its geometry fails the sanity gates.
fn nose_up(lm: &[Option<Vec3>], origin: Vec3, right: Vec3, eye_dist: f64) -> Option<Vec3> {
    let upper = mean_of(lm, 27..=28)?;
    let lower = mean_of(lm, 30..=35)?;
    // The upper dorsum sits between the eyes: far away means the nose
    // landmarks themselves mis-lifted, and the axis must not vote.
    if crate::affine::norm(sub(upper, origin)) > NOSE_NASION_MAX * eye_dist {
        return None;
    }
    let axis = sub(upper, lower);
    let a_perp = sub(axis, scale(right, dot(axis, right)));
    if crate::affine::norm(a_perp) < NOSE_AXIS_MIN * eye_dist {
        return None;
    }
    let a_perp = normalize(a_perp);
    let f0 = cross(right, a_perp);
    if crate::affine::norm(f0) < 1e-6 {
        return None;
    }
    let mut f0 = normalize(f0);
    // f0 points out of the face: the nose tip lies in front of the eye
    // plane. Too small a margin means the geometry is unreliable.
    if dot(f0, sub(lower, origin)) < 0.0 {
        f0 = scale(f0, -1.0);
    }
    if dot(f0, sub(lower, origin)) < NOSE_FORWARD_MIN * eye_dist {
        return None;
    }
    // The dorsum leans forward of the face's `up`; rotate it back.
    let lean = NOSE_LEAN_DEG.to_radians();
    Some(normalize(sub(
        scale(a_perp, lean.cos()),
        scale(f0, lean.sin()),
    )))
}

/// Build the face frame from combined 3D landmarks (68-point indexing).
///
/// The vertical axis normally comes from the mouth (eyes->mouth), but the
/// lifted mouth is validated against the nose axis: lips below the render
/// threshold (T2/PD) lift onto tissue far behind the face and would pitch
/// the whole frame (and the deface ROI with it). A mouth whose direction
/// disagrees with the nose axis by more than [`MOUTH_MAX_TILT_DEG`], or
/// whose distance falls outside [`MOUTH_DIST_RANGE`], is discarded and
/// re-placed on the nose-derived axis at the calibrated distance; the
/// frame then reports `mouth_reconstructed`.
pub fn face_frame_from_landmarks(lm: &[Option<Vec3>]) -> Result<FaceFrame, String> {
    let eye_r = mean_of(lm, 36..=41).ok_or("missing right-eye landmarks")?;
    let eye_l = mean_of(lm, 42..=47).ok_or("missing left-eye landmarks")?;
    let nose_tip = lm
        .get(30)
        .copied()
        .flatten()
        .or_else(|| mean_of(lm, 27..=35))
        .ok_or("missing nose landmarks")?;
    let mouth_lifted = mean_of(lm, 48..=67).ok_or("missing mouth landmarks")?;
    let chin = lm.get(8).copied().flatten();

    let origin = scale(add(eye_l, eye_r), 0.5);
    let eye_dist = crate::affine::norm(sub(eye_l, eye_r));
    if !(15.0..=150.0).contains(&eye_dist) {
        return Err(format!(
            "implausible inter-eye distance {:.1} mm — landmark lifting failed",
            eye_dist
        ));
    }
    let right0 = normalize(sub(eye_l, eye_r));
    let mouth_dist = crate::affine::norm(sub(origin, mouth_lifted)) / eye_dist;
    let mouth_dir = normalize(sub(origin, mouth_lifted));
    let nose = nose_up(lm, origin, right0, eye_dist);
    let mouth_ok = (MOUTH_DIST_RANGE.0..=MOUTH_DIST_RANGE.1).contains(&mouth_dist)
        && nose.is_none_or(|n| {
            dot(mouth_dir, n).clamp(-1.0, 1.0).acos().to_degrees() <= MOUTH_MAX_TILT_DEG
        });
    let (up0, mouth, mouth_reconstructed) = if mouth_ok {
        (mouth_dir, mouth_lifted, false)
    } else if let Some(n) = nose {
        // Mouth re-placed where it would be on this face if the lips had
        // lifted onto the skin: along the nose-derived down axis at the
        // calibrated distance.
        (n, sub(origin, scale(n, MOUTH_DIST * eye_dist)), true)
    } else {
        return Err(format!(
            "lifted mouth is implausible (eyes->mouth {:.2} x eye-dist) and the nose \
             axis is unusable — landmark lifting failed",
            mouth_dist
        ));
    };
    let mut forward = normalize(cross(right0, up0));
    // Orient forward out of the face: the nose tip protrudes forward of the
    // eyes/mouth plane.
    let face_center = scale(add(add(eye_l, eye_r), mouth), 1.0 / 3.0);
    if dot(forward, sub(nose_tip, face_center)) < 0.0 {
        forward = scale(forward, -1.0);
    }
    // Re-orthonormalize. Keep `up` anchored to the mouth->eyes direction
    // (Gram-Schmidt against forward) so its sign never depends on the
    // handedness of the intermediate triad; `right`'s sign is irrelevant
    // because the ROI is symmetric across it.
    let up = normalize(sub(up0, scale(forward, dot(up0, forward))));
    let right = normalize(cross(up, forward));

    Ok(FaceFrame {
        origin,
        right,
        up,
        forward,
        eye_dist,
        eye_l,
        eye_r,
        nose_tip,
        mouth,
        chin,
        mouth_reconstructed,
    })
}

/// Eyes -> mouth distance as a multiple of the inter-eye distance, and the
/// angle (degrees) by which the face frame's `up` leans forward of the
/// head's superior axis (eye landmarks lift onto the eyelids, anterior of
/// the lips). Used to place a mouth the scan does not contain.
const MOUTH_DIST: f64 = 1.15;
const UP_LEAN_DEG: f64 = 27.0;

// Nose-axis calibration, measured on 54 IXI runs (45 with a clean mouth
// lift; T1 plus the T2/PD scans whose mouth agreed with their own T1):
// dorsum lean -11.3 deg median, spread -26..+3; frames with a corrupted
// mouth sat at +32..+63, so the 25 deg gate splits the two populations
// with ~15 deg of margin on either side.
/// Lean of the nose dorsum (upper dorsum -> tip axis) forward of the
/// face's `up`, degrees.
const NOSE_LEAN_DEG: f64 = 11.3;
/// A lifted mouth must agree with the nose-derived `up` within this
/// angle, degrees.
const MOUTH_MAX_TILT_DEG: f64 = 25.0;
/// Plausible eyes->mouth distance, x inter-eye distance (measured
/// 0.82..1.14 on clean lifts).
const MOUTH_DIST_RANGE: (f64, f64) = (0.72, 1.35);
/// The upper dorsum must sit within this distance of the eye midpoint,
/// x inter-eye distance (measured <= 0.45).
const NOSE_NASION_MAX: f64 = 0.55;
/// Minimum length of the nose axis perpendicular to `right`, x inter-eye
/// distance (measured >= 0.47).
const NOSE_AXIS_MIN: f64 = 0.25;
/// The tip/nostril cluster must sit at least this far in front of the eye
/// plane, x inter-eye distance (measured >= 0.14).
const NOSE_FORWARD_MIN: f64 = 0.08;

/// Face frame for a face whose mouth is outside the field of view (an
/// axial slab cut at the nose): the mouth is placed where it would be
/// relative to the eyes in a level head (scanner superior = up), so the
/// frame matches what [`face_frame_from_landmarks`] would build on a whole
/// face. Needs eyes and nose; `chin` is left unknown.
pub fn face_frame_without_mouth(lm: &[Option<Vec3>]) -> Result<FaceFrame, String> {
    let eye_r = mean_of(lm, 36..=41).ok_or("missing right-eye landmarks")?;
    let eye_l = mean_of(lm, 42..=47).ok_or("missing left-eye landmarks")?;
    let nose_tip = lm
        .get(30)
        .copied()
        .flatten()
        .or_else(|| mean_of(lm, 27..=35))
        .ok_or("missing nose landmarks")?;
    let origin = scale(add(eye_l, eye_r), 0.5);
    let ed = crate::affine::norm(sub(eye_l, eye_r));
    let right = normalize(sub(eye_l, eye_r));
    let sup: Vec3 = [0.0, 0.0, 1.0];
    let up = sub(sup, scale(right, dot(sup, right)));
    if crate::affine::norm(up) < 1e-3 {
        return Err("eye axis is parallel to the scanner's superior axis".into());
    }
    let up = normalize(up);
    let mut fwd = normalize(cross(right, up));
    if dot(fwd, sub(nose_tip, origin)) < 0.0 {
        fwd = scale(fwd, -1.0);
    }
    let lean = UP_LEAN_DEG.to_radians();
    let down = add(scale(up, lean.cos()), scale(fwd, lean.sin()));
    let mut out = lm.to_vec();
    let mouth = sub(origin, scale(down, MOUTH_DIST * ed));
    for slot in out.iter_mut().take(68).skip(48) {
        *slot = Some(mouth);
    }
    if let Some(c) = out.get_mut(8) {
        *c = None;
    }
    face_frame_from_landmarks(&out)
}

impl FaceFrame {
    /// World point -> face-frame coordinates (u right, v up, w forward), mm.
    pub fn to_face(&self, p: Vec3) -> Vec3 {
        let d = sub(p, self.origin);
        [dot(d, self.right), dot(d, self.up), dot(d, self.forward)]
    }
}
