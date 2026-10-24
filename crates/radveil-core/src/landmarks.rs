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
}

/// Build the face frame from combined 3D landmarks (68-point indexing).
pub fn face_frame_from_landmarks(lm: &[Option<Vec3>]) -> Result<FaceFrame, String> {
    let eye_r = mean_of(lm, 36..=41).ok_or("missing right-eye landmarks")?;
    let eye_l = mean_of(lm, 42..=47).ok_or("missing left-eye landmarks")?;
    let nose_tip = lm
        .get(30)
        .copied()
        .flatten()
        .or_else(|| mean_of(lm, 27..=35))
        .ok_or("missing nose landmarks")?;
    let mouth = mean_of(lm, 48..=67).ok_or("missing mouth landmarks")?;
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
    let up0 = normalize(sub(origin, mouth));
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
    })
}

/// Eyes -> mouth distance as a multiple of the inter-eye distance, and the
/// angle (degrees) by which the face frame's `up` leans forward of the
/// head's superior axis (eye landmarks lift onto the eyelids, anterior of
/// the lips). Used to place a mouth the scan does not contain.
const MOUTH_DIST: f64 = 1.15;
const UP_LEAN_DEG: f64 = 27.0;

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
