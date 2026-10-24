// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Face anonymization: build face-aligned ROIs from the 3D landmarks and
//! rewrite only the voxels inside them.
//!
//! Modes:
//! - Blur: heavy Gaussian blur of the ROI (facial surface becomes smooth,
//!   intensities stay in-range).
//! - Cut: ROI voxels set to the background value (face removed). The
//!   background is the 1st intensity percentile: ~0 on MRI, ~air (-1000 HU)
//!   on CT -- zero-filling a CT would insert water-density voxels.
//! - Block: ROI filled with a high constant (opaque slab destroys features).
//! - Adaptive: gradient-weighted averaging — the stronger a facial feature
//!   (ridge, socket, lip line), the heavier the local averaging kernel.
//!
//! The face ROI is skin-deep. Its hull is a box in the face frame, wide
//! enough to cover both eyes with margin, from above the brows to below the
//! chin, and from the nose tip back to a few mm behind the eye surface. Its
//! posterior wall, though, follows the *measured* skin: every (u, v) column
//! is marched from the front to the first tissue, and the wall sits
//! `skin_depth_mm` below that surface (see [`build_face_wall`]).
//! Protrusions (nose, lips) are cut down to the surrounding skin level.
//! A flat wall is not safe: the frame tilts with the face, and on some heads
//! a plane a few mm behind the eyes reaches the frontal lobe at brow height.
//! Every voxel outside the ROI is preserved bit-exact.
//!
//! Ears are biometric identifiers too (pinna shape is stable and matchable
//! from surface reconstructions), so `Region::Ears` / `Region::FaceAndEars`
//! anonymize them with the same modes. No landmark detector emits ear
//! points, so ears are located from *surface evidence* instead of fixed
//! offsets: within a calibrated search window each (v, w) column of the
//! volume is ray-marched laterally to build a surface height map, a
//! morphological opening of that map gives the scalp baseline, and the
//! pinna is detected as the protrusion blob that is pinna-shaped and sits
//! where pinnae were measured to sit (see [`build_ear_roi`]). The ROI hull
//! tightens to the detected pinna; the medial cut wall follows the
//! measured head surface minus a guard and stops at the skull, so the cut
//! is brain-safe while severing the whole pinna even on asymmetric heads.
//! If no plausible pinna is found (flattened against the coil, cropped
//! FOV), a fallback box spanning every measured pinna position is used and
//! the report flags `detected = false`.
//!
//! NOTE on calibration: ears are scanned in a head-based ear frame (eye
//! axis + scanner superior, see [`ear_frame`]), not the face frame. The
//! face frame's `up` runs mouth -> eyes on the skin and tilts by 30-40
//! degrees on a truncated field of view (a clinical scan), which moved the pinna by
//! more than an ear height between scans; a prior calibrated on that scan's
//! face frame missed every IXI pinna. The window and fallback box are
//! calibrated on 106 hand-marked IXI pinnae plus a clinical T1: centre v -0.39,
//! w -1.50 (units of inter-eye distance) — re-measure on many subjects
//! before changing them, never on one scan.

use crate::affine::Vec3;
use crate::landmarks::FaceFrame;
use crate::volume::Volume;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DefaceMode {
    #[default]
    Blur,
    Cut,
    Block,
    /// Feature-weighted averaging: local gradient strength selects between a
    /// mild and a heavy kernel, so ridges/sockets/lip lines (the identifying
    /// "strong lines") are averaged hardest while flat skin keeps texture.
    Adaptive,
}

impl std::str::FromStr for DefaceMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "blur" => Ok(DefaceMode::Blur),
            "cut" => Ok(DefaceMode::Cut),
            "block" => Ok(DefaceMode::Block),
            "adaptive" => Ok(DefaceMode::Adaptive),
            other => Err(format!("unknown mode '{other}' (blur|cut|block|adaptive)")),
        }
    }
}

/// Which anatomy to anonymize. `mode` applies to every region unless
/// `DefaceParams::ear_mode` sets a separate mode for the ears.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Region {
    #[default]
    #[serde(rename = "face")]
    Face,
    #[serde(rename = "ears")]
    Ears,
    #[serde(rename = "face+ears")]
    FaceAndEars,
}

impl Region {
    pub fn includes_face(self) -> bool {
        matches!(self, Region::Face | Region::FaceAndEars)
    }
    pub fn includes_ears(self) -> bool {
        matches!(self, Region::Ears | Region::FaceAndEars)
    }
}

impl std::str::FromStr for Region {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "face" => Ok(Region::Face),
            "ears" => Ok(Region::Ears),
            "face+ears" | "ears+face" => Ok(Region::FaceAndEars),
            other => Err(format!("unknown region '{other}' (face|ears|face+ears)")),
        }
    }
}

/// All geometric parameters are optional; `None` means *auto*: the value is
/// derived from the detected inter-eye distance, so the defacer adapts to
/// head size (adult, pediatric, phantom) with zero tuning. Explicit values
/// (mm) override the auto scaling.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DefaceParams {
    pub mode: DefaceMode,
    /// Mode for the ear regions; `None` = same as `mode`. Lets one pass
    /// combine e.g. `adaptive` on the face with `cut` on the ears.
    #[serde(default)]
    pub ear_mode: Option<DefaceMode>,
    /// Anatomy to anonymize: face (default), ears, or both.
    pub region: Region,
    /// Half-width of the ROI, as a multiple of the inter-eye distance.
    pub width_factor: Option<f64>,
    /// Extra margin above the eye line, mm.
    pub above_eyes_mm: Option<f64>,
    /// Extra margin below the chin (or below the mouth if no chin), mm.
    pub below_chin_mm: Option<f64>,
    /// How far behind the eye-surface plane the ROI hull reaches, mm. The
    /// skin-following wall (`skin_depth_mm`) is never deeper than this.
    pub depth_behind_mm: Option<f64>,
    /// Face ROI: how deep below the measured skin surface voxels are
    /// rewritten, mm, measured along the face's forward axis. Protrusions
    /// (nose, lips) are removed down to the surrounding skin level. Keep it
    /// below the soft-tissue thickness (forehead ~5 mm) so the skull and
    /// brain are never reached.
    #[serde(default)]
    pub skin_depth_mm: Option<f64>,
    /// How far in front of the eye plane the ROI reaches, mm (past nose tip).
    pub depth_front_mm: Option<f64>,
    /// Face ROI: anchor the depth bounds to the *measured* eye-skin
    /// surface instead of the raw landmark plane (default: on). On
    /// thick-slice / low-resolution scans the lifted eye landmarks can sit
    /// several mm inside the head, which would push the posterior wall
    /// (the brain-safety margin) into the frontal lobe; the anchor
    /// measures that bias along the forward axis and shifts the ROI by it.
    /// On well-resolved scans the bias is ~0 and behavior is unchanged.
    /// Set `Some(false)` to opt out (raw landmark-plane anchoring).
    pub eye_surface_anchor: Option<bool>,
    /// Face ROI: explicit landmark-depth bias, mm (overrides measurement;
    /// same clamp). Pass a previously reported `eye_bias_mm` to pin the
    /// geometry across repeated calls — the bias cannot be re-measured on
    /// an already-anonymized volume (its eye surface is gone), so
    /// verification escalations use this to stay inside the first-round
    /// mask.
    pub eye_bias_mm: Option<f64>,
    /// Gaussian sigma for blur/adaptive modes, mm.
    pub blur_sigma_mm: Option<f64>,
    /// Adaptive mode: heavy-kernel sigma as a multiple of the blur sigma.
    pub adaptive_factor: Option<f64>,
    /// Ear ROIs (regions with ears): absolute medial floor, distance from
    /// the head midline, mm. The evidence-based cut never reaches medial of
    /// this, whatever the surface scan says.
    pub ear_inner_mm: Option<f64>,
    /// Ear ROIs: lateral bound, distance from the head midline, mm.
    pub ear_outer_mm: Option<f64>,
    /// Ear ROIs: cut depth below the scalp under the detected pinna (and
    /// below the measured surface on other protrusions; 0.75x on flat
    /// scalp), mm. The wall never goes medial of the skull's outer table,
    /// the local scalp minus 0.15 d, or `ear_inner_mm`.
    pub ear_guard_mm: Option<f64>,
    /// Ear ROIs: fallback box centre, distance below the eye line along the
    /// ear frame's -v (scanner superior), mm. Used when no pinna is
    /// detected; the search window grows to include the box.
    pub ear_down_mm: Option<f64>,
    /// Ear ROIs: fallback box centre, distance behind the eye plane along
    /// the ear frame's -w, mm.
    pub ear_back_mm: Option<f64>,
    /// Ear ROIs: fallback box full extent along the ear frame's v axis, mm.
    pub ear_height_mm: Option<f64>,
    /// Ear ROIs: fallback box full extent along the ear frame's w axis, mm.
    pub ear_depth_mm: Option<f64>,
    /// Leave background air untouched (default: off). A voxel inside the
    /// region is rewritten only if its original or its new value reaches
    /// the air level (halfway from background to the surface threshold):
    /// air a blur leaves as air, or a cut sets to background, keeps its
    /// original value. Tissue, the partial-volume skin edge and the halo
    /// a blur spreads into the air are still written, so the anonymized
    /// surface is the same; `block` fills its whole slab regardless.
    #[serde(default)]
    pub spare_air: bool,
    /// Face ROI: measure the skin along the level head axes (scanner
    /// superior, horizontal forward; the ear frame) instead of the face
    /// frame (default: off). The face frame's forward tilts ~25-30 degrees
    /// down, so seen along it the chin hides the lower lip and the neck
    /// hides the chin, and the skin-deep wall leaves them in place.
    #[serde(default)]
    pub level_face: bool,
    /// Skin threshold (absolute intensity) for the region geometry; `None`
    /// = the renderer's auto rule (-400 HU on CT, else 0.14 x p99).
    /// [`crate::deface_auto`] sets it to the threshold it rendered with, so
    /// the skin the cut follows is the skin the detector saw.
    #[serde(default)]
    pub surface_threshold: Option<f32>,
}

/// Auto-scaling factors, as fractions of the inter-eye distance (~60 mm on
/// adults, so the defaults below reproduce the classic absolute values).
mod auto {
    pub const WIDTH_FACTOR: f64 = 1.05;
    pub const ABOVE_EYES: f64 = 0.60; // ~35 mm
    pub const BELOW_CHIN: f64 = 0.43; // ~25 mm
    pub const DEPTH_BEHIND: f64 = 0.21; // ~12 mm hull floor
    pub const SKIN_DEPTH: f64 = 0.07; // ~4.5 mm below the skin
    pub const DEPTH_FRONT: f64 = 1.55; // ~90 mm
    pub const BLUR_SIGMA: f64 = 0.17; // ~10 mm
    pub const ADAPTIVE_FACTOR: f64 = 2.5;
    // Ear ROIs, in the head-based ear frame (see `ear_frame` in
    // deface.rs). The medial wall is NOT a fixed factor: it follows the
    // per-column detected head surface minus EAR_GUARD (measured
    // scalp+skull thickness lateral of the brain: >= 0.17), floored at the
    // median measured scalp minus 0.15 and at EAR_INNER.
    pub const EAR_INNER: f64 = 0.75; // absolute medial floor
    pub const EAR_OUTER: f64 = 2.20; // past any protruding pinna
    pub const EAR_GUARD: f64 = 0.05; // cut depth below the scalp / surface
                                     // Fallback box (no pinna detected): spans every hand-marked pinna of
                                     // 53 IXI subjects + a clinical T1 (v -1.37..0.66, w -2.19..-0.82) with margin
                                     // (v -1.50..0.80, w -2.30..-0.70).
    pub const EAR_DOWN: f64 = 0.35; // centre below the eye line
    pub const EAR_BACK: f64 = 1.50; // centre behind the eye plane
    pub const EAR_HEIGHT: f64 = 2.30; // v extent
    pub const EAR_DEPTH: f64 = 1.60; // w extent
}

/// Parameters after auto-scaling against a face frame.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ResolvedParams {
    pub width_factor: f64,
    pub above_eyes_mm: f64,
    pub below_chin_mm: f64,
    pub depth_behind_mm: f64,
    #[serde(default)]
    pub skin_depth_mm: f64,
    pub depth_front_mm: f64,
    pub blur_sigma_mm: f64,
    pub adaptive_factor: f64,
    pub ear_inner_mm: f64,
    pub ear_outer_mm: f64,
    pub ear_guard_mm: f64,
    pub ear_down_mm: f64,
    pub ear_back_mm: f64,
    pub ear_height_mm: f64,
    pub ear_depth_mm: f64,
    /// See [`DefaceParams::surface_threshold`].
    #[serde(default)]
    pub surface_threshold: Option<f32>,
}

impl DefaceParams {
    pub fn resolve(&self, eye_dist_mm: f64) -> ResolvedParams {
        let ed = eye_dist_mm;
        ResolvedParams {
            width_factor: self.width_factor.unwrap_or(auto::WIDTH_FACTOR),
            above_eyes_mm: self.above_eyes_mm.unwrap_or(auto::ABOVE_EYES * ed),
            below_chin_mm: self.below_chin_mm.unwrap_or(auto::BELOW_CHIN * ed),
            depth_behind_mm: self.depth_behind_mm.unwrap_or(auto::DEPTH_BEHIND * ed),
            skin_depth_mm: self.skin_depth_mm.unwrap_or(auto::SKIN_DEPTH * ed),
            depth_front_mm: self.depth_front_mm.unwrap_or(auto::DEPTH_FRONT * ed),
            blur_sigma_mm: self.blur_sigma_mm.unwrap_or(auto::BLUR_SIGMA * ed),
            adaptive_factor: self.adaptive_factor.unwrap_or(auto::ADAPTIVE_FACTOR),
            ear_inner_mm: self.ear_inner_mm.unwrap_or(auto::EAR_INNER * ed),
            ear_outer_mm: self.ear_outer_mm.unwrap_or(auto::EAR_OUTER * ed),
            ear_guard_mm: self.ear_guard_mm.unwrap_or(auto::EAR_GUARD * ed),
            ear_down_mm: self.ear_down_mm.unwrap_or(auto::EAR_DOWN * ed),
            ear_back_mm: self.ear_back_mm.unwrap_or(auto::EAR_BACK * ed),
            ear_height_mm: self.ear_height_mm.unwrap_or(auto::EAR_HEIGHT * ed),
            ear_depth_mm: self.ear_depth_mm.unwrap_or(auto::EAR_DEPTH * ed),
            surface_threshold: self.surface_threshold,
        }
    }
}

/// One anonymized region: hull bounds in mm (face frame for the face, the
/// head-based ear frame for ears) plus its voxel count.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoiReport {
    /// "face", "ear_left" or "ear_right" (anatomical side).
    pub label: String,
    pub u_mm: [f64; 2],
    pub v_mm: [f64; 2],
    pub w_mm: [f64; 2],
    pub voxels: usize,
    /// Ears only: whether the pinna was located by surface-protrusion
    /// evidence. `Some(false)` means the anthropometric fallback ROI was
    /// used (e.g. the ear is outside the field of view).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected: Option<bool>,
    /// Ears only, when detected: peak protrusion of the pinna above the
    /// local scalp baseline, mm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protrusion_mm: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefaceReport {
    pub mode: DefaceMode,
    /// Mode actually applied to the ear regions (equals `mode` unless
    /// `ear_mode` was set).
    pub ear_mode: DefaceMode,
    pub region: Region,
    pub voxels_changed: usize,
    pub voxels_total: usize,
    pub eye_dist_mm: f64,
    /// Measured depth of the lifted eye landmarks below the true skin
    /// surface, mm (0 when the anchor is disabled or nothing measurable).
    /// The face ROI's depth bounds are shifted anteriorly by this amount.
    #[serde(default)]
    pub eye_bias_mm: f64,
    pub rois: Vec<RoiReport>,
    pub resolved: ResolvedParams,
    /// Face region only: whether the face surface was seen through air
    /// inside the scan (false: it lies on a FOV cut and no tissue changed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub face_seen_through_air: Option<bool>,
    /// With a protected set (the brain guard): voxels of the region left
    /// untouched because they are protected. `None` without one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protected_voxels_spared: Option<usize>,
    /// With [`DefaceParams::spare_air`]: region voxels left untouched
    /// because they were air and stayed air (not in `voxels_changed`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub air_voxels_spared: Option<usize>,
}

#[derive(Clone)]
struct Roi {
    label: &'static str,
    /// Basis the bounds are expressed in: ears use the head-based
    /// [`ear_frame`]; `None` = the face frame.
    basis: Option<FaceFrame>,
    u: [f64; 2],
    v: [f64; 2],
    w: [f64; 2],
    /// Ears: per-(v,w)-column medial bound |u| following the detected head
    /// surface. Voxels are only rewritten lateral of it.
    wall: Option<EarWall>,
    /// Face: per-(u,v)-column posterior bound w following the measured
    /// skin. Voxels are only rewritten anterior of it.
    face_wall: Option<FaceWall>,
    /// Face: whether the face surface was seen through air (see
    /// `build_face_wall`); false means it lies on a FOV cut.
    seen_through_air: Option<bool>,
    detected: Option<bool>,
    protrusion_mm: Option<f64>,
}

/// Per-column posterior cut bound of the face ROI on a regular (u, v) grid.
#[derive(Clone)]
struct FaceWall {
    u0: f64,
    v0: f64,
    step: f64,
    nu: usize,
    nv: usize,
    /// Minimum w (forward coordinate) rewritten in each column.
    wall: Vec<f64>,
}

impl FaceWall {
    fn at(&self, u: f64, v: f64) -> f64 {
        let i = (((u - self.u0) / self.step).round() as isize).clamp(0, self.nu as isize - 1);
        let j = (((v - self.v0) / self.step).round() as isize).clamp(0, self.nv as isize - 1);
        self.wall[i as usize * self.nv + j as usize]
    }
}

/// Per-column medial cut bounds on a regular (v, w) grid, one ear side.
#[derive(Clone)]
struct EarWall {
    /// +1.0 => this ear is on +u (patient left), -1.0 => -u (right).
    sign: f64,
    v0: f64,
    w0: f64,
    step: f64,
    nv: usize,
    nw: usize,
    /// |u| medial bound per cell; +inf marks "do not cut this column".
    wall: Vec<f64>,
}

impl EarWall {
    fn wall_at(&self, v: f64, w: f64) -> f64 {
        let i = (((v - self.v0) / self.step).round() as isize).clamp(0, self.nv as isize - 1);
        let j = (((w - self.w0) / self.step).round() as isize).clamp(0, self.nw as isize - 1);
        self.wall[i as usize * self.nw + j as usize]
    }
}

fn build_face_roi(
    volume: &Volume,
    frame: &FaceFrame,
    params: &ResolvedParams,
    eye_bias_mm: f64,
    level: bool,
) -> Roi {
    let level_frame = level.then(|| ear_frame(frame));
    let frame = level_frame.as_ref().unwrap_or(frame);
    let half_w = params.width_factor * frame.eye_dist;
    // Chin landmarks can land on scan-boundary artifacts (e.g. when the FOV
    // cuts off below the mouth), so clamp to plausible anthropometry:
    // eyes->chin is roughly 1.5-2x the inter-eye distance.
    let chin_floor = -2.0 * frame.eye_dist;
    let chin_v = frame
        .chin
        .map(|c| frame.to_face(c)[1])
        .unwrap_or_else(|| frame.to_face(frame.mouth)[1] - 35.0)
        .max(chin_floor);
    let u = [-half_w, half_w];
    let v = [chin_v - params.below_chin_mm, params.above_eyes_mm];
    // Hull depth bounds are anchored to the measured eye-skin surface: when
    // lifted landmarks sit `eye_bias_mm` inside the head (thick-slice
    // scans), both bounds shift anteriorly by that amount.
    let w = [
        eye_bias_mm - params.depth_behind_mm,
        eye_bias_mm + params.depth_front_mm,
    ];
    let (face_wall, seen_through_air) =
        build_face_wall(volume, frame, params, eye_bias_mm, u, v, w);
    Roi {
        label: "face",
        u,
        v,
        w,
        basis: level_frame.clone(),
        wall: None,
        face_wall: Some(face_wall),
        seen_through_air: Some(seen_through_air),
        detected: None,
        protrusion_mm: None,
    }
}

/// Skin-following posterior wall of the face ROI.
///
/// 1. For every (u, v) column, march backwards along the forward axis from
///    the front of the hull and record the first above-threshold sample:
///    the skin height. Skin is only trusted after the ray has crossed some
///    air *inside* the scan: a ray that enters the volume straight into
///    tissue meets a field-of-view cut, often through the brain, not skin.
///    Its tissue is left untouched, except in the nose/lips zone below,
///    where it is a clipped nose tip.
/// 2. Protrusions stand out above a morphological opening of that height
///    map, which is the face as it would be without them. Only the nose and
///    lips count: cells in a zone around them (from the landmarks, up to the
///    nasal bridge) are cut down to that opened level, never deeper than
///    the eye-skin plane. Elsewhere, notably the forehead, the
///    wall always follows the skin itself: an edge there (an
///    already-defaced scan, a FOV cut) would otherwise interpolate the wall
///    from deep neighbours into the frontal lobe.
/// 3. The wall is that skin field minus `skin_depth_mm`, clamped to the
///    hull. A column that meets no tissue is cut only as deep as it was
///    seen to be air inside the scan (a ray can leave through a FOV cut).
///
/// Depth is measured along the forward axis, which meets the skin
/// obliquely almost everywhere, so the depth normal to the skin is at most
/// `skin_depth_mm`.
fn build_face_wall(
    volume: &Volume,
    frame: &FaceFrame,
    params: &ResolvedParams,
    eye_bias_mm: f64,
    u: [f64; 2],
    v: [f64; 2],
    w: [f64; 2],
) -> (FaceWall, bool) {
    let ed = frame.eye_dist;
    let thr = surface_threshold(volume, params.surface_threshold);
    let step = FACE_GRID_STEP_MM;
    let nu = ((u[1] - u[0]) / step).ceil() as usize + 1;
    let nv = ((v[1] - v[0]) / step).ceil() as usize + 1;
    // (skin height or NaN, whether it follows enough in-scan air to be
    // skin, deepest depth seen to be air inside the scan)
    let march: Vec<(f64, bool, f64)> = (0..nu * nv)
        .map(|idx| {
            let (cu, cv) = (
                u[0] + (idx / nv) as f64 * step,
                v[0] + (idx % nv) as f64 * step,
            );
            let mut air = 0.0;
            let mut air_floor = f64::NAN;
            let mut d = w[1];
            while d >= w[0] {
                let p = crate::affine::add(
                    frame.origin,
                    crate::affine::add(
                        crate::affine::scale(frame.right, cu),
                        crate::affine::add(
                            crate::affine::scale(frame.up, cv),
                            crate::affine::scale(frame.forward, d),
                        ),
                    ),
                );
                match sample(volume, p) {
                    Some(x) if x >= thr => return (d, air >= FACE_MIN_AIR_MM, air_floor),
                    Some(_) => {
                        air += FACE_MARCH_STEP_MM;
                        air_floor = d;
                    }
                    // Left the scan: air seen before doesn't vouch for
                    // tissue met after re-entering through another face.
                    None => air = 0.0,
                }
                d -= FACE_MARCH_STEP_MM;
            }
            (f64::NAN, true, air_floor)
        })
        .collect();
    let height: Vec<f64> = march.iter().map(|m| m.0).collect();
    let is_skin: Vec<bool> = march.iter().map(|m| m.1).collect();
    // A real face is seen through air: around the eyes and cheeks nearly
    // every column crosses in-scan air before the skin. Only then may a
    // truncated column in the nose zone be taken for a clipped nose tip; a
    // "face" found on a FOV cut plane (the brain exposed at the slab's edge)
    // fails this, and none of its truncated tissue is touched.
    let (mut band, mut band_skin) = (0usize, 0usize);
    for i in 0..nu * nv {
        let (cu, cv) = (u[0] + (i / nv) as f64 * step, v[0] + (i % nv) as f64 * step);
        if cu.abs() <= 0.8 * ed && cv.abs() <= 0.3 * ed && !height[i].is_nan() {
            band += 1;
            band_skin += is_skin[i] as usize;
        }
    }
    let seen_through_air = band > 0 && band_skin as f64 >= FACE_MIN_SKIN_FRACTION * band as f64;

    // Skin pits: a column far deeper than its neighbours missed thin, dim
    // skin (T2/PD forehead) and, through dark bone counted as air, took the
    // brain for skin. Skin is smooth: raise such a column to the local
    // median. Only ever cuts less.
    let pit_r = (FACE_PIT_RADIUS_MM / step).round().max(1.0) as isize;
    let skin_height: Vec<f64> = (0..nu * nv)
        .map(|i| {
            let h = height[i];
            if h.is_nan() {
                return h;
            }
            let (iu, iv) = ((i / nv) as isize, (i % nv) as isize);
            let mut nb: Vec<f64> = Vec::new();
            for du in -pit_r..=pit_r {
                for dv in -pit_r..=pit_r {
                    let (a, b) = (iu + du, iv + dv);
                    if a >= 0 && b >= 0 && (a as usize) < nu && (b as usize) < nv {
                        let x = height[a as usize * nv + b as usize];
                        if !x.is_nan() {
                            nb.push(x);
                        }
                    }
                }
            }
            nb.sort_by(f64::total_cmp);
            let med = nb[nb.len() / 2];
            if h < med - FACE_PIT_MM {
                med
            } else {
                h
            }
        })
        .collect();
    let radius = ((FACE_OPEN * ed) / step).round().max(1.0) as usize;
    let baseline = grid_filter(
        &grid_filter(&height, nu, nv, radius, false),
        nu,
        nv,
        radius,
        true,
    );
    // Nose and lips: centred, from below the mouth up to the nasal bridge.
    let mouth_v = frame.to_face(frame.mouth)[1];
    let in_zone = |i: usize| {
        let (cu, cv) = (u[0] + (i / nv) as f64 * step, v[0] + (i % nv) as f64 * step);
        cu.abs() <= FACE_ZONE_HALF_WIDTH * ed
            && cv >= mouth_v - FACE_ZONE_BELOW_MOUTH * ed
            && cv <= FACE_ZONE_ABOVE_EYES * ed
    };
    let protrudes: Vec<bool> = (0..nu * nv)
        .map(|i| {
            in_zone(i)
                && !height[i].is_nan()
                && !baseline[i].is_nan()
                && height[i] - baseline[i] >= FACE_PROTRUSION * ed
        })
        .collect();
    let wall = (0..nu * nv)
        .map(|i| {
            if height[i].is_nan() {
                // No tissue met, but the ray may have left the scan through
                // a FOV cut: only cut as deep as it was *seen* to be air.
                let floor = march[i].2;
                if floor.is_nan() {
                    w[1]
                } else {
                    floor.clamp(w[0], w[1])
                }
            } else if !seen_through_air {
                // Not a face seen through air: touch no tissue at all
                // (`deface_auto` refuses such volumes outright).
                height[i].clamp(w[0], w[1])
            } else if protrudes[i] {
                (baseline[i] - params.skin_depth_mm)
                    .max(eye_bias_mm)
                    .clamp(w[0], w[1])
            } else if !is_skin[i] {
                // Entered the scan straight into tissue: a FOV cut. In the
                // nose/lips zone that is a clipped nose tip, safe to take
                // skin-deep; anywhere else keep every tissue voxel.
                if in_zone(i) {
                    (height[i] - params.skin_depth_mm)
                        .max(eye_bias_mm)
                        .clamp(w[0], w[1])
                } else {
                    height[i].clamp(w[0], w[1])
                }
            } else {
                (skin_height[i] - params.skin_depth_mm).clamp(w[0], w[1])
            }
        })
        .collect();
    (
        FaceWall {
            u0: u[0],
            v0: v[0],
            step,
            nu,
            nv,
            wall,
        },
        seen_through_air,
    )
}

/// How deep the lifted eye landmarks sit below the true skin surface, mm.
///
/// Marches anteriorly-to-posteriorly along the frame's forward axis
/// through a small column grid centered on each eye and takes the median
/// first-tissue crossing (the eyelid skin). Well-resolved scans measure
/// ~0; thick-slice scans, where landmark lifting lands a few mm deep,
/// measure positive. Clamped to [0, 0.25 x eye-dist]: never negative, so a
/// re-deface of an already-anonymized volume (verification escalations)
/// can never cut deeper than the first pass.
fn eye_surface_bias(volume: &Volume, frame: &FaceFrame, thr: f32) -> f64 {
    let ed = frame.eye_dist;
    let mut crossings: Vec<f64> = Vec::new();
    for eye_u in [-0.5 * ed, 0.5 * ed] {
        for du in [-0.12 * ed, 0.0, 0.12 * ed] {
            for dv in [-0.12 * ed, 0.0, 0.12 * ed] {
                let mut w = 0.6 * ed;
                while w >= -0.4 * ed {
                    let p = crate::affine::add(
                        frame.origin,
                        crate::affine::add(
                            crate::affine::scale(frame.right, eye_u + du),
                            crate::affine::add(
                                crate::affine::scale(frame.up, dv),
                                crate::affine::scale(frame.forward, w),
                            ),
                        ),
                    );
                    if sample(volume, p).map(|x| x >= thr).unwrap_or(false) {
                        crossings.push(w);
                        break;
                    }
                    w -= 1.0;
                }
            }
        }
    }
    if crossings.is_empty() {
        return 0.0;
    }
    crossings.sort_by(|a, b| a.total_cmp(b));
    let median = crossings[crossings.len() / 2];
    median.clamp(0.0, 0.25 * ed)
}

/// Surface threshold separating tissue from air, matching the renderer's
/// auto rule: -400 HU for CT-like data, else a fraction of the p99.
fn surface_threshold(volume: &Volume, given: Option<f32>) -> f32 {
    if let Some(t) = given {
        return t;
    }
    if volume.percentile(1.0) < -500.0 {
        -400.0
    } else {
        0.14 * volume.percentile(99.0)
    }
}

/// Below this a voxel is background air for [`DefaceParams::spare_air`]:
/// halfway from the background (1st percentile) to the surface threshold.
fn air_level(volume: &Volume, thr: Option<f32>) -> f32 {
    let bg = volume.percentile(1.0);
    bg + 0.5 * (surface_threshold(volume, thr) - bg)
}

/// Nearest-voxel intensity at a world point, None outside the volume.
fn sample(volume: &Volume, p: Vec3) -> Option<f32> {
    let v = crate::affine::apply(&volume.world_to_voxel, p);
    let [nx, ny, nz] = volume.shape();
    let (i, j, k) = (v[0].round(), v[1].round(), v[2].round());
    if i < 0.0 || j < 0.0 || k < 0.0 {
        return None;
    }
    let (i, j, k) = (i as usize, j as usize, k as usize);
    if i >= nx || j >= ny || k >= nz {
        return None;
    }
    Some(volume.data[[i, j, k]])
}

/// Separable min/max filter over a NaN-aware (v, w) grid; NaN cells are
/// ignored (and stay NaN if the whole window is NaN).
fn grid_filter(src: &[f64], nv: usize, nw: usize, radius: usize, max: bool) -> Vec<f64> {
    let pick = |a: f64, b: f64| -> f64 {
        if a.is_nan() {
            b
        } else if b.is_nan() {
            a
        } else if max {
            a.max(b)
        } else {
            a.min(b)
        }
    };
    let mut tmp = vec![f64::NAN; nv * nw];
    for i in 0..nv {
        for j in 0..nw {
            let mut acc = f64::NAN;
            for dj in j.saturating_sub(radius)..(j + radius + 1).min(nw) {
                acc = pick(acc, src[i * nw + dj]);
            }
            tmp[i * nw + j] = acc;
        }
    }
    let mut out = vec![f64::NAN; nv * nw];
    for i in 0..nv {
        for j in 0..nw {
            let mut acc = f64::NAN;
            for di in i.saturating_sub(radius)..(i + radius + 1).min(nv) {
                acc = pick(acc, tmp[di * nw + j]);
            }
            out[i * nw + j] = acc;
        }
    }
    out
}

const FACE_GRID_STEP_MM: f64 = 1.0;
const FACE_MARCH_STEP_MM: f64 = 0.5;
/// Face protrusions are measured against an opening of the skin height map
/// with this radius (x inter-eye distance): wider than half a nose.
const FACE_OPEN: f64 = 0.30;
/// A face cell is a protrusion when it stands this far (x inter-eye
/// distance, ~4 mm) above the opened baseline.
const FACE_PROTRUSION: f64 = 0.06;
/// Protrusions are only looked for in the nose/lips zone: |u| below this
/// (x inter-eye distance), from this far below the mouth ...
const FACE_ZONE_HALF_WIDTH: f64 = 0.55;
const FACE_ZONE_BELOW_MOUTH: f64 = 0.30;
/// ... and up to this far above the eye line (the nasal bridge).
const FACE_ZONE_ABOVE_EYES: f64 = 0.15;
/// A face column more than this (mm) deeper than the median of its
/// neighbours within `FACE_PIT_RADIUS_MM` is a skin pit (see
/// `build_face_wall`).
const FACE_PIT_MM: f64 = 3.0;
const FACE_PIT_RADIUS_MM: f64 = 5.0;
/// In-scan air a face column must cross before its first tissue counts as
/// skin, mm.
const FACE_MIN_AIR_MM: f64 = 2.0;
/// Fraction of eye/cheek columns that must show skin behind air for the
/// face to count as seen through air (see `build_face_wall`).
const FACE_MIN_SKIN_FRACTION: f64 = 0.8;
const EAR_GRID_STEP_MM: f64 = 2.0;
const EAR_MARCH_STEP_MM: f64 = 1.0;
/// A cell belongs to the pinna when it protrudes at least this far above
/// the local scalp baseline.
const EAR_PROTRUSION_MM: f64 = 3.0;
/// Levels at which protrusion blobs are segmented, mm (the first is
/// `EAR_PROTRUSION_MM`).
const EAR_LEVELS_MM: &[f64] = &[3.0, 4.5, 6.0, 8.0, 10.0];

// Ear geometry below is in the ear frame (see [`ear_frame`]) and in units
// of the inter-eye distance `d`. Calibration: pinnae hand-marked on true
// lateral renders of 53 IXI T1 subjects (106 ears) plus the clinical
// a clinical T1, lifted to 3D through the render's position map. Pinna centre:
// v -0.39 +- 0.19, w -1.50 +- 0.12; the 2nd-98th percentile pinna surface
// spans v -1.37..0.66 and w -2.19..-0.82 across all subjects; the clinical scan lies
// inside the IXI spread.

/// Measured pinna centre (v, w) and its between-subject SD.
const EAR_PRIOR_CENTRE: (f64, f64) = (-0.39, -1.50);
const EAR_PRIOR_SD: (f64, f64) = (0.19, 0.12);
/// Search window: every measured pinna plus room for the scalp baseline
/// around it (a blob touching the border is rejected).
const EAR_SEARCH_V: (f64, f64) = (-1.7, 1.0);
const EAR_SEARCH_W: (f64, f64) = (-2.5, -0.55);
/// Lateral start of the surface scan (medial of any adult scalp; the cut
/// floor is separate).
const EAR_MEASURE_FROM: f64 = 0.6;
/// Consecutive above-threshold samples needed for a surface hit, so an
/// isolated noise voxel lateral of the head does not become the surface.
const EAR_SURFACE_RUN: usize = 2;
/// Opening radius of the scalp baseline: wider than a flattened pinna is
/// thick, small enough that the scalp's own curvature does not read as a
/// protrusion.
const EAR_OPENING: f64 = 0.30;
/// Cells this close to a column without tissue (head edge, field of view)
/// cannot be pinna: the baseline is unsupported there.
const EAR_EDGE_CELLS: usize = 5;
/// Data-driven cut floor: never deeper than this below the local median
/// scalp (scalp+skull lateral of the brain is >= 0.17 d).
const EAR_MAX_DEPTH: f64 = 0.15;
/// Radius of that local median, frame units: wide enough to outvote a
/// broken column, narrow enough to follow the head narrowing towards the
/// mastoid and neck.
const EAR_FLOOR_RADIUS: f64 = 0.3;
/// A column more than this below the median of its neighbours is a pit
/// (outer skin under the threshold), mm.
const EAR_PIT_MM: f64 = 2.0;
/// Cut depth on flat (non-protruding) scalp as a fraction of
/// `ear_guard_mm`.
const EAR_SKIN_GUARD_FRAC: f64 = 0.75;
/// Bone bound: a run of this many samples (1 mm apart) below the surface
/// threshold is the skull (or the air of the ear canal).
const EAR_BONE_RUN: usize = 2;
/// Under the pinna the bone search starts this far lateral of the
/// interpolated scalp, mm.
const EAR_BONE_TOL_MM: f64 = 2.0;
/// Pinna plausibility gates for protrusion blobs (frame units).
const EAR_MIN_AREA: f64 = 0.06;
const EAR_MAX_EXTENT_V: f64 = 1.5;
const EAR_MAX_EXTENT_W: f64 = 1.2;
const EAR_MAX_PEAK: f64 = 0.5;
/// A blob whose centre is further than this (in prior SDs) from the
/// measured pinna centre is not the pinna.
const EAR_MAX_PRIOR_Z: f64 = 3.0;
/// Detected hull: at least this (v, w) box around the blob centre, so a
/// pinna that protrudes only along its rim is still covered whole.
const EAR_MIN_HULL: (f64, f64) = (1.6, 1.4);
/// Frames whose `up` differs from the scanner's superior axis by more than
/// this are treated as reoriented volumes: the ear scan keeps the face
/// frame.
const EAR_MAX_FRAME_TILT_DEG: f64 = 60.0;

/// Head-based frame for the ear scan: the eye axis from the face frame,
/// `up` = the scanner's superior axis made orthogonal to it, `forward` on
/// the anterior side. The face frame's `up` runs mouth -> eyes on the skin
/// and tilts by 30-40 degrees when the mouth landmarks land on a truncated
/// field of view (clinical scans), which moves the pinna by more than an ear height
/// in face-frame coordinates; patients lie with the head near the scanner
/// axes, so the superior axis is the stable reference. A volume whose
/// scanner axes do not put the head upright (the frame's `up` more than
/// [`EAR_MAX_FRAME_TILT_DEG`] away) keeps the face frame.
fn ear_frame(frame: &FaceFrame) -> FaceFrame {
    use crate::affine::{cross, dot, norm, normalize, scale, sub};
    let sup: Vec3 = [0.0, 0.0, 1.0];
    let up = sub(sup, scale(frame.right, dot(sup, frame.right)));
    if norm(up) < 1e-3 {
        return frame.clone();
    }
    let up = normalize(up);
    if dot(up, frame.up) < EAR_MAX_FRAME_TILT_DEG.to_radians().cos() {
        return frame.clone();
    }
    let mut forward = normalize(cross(frame.right, up));
    if dot(forward, frame.forward) < 0.0 {
        forward = scale(forward, -1.0);
    }
    FaceFrame {
        up,
        forward,
        ..frame.clone()
    }
}

/// Fill isolated NaN cells (thin skin right at the threshold, typical over
/// cartilage) from >= 3 of their 4 neighbours.
fn fill_holes(height: &mut [f64], nv: usize, nw: usize) {
    for _ in 0..2 {
        let prev = height.to_vec();
        for i in 0..nv * nw {
            if !prev[i].is_nan() {
                continue;
            }
            let (r, c) = (i / nw, i % nw);
            let mut acc = 0.0;
            let mut n = 0;
            let mut take = |i2: usize| {
                if !prev[i2].is_nan() {
                    acc += prev[i2];
                    n += 1;
                }
            };
            if r > 0 {
                take(i - nw);
            }
            if r + 1 < nv {
                take(i + nw);
            }
            if c > 0 {
                take(i - 1);
            }
            if c + 1 < nw {
                take(i + 1);
            }
            if n >= 3 {
                height[i] = acc / n as f64;
            }
        }
    }
}

/// One connected protrusion blob and its pinna-plausibility features.
#[derive(Serialize)]
struct EarBlob {
    id: u32,
    /// Protrusion level the blob was segmented at, mm.
    level: f64,
    cells: usize,
    /// Sum of protrusion heights, mm * cells.
    mass: f64,
    /// Robust (90th percentile) protrusion, mm.
    peak: f64,
    /// Protrusion-weighted centre, frame mm.
    centre: (f64, f64),
    /// Bounding box, frame mm.
    v: [f64; 2],
    w: [f64; 2],
    /// Distance of the centre from the measured pinna centre, in SDs.
    prior_z: f64,
    interior: bool,
    plausible: bool,
    score: f64,
}

/// Evidence-based ear ROI for one side, in the ear frame ([`ear_frame`]).
/// The calibrated prior only places a *search window*; inside it the
/// volume itself is measured:
///
/// 1. For every (v, w) column, ray-march along ±u from inside the head
///    outward and record the outermost run of above-threshold samples —
///    the surface height. The outermost tissue is always scalp or pinna
///    (never brain), so the height map is trustworthy where deeper
///    tissue-run analysis is not.
/// 2. The scalp baseline is a morphological opening (rolling min then max)
///    of the height map, so the pinna stands out as a protrusion above it.
/// 3. Protrusion blobs are gated by pinna plausibility — away from the
///    window border and the head edge, pinna-sized, not too tall, and
///    centred within [`EAR_MAX_PRIOR_Z`] SDs of the measured pinna
///    position — and the plausible blob with the most protrusion mass
///    (weighted by the position prior) is the pinna. Skin speckle on the
///    cheek, the jaw angle or a coil pad is rejected rather than taken as
///    the ear. The hull is the blob's padded bounding box, grown to at
///    least a pinna-sized box around its centre. No plausible blob (ear
///    flattened against the coil, outside the FOV) falls back to the
///    prior hull, flagged `detected = false`; that hull spans every
///    measured pinna position.
/// 4. The medial wall is the *scalp field* minus `ear_guard_mm`. The scalp
///    field equals the height map on non-protruding columns and is
///    interpolated inward from them across protrusion blobs — the scalp is
///    continuous under the pinna, so the cut hugs the skull instead of
///    assuming a fixed head width. The wall is floored at the median
///    measured scalp of the hull minus [`EAR_MAX_DEPTH`] and at
///    `ear_inner_mm`, so no column cuts deeper than scalp+skull. Columns
///    with no tissue at all are cut down to the floor: brain is tissue, so
///    an empty column provably contains none.
///
/// `+u` is the patient's anatomical left: `right = up x forward` with `up`
/// anchored mouth->eyes and `forward` out of the face.
fn build_ear_roi(
    volume: &Volume,
    frame: &FaceFrame,
    params: &ResolvedParams,
    sign: f64,
    label: &'static str,
) -> Roi {
    let frame = &ear_frame(frame);
    let ed = frame.eye_dist;
    let thr = surface_threshold(volume, params.surface_threshold);
    let (vc, wc) = (-params.ear_down_mm, -params.ear_back_mm);
    let (hh, hd) = (params.ear_height_mm / 2.0, params.ear_depth_mm / 2.0);
    let (v_lo, v_hi) = (
        (vc - hh).min(EAR_SEARCH_V.0 * ed),
        (vc + hh).max(EAR_SEARCH_V.1 * ed),
    );
    let (w_lo, w_hi) = (
        (wc - hd).min(EAR_SEARCH_W.0 * ed),
        (wc + hd).max(EAR_SEARCH_W.1 * ed),
    );
    let step = EAR_GRID_STEP_MM;
    let nv = ((v_hi - v_lo) / step).ceil() as usize + 1;
    let nw = ((w_hi - w_lo) / step).ceil() as usize + 1;
    let cell = |i: usize| (v_lo + (i / nw) as f64 * step, w_lo + (i % nw) as f64 * step);

    // Column scans: outermost above-threshold |u| (surface height), mm.
    // NaN = no tissue anywhere in the column. The scan starts well inside
    // the head: a narrow head's scalp can sit at the cut floor itself.
    // The samples are kept per column for the bone bound below.
    let u0 = (EAR_MEASURE_FROM * ed).min(params.ear_inner_mm);
    let mut height = vec![f64::NAN; nv * nw];
    let mut profiles: Vec<Vec<f32>> = vec![Vec::new(); nv * nw];
    for (idx, (h, prof)) in height.iter_mut().zip(profiles.iter_mut()).enumerate() {
        let (v, w) = cell(idx);
        let mut u = u0;
        let mut run = 0usize;
        while u <= params.ear_outer_mm {
            let p = crate::affine::add(
                frame.origin,
                crate::affine::add(
                    crate::affine::scale(frame.right, sign * u),
                    crate::affine::add(
                        crate::affine::scale(frame.up, v),
                        crate::affine::scale(frame.forward, w),
                    ),
                ),
            );
            let x = sample(volume, p);
            prof.push(x.unwrap_or(f32::NAN));
            if x.map(|x| x >= thr).unwrap_or(false) {
                run += 1;
                if run >= EAR_SURFACE_RUN {
                    *h = u;
                }
            } else {
                run = 0;
            }
            u += EAR_MARCH_STEP_MM;
        }
    }
    fill_holes(&mut height, nv, nw);

    // Scalp baseline: opening of the height map, so the ear reads as a
    // protrusion above it.
    let radius = ((EAR_OPENING * ed) / step).round().max(1.0) as usize;
    let baseline = grid_filter(
        &grid_filter(&height, nv, nw, radius, false),
        nv,
        nw,
        radius,
        true,
    );

    // Cells whose whole neighbourhood has data: at the head's edge (or
    // beyond the field of view) the baseline is unsupported and the
    // head-to-neck step would read as a giant protrusion.
    let present: Vec<f64> = height
        .iter()
        .map(|h| if h.is_nan() { 0.0 } else { 1.0 })
        .collect();
    let supported = grid_filter(&present, nv, nw, EAR_EDGE_CELLS, false);
    let protrusion: Vec<f64> = (0..nv * nw)
        .map(|i| {
            if supported[i] >= 1.0 {
                height[i] - baseline[i]
            } else {
                f64::NAN
            }
        })
        .collect();
    let is_ear: Vec<bool> = protrusion.iter().map(|&p| p >= EAR_PROTRUSION_MM).collect();

    // Protrusion blobs at several levels and their plausibility features.
    // At the base level a flattened pinna often merges with neighbouring
    // scalp bulges into one implausibly large blob; the pinna rises
    // higher than those, so a higher level separates it.
    let min_cells = (EAR_MIN_AREA * ed * ed / (step * step)).ceil() as usize;
    let (pv, pw) = (EAR_PRIOR_CENTRE.0 * ed, EAR_PRIOR_CENTRE.1 * ed);
    let (sv, sw) = (EAR_PRIOR_SD.0 * ed, EAR_PRIOR_SD.1 * ed);
    let mut blobs: Vec<EarBlob> = Vec::new();
    let mut blob_cells: Vec<Vec<usize>> = Vec::new();
    for &level in EAR_LEVELS_MM {
        let on: Vec<bool> = protrusion.iter().map(|&p| p >= level).collect();
        let mut seen = vec![false; nv * nw];
        for start in 0..nv * nw {
            if !on[start] || seen[start] {
                continue;
            }
            let mut stack = vec![start];
            seen[start] = true;
            let mut members = Vec::new();
            while let Some(c) = stack.pop() {
                members.push(c);
                let (i, j) = (c / nw, c % nw);
                let mut push = |c2: usize| {
                    if on[c2] && !seen[c2] {
                        seen[c2] = true;
                        stack.push(c2);
                    }
                };
                if i > 0 {
                    push(c - nw);
                }
                if i + 1 < nv {
                    push(c + nw);
                }
                if j > 0 {
                    push(c - 1);
                }
                if j + 1 < nw {
                    push(c + 1);
                }
            }
            let (mut bv0, mut bv1, mut bw0, mut bw1) = (usize::MAX, 0usize, usize::MAX, 0usize);
            let (mut mass, mut mv, mut mw) = (0.0, 0.0, 0.0);
            let mut hs = Vec::with_capacity(members.len());
            for &c in &members {
                let (i, j) = (c / nw, c % nw);
                (bv0, bv1) = (bv0.min(i), bv1.max(i));
                (bw0, bw1) = (bw0.min(j), bw1.max(j));
                let h = protrusion[c];
                let (v, w) = cell(c);
                mass += h;
                mv += h * v;
                mw += h * w;
                hs.push(h);
            }
            let k = ((hs.len() as f64 * 0.9).floor() as usize).min(hs.len() - 1);
            let (_, peak, _) = hs.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
            let peak = *peak;
            let centre = (mv / mass, mw / mass);
            let prior_z = (((centre.0 - pv) / sv).powi(2) + ((centre.1 - pw) / sw).powi(2)).sqrt();
            let interior = bv0 > 0 && bv1 + 1 < nv && bw0 > 0 && bw1 + 1 < nw;
            let ext_v = (bv1 - bv0 + 1) as f64 * step;
            let ext_w = (bw1 - bw0 + 1) as f64 * step;
            let plausible = interior
                && members.len() >= min_cells
                && ext_v <= EAR_MAX_EXTENT_V * ed
                && ext_w <= EAR_MAX_EXTENT_W * ed
                && peak <= EAR_MAX_PEAK * ed
                && prior_z <= EAR_MAX_PRIOR_Z;
            blobs.push(EarBlob {
                id: blobs.len() as u32,
                level,
                cells: members.len(),
                mass,
                peak,
                centre,
                v: [v_lo + bv0 as f64 * step, v_lo + bv1 as f64 * step],
                w: [w_lo + bw0 as f64 * step, w_lo + bw1 as f64 * step],
                prior_z,
                interior,
                plausible,
                score: mass * (-0.5 * prior_z * prior_z).exp(),
            });
            blob_cells.push(members);
        }
    }
    let best = blobs
        .iter()
        .filter(|b| b.plausible)
        .max_by(|a, b| a.score.total_cmp(&b.score));
    let detected = best.is_some();

    // Hull: the blob's padded box, grown to a pinna-sized box around its
    // centre; the prior box when nothing plausible was found.
    let (hull_v, hull_w, protrusion_mm) = match best {
        Some(b) => {
            let pad = 0.18 * ed;
            let (mh, mw) = (EAR_MIN_HULL.0 * ed / 2.0, EAR_MIN_HULL.1 * ed / 2.0);
            (
                [
                    (b.v[0] - pad).min(b.centre.0 - mh).max(v_lo),
                    (b.v[1] + pad).max(b.centre.0 + mh).min(v_hi),
                ],
                [
                    (b.w[0] - pad).min(b.centre.1 - mw).max(w_lo),
                    (b.w[1] + pad).max(b.centre.1 + mw).min(w_hi),
                ],
                Some(b.peak),
            )
        }
        None => ([vc - hh, vc + hh], [wc - hd, wc + hd], None),
    };

    // Scalp field: the measured height, except under the detected pinna,
    // where it is interpolated inward from the blob's rim (the scalp runs
    // continuously under the pinna). Other protrusions are scalp bulges
    // (temporal region, mastoid) and keep their measured surface:
    // interpolating under them would put the cut below the real scalp.
    let mut under_pinna = vec![false; nv * nw];
    if let Some(b) = best {
        for &c in &blob_cells[b.id as usize] {
            under_pinna[c] = true;
        }
    }
    // Isolated pits in flat scalp (a column whose outer skin fell under
    // the threshold, so the march stopped on a deeper layer) are raised to
    // the median of their neighbours: the cut follows the surface and
    // never spikes deeper than the columns around it. Columns next to a
    // protrusion are left alone — the folds of the pinna are real grooves.
    let surface: Vec<f64> = (0..nv * nw)
        .map(|i| {
            let h = height[i];
            if h.is_nan() {
                return h;
            }
            let (r, c) = ((i / nw) as isize, (i % nw) as isize);
            let mut near: Vec<f64> = Vec::with_capacity(8);
            for rr in r - 1..=r + 1 {
                for cc in c - 1..=c + 1 {
                    if rr < 0 || cc < 0 || rr >= nv as isize || cc >= nw as isize {
                        continue;
                    }
                    let j = rr as usize * nw + cc as usize;
                    if is_ear[j] {
                        return h;
                    }
                    if j == i {
                        continue;
                    }
                    let x = height[j];
                    if !x.is_nan() {
                        near.push(x);
                    }
                }
            }
            if near.len() < 5 {
                return h;
            }
            let mid = near.len() / 2;
            let (_, med, _) = near.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            if *med - h > EAR_PIT_MM {
                *med
            } else {
                h
            }
        })
        .collect();
    let mut scalp: Vec<f64> = (0..nv * nw)
        .map(|i| if under_pinna[i] { f64::NAN } else { surface[i] })
        .collect();
    loop {
        let mut filled = 0usize;
        let prev = scalp.clone();
        for i in 0..nv * nw {
            if !under_pinna[i] || !scalp[i].is_nan() {
                continue;
            }
            let (r, c) = (i / nw, i % nw);
            let mut acc = 0.0;
            let mut n = 0.0;
            let mut take = |i2: usize| {
                if !prev[i2].is_nan() {
                    acc += prev[i2];
                    n += 1.0;
                }
            };
            if r > 0 {
                take(i - nw);
            }
            if r + 1 < nv {
                take(i + nw);
            }
            if c > 0 {
                take(i - 1);
            }
            if c + 1 < nw {
                take(i + 1);
            }
            if n > 0.0 {
                scalp[i] = acc / n;
                filled += 1;
            }
        }
        if filled == 0 {
            break;
        }
    }

    // Cut floor per column: the local median scalp minus EAR_MAX_DEPTH,
    // never medial of `ear_inner_mm`, so a broken column cannot cut deeper
    // than scalp+skull while the floor still follows the head's width.
    let fr = ((EAR_FLOOR_RADIUS * ed) / step).round().max(1.0) as isize;
    let floor: Vec<f64> = (0..nv * nw)
        .map(|i| {
            let (r, c) = ((i / nw) as isize, (i % nw) as isize);
            let mut near: Vec<f64> = Vec::new();
            for rr in (r - fr).max(0)..(r + fr + 1).min(nv as isize) {
                for cc in (c - fr).max(0)..(c + fr + 1).min(nw as isize) {
                    let x = scalp[rr as usize * nw + cc as usize];
                    if !x.is_nan() {
                        near.push(x);
                    }
                }
            }
            if near.is_empty() {
                return params.ear_inner_mm;
            }
            let mid = near.len() / 2;
            let (_, med, _) = near.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            params.ear_inner_mm.max(*med - EAR_MAX_DEPTH * ed)
        })
        .collect();

    // Medial wall per column, floored: under the detected pinna the
    // interpolated scalp minus the guard (severs the pinna at its root); on
    // other protrusions (an undetected pinna) the surface minus the guard;
    // on flat scalp only the surface minus the shallower skin guard — the
    // scalp above the ear can be thin, and there is no pinna to sever. An
    // empty column has no tissue at all — brain included — so it is cut
    // down to the floor.
    let skin_guard = params.ear_guard_mm * EAR_SKIN_GUARD_FRAC;
    let wall: Vec<f64> = (0..nv * nw)
        .map(|i| {
            if height[i].is_nan() {
                floor[i]
            } else if under_pinna[i] && !scalp[i].is_nan() {
                (scalp[i] - params.ear_guard_mm).max(floor[i])
            } else if is_ear[i] {
                (surface[i] - params.ear_guard_mm).max(floor[i])
            } else {
                (surface[i] - skin_guard).max(floor[i])
            }
        })
        .collect();
    // Bone bound: march medially from the scalp and stop the cut at the
    // first dark run — the skull's outer table (or the air of the ear
    // canal). The pinna and the scalp lie lateral of it and the brain
    // medial, whatever the surface estimates above say: on T2 / PD
    // contrasts the scalp can be thin or dark and the scalp under the
    // pinna, interpolated from the blob's rim, can land below the real
    // skin.
    // Median surface of the 3x3 neighbourhood: a column whose thin outer
    // skin was missed (a single bright voxel fails the surface run) starts
    // its bone search at its neighbours' skin level instead of inside the
    // head.
    let nbhd_surface: Vec<f64> = (0..nv * nw)
        .map(|i| {
            let (r, c) = ((i / nw) as isize, (i % nw) as isize);
            let mut near: Vec<f64> = Vec::with_capacity(9);
            for rr in (r - 1).max(0)..(r + 2).min(nv as isize) {
                for cc in (c - 1).max(0)..(c + 2).min(nw as isize) {
                    let x = surface[rr as usize * nw + cc as usize];
                    if !x.is_nan() {
                        near.push(x);
                    }
                }
            }
            if near.is_empty() {
                return f64::NAN;
            }
            let mid = near.len() / 2;
            *near.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
        })
        .collect();
    // Outer edge of the first dark run medial of `start` (leading air
    // lateral of the skin skipped), or None if the cut `limit` comes
    // first.
    let first_dark = |i: usize, start: f64, limit: f64| -> Option<f64> {
        let prof = &profiles[i];
        let mut k =
            (((start - u0) / EAR_MARCH_STEP_MM).floor() as isize).min(prof.len() as isize - 1);
        while k >= 0 && u0 + k as f64 * EAR_MARCH_STEP_MM > height[i] && {
            let x = prof[k as usize];
            x.is_nan() || x < thr
        } {
            k -= 1;
        }
        let mut run = 0usize;
        while k >= 0 {
            let u = u0 + k as f64 * EAR_MARCH_STEP_MM;
            if u < limit {
                return None;
            }
            if prof[k as usize] < thr {
                run += 1;
                if run >= EAR_BONE_RUN {
                    return Some(u + (EAR_BONE_RUN - 1) as f64 * EAR_MARCH_STEP_MM);
                }
            } else {
                run = 0;
            }
            k -= 1;
        }
        None
    };
    let mut wall = wall;
    for i in 0..nv * nw {
        if height[i].is_nan() {
            continue;
        }
        // Under the pinna, search from just lateral of the interpolated
        // scalp (the pinna's own folds and the gap behind it lie further
        // out). If no skull turns up down to the floor, the interpolation
        // landed below the skull: search again from the outer surface.
        // Elsewhere, search from the surface — or from the neighbours'
        // surface when that is further out (a thin outer skin missed by
        // the surface run).
        let surf = surface[i].min(height[i]).max(nbhd_surface[i]);
        let bone = if under_pinna[i] && !scalp[i].is_nan() {
            first_dark(i, (scalp[i] + EAR_BONE_TOL_MM).min(height[i]), floor[i])
                .or_else(|| first_dark(i, surf, wall[i]))
        } else {
            first_dark(i, surf, wall[i])
        };
        if let Some(outer) = bone {
            wall[i] = wall[i].max(outer);
        }
    }
    let inner = wall.iter().copied().fold(params.ear_outer_mm, f64::min);

    if let Ok(prefix) = std::env::var("RADVEIL_EAR_DEBUG") {
        let class: Vec<u8> = (0..nv * nw)
            .map(|i| {
                if under_pinna[i] {
                    2
                } else if is_ear[i] {
                    1
                } else {
                    0
                }
            })
            .collect();
        let dump = serde_json::json!({
            "label": label, "eye_dist": ed, "nv": nv, "nw": nw, "v_lo": v_lo, "w_lo": w_lo,
            "step": step, "height": height, "baseline": baseline, "blobs": blobs,
            "best": best.map(|b| b.id), "wall": wall, "class": class,
            "frame": {"origin": frame.origin, "right": frame.right, "up": frame.up, "forward": frame.forward},
        });
        let _ = std::fs::write(format!("{prefix}_{label}.json"), dump.to_string());
    }

    Roi {
        label,
        basis: Some(frame.clone()),
        u: if sign > 0.0 {
            [inner, params.ear_outer_mm]
        } else {
            [-params.ear_outer_mm, -inner]
        },
        v: hull_v,
        w: hull_w,
        wall: Some(EarWall {
            sign,
            v0: v_lo,
            w0: w_lo,
            step,
            nv,
            nw,
            wall,
        }),
        face_wall: None,
        seen_through_air: None,
        detected: Some(detected),
        protrusion_mm,
    }
}

/// Voxel-index bounding box of an ROI (transform the 8 ROI corners to
/// voxel space, then clamp), and mark its voxels in the mask. Returns the
/// bbox plus the number of voxels newly set (overlap with an earlier ROI
/// is not double-counted).
fn mark_roi(
    volume: &Volume,
    frame: &FaceFrame,
    roi: &Roi,
    mask: &mut Array3<u8>,
    label: u8,
    protect: Option<&Array3<u8>>,
    spared: &mut usize,
) -> ([isize; 3], [isize; 3], usize) {
    let frame = roi.basis.as_ref().unwrap_or(frame);
    let [nx, ny, nz] = volume.shape();
    let mut lo = [isize::MAX; 3];
    let mut hi = [isize::MIN; 3];
    for &u in &roi.u {
        for &v in &roi.v {
            for &w in &roi.w {
                let p: Vec3 = crate::affine::add(
                    frame.origin,
                    crate::affine::add(
                        crate::affine::add(
                            crate::affine::scale(frame.right, u),
                            crate::affine::scale(frame.up, v),
                        ),
                        crate::affine::scale(frame.forward, w),
                    ),
                );
                let vx = crate::affine::apply(&volume.world_to_voxel, p);
                for a in 0..3 {
                    lo[a] = lo[a].min(vx[a].floor() as isize - 1);
                    hi[a] = hi[a].max(vx[a].ceil() as isize + 1);
                }
            }
        }
    }
    let shape = [nx as isize, ny as isize, nz as isize];
    for a in 0..3 {
        lo[a] = lo[a].clamp(0, shape[a] - 1);
        hi[a] = hi[a].clamp(0, shape[a] - 1);
    }

    // Mark mask voxels.
    let mut changed = 0usize;
    for i in lo[0]..=hi[0] {
        for j in lo[1]..=hi[1] {
            for k in lo[2]..=hi[2] {
                let wp = crate::affine::apply(&volume.affine, [i as f64, j as f64, k as f64]);
                let f = frame.to_face(wp);
                if f[0] >= roi.u[0]
                    && f[0] <= roi.u[1]
                    && f[1] >= roi.v[0]
                    && f[1] <= roi.v[1]
                    && f[2] >= roi.w[0]
                    && f[2] <= roi.w[1]
                    && roi
                        .wall
                        .as_ref()
                        .is_none_or(|wg| wg.sign * f[0] >= wg.wall_at(f[1], f[2]))
                    && roi
                        .face_wall
                        .as_ref()
                        .is_none_or(|fw| f[2] >= fw.at(f[0], f[1]))
                    && mask[[i as usize, j as usize, k as usize]] == 0
                {
                    let idx = [i as usize, j as usize, k as usize];
                    if protect.is_some_and(|p| p[idx] != 0) {
                        *spared += 1;
                    } else {
                        mask[idx] = label;
                        changed += 1;
                    }
                }
            }
        }
    }
    (lo, hi, changed)
}

/// Where to anonymize: the ROIs measured on the *original* volume. Build
/// it once with [`plan`] and [`apply`] it as often as needed; re-applying a
/// plan never reaches beyond its first mask, whereas re-planning on an
/// already-anonymized volume would measure a receded surface and cut
/// deeper each time.
#[derive(Clone)]
pub struct DefacePlan {
    rois: Vec<Roi>,
    region: Region,
    ear_mode: Option<DefaceMode>,
    eye_bias_mm: f64,
    resolved: ResolvedParams,
    /// See [`DefaceParams::spare_air`].
    spare_air: bool,
    /// With `spare_air`: the smooth shell around the *original* tissue a
    /// block may fill (see [`block_shell`]). Measured at plan time, so a
    /// block escalation after a cut does not wrap the exposed skull.
    shell: Option<std::sync::Arc<Array3<bool>>>,
    /// Voxels that must never be rewritten (the brain guard), same grid
    /// as the volume.
    protect: Option<std::sync::Arc<Array3<u8>>>,
}

impl DefacePlan {
    /// Never rewrite the voxels set in `protect` (same grid as the volume),
    /// whatever the region says. [`crate::deface_auto`] uses this for the
    /// brain guard; the report counts the voxels spared.
    pub fn with_protected(mut self, protect: Array3<u8>) -> DefacePlan {
        self.protect = Some(std::sync::Arc::new(protect));
        self
    }

    /// Whether the face surface was seen through air inside the scan. `false`
    /// means the "face" lies on a field-of-view cut: nothing plausible to
    /// anonymize, and no tissue is touched. `None` without a face region.
    pub fn face_seen_through_air(&self) -> Option<bool> {
        self.rois.iter().find_map(|r| r.seen_through_air)
    }

    /// Whether this plan spares air (see [`DefaceParams::spare_air`]).
    pub fn spares_air(&self) -> bool {
        self.spare_air
    }

    /// The same plan without sparing air: every region voxel may be
    /// rewritten (the last verification escalation when a shell-bounded
    /// fill still leaves a face).
    pub fn full_hull(&self) -> DefacePlan {
        DefacePlan {
            spare_air: false,
            shell: None,
            ..self.clone()
        }
    }

    /// The same plan restricted to the face ROI (ear evidence cannot be
    /// re-measured on an anonymized volume, and escalations target the face).
    pub fn face_only(&self) -> DefacePlan {
        DefacePlan {
            rois: self
                .rois
                .iter()
                .filter(|r| r.label == "face")
                .cloned()
                .collect(),
            region: Region::Face,
            ..self.clone()
        }
    }
}

/// Measure the ROIs for `params` on `volume` (see [`DefacePlan`]).
pub fn plan(volume: &Volume, frame: &FaceFrame, params: &DefaceParams) -> DefacePlan {
    let resolved = params.resolve(frame.eye_dist);
    let mut rois = Vec::new();
    let mut eye_bias_mm = 0.0;
    if params.region.includes_face() {
        if let Some(b) = params.eye_bias_mm {
            eye_bias_mm = b.clamp(0.0, 0.25 * frame.eye_dist);
        } else if params.eye_surface_anchor.unwrap_or(true) {
            eye_bias_mm = eye_surface_bias(
                volume,
                frame,
                surface_threshold(volume, resolved.surface_threshold),
            );
        }
        rois.push(build_face_roi(
            volume,
            frame,
            &resolved,
            eye_bias_mm,
            params.level_face,
        ));
    }
    if params.region.includes_ears() {
        rois.push(build_ear_roi(volume, frame, &resolved, 1.0, "ear_left"));
        rois.push(build_ear_roi(volume, frame, &resolved, -1.0, "ear_right"));
    }
    DefacePlan {
        rois,
        region: params.region,
        ear_mode: params.ear_mode,
        eye_bias_mm,
        resolved,
        spare_air: params.spare_air,
        shell: params.spare_air.then(|| {
            std::sync::Arc::new(block_shell(
                volume,
                resolved.blur_sigma_mm,
                resolved.surface_threshold,
            ))
        }),
        protect: None,
    }
}

/// Apply anonymization in place. Returns a report plus the binary mask
/// (1 = anonymized voxel).
pub fn deface(
    volume: &mut Volume,
    frame: &FaceFrame,
    params: &DefaceParams,
) -> (DefaceReport, Array3<u8>) {
    let p = plan(volume, frame, params);
    apply(volume, frame, &p, params.mode, None)
}

// Face voxels are labelled 1, ear voxels 2 (the face is marked first, so a
// voxel in both keeps the face label); each label is rewritten with its own
// mode and the exported mask is binarized.
const FACE: u8 = 1;
const EARS: u8 = 2;

/// Rewrite the voxels of a [`DefacePlan`]: the face with `mode`, the ears
/// with the plan's `ear_mode` (default `mode`); `blur_sigma_mm` overrides
/// the plan's sigma (verification escalations).
pub fn apply(
    volume: &mut Volume,
    frame: &FaceFrame,
    plan: &DefacePlan,
    mode: DefaceMode,
    blur_sigma_mm: Option<f64>,
) -> (DefaceReport, Array3<u8>) {
    let mut resolved = plan.resolved;
    if let Some(sigma) = blur_sigma_mm {
        resolved.blur_sigma_mm = sigma;
    }
    let ear_mode = plan.ear_mode.unwrap_or(mode);
    let (mut report, mut mask, boxes) = mark_plan(volume, frame, plan, mode);
    report.resolved = resolved;
    let params_mode = mode;
    // Fill values come from the untouched input, before any region is
    // rewritten, so the order of the two groups cannot change them.
    let cut_fill = volume.percentile(1.0).min(0.0);
    let block_fill = volume.percentile(98.0);
    let air = plan
        .spare_air
        .then(|| air_level(volume, plan.resolved.surface_threshold));
    // Blur-family groups run before fills, so a blur kernel reaching past
    // its own region samples original tissue rather than another region's
    // cut or block fill.
    let mut groups = [(FACE, params_mode), (EARS, ear_mode)];
    groups.sort_by_key(|g| matches!(g.1, DefaceMode::Cut | DefaceMode::Block));
    for (label, mode) in groups {
        if !boxes.iter().any(|b| b.3 == label && b.2 > 0) {
            continue;
        }
        // Sparing air, nothing is written beyond a smooth shell around the
        // original tissue: no far-field blur halo, no slab out to the hull.
        if let Some(shell) = &plan.shell {
            ndarray::Zip::from(&mut mask)
                .and(&**shell)
                .for_each(|m, &near| {
                    if *m == label && !near {
                        *m = 0;
                    }
                });
        }
        match mode {
            // Background fill: ~0 on MRI; ~air on CT (never insert
            // water-density 0 HU into a CT).
            DefaceMode::Cut | DefaceMode::Block => {
                let fill = if mode == DefaceMode::Cut {
                    cut_fill
                } else {
                    block_fill
                };
                ndarray::Zip::from(&mut volume.data)
                    .and(&mut mask)
                    .for_each(|v, m| {
                        if *m == label {
                            if air.is_some_and(|a| *v < a && fill < a) {
                                *m = 0;
                            } else {
                                *v = fill;
                            }
                        }
                    });
            }
            // Blur per ROI bounding box: the boxes are small and disjoint
            // (face vs. ears), so blurring each is much cheaper than
            // blurring their joint bounding box.
            DefaceMode::Blur | DefaceMode::Adaptive => {
                let factor = if mode == DefaceMode::Adaptive {
                    Some(resolved.adaptive_factor)
                } else {
                    None
                };
                for &(lo, hi, n, l) in &boxes {
                    if l == label && n > 0 {
                        blur_roi(
                            volume,
                            &mut mask,
                            lo,
                            hi,
                            resolved.blur_sigma_mm,
                            factor,
                            label,
                            air,
                        );
                    }
                }
            }
        }
    }
    mask.mapv_inplace(|m| u8::from(m != 0));
    if air.is_some() {
        let written = mask.iter().filter(|&&m| m != 0).count();
        report.air_voxels_spared = Some(report.voxels_changed - written);
        report.voxels_changed = written;
    }

    volume.accel.invalidate();
    (report, mask)
}

/// The anonymization mask of a [`DefacePlan`] (1 = a voxel [`apply`] would
/// rewrite), leaving the volume untouched: an escape hatch for callers who
/// want to anonymize or review the region themselves. Nothing is verified;
/// the report's `voxels_changed` counts the voxels that *would* change.
pub fn plan_mask(
    volume: &Volume,
    frame: &FaceFrame,
    plan: &DefacePlan,
    mode: DefaceMode,
) -> (DefaceReport, Array3<u8>) {
    let (report, mut mask, _) = mark_plan(volume, frame, plan, mode);
    mask.mapv_inplace(|m| u8::from(m != 0));
    (report, mask)
}

type RoiBox = ([isize; 3], [isize; 3], usize, u8);

/// Rasterize every ROI of `plan` into a labelled mask (face 1, ears 2);
/// also returns each ROI's voxel bounding box and label (for the blur
/// passes).
fn mark_plan(
    volume: &Volume,
    frame: &FaceFrame,
    plan: &DefacePlan,
    mode: DefaceMode,
) -> (DefaceReport, Array3<u8>, Vec<RoiBox>) {
    let [nx, ny, nz] = volume.shape();
    let mut mask = Array3::<u8>::zeros((nx, ny, nz));
    let mut changed = 0usize;
    let mut boxes = Vec::with_capacity(plan.rois.len());
    let mut roi_reports = Vec::with_capacity(plan.rois.len());
    let mut spared = 0usize;
    for roi in &plan.rois {
        let label = if roi.label == "face" { FACE } else { EARS };
        let (lo, hi, n) = mark_roi(
            volume,
            frame,
            roi,
            &mut mask,
            label,
            plan.protect.as_deref(),
            &mut spared,
        );
        changed += n;
        boxes.push((lo, hi, n, label));
        roi_reports.push(RoiReport {
            label: roi.label.to_string(),
            u_mm: roi.u,
            v_mm: roi.v,
            w_mm: roi.w,
            voxels: n,
            detected: roi.detected,
            protrusion_mm: roi.protrusion_mm,
        });
    }
    let report = DefaceReport {
        mode,
        ear_mode: plan.ear_mode.unwrap_or(mode),
        region: plan.region,
        voxels_changed: changed,
        voxels_total: nx * ny * nz,
        eye_dist_mm: frame.eye_dist,
        eye_bias_mm: plan.eye_bias_mm,
        rois: roi_reports,
        resolved: plan.resolved,
        face_seen_through_air: plan.face_seen_through_air(),
        protected_voxels_spared: plan.protect.as_ref().map(|_| spared),
        air_voxels_spared: None,
    };
    (report, mask, boxes)
}

/// Fast Gaussian approximation: three box-filter passes per axis
/// (Kuckir/W3C radii), each O(n) independent of sigma, lanes processed in
/// parallel with rayon. Accuracy is well within what feature destruction
/// needs, and it turns the previous O(n*k) separable convolution (k up to
/// hundreds of taps on fine grids) into a flat O(n).
fn box_radii_for_gauss(sigma: f64) -> [usize; 3] {
    let n = 3.0;
    let w_ideal = (12.0 * sigma * sigma / n + 1.0).sqrt();
    let mut wl = w_ideal.floor() as i64;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wl = wl.max(1);
    let wu = wl + 2;
    let m_ideal = (12.0 * sigma * sigma - n * (wl * wl) as f64 - 4.0 * n * wl as f64 - 3.0 * n)
        / (-4.0 * wl as f64 - 4.0);
    let m = m_ideal.round().clamp(0.0, 3.0) as usize;
    let mut out = [0usize; 3];
    for (i, o) in out.iter_mut().enumerate() {
        let w = if i < m { wl } else { wu };
        *o = ((w - 1) / 2) as usize;
    }
    out
}

/// Gaussian approximation over a C-order buffer via the SIMD-friendly
/// kernels in [`crate::kernels`].
fn gaussian3(src: &Array3<f32>, sigmas: [f64; 3]) -> Array3<f32> {
    let (nx, ny, nz) = src.dim();
    let mut data = src
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_else(|| src.iter().copied().collect());
    gaussian3_slices(&mut data, [nx, ny, nz], sigmas);
    Array3::from_shape_vec((nx, ny, nz), data).expect("shape preserved")
}

fn gaussian3_slices(data: &mut [f32], dims: [usize; 3], sigmas: [f64; 3]) {
    let mut scratch = Vec::with_capacity(data.len());
    for (axis, &sigma) in sigmas.iter().enumerate() {
        if sigma <= 0.2 {
            continue;
        }
        for r in box_radii_for_gauss(sigma) {
            crate::kernels::box_pass_axis(data, &mut scratch, dims, axis, r);
        }
    }
}

/// Blurred-indicator level that bounds a block's shell: about 1.6 sigma
/// beyond a flat skin surface.
const BLOCK_SHELL_LEVEL: f32 = 0.05;

/// Smooth shell around the tissue (voxels at or above [`air_level`]): the
/// tissue indicator blurred with `sigma_mm` reaches [`BLOCK_SHELL_LEVEL`].
/// The blur rounds the shell off, so its outline is not the face's.
fn block_shell(volume: &Volume, sigma_mm: f64, thr: Option<f32>) -> Array3<bool> {
    let air = air_level(volume, thr);
    let ind = volume.data.mapv(|v| f32::from(v >= air));
    let sigmas: [f64; 3] = std::array::from_fn(|a| sigma_mm / volume.spacing[a]);
    gaussian3(&ind, sigmas).mapv(|x| x >= BLOCK_SHELL_LEVEL)
}

/// Blur the ROI bounding box and splat into masked voxels only.
///
/// With `adaptive_factor` = None this is a plain Gaussian of `sigma_mm`.
/// Otherwise a second, heavier Gaussian (sigma * factor) is computed and the
/// two are mixed per voxel by normalized local gradient magnitude: strong
/// facial lines (high gradient) receive the heavy averaging, flat regions
/// the mild one.
#[allow(clippy::too_many_arguments)]
fn blur_roi(
    volume: &mut Volume,
    mask: &mut Array3<u8>,
    lo: [isize; 3],
    hi: [isize; 3],
    sigma_mm: f64,
    adaptive_factor: Option<f64>,
    label: u8,
    air: Option<f32>,
) {
    let b0 = [lo[0] as usize, lo[1] as usize, lo[2] as usize];
    let b1 = [hi[0] as usize, hi[1] as usize, hi[2] as usize];
    let bs = [b1[0] - b0[0] + 1, b1[1] - b0[1] + 1, b1[2] - b0[2] + 1];

    let max_sigma_mm = sigma_mm * adaptive_factor.unwrap_or(1.0);
    let sigmas: [f64; 3] = std::array::from_fn(|a| sigma_mm / volume.spacing[a]);
    let sigmas_hi: [f64; 3] = std::array::from_fn(|a| max_sigma_mm / volume.spacing[a]);
    let radii: [usize; 3] = std::array::from_fn(|a| (sigmas_hi[a] * 3.0).ceil() as usize);

    // Padded subvolume (pad by the largest kernel radius, clamped).
    let p0 = [
        b0[0].saturating_sub(radii[0]),
        b0[1].saturating_sub(radii[1]),
        b0[2].saturating_sub(radii[2]),
    ];
    let shape = volume.shape();
    let p1 = [
        (b1[0] + radii[0]).min(shape[0] - 1),
        (b1[1] + radii[1]).min(shape[1] - 1),
        (b1[2] + radii[2]).min(shape[2] - 1),
    ];
    let ps = [p1[0] - p0[0] + 1, p1[1] - p0[1] + 1, p1[2] - p0[2] + 1];

    let mut src = Array3::<f32>::zeros((ps[0], ps[1], ps[2]));
    for i in 0..ps[0] {
        for j in 0..ps[1] {
            for k in 0..ps[2] {
                src[[i, j, k]] = volume.data[[p0[0] + i, p0[1] + j, p0[2] + k]];
            }
        }
    }

    let mild = gaussian3(&src, sigmas);
    let result = match adaptive_factor {
        None => mild,
        Some(_) => {
            let dims = [ps[0], ps[1], ps[2]];
            let n = ps[0] * ps[1] * ps[2];
            let heavy = gaussian3(&src, sigmas_hi);
            let mild_sl = mild.as_slice().expect("C-order");
            // Gradient magnitude of the mildly smoothed field (mm^-1 scale).
            let mut grad = vec![0.0f32; n];
            crate::kernels::gradient_mag(mild_sl, dims, volume.spacing, &mut grad);
            // Normalize by a robust high percentile (subsampled).
            let stride = (n / 1_000_000).max(1);
            let mut gv: Vec<f32> = grad
                .iter()
                .step_by(stride)
                .copied()
                .filter(|v| *v > 0.0)
                .collect();
            let scale = if gv.is_empty() {
                1.0
            } else {
                let n95 = ((gv.len() - 1) as f64 * 0.95) as usize;
                let (_, v, _) = gv.select_nth_unstable_by(n95, |a, b| a.total_cmp(b));
                (*v).max(1e-6)
            };
            crate::kernels::clamp_scale(&mut grad, scale);
            // Smooth the weight field so kernel choice varies gently.
            gaussian3_slices(&mut grad, dims, sigmas);
            let mut out = mild;
            crate::kernels::mix_buffers(
                out.as_slice_mut().expect("C-order"),
                heavy.as_slice().expect("C-order"),
                &grad,
            );
            out
        }
    };

    // Splat into masked voxels only.
    for i in 0..bs[0] {
        for j in 0..bs[1] {
            for k in 0..bs[2] {
                let gi = b0[0] + i;
                let gj = b0[1] + j;
                let gk = b0[2] + k;
                if mask[[gi, gj, gk]] == label {
                    let new = result[[gi - p0[0], gj - p0[1], gk - p0[2]]];
                    if air.is_some_and(|a| volume.data[[gi, gj, gk]] < a && new < a) {
                        mask[[gi, gj, gk]] = 0;
                    } else {
                        volume.data[[gi, gj, gk]] = new;
                    }
                }
            }
        }
    }
}
