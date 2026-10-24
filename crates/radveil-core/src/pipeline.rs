// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Automatic pipeline: render -> detect -> validate -> lift -> anonymize ->
//! verify. Detector-agnostic: pass the built-in [`MediaPipeDetector`]
//! (feature `detector`) or any other [`FaceDetector`] (the Python package
//! accepts any callable detector this way).
//!
//! [`MediaPipeDetector`]: crate::detect::MediaPipeDetector

use crate::affine::{add, cross, dot, norm, scale, sub, Vec3};
use crate::deface::{
    apply, plan, plan_mask, DefaceMode, DefaceParams, DefacePlan, DefaceReport, Region,
};
use crate::landmarks::{combine_views, face_frame_from_landmarks, unproject, FaceFrame};
use crate::render::{render_view, RenderParams, RenderView};
use crate::volume::Volume;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// A face found on a rendered view.
#[derive(Clone, Debug)]
pub struct Detection {
    /// 68 landmarks in 68-point order, pixel coordinates of the image.
    pub landmarks: Vec<[f64; 2]>,
    /// Detector confidence in 0..1 (detectors without a calibrated score
    /// report 1.0).
    pub score: f64,
}

/// A 2D face-landmark detector run on rendered views.
pub trait FaceDetector {
    /// Detect the most prominent face in an RGB8 image (`w` x `h`,
    /// row-major), or `None` when there is no face. `try_rotations` asks
    /// the detector to also consider the image rotated by 90/180/270°
    /// (used for volumes with a wrong stored orientation).
    fn detect(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
        try_rotations: bool,
    ) -> Result<Option<Detection>, String>;
}

#[cfg(feature = "detector")]
impl FaceDetector for crate::detect::MediaPipeDetector {
    fn detect(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
        try_rotations: bool,
    ) -> Result<Option<Detection>, String> {
        Ok(
            crate::detect::MediaPipeDetector::detect(self, rgb, w, h, try_rotations)?.map(|m| {
                Detection {
                    landmarks: m.landmarks68(),
                    score: m.detection.score,
                }
            }),
        )
    }
}

/// Plausible adult/pediatric inter-eye distance for a single view, mm.
const EYE_DIST_RANGE: std::ops::RangeInclusive<f64> = 25.0..=120.0;
/// Two views show the same face when the median 3D distance between their
/// lifted central landmarks (brows, nose, eyes, mouth) is below this
/// fraction of the inter-eye distance. Measured: real faces <= 0.34
/// (FOV-cut clinical FLAIR), IXI <= 0.10; detector hallucinations on
/// already-defaced heads >= 0.58.
const SHAPE_AGREEMENT: f64 = 0.45;
/// Single-view validation: offsets (yaw, pitch; degrees) around the
/// implied face axis at which the face is re-detected, the (stricter)
/// agreement each re-detection must reach, and how many must agree.
/// Measured: a real face is re-found within 0.18·d from 6-7 of these 7
/// angles; hallucinations on smooth or defaced heads (which *are*
/// re-hallucinated from nearby angles) reach at most 2.
const VALIDATION_OFFSETS: [(f64, f64); 7] = [
    (0.0, 0.0),
    (-20.0, 0.0),
    (20.0, 0.0),
    (-35.0, 0.0),
    (35.0, 0.0),
    (0.0, -20.0),
    (0.0, 20.0),
];
const VALIDATION_AGREEMENT: f64 = 0.25;
const VALIDATION_MIN_VIEWS: usize = 4;
/// Verification counts a re-detection as residual when its lifted central
/// landmarks lie within this multiple of the eye distance of the
/// anonymized face (detections elsewhere on the head -- e.g. hallucinated
/// at the ear -- must not force further escalation of the face).
const RESIDUAL_RADIUS: f64 = 0.75;

/// Pipeline settings. `Default` reproduces the Python `deface_auto`
/// defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoParams {
    /// Anonymization parameters (mode, region, geometry overrides).
    pub deface: DefaceParams,
    /// Yaw angles of the detection views, degrees.
    pub yaws: Vec<f64>,
    pub pitch: f64,
    /// Render size, pixels.
    pub size: usize,
    pub threshold_frac: f32,
    pub threshold: Option<f32>,
    /// Re-render and re-detect after anonymization, escalating until the
    /// detector no longer finds a face.
    pub verify: bool,
    /// Extra pitches (degrees) every detected view is re-checked at during
    /// verification, besides its own: a cut or heavy blur can expose
    /// orbit structure that only a view from below sees.
    #[serde(default = "default_verify_pitches")]
    pub verify_pitches: Vec<f64>,
    /// How many angles verification attacks the output from (speed vs
    /// quality); see [`VerifyEffort`].
    #[serde(default)]
    pub verify_effort: VerifyEffort,
    /// Also verify with the built-in MediaPipe detector (feature
    /// `detector`), whatever `det` is. Default on; pointless only when `det`
    /// is the built-in detector itself.
    #[serde(default = "default_true")]
    pub verify_with_builtin: bool,
    pub max_verify_rounds: usize,
    pub escalate_to_block: bool,
    /// Also return the anonymization mask.
    pub return_mask: bool,
    /// Safety check: extract the brain from the original volume
    /// (`deskull`, SynthStrip) and report how many brain voxels the
    /// anonymization modified. Needs the `synthstrip` feature.
    pub brain_check: bool,
    /// Brain guard (opt-in; needs the `synthstrip` feature): extract the
    /// brain from the original volume and never rewrite it, nor anything
    /// within `brain_guard_mm` of it, whatever the region geometry says.
    /// Costs one brain extraction (~3 s, ~3.2 GB peak). The report's
    /// `deface.protected_voxels_spared` counts the region voxels it kept.
    #[serde(default)]
    pub brain_guard: bool,
    /// Brain guard margin, mm.
    #[serde(default = "default_brain_guard_mm")]
    pub brain_guard_mm: f64,
    /// Faces cut by the field of view (opt-in): when the upright views
    /// validate no face, also look from slightly above and below
    /// ([`PARTIAL_PITCHES`]); build the frame without a mouth when the scan
    /// has none; and measure the skin along the level head axes
    /// (`level_face`). For axial slabs that end at the nose.
    #[serde(default)]
    pub partial_face: bool,
    /// Noisy, dim scans (infants; opt-in): when no face is found at the
    /// default surface threshold, retry at higher ones
    /// ([`NOISY_THRESHOLD_FRACS`]), where background noise no longer renders
    /// as a solid shell around the head. Ignored when the threshold is set
    /// explicitly or the volume is CT.
    #[serde(default)]
    pub auto_threshold: bool,
}

/// Extra detection pitches (degrees) for `partial_face`.
const PARTIAL_PITCHES: [f64; 2] = [-15.0, 15.0];
/// Extra verification pitches for `partial_face`: a partial face is seen
/// from fewer angles, so its output is attacked from more.
const PARTIAL_VERIFY_PITCHES: [f64; 4] = [-30.0, -15.0, 15.0, 30.0];
/// `partial_face`: landmarks within this many voxels of the volume's edge
/// lie on the field-of-view cut.
const FOV_EDGE_VOXELS: f64 = 1.5;

/// Whether most lifted landmarks in `range` lie on the volume's boundary,
/// i.e. where the field of view cut the head.
fn on_fov_edge(vol: &Volume, pts: &[Option<Vec3>], range: std::ops::Range<usize>) -> bool {
    let shape = vol.shape();
    let (mut n, mut edge) = (0usize, 0usize);
    for p in pts.iter().take(range.end).skip(range.start).flatten() {
        let v = crate::affine::apply(&vol.world_to_voxel, *p);
        n += 1;
        if (0..3).any(|a| v[a] < FOV_EDGE_VOXELS || v[a] > shape[a] as f64 - 1.0 - FOV_EDGE_VOXELS)
        {
            edge += 1;
        }
    }
    n > 0 && 2 * edge > n
}

fn default_brain_guard_mm() -> f64 {
    3.0
}

fn default_true() -> bool {
    true
}

/// Verification effort: besides every detection view (at its own pitch
/// and at `verify_pitches`), the output is re-rendered on a yaw x pitch
/// grid centred on the detected face direction. Every extra view costs
/// a render and a detection per verification round.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VerifyEffort {
    /// Detection views only (the behaviour before 0.2).
    Fast,
    /// Plus a 3 x 3 grid: yaw -25/0/+25 x pitch -20/0/+20 degrees.
    #[default]
    Balanced,
    /// Plus a 5 x 5 grid: yaw -45..+45 x pitch -30..+30 degrees.
    Thorough,
}

impl VerifyEffort {
    /// Grid (yaw, pitch) offsets, degrees.
    fn grid(self) -> Vec<(f64, f64)> {
        let (yaws, pitches): (&[f64], &[f64]) = match self {
            VerifyEffort::Fast => (&[], &[]),
            VerifyEffort::Balanced => (&[-25.0, 0.0, 25.0], &[-20.0, 0.0, 20.0]),
            VerifyEffort::Thorough => (
                &[-45.0, -25.0, 0.0, 25.0, 45.0],
                &[-30.0, -20.0, 0.0, 20.0, 30.0],
            ),
        };
        yaws.iter()
            .flat_map(|&y| pitches.iter().map(move |&p| (y, p)))
            .collect()
    }
}

impl std::str::FromStr for VerifyEffort {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "fast" => Ok(VerifyEffort::Fast),
            "balanced" => Ok(VerifyEffort::Balanced),
            "thorough" => Ok(VerifyEffort::Thorough),
            other => Err(format!(
                "unknown verify effort '{other}' (fast|balanced|thorough)"
            )),
        }
    }
}

fn default_verify_pitches() -> Vec<f64> {
    vec![0.0, -15.0]
}

impl Default for AutoParams {
    fn default() -> Self {
        AutoParams {
            deface: DefaceParams::default(),
            yaws: vec![-20.0, 0.0, 20.0],
            pitch: 0.0,
            size: 512,
            threshold_frac: 0.14,
            threshold: None,
            verify: true,
            verify_pitches: default_verify_pitches(),
            verify_effort: VerifyEffort::default(),
            verify_with_builtin: true,
            max_verify_rounds: 3,
            escalate_to_block: true,
            return_mask: false,
            brain_check: false,
            brain_guard: false,
            brain_guard_mm: default_brain_guard_mm(),
            partial_face: false,
            auto_threshold: false,
        }
    }
}

/// One detection view that contributed to the landmarks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UsedView {
    pub yaw_deg: f64,
    pub pitch_deg: f64,
    /// 68 landmarks (68-point order) in that view's pixels.
    pub landmarks: Vec<[f64; 2]>,
    /// Detector confidence for this view.
    pub score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoReport {
    /// Report of the first anonymization round.
    pub deface: DefaceReport,
    /// Face frame the anonymization was placed in.
    pub frame: FaceFrame,
    pub views_detected: usize,
    pub orientation: String,
    pub orientation_sweep: bool,
    pub landmarks_3d: Vec<Option<Vec3>>,
    pub views: Vec<UsedView>,
    /// Verification (absent when `verify` is off or region is ears-only).
    pub verify_rounds: Option<usize>,
    pub residual_detections: Option<usize>,
    pub escalated_to_block: Option<bool>,
    pub final_blur_sigma_mm: Option<f64>,
    /// Anonymization mask (1 = modified), when requested.
    #[serde(skip)]
    pub mask: Option<Array3<u8>>,
    /// Result of the brain safety check (`brain_check`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brain: Option<BrainCheck>,
}

/// How the anonymization related to the brain (see [`AutoParams::brain_check`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrainCheck {
    /// Brain voxels the anonymization modified. 0 means the brain is
    /// untouched.
    pub voxels_modified: usize,
    /// Extracted brain volume, millilitres.
    pub brain_volume_ml: f64,
    /// Smallest distance from a modified voxel to the brain, mm; `None`
    /// when the brain was touched.
    pub margin_mm: Option<f64>,
}

struct Candidate {
    view: RenderView,
    lm: Vec<[f64; 2]>,
    score: f64,
    pts: Vec<Option<Vec3>>,
    eye_dist: f64,
}

fn mean(pts: &[Option<Vec3>], idx: std::ops::Range<usize>) -> Option<Vec3> {
    let v: Vec<Vec3> = idx.filter_map(|i| pts.get(i).copied().flatten()).collect();
    if v.is_empty() {
        return None;
    }
    let s = v.iter().fold([0.0; 3], |a, p| add(a, *p));
    Some(scale(s, 1.0 / v.len() as f64))
}

/// Eye centers, nose tip, mouth center of a lifted landmark set.
struct KeyPoints {
    eye_r: Vec3,
    eye_l: Vec3,
    nose: Vec3,
    mouth: Option<Vec3>,
}

fn key_points(pts: &[Option<Vec3>]) -> Option<KeyPoints> {
    Some(KeyPoints {
        eye_r: mean(pts, 36..42)?,
        eye_l: mean(pts, 42..48)?,
        nose: pts.get(30).copied().flatten()?,
        mouth: mean(pts, 48..68),
    })
}

/// Outward face normal implied by the key points.
fn face_forward(kp: &KeyPoints) -> Option<Vec3> {
    let eye_mid = scale(add(kp.eye_r, kp.eye_l), 0.5);
    if let Some(mouth) = kp.mouth {
        let f = cross(sub(kp.eye_l, kp.eye_r), sub(eye_mid, mouth));
        let n = norm(f);
        if n > 1e-6 {
            let f = scale(f, 1.0 / n);
            let center = scale(add(add(kp.eye_r, kp.eye_l), mouth), 1.0 / 3.0);
            return Some(if dot(f, sub(kp.nose, center)) < 0.0 {
                scale(f, -1.0)
            } else {
                f
            });
        }
    }
    let f = sub(kp.nose, eye_mid);
    let n = norm(f);
    (n > 1e-6).then(|| scale(f, 1.0 / n))
}

/// Median 3D distance between two lifted landmark sets over the central
/// face (brows, nose, eyes, mouth), relative to their eye distance.
fn shape_disagreement(a: &Candidate, b: &Candidate) -> f64 {
    let mut d: Vec<f64> = (17..68)
        .filter_map(|i| match (a.pts.get(i), b.pts.get(i)) {
            (Some(Some(p)), Some(Some(q))) => Some(norm(sub(*p, *q))),
            _ => None,
        })
        .collect();
    if d.len() < 10 {
        return f64::INFINITY;
    }
    d.sort_by(f64::total_cmp);
    d[d.len() / 2] / (0.5 * (a.eye_dist + b.eye_dist))
}

/// Run the detector on a view and lift the result; keeps it only if its
/// implied inter-eye distance is anatomical (rejects tiny false positives).
fn candidate(
    det: &mut dyn FaceDetector,
    view: RenderView,
    rot: bool,
) -> Result<Option<Candidate>, String> {
    let (w, h) = (view.camera.width, view.camera.height);
    let Some(Detection {
        landmarks: lm,
        score,
    }) = det.detect(&view.rgb, w, h, rot)?
    else {
        return Ok(None);
    };
    let pts = unproject(&view, &lm);
    let Some(kp) = key_points(&pts) else {
        return Ok(None);
    };
    let eye_dist = norm(sub(kp.eye_l, kp.eye_r));
    if !EYE_DIST_RANGE.contains(&eye_dist) {
        return Ok(None);
    }
    Ok(Some(Candidate {
        view,
        lm,
        score,
        pts,
        eye_dist,
    }))
}

/// Group candidate views by shape consensus and return the first group
/// that validates, or `None`.
///
/// A group of two or more views that agree in 3D is geometrically
/// self-validating (hallucinations land at different 3D spots from
/// different angles). A single view must be re-found with nearly the same
/// 3D shape from most of the [`VALIDATION_OFFSETS`] renders around its
/// implied face axis. (One head-on re-detection is not enough: a
/// hallucinated "face" on a smooth surface tends to be hallucinated again
/// when looked at straight on.)
fn validate(
    det: &mut dyn FaceDetector,
    vol: &Volume,
    render: &dyn Fn(&Volume, f64, f64) -> RenderView,
    cands: Vec<Candidate>,
) -> Result<Option<Vec<Candidate>>, String> {
    let mut clusters: Vec<Vec<Candidate>> = Vec::new();
    for c in cands {
        match clusters
            .iter_mut()
            .find(|cl| shape_disagreement(&cl[0], &c) <= SHAPE_AGREEMENT)
        {
            Some(cl) => cl.push(c),
            None => clusters.push(vec![c]),
        }
    }
    clusters.sort_by_key(|cl| std::cmp::Reverse(cl.len()));
    for mut cl in clusters {
        if cl.len() >= 2 {
            return Ok(Some(cl));
        }
        let Some(f) = key_points(&cl[0].pts).as_ref().and_then(face_forward) else {
            continue;
        };
        let yaw = f[0].atan2(f[1]).to_degrees();
        let pitch = f[2].clamp(-1.0, 1.0).asin().to_degrees().clamp(-89.0, 89.0);
        for (dy, dp) in VALIDATION_OFFSETS {
            let view = render(vol, yaw + dy, (pitch + dp).clamp(-89.0, 89.0));
            if let Some(v) = candidate(det, view, true)? {
                if shape_disagreement(&cl[0], &v) <= VALIDATION_AGREEMENT {
                    cl.push(v);
                }
            }
        }
        if cl.len() > VALIDATION_MIN_VIEWS {
            return Ok(Some(cl));
        }
    }
    Ok(None)
}

/// A validated face located in a volume (see [`locate_face`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocatedFace {
    pub frame: FaceFrame,
    pub landmarks_3d: Vec<Option<Vec3>>,
    pub views: Vec<UsedView>,
    pub orientation_sweep: bool,
}

/// Render -> detect -> validate -> lift: find the face and its 3D frame
/// without modifying the volume. Shared by [`deface_auto`] and brain
/// extraction (which uses the face as an anatomical prior).
///
/// Views that detect a face are grouped by *shape consensus*: two views
/// agree when their lifted central landmarks coincide in 3D (see
/// [`SHAPE_AGREEMENT`] and `validate`). If the upright views yield no
/// validated face, a 6-direction orientation sweep (with in-plane
/// rotations) handles volumes whose stored orientation is wrong.
pub fn locate_face(
    vol: &Volume,
    det: &mut dyn FaceDetector,
    params: &AutoParams,
) -> Result<LocatedFace, String> {
    let rp = RenderParams {
        size: params.size,
        threshold_frac: params.threshold_frac,
        threshold_abs: params.threshold,
        step_mm: 0.0,
    };
    let render = |vol: &Volume, yaw: f64, pitch: f64| render_view(vol, yaw, pitch, &rp);

    let mut upright = Vec::new();
    for &y in &params.yaws {
        if let Some(c) = candidate(det, render(vol, y, params.pitch), false)? {
            upright.push(c);
        }
    }
    let mut n_cands = upright.len();
    let mut accepted = validate(det, vol, &render, upright)?;
    if accepted.is_none() && params.partial_face {
        // A face cut at the nose is often found only from above or below
        // the horizontal: pool those views with the upright ones (rendered
        // again; only scans that would be refused pay for it).
        let mut pool = Vec::new();
        for &y in &params.yaws {
            for dp in std::iter::once(0.0).chain(PARTIAL_PITCHES) {
                if let Some(c) = candidate(det, render(vol, y, params.pitch + dp), false)? {
                    pool.push(c);
                }
            }
        }
        n_cands = pool.len();
        accepted = validate(det, vol, &render, pool)?;
    }
    let mut orientation_sweep = false;
    if accepted.is_none() {
        // Orientation sweep: the stored affine may be wrong (e.g. Analyze
        // conversions). Look at the head from all six axis directions and
        // let the detector try in-plane rotations too.
        orientation_sweep = true;
        let sweep = [
            (0.0, 0.0),
            (90.0, 0.0),
            (180.0, 0.0),
            (270.0, 0.0),
            (0.0, 89.0),
            (0.0, -89.0),
        ];
        let mut cands = Vec::new();
        for (y, p) in sweep {
            if let Some(c) = candidate(det, render(vol, y, p), true)? {
                cands.push(c);
            }
        }
        n_cands += cands.len();
        accepted = validate(det, vol, &render, cands)?;
    }
    let Some(accepted) = accepted else {
        return Err(if n_cands == 0 {
            format!(
                "no plausible face in any rendered view, including a 6-direction \
                 orientation sweep (yaws={:?}, pitch={}); check the render output visually",
                params.yaws, params.pitch
            )
        } else {
            "face candidates did not survive validation -- the detections were likely \
             false positives; check the render output visually"
                .into()
        });
    };

    let per_view: Vec<Vec<Option<Vec3>>> = accepted.iter().map(|c| c.pts.clone()).collect();
    let pts3d = combine_views(&per_view);
    let frame = match face_frame_from_landmarks(&pts3d) {
        Err(e) if params.partial_face && e.contains("mouth") => {
            crate::landmarks::face_frame_without_mouth(&pts3d)?
        }
        // Mouth landmarks lifted onto the field-of-view cut (the slab's end
        // below the nose) tip the frame forward; the scanner axes are trusted
        // only when no orientation sweep was needed.
        Ok(_) if params.partial_face && !orientation_sweep && on_fov_edge(vol, &pts3d, 48..68) => {
            crate::landmarks::face_frame_without_mouth(&pts3d)?
        }
        r => r?,
    };
    let views: Vec<UsedView> = accepted
        .iter()
        .map(|c| UsedView {
            yaw_deg: c.view.camera.yaw_deg,
            pitch_deg: c.view.camera.pitch_deg,
            landmarks: c.lm.clone(),
            score: c.score,
        })
        .collect();
    Ok(LocatedFace {
        frame,
        landmarks_3d: pts3d,
        views,
        orientation_sweep,
    })
}

/// In-memory pipeline: [`locate_face`] -> anonymize -> verify. Modifies
/// `vol` in place.
///
/// With `verify`, the anonymized volume is re-rendered from the detection
/// angles and the detector run again; blur-family modes escalate sigma
/// x1.6 per round, and an opaque `block` fill is the final guaranteed step
/// (unless `escalate_to_block` is off).
pub fn deface_auto(
    vol: &mut Volume,
    det: &mut dyn FaceDetector,
    params: &AutoParams,
) -> Result<AutoReport, String> {
    let (face, plan, brain, eff) = prepare(vol, det, params)?;
    // After a noisy-scan retry, verification renders at the threshold the
    // face was found at, from the original volume's percentiles; otherwise
    // exactly as requested.
    let retried = eff.threshold_frac != params.threshold_frac;
    let params = &eff;
    let rp = RenderParams {
        size: params.size,
        threshold_frac: params.threshold_frac,
        threshold_abs: if retried {
            Some(render_threshold(vol, params))
        } else {
            params.threshold
        },
        step_mm: 0.0,
    };
    let render = |vol: &Volume, yaw: f64, pitch: f64| render_view(vol, yaw, pitch, &rp);
    let frame = face.frame;
    let track_mask = params.return_mask || brain.is_some();
    let (report, mut mask) = apply(vol, &frame, &plan, params.deface.mode, None);

    let mut out = AutoReport {
        views_detected: face.views.len(),
        orientation: vol.orientation(),
        orientation_sweep: face.orientation_sweep,
        landmarks_3d: face.landmarks_3d,
        views: face.views,
        verify_rounds: None,
        residual_detections: None,
        escalated_to_block: None,
        final_blur_sigma_mm: None,
        mask: None,
        brain: None,
        deface: report,
        frame: frame.clone(),
    };

    let mode = params.deface.mode;
    if params.verify && params.deface.region != Region::Ears {
        let mut sigma = out.deface.resolved.blur_sigma_mm;
        let mut rounds = 0;
        let mut escalated = false;
        let mut full_hull = false;
        // A partial face's mouth landmarks may lie on the field-of-view cut:
        // centre it on brows, eyes and nose.
        let central_end = if params.partial_face { 48 } else { 68 };
        let face_center = mean(&out.landmarks_3d, 17..central_end).unwrap_or(frame.origin);
        let radius = RESIDUAL_RADIUS * frame.eye_dist;
        // Every detected view at its own pitch and at each verification
        // pitch (deduplicated).
        let mut angles: Vec<(f64, f64)> = Vec::new();
        let mut add_angle = |yaw: f64, pitch: f64| {
            let a = (yaw, pitch.clamp(-89.0, 89.0));
            if !angles
                .iter()
                .any(|b| (b.0 - a.0).abs() < 1e-6 && (b.1 - a.1).abs() < 1e-6)
            {
                angles.push(a);
            }
        };
        for v in &out.views {
            for p in std::iter::once(v.pitch_deg).chain(params.verify_pitches.iter().copied()) {
                add_angle(v.yaw_deg, p);
            }
        }
        // The effort grid, centred on the best detection view (on a wrongly
        // stored volume the orientation sweep finds the face from wherever
        // it is, so the scanner's front is no reference).
        if let Some(best) = out.views.iter().max_by(|a, b| a.score.total_cmp(&b.score)) {
            let (yaw0, pitch0) = (best.yaw_deg, best.pitch_deg);
            for (dy, dp) in params.verify_effort.grid() {
                add_angle(yaw0 + dy, pitch0 + dp);
            }
        }
        // A partial face is seen from fewer angles: attack its output from
        // every detection yaw at every partial pitch too.
        if params.partial_face {
            for &y in &params.yaws {
                for p in std::iter::once(0.0).chain(PARTIAL_VERIFY_PITCHES) {
                    add_angle(y, params.pitch + p);
                }
            }
        }
        // The built-in detector also attacks the output whatever detector
        // found the face: a weaker one (dlib) stops seeing a blurred face
        // long before the built-in one does.
        #[cfg(feature = "detector")]
        let mut builtin = params
            .verify_with_builtin
            .then(crate::detect::MediaPipeDetector::default);
        let residual = loop {
            let mut residual = 0;
            for &(yaw, pitch) in &angles {
                let view = render(vol, yaw, pitch);
                let (w, h) = (view.camera.width, view.camera.height);
                #[cfg_attr(not(feature = "detector"), allow(unused_mut))]
                let mut found = vec![det.detect(&view.rgb, w, h, out.orientation_sweep)?];
                #[cfg(feature = "detector")]
                if let Some(b) = builtin.as_mut() {
                    found.push(b.detect(&view.rgb, w, h, out.orientation_sweep)?);
                }
                // Conservative: a detection that cannot be placed in 3D
                // counts as residual.
                let at_face = |d: &Detection| {
                    let pts = unproject(&view, &d.landmarks);
                    // A partial face's slab ends below the nose; seen from
                    // below, its cross-section can read as a "face" that no
                    // anonymization of the skin can remove.
                    if params.partial_face && on_fov_edge(vol, &pts, 0..68) {
                        return false;
                    }
                    mean(&pts, 17..central_end).is_none_or(|c| norm(sub(c, face_center)) <= radius)
                };
                if found.iter().flatten().any(at_face) {
                    residual += 1;
                }
            }
            // A block is final unless it was bounded to the air-sparing
            // shell (then the full hull is still to come).
            if residual == 0 || (mode == DefaceMode::Block && (!plan.spares_air() || full_hull)) {
                break residual;
            }
            // Escalations re-apply the first round's face ROI: the skin and
            // ear evidence cannot be re-measured on an anonymized volume, and
            // re-measuring would push the wall deeper each round.
            let face = plan.face_only();
            if matches!(mode, DefaceMode::Blur | DefaceMode::Adaptive)
                && rounds < params.max_verify_rounds
            {
                rounds += 1;
                sigma *= 1.6;
                let (_, m) = apply(vol, &frame, &face, mode, Some(sigma));
                if track_mask {
                    union_mask(&mut mask, &m);
                }
            } else if params.escalate_to_block && !escalated {
                // Last resort for blur-family and cut: an opaque fill
                // destroys whatever the detector still keys on (heavy blur
                // and cut can expose skull orbits that read as a face).
                rounds += 1;
                escalated = true;
                let (_, m) = apply(vol, &frame, &face, DefaceMode::Block, None);
                if track_mask {
                    union_mask(&mut mask, &m);
                }
            } else if (escalated || mode == DefaceMode::Block) && face.spares_air() && !full_hull {
                // A block bounded to the shell around the tissue can still
                // read as a face: fill the whole hull, as without sparing
                // air.
                rounds += 1;
                full_hull = true;
                let (_, m) = apply(vol, &frame, &face.full_hull(), DefaceMode::Block, None);
                if track_mask {
                    union_mask(&mut mask, &m);
                }
            } else {
                break residual;
            }
        };
        out.verify_rounds = Some(rounds);
        out.residual_detections = Some(residual);
        out.escalated_to_block = Some(escalated);
        out.final_blur_sigma_mm =
            matches!(mode, DefaceMode::Blur | DefaceMode::Adaptive).then_some(sigma);
    }
    if let Some((brain, report)) = brain {
        out.brain = Some(brain_check(vol, &brain, &report, &mask));
    }
    if params.return_mask {
        out.mask = Some(mask);
    }
    Ok(out)
}

fn union_mask(into: &mut Array3<u8>, from: &Array3<u8>) {
    ndarray::Zip::from(into).and(from).for_each(|a, &b| {
        *a |= b;
    });
}

type Brain = Option<(Array3<u8>, crate::deskull::DeskullReport)>;

/// Detection, the safety refusals and the anonymization plan, shared by
/// [`deface_auto`] and [`anonymization_mask`]. The plan is measured once on
/// the original volume; verification escalations re-apply it, so they never
/// reach deeper than the first round.
/// Surface threshold fractions tried, in order, by `auto_threshold` when no
/// face is found at the default one: noisy, dim scans (infants) render
/// their background noise as a solid shell at 0.14 x p99.
const NOISY_THRESHOLD_FRACS: [f32; 2] = [0.3, 0.45];

/// The absolute surface threshold the renderer uses for `params`.
fn render_threshold(vol: &Volume, params: &AutoParams) -> f32 {
    params.threshold.unwrap_or_else(|| {
        if vol.percentile(1.0) < -500.0 {
            -400.0
        } else {
            params.threshold_frac * vol.percentile(99.0)
        }
    })
}

/// [`locate_face`], retried at higher surface thresholds for noisy scans
/// (only when the caller left the threshold on auto). Returns the face and
/// the parameters it was found with.
fn locate_face_any_threshold(
    vol: &Volume,
    det: &mut dyn FaceDetector,
    params: &AutoParams,
) -> Result<(LocatedFace, AutoParams), String> {
    let first = match locate_face(vol, det, params) {
        Ok(face) => return Ok((face, params.clone())),
        Err(e) => e,
    };
    let ct = vol.percentile(1.0) < -500.0;
    let auto =
        params.threshold.is_none() && params.threshold_frac == AutoParams::default().threshold_frac;
    if !params.auto_threshold || ct || !auto {
        return Err(first);
    }
    for frac in NOISY_THRESHOLD_FRACS {
        let p = AutoParams {
            threshold_frac: frac,
            ..params.clone()
        };
        if let Ok(face) = locate_face(vol, det, &p) {
            return Ok((face, p));
        }
    }
    Err(first)
}

fn prepare(
    vol: &Volume,
    det: &mut dyn FaceDetector,
    params: &AutoParams,
) -> Result<(LocatedFace, DefacePlan, Brain, AutoParams), String> {
    let (face, params) = locate_face_any_threshold(vol, det, params)?;
    let params = &params;
    let mut deface = params.deface.clone();
    // The cut follows the skin the detector saw.
    deface.surface_threshold = Some(render_threshold(vol, params));
    deface.level_face |= params.partial_face;
    let plan = plan(vol, &face.frame, &deface);
    if plan.face_seen_through_air() == Some(false) {
        return Err("the detected face is not seen through air: it lies on a \
                    field-of-view cut, not on skin; refusing to modify the volume"
            .into());
    }
    // The guard and the check need the brain of the *original* volume.
    let brain = if params.brain_check || params.brain_guard {
        #[cfg(feature = "synthstrip")]
        {
            let params = crate::deskull::DeskullParams {
                method: crate::deskull::Method::SynthStrip,
                ..Default::default()
            };
            Some(crate::deskull::brain_mask(vol, &params)?)
        }
        #[cfg(not(feature = "synthstrip"))]
        return Err("brain_check and brain_guard need the `synthstrip` feature".into());
    } else {
        None
    };
    let plan = match &brain {
        Some((b, _)) if params.brain_guard => {
            plan.with_protected(dilate_mm(b, vol.spacing, params.brain_guard_mm))
        }
        _ => plan,
    };
    let brain = if params.brain_check { brain } else { None };
    Ok((face, plan, brain, params.clone()))
}

/// `mask` grown by `mm` in every direction (a box of the per-axis voxel
/// radius, so at least `mm` everywhere: conservative for a guard).
fn dilate_mm(mask: &Array3<u8>, spacing: [f64; 3], mm: f64) -> Array3<u8> {
    let mut out = mask.clone();
    for (axis, &sp) in spacing.iter().enumerate() {
        let r = (mm / sp.max(1e-6)).ceil() as usize;
        if r == 0 {
            continue;
        }
        let src = out.clone();
        // Separable running max along `axis` (dilation only adds voxels, so
        // `out`, which starts as `src`, just gains the ones within `r`).
        ndarray::Zip::from(out.lanes_mut(ndarray::Axis(axis)))
            .and(src.lanes(ndarray::Axis(axis)))
            .for_each(|mut dst, s| {
                let n = s.len();
                let mut last_on: Option<usize> = None;
                for i in 0..n {
                    if s[i] != 0 {
                        last_on = Some(i);
                    }
                    if last_on.is_some_and(|j| i - j <= r) {
                        dst[i] = 1;
                    }
                }
                let mut next_on: Option<usize> = None;
                for i in (0..n).rev() {
                    if s[i] != 0 {
                        next_on = Some(i);
                    }
                    if next_on.is_some_and(|j| j - i <= r) {
                        dst[i] = 1;
                    }
                }
            });
    }
    out
}

/// Escape hatch: the region [`deface_auto`] would anonymize, as a mask
/// (1 = voxel it would rewrite), without modifying the volume. Runs the same
/// detection, validation, safety refusals and (with `brain_check`) brain
/// check, then stops: nothing is rewritten and nothing is *verified*, so the
/// mask says where RadVeil would anonymize, not that anonymizing there
/// defeats re-identification. Callers who process the region themselves own
/// that check. The mask is returned in `AutoReport::mask`; the verification
/// fields stay `None`.
///
/// Verification escalations in [`deface_auto`] re-apply the same region with
/// stronger modes, so for the face this mask is also the final one.
pub fn anonymization_mask(
    vol: &Volume,
    det: &mut dyn FaceDetector,
    params: &AutoParams,
) -> Result<AutoReport, String> {
    let (face, plan, brain, _) = prepare(vol, det, params)?;
    let (report, mask) = plan_mask(vol, &face.frame, &plan, params.deface.mode);
    let brain = brain.map(|(b, r)| brain_check(vol, &b, &r, &mask));
    Ok(AutoReport {
        views_detected: face.views.len(),
        orientation: vol.orientation(),
        orientation_sweep: face.orientation_sweep,
        landmarks_3d: face.landmarks_3d,
        views: face.views,
        verify_rounds: None,
        residual_detections: None,
        escalated_to_block: None,
        final_blur_sigma_mm: None,
        mask: Some(mask),
        brain,
        deface: report,
        frame: face.frame,
    })
}

/// Overlap between the anonymization mask and the extracted brain, plus
/// the distance from the brain to the nearest modified voxel.
fn brain_check(
    vol: &Volume,
    brain: &Array3<u8>,
    report: &crate::deskull::DeskullReport,
    mask: &Array3<u8>,
) -> BrainCheck {
    let modified = ndarray::Zip::from(brain)
        .and(mask)
        .fold(0usize, |n, &b, &m| n + usize::from(b != 0 && m != 0));
    // Distance from every voxel to the brain (chamfer, mm), then the
    // smallest value over the modified voxels: how close the
    // anonymization came to the brain.
    let margin_mm = (modified == 0).then(|| {
        let [nx, ny, nz] = vol.shape();
        let inside: Vec<bool> = brain.iter().map(|&b| b != 0).collect();
        let dist = crate::deskull::distance_to(&inside, [nx, ny, nz], vol.spacing, true);
        mask.iter()
            .zip(&dist)
            .filter(|(&m, _)| m != 0)
            .map(|(_, &d)| d as f64)
            .fold(f64::INFINITY, f64::min)
    });
    let margin_mm = margin_mm.filter(|d| d.is_finite());
    BrainCheck {
        voxels_modified: modified,
        brain_volume_ml: report.brain_volume_ml,
        margin_mm,
    }
}
