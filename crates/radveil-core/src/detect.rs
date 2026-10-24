// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Built-in face detector: MediaPipe BlazeFace (short range) + Face Mesh v2,
//! run by RadVeil's own small inference engine ([`crate::nn`]): pure Rust,
//! no MediaPipe framework, no Bazel, no ML runtime, no native libraries.
//!
//! The two networks are Google's Apache-2.0 MediaPipe Face Landmarker
//! models, exported once offline from their TFLite files to `.rvnn` (see
//! `models/README.md`) and embedded in the binary. Everything MediaPipe's
//! graph does around them is
//! reimplemented here: SSD anchor decoding + weighted NMS for the detector,
//! the eye-aligned 1.5x ROI crop for the landmark model, and the mapping of
//! the 478-point mesh onto the 68-point indexing the rest of radveil uses.

use crate::nn::{Net, Workspace};
use std::cell::RefCell;
use std::sync::OnceLock;

const DETECTOR_NET: &[u8] = include_bytes!("../models/blazeface_short_range.rvnn");
const MESH_NET: &[u8] = include_bytes!("../models/face_mesh_v2.rvnn");

const DET_SIZE: usize = 128;
const MESH_SIZE: usize = 256;
const NMS_IOU: f64 = 0.3;

/// Face Mesh v2 vertex indices corresponding to the 68-point landmarks
/// (jaw 0-16, brows 17-26, nose 27-35, eyes 36-47, mouth 48-67), in the
/// same image-side convention (index 36 = image-left eye outer corner).
pub const MESH_TO_68: [usize; 68] = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389, // jaw
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301, // brows
    168, 197, 5, 4, 75, 97, 2, 326, 305, // nose
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, // eyes
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14,
    87, // mouth
];

/// One BlazeFace detection in image pixel coordinates.
#[derive(Clone, Debug)]
pub struct FaceBox {
    pub score: f64,
    /// Box center and side length (the longer box side), in continuous
    /// image coordinates (pixel i spans [i, i+1)).
    pub center: [f64; 2],
    pub size: f64,
    /// right eye, left eye, nose, mouth, right ear, left ear (image px).
    pub keypoints: [[f64; 2]; 6],
}

/// Dense landmarks for one face.
#[derive(Clone, Debug)]
pub struct FaceMesh {
    /// Detection that seeded the mesh.
    pub detection: FaceBox,
    /// 478 points: image x, y (pixel-index coordinates, pixel centers at
    /// integers) and relative depth z (px scale).
    pub points: Vec<[f64; 3]>,
    /// Face-presence score of the landmark model (0..1).
    pub presence: f64,
}

impl FaceMesh {
    /// The 68 landmarks (image x, y), dlib-68 index order.
    pub fn landmarks68(&self) -> Vec<[f64; 2]> {
        MESH_TO_68
            .iter()
            .map(|&i| [self.points[i][0], self.points[i][1]])
            .collect()
    }
}

/// Thresholds for [`MediaPipeDetector`].
#[derive(Clone, Copy, Debug)]
pub struct DetectorParams {
    /// BlazeFace score threshold (MediaPipe default 0.5).
    pub min_score: f64,
    /// Face Mesh presence threshold (MediaPipe default 0.5).
    pub min_presence: f64,
}

impl Default for DetectorParams {
    fn default() -> Self {
        DetectorParams {
            min_score: 0.5,
            min_presence: 0.5,
        }
    }
}

struct Models {
    detector: Net,
    mesh: Net,
    anchors: Vec<[f64; 2]>,
}

/// BlazeFace short-range SSD anchors: stride 8 (2 per cell) then stride 16
/// (6 per cell) over a 128x128 input — 896 anchors, fixed size 1.
fn anchors() -> Vec<[f64; 2]> {
    let mut a = Vec::with_capacity(896);
    for (stride, per_cell) in [(8usize, 2usize), (16, 6)] {
        let g = DET_SIZE / stride;
        for y in 0..g {
            for x in 0..g {
                for _ in 0..per_cell {
                    a.push([(x as f64 + 0.5) / g as f64, (y as f64 + 0.5) / g as f64]);
                }
            }
        }
    }
    a
}

fn models() -> Result<&'static Models, String> {
    static MODELS: OnceLock<Result<Models, String>> = OnceLock::new();
    MODELS
        .get_or_init(|| {
            let net = |bytes: &[u8], what: &str| -> Result<Net, String> {
                Net::from_bytes(bytes).map_err(|e| format!("loading {what}: {e}"))
            };
            let models = Models {
                detector: net(DETECTOR_NET, "face detector")?,
                mesh: net(MESH_NET, "face mesh")?,
                anchors: anchors(),
            };
            if models.detector.input_shape() != [DET_SIZE, DET_SIZE, 3]
                || models.mesh.input_shape() != [MESH_SIZE, MESH_SIZE, 3]
            {
                return Err("embedded face models have unexpected input shapes".into());
            }
            Ok(models)
        })
        .as_ref()
        .map_err(|e| e.clone())
}

#[derive(Clone, Copy)]
enum Which {
    Detector,
    Mesh,
}

/// Run one of the embedded networks with this thread's activation
/// buffers (allocated on first use, then reused).
fn run_net(which: Which, input: &[f32]) -> Result<Vec<Vec<f32>>, String> {
    thread_local! {
        static WS: RefCell<Option<(Workspace, Workspace)>> = const { RefCell::new(None) };
    }
    let m = models()?;
    Ok(WS.with(|ws| {
        let mut ws = ws.borrow_mut();
        let (det_ws, mesh_ws) =
            ws.get_or_insert_with(|| (m.detector.workspace(), m.mesh.workspace()));
        match which {
            Which::Detector => m.detector.run(det_ws, input),
            Which::Mesh => m.mesh.run(mesh_ws, input),
        }
    }))
}

/// 2x3 affine map from output-crop pixel coordinates to image pixel
/// coordinates: `img = m * [u, v, 1]`.
#[derive(Clone, Copy, Debug)]
struct Affine2 {
    m: [[f64; 3]; 2],
}

impl Affine2 {
    /// Crop of `out` pixels centered at `center`, covering `size` image
    /// pixels, rotated by `angle` (radians, crop x-axis -> image direction
    /// (cos, sin)).
    fn crop(center: [f64; 2], size: f64, angle: f64, out: usize) -> Self {
        let k = size / out as f64;
        let (s, c) = angle.sin_cos();
        let h = out as f64 / 2.0;
        let a = [[c * k, -s * k], [s * k, c * k]];
        let t = [
            center[0] - (a[0][0] * h + a[0][1] * h),
            center[1] - (a[1][0] * h + a[1][1] * h),
        ];
        Affine2 {
            m: [[a[0][0], a[0][1], t[0]], [a[1][0], a[1][1], t[1]]],
        }
    }

    fn apply(&self, u: f64, v: f64) -> [f64; 2] {
        [
            self.m[0][0] * u + self.m[0][1] * v + self.m[0][2],
            self.m[1][0] * u + self.m[1][1] * v + self.m[1][2],
        ]
    }

    /// Image pixels per crop pixel.
    fn scale(&self) -> f64 {
        (self.m[0][0].powi(2) + self.m[1][0].powi(2)).sqrt()
    }
}

/// Resample an RGB8 image through `map` into an `out x out` NHWC float
/// image, value = pixel * `mul` + `add`. Bilinear, with supersampling when
/// shrinking so downscaled renders are area-averaged rather than aliased.
/// Samples outside the image read as black.
fn warp(rgb: &[u8], w: usize, h: usize, map: &Affine2, out: usize, mul: f32, add: f32) -> Vec<f32> {
    let ss = map.scale().ceil().clamp(1.0, 8.0) as usize;
    let inv = 1.0 / (ss * ss) as f32;
    let fetch = |x: i64, y: i64, c: usize| -> f32 {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            0.0
        } else {
            rgb[(y as usize * w + x as usize) * 3 + c] as f32
        }
    };
    let mut data = vec![0f32; out * out * 3];
    for v in 0..out {
        for u in 0..out {
            let mut acc = [0f32; 3];
            for sy in 0..ss {
                for sx in 0..ss {
                    let fu = u as f64 + (sx as f64 + 0.5) / ss as f64;
                    let fv = v as f64 + (sy as f64 + 0.5) / ss as f64;
                    // pixel centers: image pixel i covers [i, i+1)
                    let [x, y] = map.apply(fu, fv);
                    let (x, y) = (x - 0.5, y - 0.5);
                    let (x0, y0) = (x.floor(), y.floor());
                    let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
                    let (x0, y0) = (x0 as i64, y0 as i64);
                    for (c, a) in acc.iter_mut().enumerate() {
                        let top = fetch(x0, y0, c) * (1.0 - fx) + fetch(x0 + 1, y0, c) * fx;
                        let bot = fetch(x0, y0 + 1, c) * (1.0 - fx) + fetch(x0 + 1, y0 + 1, c) * fx;
                        *a += top * (1.0 - fy) + bot * fy;
                    }
                }
            }
            let o = (v * out + u) * 3;
            for c in 0..3 {
                data[o + c] = acc[c] * inv * mul + add;
            }
        }
    }
    data
}

fn sigmoid(x: f32) -> f64 {
    1.0 / (1.0 + (-(x.clamp(-100.0, 100.0) as f64)).exp())
}

fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let area = |r: &[f64; 4]| (r[2] - r[0]) * (r[3] - r[1]);
    inter / (area(a) + area(b) - inter).max(1e-12)
}

/// Framework-free MediaPipe face pipeline. Cheap to construct; the models
/// are parsed once per process and shared.
#[derive(Clone, Copy, Debug, Default)]
pub struct MediaPipeDetector {
    pub params: DetectorParams,
}

impl MediaPipeDetector {
    pub fn new(params: DetectorParams) -> Self {
        MediaPipeDetector { params }
    }

    /// Parse the embedded models now (otherwise done on first use).
    pub fn warm_up() -> Result<(), String> {
        models().map(|_| ())
    }

    /// Run BlazeFace on an RGB8 image rotated by `quarter_turns` x 90°
    /// counter-clockwise (the detector is roll-sensitive). Returned boxes
    /// and keypoints are in the *original* image frame.
    pub fn detect_faces(
        &self,
        rgb: &[u8],
        w: usize,
        h: usize,
        quarter_turns: u32,
    ) -> Result<Vec<FaceBox>, String> {
        let m = models()?;
        // Letterbox the image into a square canvas, rotated so the detector
        // sees an upright face; `map` takes detector pixels to image pixels.
        let side = w.max(h) as f64;
        let angle = (quarter_turns % 4) as f64 * std::f64::consts::FRAC_PI_2;
        let map = Affine2::crop([w as f64 / 2.0, h as f64 / 2.0], side, angle, DET_SIZE);
        let input = warp(rgb, w, h, &map, DET_SIZE, 1.0 / 127.5, -1.0);
        let out = run_net(Which::Detector, &input)?;
        let (reg, cls) = (&out[0], &out[1]);
        if reg.len() != 896 * 16 || cls.len() != 896 {
            return Err("face detector output size".into());
        }

        let det = DET_SIZE as f64;
        let mut cand: Vec<(f64, [f64; 4], [[f64; 2]; 6])> = Vec::new();
        for (i, a) in m.anchors.iter().enumerate() {
            let score = sigmoid(cls[i]);
            if score < self.params.min_score {
                continue;
            }
            let r = &reg[i * 16..i * 16 + 16];
            let cx = r[0] as f64 / det + a[0];
            let cy = r[1] as f64 / det + a[1];
            let bw = r[2] as f64 / det;
            let bh = r[3] as f64 / det;
            let mut kps = [[0.0; 2]; 6];
            for (k, kp) in kps.iter_mut().enumerate() {
                *kp = [
                    r[4 + 2 * k] as f64 / det + a[0],
                    r[5 + 2 * k] as f64 / det + a[1],
                ];
            }
            cand.push((
                score,
                [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
                kps,
            ));
        }
        cand.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Weighted NMS (MediaPipe's default for BlazeFace): overlapping
        // candidates are averaged by score, the top score is kept.
        let mut faces = Vec::new();
        let mut used = vec![false; cand.len()];
        for i in 0..cand.len() {
            if used[i] {
                continue;
            }
            let mut wsum = 0.0;
            let mut bx = [0.0; 4];
            let mut kp = [[0.0; 2]; 6];
            for j in i..cand.len() {
                if used[j] || iou(&cand[i].1, &cand[j].1) <= NMS_IOU {
                    continue;
                }
                used[j] = true;
                let s = cand[j].0;
                wsum += s;
                for (acc, v) in bx.iter_mut().zip(cand[j].1) {
                    *acc += s * v;
                }
                for (acc, v) in kp.iter_mut().zip(cand[j].2) {
                    acc[0] += s * v[0];
                    acc[1] += s * v[1];
                }
            }
            let to_img = |p: [f64; 2]| map.apply(p[0] * det, p[1] * det);
            let center = to_img([
                (bx[0] + bx[2]) / (2.0 * wsum),
                (bx[1] + bx[3]) / (2.0 * wsum),
            ]);
            let size = ((bx[2] - bx[0]).max(bx[3] - bx[1]) / wsum) * det * map.scale();
            let mut keypoints = [[0.0; 2]; 6];
            for k in 0..6 {
                keypoints[k] = to_img([kp[k][0] / wsum, kp[k][1] / wsum]);
            }
            faces.push(FaceBox {
                score: cand[i].0,
                center,
                size,
                keypoints,
            });
        }
        Ok(faces)
    }

    /// Run Face Mesh v2 on the eye-aligned crop around a detection.
    pub fn mesh(&self, rgb: &[u8], w: usize, h: usize, face: &FaceBox) -> Result<FaceMesh, String> {
        let [re, le] = [face.keypoints[0], face.keypoints[1]];
        // Crop rotation that levels the eyes (MediaPipe: target angle 0).
        let angle = (le[1] - re[1]).atan2(le[0] - re[0]);
        let map = Affine2::crop(face.center, face.size * 1.5, angle, MESH_SIZE);
        let input = warp(rgb, w, h, &map, MESH_SIZE, 1.0 / 255.0, 0.0);
        let out = run_net(Which::Mesh, &input)?;
        let (lm, pres) = (&out[0], &out[1]);
        if lm.len() != 478 * 3 || pres.is_empty() {
            return Err("face mesh output size".into());
        }
        let presence = sigmoid(pres[0]);
        let k = map.scale();
        let points = lm
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| {
                // continuous crop coords -> pixel-index coords (center of
                // pixel i is i, the convention of RenderView::world_at)
                let [x, y] = map.apply(p[0] as f64, p[1] as f64);
                [x - 0.5, y - 0.5, p[2] as f64 * k]
            })
            .collect();
        Ok(FaceMesh {
            detection: face.clone(),
            points,
            presence,
        })
    }

    /// Detect the most confident face and return its mesh, or `None`.
    /// With `try_rotations`, the detector also looks at the image rotated
    /// by 90/180/270° (for volumes whose stored orientation is wrong) and
    /// the most confident detection over all rotations wins; the mesh crop
    /// is aligned to the detected eyes, so any roll is handled.
    pub fn detect(
        &self,
        rgb: &[u8],
        w: usize,
        h: usize,
        try_rotations: bool,
    ) -> Result<Option<FaceMesh>, String> {
        let mut best: Option<FaceBox> = None;
        for turns in 0..if try_rotations { 4 } else { 1 } {
            for f in self.detect_faces(rgb, w, h, turns)? {
                if best.as_ref().is_none_or(|b| f.score > b.score) {
                    best = Some(f);
                }
            }
        }
        let Some(best) = best else {
            return Ok(None);
        };
        let mesh = self.mesh(rgb, w, h, &best)?;
        Ok((mesh.presence >= self.params.min_presence).then_some(mesh))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_count() {
        assert_eq!(anchors().len(), 896);
    }

    #[test]
    fn crop_affine_roundtrip() {
        let m = Affine2::crop([100.0, 50.0], 64.0, 0.3, 256);
        let c = m.apply(128.0, 128.0);
        assert!((c[0] - 100.0).abs() < 1e-9 && (c[1] - 50.0).abs() < 1e-9);
        assert!((m.scale() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn models_load_and_blank_image_has_no_face() {
        let rgb = vec![10u8; 64 * 64 * 3];
        let d = MediaPipeDetector::default();
        assert!(d.detect(&rgb, 64, 64, true).unwrap().is_none());
    }
}
