// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Brain extraction ("deskulling"): a binary brain mask and, optionally,
//! the volume with every non-brain voxel replaced by background.
//!
//! Extraction goes through the [`BrainExtractor`] trait so learned models
//! can plug in beside the built-in classical method ([`ClassicalExtractor`]):
//!
//! * **MRI** — a deformable-surface model after Smith, "Fast robust
//!   automated brain extraction" (Hum. Brain Mapp. 2002), implemented from
//!   the paper: a tessellated sphere starts inside the head and each vertex
//!   moves under three forces (tangential regularisation, curvature-
//!   dependent smoothing, and an intensity term that pushes outwards while
//!   the local minimum/maximum profile along the inward normal still looks
//!   like brain). The surface is then filled into a voxel mask.
//! * **CT** — bone enclosure: soft tissue (brain/CSF HU range) that has
//!   bone in at least five of the six voxel-axis directions lies inside the
//!   skull. It is eroded to cut thin bridges (foramina, orbits), the
//!   largest component kept, dilated back inside the candidate, and holes
//!   filled.
//!
//! All geometry works in voxel-index × spacing millimetres, so any voxel
//! orientation (RAS, LPS, oblique...) is handled without resampling.

use crate::landmarks::FaceFrame;
use crate::volume::Volume;
use ndarray::Array3;
use rayon::prelude::*;
use serde::Serialize;

/// Image modality; decides which classical algorithm runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// CT when the 1st intensity percentile is below -500 (Hounsfield
    /// air), MRI otherwise.
    Auto,
    Mri,
    Ct,
}

impl Modality {
    pub fn parse(s: &str) -> Option<Modality> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Modality::Auto),
            "mri" | "mr" => Some(Modality::Mri),
            "ct" => Some(Modality::Ct),
            _ => None,
        }
    }

    /// Resolve `Auto` against the data.
    pub fn resolve(self, volume: &Volume) -> Modality {
        match self {
            Modality::Auto if volume.percentile(1.0) < -500.0 => Modality::Ct,
            Modality::Auto => Modality::Mri,
            m => m,
        }
    }
}

/// Brain-extraction method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// The SynthStrip network (feature `synthstrip`, the default): any MRI
    /// contrast, CT and PET; mask includes CSF.
    SynthStrip,
    /// RadVeil's own small network (feature `rvbrain`): a few times faster
    /// than SynthStrip on a coarser grid. Needs a `.rvb` model in
    /// [`DeskullParams::model`] unless one is compiled in.
    RvBrain,
    /// Model-free classical extractor (see the module docs).
    Classical,
}

impl Method {
    pub fn parse(s: &str) -> Option<Method> {
        match s.to_ascii_lowercase().as_str() {
            "synthstrip" => Some(Method::SynthStrip),
            "rvbrain" => Some(Method::RvBrain),
            "classical" => Some(Method::Classical),
            _ => None,
        }
    }
}

impl Default for Method {
    fn default() -> Self {
        if cfg!(feature = "synthstrip") {
            Method::SynthStrip
        } else {
            Method::Classical
        }
    }
}

/// Parameters for [`deskull`] / [`brain_mask`].
#[derive(Clone, Debug)]
pub struct DeskullParams {
    pub method: Method,
    pub modality: Modality,
    /// MRI: fractional intensity threshold (0..1). Smaller values give a
    /// larger brain estimate. BET's `-f`.
    pub frac: f64,
    /// MRI: surface-evolution iterations.
    pub iterations: usize,
    /// MRI: icosphere subdivision level (4 = 2562 vertices).
    pub subdivisions: u32,
    /// MRI: re-centre on the extracted brain and re-run until the centre
    /// moves less than 1 mm (at most 3 passes). Helps when the neck or
    /// shoulders pull the initial centre of gravity down.
    pub robust: bool,
    /// CT: soft-tissue Hounsfield window considered brain/CSF.
    pub ct_window: (f32, f32),
    /// CT: HU at and above which a voxel is bone.
    pub ct_bone_hu: f32,
    /// CT: erosion radius (mm) that separates the brain from extracranial
    /// soft tissue before the component is chosen.
    pub ct_erode_mm: f64,
    /// Value written outside the brain by [`strip`]; `None` = 0 for MRI,
    /// the air value (1st percentile, at most 0) for CT.
    pub fill: Option<f32>,
    /// Located face (e.g. from [`crate::locate_face`]). When present, MRI
    /// extraction starts from the brain centre implied by the face and the
    /// surface may not enter the measured non-brain region below the brain
    /// floor (orbits, nasal cavity, skull base, neck); the mask is clipped
    /// there too.
    pub face: Option<FaceFrame>,
    /// Settings for [`Method::SynthStrip`].
    #[cfg(feature = "synthstrip")]
    pub synthstrip: crate::synthstrip::SynthStripParams,
    /// Settings for [`Method::RvBrain`].
    #[cfg(feature = "rvbrain")]
    pub rvbrain: crate::rvbrain::RvBrainParams,
    /// Model file for [`Method::RvBrain`] (`.rvb`). `None` uses the model
    /// compiled into this build, if there is one.
    pub model: Option<std::path::PathBuf>,
}

impl Default for DeskullParams {
    fn default() -> Self {
        DeskullParams {
            method: Method::default(),
            modality: Modality::Auto,
            frac: 0.5,
            iterations: 1000,
            subdivisions: 4,
            robust: true,
            ct_window: (-15.0, 100.0),
            ct_bone_hu: 250.0,
            ct_erode_mm: 3.0,
            fill: None,
            face: None,
            #[cfg(feature = "synthstrip")]
            synthstrip: Default::default(),
            #[cfg(feature = "rvbrain")]
            rvbrain: Default::default(),
            model: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DeskullReport {
    pub method: String,
    pub modality: Modality,
    /// Brain volume in millilitres.
    pub brain_volume_ml: f64,
    pub brain_voxels: usize,
    /// Mask centroid in world RAS+ mm.
    pub centroid_world: [f64; 3],
    /// Value written outside the brain (when a stripped volume was made).
    pub fill: Option<f32>,
    /// MRI surface passes run (1 unless `robust` re-centred).
    pub passes: usize,
    /// Whether the face prior was used.
    pub face_prior: bool,
}

/// A brain-extraction backend. Returns a binary mask (1 = brain) on the
/// volume's voxel grid.
pub trait BrainExtractor {
    fn name(&self) -> &str;
    fn extract(&self, volume: &Volume, modality: Modality) -> Result<Array3<u8>, String>;
}

/// Built-in classical extractor (no model; see the module docs).
pub struct ClassicalExtractor {
    pub params: DeskullParams,
}

impl BrainExtractor for ClassicalExtractor {
    fn name(&self) -> &str {
        "classical"
    }

    fn extract(&self, volume: &Volume, modality: Modality) -> Result<Array3<u8>, String> {
        Ok(match modality.resolve(volume) {
            Modality::Ct => ct_mask(volume, &self.params),
            _ => mri_mask(volume, &self.params).0,
        })
    }
}

/// Compute the brain mask with the method selected in `params`.
pub fn brain_mask(
    volume: &Volume,
    params: &DeskullParams,
) -> Result<(Array3<u8>, DeskullReport), String> {
    method_mask(volume, params)
}

fn method_mask(
    volume: &Volume,
    params: &DeskullParams,
) -> Result<(Array3<u8>, DeskullReport), String> {
    match params.method {
        #[cfg(feature = "synthstrip")]
        Method::SynthStrip => brain_mask_with(
            volume,
            &crate::synthstrip::SynthStrip {
                params: params.synthstrip.clone(),
            },
            params.modality,
        ),
        #[cfg(not(feature = "synthstrip"))]
        Method::SynthStrip => Err("built without the `synthstrip` feature".into()),
        #[cfg(feature = "rvbrain")]
        Method::RvBrain => {
            let mut net = match &params.model {
                Some(path) => crate::rvbrain::RvBrain::load(path)?,
                None => crate::rvbrain::RvBrain::embedded()?,
            };
            net.params = params.rvbrain.clone();
            brain_mask_with(volume, &net, params.modality)
        }
        #[cfg(not(feature = "rvbrain"))]
        Method::RvBrain => Err("built without the `rvbrain` feature".into()),
        Method::Classical => Ok(classical_mask(volume, params)),
    }
}

fn classical_mask(volume: &Volume, params: &DeskullParams) -> (Array3<u8>, DeskullReport) {
    let modality = params.modality.resolve(volume);
    let (mask, passes, face_prior) = match modality {
        Modality::Ct => (ct_mask(volume, params), 0, false),
        _ => mri_mask(volume, params),
    };
    let mut report = report_for(volume, &mask, "classical", modality, None, passes);
    report.face_prior = face_prior;
    (mask, report)
}

/// Run any extractor and build the report.
pub fn brain_mask_with(
    volume: &Volume,
    extractor: &dyn BrainExtractor,
    modality: Modality,
) -> Result<(Array3<u8>, DeskullReport), String> {
    let modality = modality.resolve(volume);
    let mask = extractor.extract(volume, modality)?;
    if mask.shape() != volume.data.shape() {
        return Err(format!(
            "extractor '{}' returned mask shape {:?}, volume is {:?}",
            extractor.name(),
            mask.shape(),
            volume.data.shape()
        ));
    }
    let report = report_for(volume, &mask, extractor.name(), modality, None, 1);
    Ok((mask, report))
}

/// Default background for [`strip`].
pub fn default_fill(volume: &Volume, modality: Modality) -> f32 {
    match modality.resolve(volume) {
        Modality::Ct => volume.percentile(1.0).min(0.0),
        _ => 0.0,
    }
}

/// Replace every voxel outside `mask` with `fill`, in place.
pub fn strip(volume: &mut Volume, mask: &Array3<u8>, fill: f32) {
    ndarray::Zip::from(&mut volume.data)
        .and(mask)
        .par_for_each(|v, &m| {
            if m == 0 {
                *v = fill;
            }
        });
    volume.accel.invalidate();
}

/// Extract the brain and strip the volume in place. Returns the mask.
pub fn deskull(
    volume: &mut Volume,
    params: &DeskullParams,
) -> Result<(Array3<u8>, DeskullReport), String> {
    let (mask, mut report) = brain_mask(volume, params)?;
    let fill = params
        .fill
        .unwrap_or_else(|| default_fill(volume, report.modality));
    strip(volume, &mask, fill);
    report.fill = Some(fill);
    Ok((mask, report))
}

fn report_for(
    volume: &Volume,
    mask: &Array3<u8>,
    method: &str,
    modality: Modality,
    fill: Option<f32>,
    passes: usize,
) -> DeskullReport {
    let mut n = 0usize;
    let mut c = [0.0f64; 3];
    for ((i, j, k), &m) in mask.indexed_iter() {
        if m != 0 {
            n += 1;
            c[0] += i as f64;
            c[1] += j as f64;
            c[2] += k as f64;
        }
    }
    let centroid = if n > 0 {
        crate::affine::apply(
            &volume.affine,
            [c[0] / n as f64, c[1] / n as f64, c[2] / n as f64],
        )
    } else {
        [f64::NAN; 3]
    };
    let voxel_ml = volume.spacing.iter().product::<f64>() / 1000.0;
    DeskullReport {
        method: method.to_string(),
        modality,
        brain_volume_ml: n as f64 * voxel_ml,
        brain_voxels: n,
        centroid_world: centroid,
        fill,
        passes,
        face_prior: false,
    }
}

// ---------------------------------------------------------------------------
// MRI: deformable surface
// ---------------------------------------------------------------------------

type V3 = [f64; 3];

#[inline]
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

struct Mesh {
    verts: Vec<V3>,
    faces: Vec<[u32; 3]>,
    /// Vertex -> adjacent vertices.
    nbrs: Vec<Vec<u32>>,
    /// Vertex -> incident faces.
    vfaces: Vec<Vec<u32>>,
}

/// Unit icosphere with outward-oriented faces.
fn icosphere(level: u32) -> Mesh {
    let t = (1.0 + 5f64.sqrt()) / 2.0;
    let mut verts: Vec<V3> = vec![
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    for v in verts.iter_mut() {
        *v = scale(*v, 1.0 / norm(*v));
    }
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..level {
        let mut cache = std::collections::HashMap::<(u32, u32), u32>::new();
        let mut mid = |a: u32, b: u32, verts: &mut Vec<V3>| -> u32 {
            let key = (a.min(b), a.max(b));
            *cache.entry(key).or_insert_with(|| {
                let m = scale(add(verts[a as usize], verts[b as usize]), 0.5);
                verts.push(scale(m, 1.0 / norm(m)));
                (verts.len() - 1) as u32
            })
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for &[a, b, c] in &faces {
            let ab = mid(a, b, &mut verts);
            let bc = mid(b, c, &mut verts);
            let ca = mid(c, a, &mut verts);
            next.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = next;
    }
    // Orient outward.
    for f in faces.iter_mut() {
        let [a, b, c] = f.map(|i| verts[i as usize]);
        if dot(cross(sub(b, a), sub(c, a)), a) < 0.0 {
            f.swap(1, 2);
        }
    }
    let n = verts.len();
    let mut nbrs = vec![Vec::new(); n];
    let mut vfaces = vec![Vec::new(); n];
    for (fi, &[a, b, c]) in faces.iter().enumerate() {
        for (x, y) in [(a, b), (b, c), (c, a)] {
            if !nbrs[x as usize].contains(&y) {
                nbrs[x as usize].push(y);
            }
            if !nbrs[y as usize].contains(&x) {
                nbrs[y as usize].push(x);
            }
        }
        for v in [a, b, c] {
            vfaces[v as usize].push(fi as u32);
        }
    }
    Mesh {
        verts,
        faces,
        nbrs,
        vfaces,
    }
}

/// Global intensity statistics BET derives before evolving the surface.
struct HeadStats {
    t2: f64,
    t: f64,
    tm: f64,
    cog: V3,
    radius: f64,
}

/// Trilinear sample at a point in voxel-mm space.
#[inline]
fn sample_mm(vol: &Volume, p: V3) -> f64 {
    let s = vol.spacing;
    vol.sample_voxel([p[0] / s[0], p[1] / s[1], p[2] / s[2]]) as f64
}

fn head_stats(vol: &Volume) -> HeadStats {
    let t2 = vol.percentile(2.0) as f64;
    let t98 = vol.percentile(98.0) as f64;
    let t = t2 + 0.1 * (t98 - t2);
    let s = vol.spacing;
    let (mut w, mut c, mut count) = (0.0f64, [0.0f64; 3], 0usize);
    for ((i, j, k), &v) in vol.data.indexed_iter() {
        let v = v as f64;
        if v > t {
            let wv = v.min(t98) - t2;
            w += wv;
            c = add(
                c,
                scale([i as f64 * s[0], j as f64 * s[1], k as f64 * s[2]], wv),
            );
            count += 1;
        }
    }
    let cog = if w > 0.0 {
        scale(c, 1.0 / w)
    } else {
        let [nx, ny, nz] = vol.shape();
        [
            nx as f64 * s[0] / 2.0,
            ny as f64 * s[1] / 2.0,
            nz as f64 * s[2] / 2.0,
        ]
    };
    let voxvol = s[0] * s[1] * s[2];
    let radius = (3.0 * count as f64 * voxvol / (4.0 * std::f64::consts::PI)).cbrt();
    let tm = median_in_sphere(vol, cog, radius, t2, t98);
    HeadStats {
        t2,
        t,
        tm,
        cog,
        radius,
    }
}

fn median_in_sphere(vol: &Volume, c: V3, r: f64, lo: f64, hi: f64) -> f64 {
    let s = vol.spacing;
    let [nx, ny, nz] = vol.shape();
    let mut vals: Vec<f32> = Vec::new();
    let rng = |ci: f64, sp: f64, n: usize| {
        let a = ((ci - r) / sp).floor().max(0.0) as usize;
        let b = (((ci + r) / sp).ceil() as usize).min(n.saturating_sub(1));
        a..=b
    };
    for i in rng(c[0], s[0], nx) {
        for j in rng(c[1], s[1], ny) {
            for k in rng(c[2], s[2], nz) {
                let p = [i as f64 * s[0], j as f64 * s[1], k as f64 * s[2]];
                if norm(sub(p, c)) <= r {
                    let v = vol.data[[i, j, k]];
                    if (v as f64) > lo && (v as f64) < hi {
                        vals.push(v);
                    }
                }
            }
        }
    }
    if vals.is_empty() {
        return (lo + hi) / 2.0;
    }
    let mid = vals.len() / 2;
    let (_, m, _) = vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    *m as f64
}

fn vertex_normals(mesh: &Mesh, verts: &[V3]) -> Vec<V3> {
    (0..verts.len())
        .into_par_iter()
        .map(|v| {
            let mut n = [0.0; 3];
            for &f in &mesh.vfaces[v] {
                let [a, b, c] = mesh.faces[f as usize].map(|i| verts[i as usize]);
                n = add(n, cross(sub(b, a), sub(c, a)));
            }
            let l = norm(n);
            if l > 0.0 {
                scale(n, 1.0 / l)
            } else {
                n
            }
        })
        .collect()
}

fn mean_edge(mesh: &Mesh, verts: &[V3]) -> f64 {
    let (mut sum, mut n) = (0.0, 0usize);
    for (v, nb) in mesh.nbrs.iter().enumerate() {
        for &u in nb {
            sum += norm(sub(verts[v], verts[u as usize]));
            n += 1;
        }
    }
    sum / n.max(1) as f64
}

/// Evolve the surface from a sphere at `cog`; returns vertex positions.
fn evolve(
    vol: &Volume,
    mesh: &Mesh,
    st: &HeadStats,
    cog: V3,
    r0: f64,
    p: &DeskullParams,
    prior: Option<&BrainFrame>,
) -> Vec<V3> {
    // Smoothing: radius of curvature below RMIN is flattened hard, above
    // RMAX barely (mm).
    const RMIN: f64 = 3.33;
    const RMAX: f64 = 10.0;
    const D1: f64 = 20.0;
    const D2: f64 = D1 / 2.0;
    let e = 0.5 * (1.0 / RMIN + 1.0 / RMAX);
    let f = 6.0 / (1.0 / RMIN - 1.0 / RMAX);
    let bt = p.frac.clamp(0.0, 1.0);
    let step = vol
        .spacing
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .clamp(0.5, 1.0);

    let mut verts: Vec<V3> = mesh.verts.iter().map(|&u| add(cog, scale(u, r0))).collect();
    // Volume box in voxel-mm; vertices are kept inside it so the surface
    // always rasterises to a closed wall.
    let hi: V3 = {
        let [nx, ny, nz] = vol.shape();
        let s = vol.spacing;
        [
            (nx - 1) as f64 * s[0],
            (ny - 1) as f64 * s[1],
            (nz - 1) as f64 * s[2],
        ]
    };
    // BET scales the intensity step by the mean edge length. A few edges
    // stretched along the volume border would inflate that mean and feed
    // back into ever larger steps, so the step length follows the
    // sphere's edge length scaled by the mean radius instead.
    let l0 = mean_edge(mesh, &verts);
    for _ in 0..p.iterations {
        let centre = scale(
            verts.iter().fold([0.0; 3], |a, &v| add(a, v)),
            1.0 / verts.len() as f64,
        );
        let mean_r = verts.iter().map(|&v| norm(sub(v, centre))).sum::<f64>() / verts.len() as f64;
        let l = l0 * mean_r / r0;
        let normals = vertex_normals(mesh, &verts);
        let next: Vec<V3> = (0..verts.len())
            .into_par_iter()
            .map(|vi| {
                let v = verts[vi];
                let n = normals[vi];
                let nb = &mesh.nbrs[vi];
                let mut m = [0.0; 3];
                let mut ll = 0.0;
                for &u in nb {
                    let w = verts[u as usize];
                    m = add(m, w);
                    ll += norm(sub(w, v));
                }
                m = scale(m, 1.0 / nb.len() as f64);
                ll /= nb.len() as f64;
                let s = sub(m, v);
                let sn_len = dot(s, n);
                let sn = scale(n, sn_len);
                let stan = sub(s, sn);
                let u1 = scale(stan, 0.5);
                // Local curvature radius from the normal offset of the
                // neighbourhood mean: r = l^2 / (2 |sn|).
                let f2 = if sn_len.abs() > 1e-12 && ll > 0.0 {
                    let rinv = 2.0 * sn_len.abs() / (ll * ll);
                    0.5 * (1.0 + (f * (rinv - e)).tanh())
                } else {
                    0.0
                };
                let u2 = scale(sn, f2);
                // Intensity profile along the inward normal.
                let mut imin = st.tm;
                let mut imax = st.t;
                let mut d = 0.0;
                while d <= D1 {
                    let iv = sample_mm(vol, sub(v, scale(n, d)));
                    imin = imin.min(iv);
                    if d <= D2 {
                        imax = imax.max(iv);
                    }
                    d += step;
                }
                let imin = imin.max(st.t2);
                let imax = imax.min(st.tm);
                let tl = (imax - st.t2) * bt + st.t2;
                let denom = (imax - st.t2).max(1e-6);
                let mut f3 = 2.0 * (imin - tl) / denom;
                if prior.is_some_and(|b| b.below_floor(v)) {
                    f3 = -1.0;
                }
                let u3 = scale(n, 0.05 * f3 * l);
                let q = add(v, add(u1, add(u2, u3)));
                [
                    q[0].clamp(0.0, hi[0]),
                    q[1].clamp(0.0, hi[1]),
                    q[2].clamp(0.0, hi[2]),
                ]
            })
            .collect();
        verts = next;
    }
    verts
}

/// Fill a closed triangle surface (voxel-mm coordinates) into a mask:
/// rasterise the triangles densely, then flood the outside from the
/// volume border; everything not reached is inside.
fn fill_surface(vol: &Volume, mesh: &Mesh, verts: &[V3]) -> Array3<u8> {
    let s = vol.spacing;
    let [nx, ny, nz] = vol.shape();
    let mut surf = vec![false; nx * ny * nz];
    let idx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    let vox_step = 0.3 * s.iter().cloned().fold(f64::INFINITY, f64::min);
    for &[a, b, c] in &mesh.faces {
        let (a, b, c) = (verts[a as usize], verts[b as usize], verts[c as usize]);
        let longest = norm(sub(b, a)).max(norm(sub(c, a))).max(norm(sub(c, b)));
        let n = (longest / vox_step).ceil().max(1.0) as usize;
        for u in 0..=n {
            for w in 0..=(n - u) {
                let (fu, fw) = (u as f64 / n as f64, w as f64 / n as f64);
                let q = add(a, add(scale(sub(b, a), fu), scale(sub(c, a), fw)));
                let (i, j, k) = (
                    (q[0] / s[0]).round(),
                    (q[1] / s[1]).round(),
                    (q[2] / s[2]).round(),
                );
                if i >= 0.0 && j >= 0.0 && k >= 0.0 {
                    let (i, j, k) = (i as usize, j as usize, k as usize);
                    if i < nx && j < ny && k < nz {
                        surf[idx(i, j, k)] = true;
                    }
                }
            }
        }
    }
    let outside = flood_from_border(&surf, [nx, ny, nz]);
    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        let x = idx(i, j, k);
        u8::from(surf[x] || !outside[x])
    })
}

/// Smallest plausible brain (ml); passes below this are rejected. Adult
/// brains are ~1000-1600 ml, neonates ~350 ml.
const MIN_BRAIN_ML: f64 = 200.0;

/// MRI extraction. Returns the mask, the number of surface passes and
/// whether the face prior was actually used.
fn mri_mask(vol: &Volume, p: &DeskullParams) -> (Array3<u8>, usize, bool) {
    let st = head_stats(vol);
    let mesh = icosphere(p.subdivisions.min(6));
    // The prior is only trusted when the brain centre it implies lies in
    // head tissue inside the volume (a mislifted or misoriented face can
    // put it in the air or outside the field of view).
    let prior = p
        .face
        .as_ref()
        .and_then(|f| brain_frame(vol, f, st.cog))
        .filter(|b| sample_mm(vol, b.point(BRAIN_CENTRE)) > st.t);
    if let Some(b) = &prior {
        let (mask, passes) = surface_passes(vol, p, &st, &mesh, Some(b));
        if mask_ml(vol, &mask) >= MIN_BRAIN_ML {
            return (mask, passes, true);
        }
    }
    let (mask, passes) = surface_passes(vol, p, &st, &mesh, None);
    (mask, passes, false)
}

fn mask_ml(vol: &Volume, mask: &Array3<u8>) -> f64 {
    let n = mask.iter().filter(|&&m| m != 0).count();
    n as f64 * vol.spacing.iter().product::<f64>() / 1000.0
}

fn surface_passes(
    vol: &Volume,
    p: &DeskullParams,
    st: &HeadStats,
    mesh: &Mesh,
    prior: Option<&BrainFrame>,
) -> (Array3<u8>, usize) {
    let (mut cog, mut r0) = match prior {
        Some(b) => (b.point(BRAIN_CENTRE), BRAIN_START_RADIUS * b.d),
        None => (st.cog, st.radius / 2.0),
    };
    let mut best: Option<Array3<u8>> = None;
    let mut passes = 0;
    loop {
        passes += 1;
        let verts = evolve(vol, mesh, st, cog, r0, p, prior);
        let mut mask = fill_surface(vol, mesh, &verts);
        if let Some(b) = prior {
            clip_below_floor(vol, b, &mut mask);
        }
        // A re-centred pass that collapses keeps the previous result.
        if mask_ml(vol, &mask) < MIN_BRAIN_ML {
            if let Some(prev) = best {
                return (prev, passes - 1);
            }
        }
        if !p.robust || passes >= 3 {
            return (mask, passes);
        }
        // Re-centre on the brain found and start from a sphere of half
        // its equivalent radius, as for the first pass.
        let s = vol.spacing;
        let (mut c, mut n) = ([0.0f64; 3], 0usize);
        for ((i, j, k), &m) in mask.indexed_iter() {
            if m != 0 {
                c = add(c, [i as f64 * s[0], j as f64 * s[1], k as f64 * s[2]]);
                n += 1;
            }
        }
        if n == 0 {
            return (mask, passes);
        }
        let new_cog = scale(c, 1.0 / n as f64);
        if norm(sub(new_cog, cog)) < 1.0 {
            return (mask, passes);
        }
        let vol_mm3 = n as f64 * s[0] * s[1] * s[2];
        r0 = (3.0 * vol_mm3 / (4.0 * std::f64::consts::PI)).cbrt() / 2.0;
        cog = new_cog;
        best = Some(mask);
    }
}

// ---------------------------------------------------------------------------
// Face prior
// ---------------------------------------------------------------------------

/// Face-anchored frame for the brain prior, in voxel-mm coordinates.
/// Built from the eyes and nose tip only (the mouth often lands on the
/// field-of-view cut plane): `r` = eye axis, `u` = eyes-minus-nose
/// direction orthogonalised against `r`, `f` = `r` x `u` oriented away
/// from the head's centre of gravity. Coordinates are in units of the
/// inter-eye distance `d`.
struct BrainFrame {
    o: V3,
    r: V3,
    u: V3,
    f: V3,
    d: f64,
}

fn to_vmm(vol: &Volume, w: [f64; 3]) -> V3 {
    let v = crate::affine::apply(&vol.world_to_voxel, w);
    let s = vol.spacing;
    [v[0] * s[0], v[1] * s[1], v[2] * s[2]]
}

fn brain_frame(vol: &Volume, face: &FaceFrame, head_cog: V3) -> Option<BrainFrame> {
    let el = to_vmm(vol, face.eye_l);
    let er = to_vmm(vol, face.eye_r);
    let nose = to_vmm(vol, face.nose_tip);
    let o = scale(add(el, er), 0.5);
    let d = norm(sub(el, er));
    if d < 1.0 {
        return None;
    }
    let r = scale(sub(el, er), 1.0 / d);
    let up0 = sub(o, nose);
    let u = sub(up0, scale(r, dot(up0, r)));
    let ul = norm(u);
    if ul < 1e-6 {
        return None;
    }
    let u = scale(u, 1.0 / ul);
    let mut f = cross(r, u);
    if dot(f, sub(o, head_cog)) < 0.0 {
        f = scale(f, -1.0);
    }
    Some(BrainFrame { o, r, u, f, d })
}

impl BrainFrame {
    /// (|u|, v, w) of a voxel-mm point, in units of d.
    #[inline]
    fn coords(&self, p: V3) -> V3 {
        let q = sub(p, self.o);
        [
            dot(q, self.r).abs() / self.d,
            dot(q, self.u) / self.d,
            dot(q, self.f) / self.d,
        ]
    }

    #[inline]
    fn point(&self, c: V3) -> V3 {
        add(
            self.o,
            scale(
                add(
                    scale(self.r, c[0]),
                    add(scale(self.u, c[1]), scale(self.f, c[2])),
                ),
                self.d,
            ),
        )
    }

    /// True when `p` lies below the measured brain floor (with margin).
    #[inline]
    fn below_floor(&self, p: V3) -> bool {
        let [au, v, w] = self.coords(p);
        v < floor_at(au, w) - FLOOR_MARGIN
    }
}

/// Bilinear lookup in the calibrated floor table; outside the table the
/// nearest edge value applies. `f64::INFINITY` = no brain was ever
/// observed in that column (e.g. in front of the face).
fn floor_at(au: f64, w: f64) -> f64 {
    let fu = ((au - FLOOR_U0) / FLOOR_STEP).clamp(0.0, (FLOOR_NU - 1) as f64);
    let fw = ((w - FLOOR_W0) / FLOOR_STEP).clamp(0.0, (FLOOR_NW - 1) as f64);
    let (iu, iw) = (fu.floor() as usize, fw.floor() as usize);
    let (iu1, iw1) = ((iu + 1).min(FLOOR_NU - 1), (iw + 1).min(FLOOR_NW - 1));
    let (tu, tw) = (fu - iu as f64, fw - iw as f64);
    let g = |a: usize, b: usize| FLOOR[a][b] as f64;
    let corners = [g(iu, iw), g(iu1, iw), g(iu, iw1), g(iu1, iw1)];
    if corners.iter().any(|c| !c.is_finite()) {
        // Any forbidden corner: be conservative only when the nearest
        // cell is forbidden.
        return g(fu.round() as usize, fw.round() as usize);
    }
    let a = corners[0] + (corners[1] - corners[0]) * tu;
    let b = corners[2] + (corners[3] - corners[2]) * tu;
    a + (b - a) * tw
}

include!("deskull_floor.rs");

fn clip_below_floor(vol: &Volume, b: &BrainFrame, mask: &mut Array3<u8>) {
    let s = vol.spacing;
    ndarray::Zip::indexed(mask).par_for_each(|(i, j, k), m| {
        if *m != 0 && b.below_floor([i as f64 * s[0], j as f64 * s[1], k as f64 * s[2]]) {
            *m = 0;
        }
    });
}

// ---------------------------------------------------------------------------
// CT: bone enclosure + morphology
// ---------------------------------------------------------------------------

fn ct_mask(vol: &Volume, p: &DeskullParams) -> Array3<u8> {
    let [nx, ny, nz] = vol.shape();
    let dims = [nx, ny, nz];
    let n = nx * ny * nz;
    let data: Vec<f32> = vol.data.iter().copied().collect();
    let bone: Vec<bool> = data.par_iter().map(|&v| v >= p.ct_bone_hu).collect();
    let enclosed = bone_enclosure(&bone, dims);
    let (lo, hi) = p.ct_window;
    let candidate: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|x| data[x] >= lo && data[x] <= hi && enclosed[x] >= 5)
        .collect();
    // Erode, keep the largest component, dilate back inside the candidate.
    let d_out = distance_to(&candidate, dims, vol.spacing, false);
    let core: Vec<bool> = d_out
        .par_iter()
        .map(|&d| d > p.ct_erode_mm as f32)
        .collect();
    let core = largest_component(&core, dims);
    let d_core = distance_to(&core, dims, vol.spacing, true);
    let grow = p.ct_erode_mm as f32 + 0.5 * vol.spacing.iter().cloned().fold(0.0, f64::max) as f32;
    let mut brain: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|x| candidate[x] && d_core[x] <= grow)
        .collect();
    brain = largest_component(&brain, dims);
    let outside = flood_from_border(&brain, dims);
    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        let x = (i * ny + j) * nz + k;
        u8::from(brain[x] || !outside[x])
    })
}

/// For every voxel, how many of the six voxel-axis directions contain a
/// bone voxel somewhere along the ray (0..=6).
fn bone_enclosure(bone: &[bool], [nx, ny, nz]: [usize; 3]) -> Vec<u8> {
    let idx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    let mut count = vec![0u8; bone.len()];
    // Sweep each axis in both directions; `seen` = bone passed so far.
    for axis in 0..3 {
        let (la, lb, lc) = match axis {
            0 => (ny, nz, nx),
            1 => (nx, nz, ny),
            _ => (nx, ny, nz),
        };
        for a in 0..la {
            for b in 0..lb {
                let at = |c: usize| match axis {
                    0 => idx(c, a, b),
                    1 => idx(a, c, b),
                    _ => idx(a, b, c),
                };
                let mut seen = false;
                for c in 0..lc {
                    let x = at(c);
                    if seen {
                        count[x] += 1;
                    }
                    seen |= bone[x];
                }
                let mut seen = false;
                for c in (0..lc).rev() {
                    let x = at(c);
                    if seen {
                        count[x] += 1;
                    }
                    seen |= bone[x];
                }
            }
        }
    }
    count
}

/// Chamfer (26-neighbour, spacing-weighted) distance in mm from each voxel
/// to the nearest voxel where `set == target`; 0 on those voxels.
pub(crate) fn distance_to(
    set: &[bool],
    [nx, ny, nz]: [usize; 3],
    s: [f64; 3],
    target: bool,
) -> Vec<f32> {
    let idx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    let mut d: Vec<f32> = set
        .iter()
        .map(|&v| if v == target { 0.0 } else { f32::INFINITY })
        .collect();
    let mut offs: Vec<([isize; 3], f32)> = Vec::new();
    for di in -1isize..=1 {
        for dj in -1isize..=1 {
            for dk in -1isize..=1 {
                if (di, dj, dk) < (0, 0, 0) {
                    let w = ((di as f64 * s[0]).powi(2)
                        + (dj as f64 * s[1]).powi(2)
                        + (dk as f64 * s[2]).powi(2))
                    .sqrt() as f32;
                    offs.push(([di, dj, dk], w));
                }
            }
        }
    }
    let (nxi, nyi, nzi) = (nx as isize, ny as isize, nz as isize);
    // Forward raster pass pulls from lexicographically earlier neighbours,
    // the backward pass (reversed order, negated offsets) from later ones.
    for forward in [true, false] {
        let sign: isize = if forward { 1 } else { -1 };
        for step in 0..n_total(nx, ny, nz) {
            let x = if forward {
                step
            } else {
                n_total(nx, ny, nz) - 1 - step
            };
            let mut best = d[x];
            if best == 0.0 {
                continue;
            }
            let (i, j, k) = (x / (ny * nz), (x / nz) % ny, x % nz);
            for &(o, w) in &offs {
                let (a, b, c) = (
                    i as isize + sign * o[0],
                    j as isize + sign * o[1],
                    k as isize + sign * o[2],
                );
                if a < 0 || b < 0 || c < 0 || a >= nxi || b >= nyi || c >= nzi {
                    continue;
                }
                let cand = d[idx(a as usize, b as usize, c as usize)] + w;
                if cand < best {
                    best = cand;
                }
            }
            d[x] = best;
        }
    }
    d
}

#[inline]
fn n_total(nx: usize, ny: usize, nz: usize) -> usize {
    nx * ny * nz
}

/// Mark voxels reachable from the volume border through `!wall` voxels
/// (6-connectivity).
pub(crate) fn flood_from_border(wall: &[bool], [nx, ny, nz]: [usize; 3]) -> Vec<bool> {
    let idx = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    let mut out = vec![false; wall.len()];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 {
                    let x = idx(i, j, k);
                    if !wall[x] && !out[x] {
                        out[x] = true;
                        stack.push(x);
                    }
                }
            }
        }
    }
    flood(&mut out, &mut stack, [nx, ny, nz], |x| !wall[x]);
    out
}

fn flood(
    mark: &mut [bool],
    stack: &mut Vec<usize>,
    [nx, ny, nz]: [usize; 3],
    pass: impl Fn(usize) -> bool,
) -> usize {
    let (sj, si) = (nz, ny * nz);
    let mut count = 0usize;
    while let Some(x) = stack.pop() {
        let k = x % nz;
        let j = (x / nz) % ny;
        let i = x / si;
        let mut push = |y: usize| {
            if !mark[y] && pass(y) {
                mark[y] = true;
                stack.push(y);
                count += 1;
            }
        };
        if i > 0 {
            push(x - si);
        }
        if i + 1 < nx {
            push(x + si);
        }
        if j > 0 {
            push(x - sj);
        }
        if j + 1 < ny {
            push(x + sj);
        }
        if k > 0 {
            push(x - 1);
        }
        if k + 1 < nz {
            push(x + 1);
        }
    }
    count
}

/// Largest 6-connected component of `set`.
pub(crate) fn largest_component(set: &[bool], dims: [usize; 3]) -> Vec<bool> {
    let mut seen = vec![false; set.len()];
    let (mut best_seed, mut best_n) = (None, 0usize);
    let mut stack = Vec::new();
    for x in 0..set.len() {
        if set[x] && !seen[x] {
            seen[x] = true;
            stack.push(x);
            let n = flood(&mut seen, &mut stack, dims, |y| set[y]) + 1;
            if n > best_n {
                best_n = n;
                best_seed = Some(x);
            }
        }
    }
    let mut out = vec![false; set.len()];
    if let Some(x) = best_seed {
        out[x] = true;
        stack.push(x);
        flood(&mut out, &mut stack, dims, |y| set[y]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_is_closed_and_outward() {
        let m = icosphere(3);
        assert_eq!(m.verts.len(), 642);
        // Euler characteristic of a sphere.
        let edges: usize = m.nbrs.iter().map(|n| n.len()).sum::<usize>() / 2;
        assert_eq!(m.verts.len() + m.faces.len() - edges, 2);
        let normals = vertex_normals(&m, &m.verts);
        assert!(normals.iter().zip(&m.verts).all(|(n, v)| dot(*n, *v) > 0.9));
    }

    #[test]
    fn distance_transform_is_close_to_euclidean() {
        let dims = [41, 41, 41];
        let mut set = vec![false; 41 * 41 * 41];
        set[(20 * 41 + 20) * 41 + 20] = true;
        let d = distance_to(&set, dims, [1.0, 1.0, 2.0], true);
        let at = |i: usize, j: usize, k: usize| d[(i * 41 + j) * 41 + k];
        assert_eq!(at(20, 20, 20), 0.0);
        assert!((at(30, 20, 20) - 10.0).abs() < 1e-4);
        assert!((at(20, 20, 25) - 10.0).abs() < 1e-4);
        let diag = at(28, 26, 20);
        assert!((diag - 10.0).abs() < 1.0, "{diag}");
    }
}
