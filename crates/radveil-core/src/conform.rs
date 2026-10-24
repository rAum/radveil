// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! The resampled grid the brain-extraction networks see, and its
//! surfa-compatible construction.
//!
//! The networks see a resampled, cropped and intensity-normalised grid.
//! Reproducing surfa's exact conventions matters: getting the resize centre
//! or the crop predicate wrong moves the mask boundary, and picking the
//! voxel axes from the inverse affine (whose rows are scaled by voxel size)
//! picks the wrong axis on anisotropic scans.

use crate::affine::{self, Mat4};
use crate::volume::Volume;
use rayon::prelude::*;

/// Target grid for [`conform`]. Each network's own spec comes from its
/// weights: SynthStrip is fixed at 1 mm, a `.rvb` model carries its own.
#[derive(Clone, Copy, Debug)]
pub struct ConformSpec {
    /// Isotropic voxel size (mm) the network runs at.
    pub voxel_mm: f64,
    /// Each axis is padded/cropped to a multiple of this (the number of
    /// pooling steps sets it: 2^levels).
    pub multiple: usize,
    /// Clamp for the per-axis size, in voxels.
    pub min: usize,
    pub max: usize,
}

/// A conformed isotropic LIA grid: exactly what a network is fed.
///
/// `data` is C-order over `shape` (last axis fastest) and intensity
/// normalised by [`normalise`]: shifted to a zero minimum, divided by the
/// 99th percentile and clipped to 0..1.
pub struct Conformed {
    pub shape: [usize; 3],
    /// Grid voxel -> world RAS.
    pub affine: Mat4,
    pub data: Vec<f32>,
}

/// Reproduces SynthStrip's preprocessing (surfa `conform(voxsize=1,
/// orientation='LIA', method='nearest')`, `crop_to_bbox()`, `reshape()`):
///
/// 1. reorient: permute/flip voxel axes to the closest LIA order (no
///    resampling; any obliquity is kept);
/// 2. resize to 1 mm along those axes about the image centre (surfa's
///    centre is the world position of voxel `shape / 2`), nearest
///    neighbour, `ceil(voxsize * shape)` voxels;
/// 3. crop to the bounding box of voxels > 0;
/// 4. centre-pad/crop to a multiple of 64 in 192..=320, offset truncated
///    toward zero.
pub fn conform(vol: &Volume, spec: &ConformSpec) -> Conformed {
    let src = &vol.affine;
    let [nx, ny, nz] = vol.shape();
    let nsrc = [nx, ny, nz];
    // 1. Reorientation. Like surfa, each voxel axis is labelled by the
    //    world axis its (normalised) direction is most aligned with; target
    //    axis a is L, I, A = world axes 0, 2, 1 with signs -, -, +.
    let dominant = |j: usize| -> (usize, f64) {
        let col = [src[0][j], src[1][j], src[2][j]];
        let n = affine::norm(col).max(1e-12);
        let wa = (0..3)
            .max_by(|&x, &y| col[x].abs().partial_cmp(&col[y].abs()).unwrap())
            .unwrap();
        (wa, col[wa] / n)
    };
    let world_axis = [0usize, 2, 1];
    let want_sign = [-1.0f64, -1.0, 1.0];
    let mut perm = [0usize, 1, 2];
    let mut flip = [false; 3];
    let labels: Vec<(usize, f64)> = (0..3).map(dominant).collect();
    let distinct =
        labels[0].0 != labels[1].0 && labels[1].0 != labels[2].0 && labels[0].0 != labels[2].0;
    for a in 0..3 {
        let wa = world_axis[a];
        let j = if distinct {
            (0..3).find(|&j| labels[j].0 == wa).unwrap()
        } else {
            a
        };
        perm[a] = j;
        flip[a] = src[wa][j] * want_sign[a] < 0.0;
    }
    // Reoriented grid r: index n_r -> source index.
    let shape_r = [nsrc[perm[0]], nsrc[perm[1]], nsrc[perm[2]]];
    let mut r2s: Mat4 = [[0.0; 4]; 4];
    for a in 0..3 {
        let j = perm[a];
        if flip[a] {
            r2s[j][a] = -1.0;
            r2s[j][3] = (shape_r[a] - 1) as f64;
        } else {
            r2s[j][a] = 1.0;
        }
    }
    r2s[3][3] = 1.0;
    let v2w_r = affine::mul(src, &r2s);
    let vs_r = [
        vol.spacing[perm[0]],
        vol.spacing[perm[1]],
        vol.spacing[perm[2]],
    ];
    // 2. Resize to 1 mm about the image centre.
    let (shape1, v2w1) = if vs_r.iter().all(|v| (v - spec.voxel_mm).abs() <= 1e-5) {
        (shape_r, v2w_r)
    } else {
        let shape1 =
            [0, 1, 2].map(|a| (vs_r[a] * shape_r[a] as f64 / spec.voxel_mm - 1e-9).ceil() as usize);
        let centre = affine::apply(&v2w_r, [0, 1, 2].map(|a| shape_r[a] as f64 / 2.0));
        let mut m: Mat4 = [[0.0; 4]; 4];
        for a in 0..3 {
            for r in 0..3 {
                m[r][a] = v2w_r[r][a] / vs_r[a] * spec.voxel_mm;
            }
        }
        for r in 0..3 {
            m[r][3] = centre[r]
                - (0..3)
                    .map(|a| m[r][a] * shape1[a] as f64 / 2.0)
                    .sum::<f64>();
        }
        m[3][3] = 1.0;
        (shape1, m)
    };
    // Nearest-neighbour samples of the 1 mm grid.
    let to_src = affine::mul(&vol.world_to_voxel, &v2w1);
    let [d1, h1, w1] = shape1;
    let mut g1 = vec![0.0f32; d1 * h1 * w1];
    g1.par_chunks_mut(h1 * w1)
        .enumerate()
        .for_each(|(i, plane)| {
            for j in 0..h1 {
                for k in 0..w1 {
                    let v = affine::apply(&to_src, [i as f64, j as f64, k as f64]);
                    let (a, b, c) = (v[0].round(), v[1].round(), v[2].round());
                    if a >= 0.0 && b >= 0.0 && c >= 0.0 {
                        let (a, b, c) = (a as usize, b as usize, c as usize);
                        if a < nx && b < ny && c < nz {
                            plane[j * w1 + k] = vol.data[[a, b, c]];
                        }
                    }
                }
            }
        });
    // 3. Bounding box of voxels > 0.
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for i in 0..d1 {
        for j in 0..h1 {
            for k in 0..w1 {
                if g1[(i * h1 + j) * w1 + k] > 0.0 {
                    for (a, x) in [i, j, k].into_iter().enumerate() {
                        lo[a] = lo[a].min(x);
                        hi[a] = hi[a].max(x);
                    }
                }
            }
        }
    }
    if lo[0] == usize::MAX {
        lo = [0; 3];
        hi = [d1 - 1, h1 - 1, w1 - 1];
    }
    let ext = [0, 1, 2].map(|a| hi[a] - lo[a] + 1);
    // 4. Target shape and centred offset (truncated toward zero).
    let shape = ext.map(|e| (e.div_ceil(spec.multiple) * spec.multiple).clamp(spec.min, spec.max));
    let off = [0, 1, 2].map(|a| {
        let delta = (shape[a] as f64 - ext[a] as f64) / 2.0;
        lo[a] as isize - delta.trunc() as isize
    });
    let [d, h, w] = shape;
    let mut data = vec![0.0f32; d * h * w];
    data.par_chunks_mut(h * w)
        .enumerate()
        .for_each(|(i, plane)| {
            let si = i as isize + off[0];
            if si < lo[0] as isize || si > hi[0] as isize {
                return;
            }
            for j in 0..h {
                let sj = j as isize + off[1];
                if sj < lo[1] as isize || sj > hi[1] as isize {
                    continue;
                }
                for k in 0..w {
                    let sk = k as isize + off[2];
                    if sk < lo[2] as isize || sk > hi[2] as isize {
                        continue;
                    }
                    plane[j * w + k] = g1[((si as usize) * h1 + sj as usize) * w1 + sk as usize];
                }
            }
        });
    let mut shift: Mat4 = [[0.0; 4]; 4];
    for a in 0..3 {
        shift[a][a] = 1.0;
        shift[a][3] = off[a] as f64;
    }
    shift[3][3] = 1.0;
    let aff = affine::mul(&v2w1, &shift);
    normalise(&mut data);
    Conformed {
        shape,
        affine: aff,
        data,
    }
}

/// Shift to a zero minimum, divide by the 99th percentile, clip to 0..1.
/// This is what the networks were trained with, so it has to be applied to
/// the whole grid, not per patch.
pub fn normalise(data: &mut [f32]) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    data.par_iter_mut().for_each(|v| *v -= min);
    let mut sorted = data.to_vec();
    let idx = ((sorted.len() - 1) as f64 * 0.99).round() as usize;
    let (_, p99, _) = sorted.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).unwrap());
    let p99 = if *p99 > 0.0 { *p99 } else { 1.0 };
    data.par_iter_mut()
        .for_each(|v| *v = (*v / p99).clamp(0.0, 1.0));
}

pub(crate) fn trilinear(v: &[f32], [d, h, w]: [usize; 3], p: [f64; 3]) -> Option<f32> {
    let (x, y, z) = (p[0], p[1], p[2]);
    if x < 0.0
        || y < 0.0
        || z < 0.0
        || x > (d - 1) as f64
        || y > (h - 1) as f64
        || z > (w - 1) as f64
    {
        return None;
    }
    let (x0, y0, z0) = (x.floor() as usize, y.floor() as usize, z.floor() as usize);
    let (x1, y1, z1) = (
        (x0 + 1).min(d - 1),
        (y0 + 1).min(h - 1),
        (z0 + 1).min(w - 1),
    );
    let (fx, fy, fz) = (
        (x - x0 as f64) as f32,
        (y - y0 as f64) as f32,
        (z - z0 as f64) as f32,
    );
    let g = |a: usize, b: usize, c: usize| v[(a * h + b) * w + c];
    let c00 = g(x0, y0, z0) + (g(x1, y0, z0) - g(x0, y0, z0)) * fx;
    let c10 = g(x0, y1, z0) + (g(x1, y1, z0) - g(x0, y1, z0)) * fx;
    let c01 = g(x0, y0, z1) + (g(x1, y0, z1) - g(x0, y0, z1)) * fx;
    let c11 = g(x0, y1, z1) + (g(x1, y1, z1) - g(x0, y1, z1)) * fx;
    let c0 = c00 + (c10 - c00) * fy;
    let c1 = c01 + (c11 - c01) * fy;
    Some(c0 + (c1 - c0) * fz)
}
