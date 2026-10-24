// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU software volume renderer (orthographic surface ray caster).
//!
//! The camera orbits the volume center in world (RAS+) space: yaw rotates
//! around the superior axis (0 = looking at the face from anterior),
//! pitch tilts the camera up/down. Each pixel marches a ray until the
//! intensity crosses a threshold, refines the hit by bisection, and shades
//! with a skin-tone Blinn-Phong headlight so face detectors see a
//! natural-looking portrait. Alongside RGB we keep the 3D world position of
//! every surface hit, which is what makes 2D landmark -> 3D unprojection
//! trivial and exact.

use crate::affine::{add, cross, dot, normalize, scale, sub, Vec3};
use crate::volume::Volume;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderParams {
    /// Image width/height in pixels.
    pub size: usize,
    /// Surface threshold as a fraction of the 99th intensity percentile
    /// (MRI-like data). Used when `threshold_abs` is None. For CT-like
    /// data (HU: 1st percentile below -500) this is ignored and the
    /// air/soft-tissue boundary at -400 HU is used instead.
    pub threshold_frac: f32,
    /// Absolute intensity threshold override.
    pub threshold_abs: Option<f32>,
    /// Ray step in mm (0 = auto: 0.4 * min voxel spacing).
    pub step_mm: f64,
}

impl Default for RenderParams {
    fn default() -> Self {
        RenderParams {
            size: 512,
            threshold_frac: 0.14,
            threshold_abs: None,
            step_mm: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Camera {
    pub yaw_deg: f64,
    pub pitch_deg: f64,
    /// World-space center the camera orbits.
    pub center: Vec3,
    /// Unit vectors: viewing direction, image right, image up (world space).
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    /// mm per pixel.
    pub pixel_mm: f64,
    pub width: usize,
    pub height: usize,
}

pub struct RenderView {
    pub camera: Camera,
    /// RGB8, row-major, y down.
    pub rgb: Vec<u8>,
    /// Per-pixel world position of the surface hit (x,y,z), NaN if no hit.
    pub pos: Vec<f32>,
    pub hit: Vec<bool>,
    /// True where the surface hit lies on the volume boundary (scan cut
    /// plane) rather than real anatomy; such positions are unreliable for
    /// landmark lifting.
    pub boundary: Vec<bool>,
    pub threshold: f32,
}

impl RenderView {
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let f = std::fs::File::create(path)?;
        let mut enc = png::Encoder::new(
            std::io::BufWriter::new(f),
            self.camera.width as u32,
            self.camera.height as u32,
        );
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().map_err(std::io::Error::other)?;
        writer
            .write_image_data(&self.rgb)
            .map_err(std::io::Error::other)?;
        Ok(())
    }

    /// World position for an image pixel, searching a small neighborhood if
    /// the exact pixel missed the surface. Returns None outside the head.
    pub fn world_at(&self, x: f64, y: f64, max_radius: i64) -> Option<Vec3> {
        self.world_at_impl(x, y, max_radius, false)
    }

    /// Like [`world_at`](Self::world_at) but refuses positions on the volume
    /// boundary (scan cut planes). Use for landmarks where a boundary hit
    /// means "feature outside the field of view" (e.g. the jaw line).
    pub fn world_at_anatomical(&self, x: f64, y: f64, max_radius: i64) -> Option<Vec3> {
        self.world_at_impl(x, y, max_radius, true)
    }

    fn world_at_impl(&self, x: f64, y: f64, max_radius: i64, skip_boundary: bool) -> Option<Vec3> {
        let w = self.camera.width as i64;
        let h = self.camera.height as i64;
        let cx = x.round() as i64;
        let cy = y.round() as i64;
        let mut best: Option<(i64, Vec3)> = None;
        for r in 0..=max_radius {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs().max(dy.abs()) != r {
                        continue;
                    }
                    let px = cx + dx;
                    let py = cy + dy;
                    if px < 0 || py < 0 || px >= w || py >= h {
                        continue;
                    }
                    let idx = (py * w + px) as usize;
                    if self.hit[idx] && !(skip_boundary && self.boundary[idx]) {
                        let d2 = dx * dx + dy * dy;
                        let p = [
                            self.pos[idx * 3] as f64,
                            self.pos[idx * 3 + 1] as f64,
                            self.pos[idx * 3 + 2] as f64,
                        ];
                        if best.is_none_or(|(bd, _)| d2 < bd) {
                            best = Some((d2, p));
                        }
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        best.map(|(_, p)| p)
    }
}

fn camera_for(volume: &Volume, yaw_deg: f64, pitch_deg: f64, size: usize) -> (Camera, f64) {
    let (lo, hi) = volume.world_bounds();
    let center = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let radius =
        0.5 * ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let yaw = yaw_deg.to_radians();
    let pitch = pitch_deg.to_radians();
    // Camera sits on the anterior side (+A) at yaw 0, orbiting around S.
    let cam_dir = [
        yaw.sin() * pitch.cos(),
        yaw.cos() * pitch.cos(),
        pitch.sin(),
    ];
    let forward = scale(cam_dir, -1.0);
    let world_up = [0.0, 0.0, 1.0];
    let mut right = cross(forward, world_up);
    if crate::affine::norm(right) < 1e-6 {
        right = [1.0, 0.0, 0.0];
    }
    let right = normalize(right);
    let up = normalize(cross(right, forward));
    let pixel_mm = (2.0 * radius * 1.02) / size as f64;
    (
        Camera {
            yaw_deg,
            pitch_deg,
            center,
            forward,
            right,
            up,
            pixel_mm,
            width: size,
            height: size,
        },
        radius,
    )
}

#[allow(clippy::needless_range_loop)]
pub fn render_view(
    volume: &Volume,
    yaw_deg: f64,
    pitch_deg: f64,
    params: &RenderParams,
) -> RenderView {
    let size = params.size;
    let (camera, radius) = camera_for(volume, yaw_deg, pitch_deg, size);
    let threshold = params.threshold_abs.unwrap_or_else(|| {
        if volume.percentile(1.0) < -500.0 {
            // CT in Hounsfield units: air is ~-1000 (FOV padding often
            // lower), skin sits near 0; -400 crosses at the skin surface.
            -400.0
        } else {
            params.threshold_frac * volume.percentile(99.0)
        }
    });
    let min_sp = volume.spacing.iter().cloned().fold(f64::INFINITY, f64::min);
    let step = if params.step_mm > 0.0 {
        params.step_mm
    } else {
        (0.4 * min_sp).max(0.05)
    };
    let n_steps = ((2.0 * radius) / step).ceil() as usize + 2;

    // Light: headlight tilted a bit up-right of the camera.
    let light = normalize(add(
        add(scale(camera.forward, -1.0), scale(camera.up, 0.45)),
        scale(camera.right, 0.25),
    ));
    let skin = [0.96f32, 0.78, 0.66];

    let mut rgb = vec![0u8; size * size * 3];
    let mut pos = vec![f32::NAN; size * size * 3];
    let mut hit = vec![false; size * size];
    let mut boundary = vec![false; size * size];
    let [nx, ny, nz] = volume.shape();
    let grid = volume.macro_grid();

    type Row = (Vec<u8>, Vec<f32>, Vec<bool>, Vec<bool>);
    let rows: Vec<usize> = (0..size).collect();
    let results: Vec<Row> = rows
        .par_iter()
        .map(|&py| {
            let mut row_rgb = vec![0u8; size * 3];
            let mut row_pos = vec![f32::NAN; size * 3];
            let mut row_hit = vec![false; size];
            let mut row_boundary = vec![false; size];
            for px in 0..size {
                let u = (px as f64 - size as f64 * 0.5 + 0.5) * camera.pixel_mm;
                let v = (size as f64 * 0.5 - py as f64 - 0.5) * camera.pixel_mm;
                let origin = add(
                    add(
                        add(camera.center, scale(camera.right, u)),
                        scale(camera.up, v),
                    ),
                    scale(camera.forward, -radius),
                );
                // Clip the ray to the voxel box (slab test in voxel space).
                // Marching only inside the volume matters for CT: outside
                // samples read as 0, which is above an HU threshold.
                let ov = crate::affine::apply(&volume.world_to_voxel, origin);
                let dv = crate::affine::apply_linear(&volume.world_to_voxel, camera.forward);
                let mut t_in = 0.0f64;
                let mut t_out = 2.0 * radius;
                let dims = [nx as f64, ny as f64, nz as f64];
                for a in 0..3 {
                    let lo = -0.49;
                    let hi = dims[a] - 0.51;
                    if dv[a].abs() < 1e-12 {
                        if ov[a] < lo || ov[a] > hi {
                            t_out = -1.0;
                            break;
                        }
                    } else {
                        let ta = (lo - ov[a]) / dv[a];
                        let tb = (hi - ov[a]) / dv[a];
                        let (ta, tb) = if ta < tb { (ta, tb) } else { (tb, ta) };
                        t_in = t_in.max(ta);
                        t_out = t_out.min(tb);
                    }
                }
                // March with macro-grid empty-space skipping: whole 8^3
                // blocks whose max intensity is below the threshold are
                // stepped over in one jump.
                let bsz = grid.block as f64;
                let mut prev_t = t_in;
                let mut found: Option<f64> = None;
                let mut t = t_in;
                let mut guard = 0usize;
                while t <= t_out && found.is_none() {
                    guard += 1;
                    if guard > 4 * n_steps + 64 {
                        break;
                    }
                    let pv = [ov[0] + t * dv[0], ov[1] + t * dv[1], ov[2] + t * dv[2]];
                    let blk = [
                        ((pv[0].max(0.0) as usize) / grid.block).min(grid.dims[0] - 1),
                        ((pv[1].max(0.0) as usize) / grid.block).min(grid.dims[1] - 1),
                        ((pv[2].max(0.0) as usize) / grid.block).min(grid.dims[2] - 1),
                    ];
                    // exit distance of the current block along the ray
                    let mut t_exit = t_out;
                    for a in 0..3 {
                        if dv[a].abs() > 1e-12 {
                            let bound = if dv[a] > 0.0 {
                                (blk[a] + 1) as f64 * bsz
                            } else {
                                blk[a] as f64 * bsz
                            };
                            let ta = (bound - ov[a]) / dv[a];
                            if ta > t {
                                t_exit = t_exit.min(ta);
                            }
                        }
                    }
                    let bidx = (blk[0] * grid.dims[1] + blk[1]) * grid.dims[2] + blk[2];
                    if grid.max[bidx] < threshold {
                        // empty block: jump to its far side
                        prev_t = t_exit;
                        t = t_exit.max(t) + 0.25 * step;
                        continue;
                    }
                    // fine march through the occupied block
                    let t_end = (t_exit + 0.25 * step).min(t_out);
                    // Sample in voxel space: the ray is a straight line
                    // there too, so this skips a full affine per sample.
                    let at = |t: f64| [ov[0] + t * dv[0], ov[1] + t * dv[1], ov[2] + t * dv[2]];
                    while t <= t_end {
                        let val = volume.sample_voxel(at(t));
                        if val >= threshold {
                            // refine by bisection between prev_t and t
                            let mut a = prev_t;
                            let mut b = t;
                            for _ in 0..5 {
                                let m = 0.5 * (a + b);
                                let vm = volume.sample_voxel(at(m));
                                if vm >= threshold {
                                    b = m;
                                } else {
                                    a = m;
                                }
                            }
                            found = Some(b);
                            break;
                        }
                        prev_t = t;
                        t += step;
                    }
                }
                if let Some(t_hit) = found {
                    let p = add(origin, scale(camera.forward, t_hit));
                    // Flag hits on the volume boundary (scan cut planes).
                    let vx = crate::affine::apply(&volume.world_to_voxel, p);
                    let margin = 1.2;
                    if vx[0] < margin
                        || vx[1] < margin
                        || vx[2] < margin
                        || vx[0] > nx as f64 - 1.0 - margin
                        || vx[1] > ny as f64 - 1.0 - margin
                        || vx[2] > nz as f64 - 1.0 - margin
                    {
                        row_boundary[px] = true;
                    }
                    // Gradient normal via central differences (world axes).
                    let d = (min_sp * 0.75).max(0.3);
                    let gx = volume.sample_world([p[0] + d, p[1], p[2]])
                        - volume.sample_world([p[0] - d, p[1], p[2]]);
                    let gy = volume.sample_world([p[0], p[1] + d, p[2]])
                        - volume.sample_world([p[0], p[1] - d, p[2]]);
                    let gz = volume.sample_world([p[0], p[1], p[2] + d])
                        - volume.sample_world([p[0], p[1], p[2] - d]);
                    let mut n = normalize([-(gx as f64), -(gy as f64), -(gz as f64)]);
                    if dot(n, camera.forward) > 0.0 {
                        n = scale(n, -1.0);
                    }
                    let ndl = dot(n, light).max(0.0) as f32;
                    let half = normalize(sub(light, camera.forward));
                    let spec = (dot(n, half).max(0.0) as f32).powf(24.0) * 0.25;
                    let ambient = 0.22f32;
                    let idx = px * 3;
                    for c in 0..3 {
                        let val = (skin[c] * (ambient + 0.85 * ndl) + spec).clamp(0.0, 1.0);
                        row_rgb[idx + c] = (val * 255.0) as u8;
                    }
                    row_pos[idx] = p[0] as f32;
                    row_pos[idx + 1] = p[1] as f32;
                    row_pos[idx + 2] = p[2] as f32;
                    row_hit[px] = true;
                } else {
                    // dark background
                    let idx = px * 3;
                    row_rgb[idx] = 12;
                    row_rgb[idx + 1] = 12;
                    row_rgb[idx + 2] = 14;
                }
            }
            (row_rgb, row_pos, row_hit, row_boundary)
        })
        .collect();

    for (py, (row_rgb, row_pos, row_hit, row_boundary)) in results.into_iter().enumerate() {
        rgb[py * size * 3..(py + 1) * size * 3].copy_from_slice(&row_rgb);
        pos[py * size * 3..(py + 1) * size * 3].copy_from_slice(&row_pos);
        hit[py * size..(py + 1) * size].copy_from_slice(&row_hit);
        boundary[py * size..(py + 1) * size].copy_from_slice(&row_boundary);
    }

    RenderView {
        camera,
        rgb,
        pos,
        hit,
        boundary,
        threshold,
    }
}
