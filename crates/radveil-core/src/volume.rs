// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! NIfTI volume loading/saving with orientation handling.
//!
//! The voxel->world affine is resolved the standard NIfTI way (sform if
//! present, else qform, else pixdim fallback) and always maps into the
//! RAS+ world frame ("world" below). Data stored in any orientation
//! (LPS-ordered voxels, axial/sagittal/coronal, flipped axes...) is handled
//! transparently because all geometry passes through the affine.
//!
//! Saving preserves the original datatype, scl_slope/scl_inter scaling and
//! header fields; voxels never touched by the defacer round-trip bit-exact
//! for integer datatypes.

use crate::affine::{self, Mat4};
use ndarray::{Array3, ArrayD, Ix3};
use nifti::{
    object::NiftiObject, writer::WriterOptions, IntoNdArray, NiftiHeader, NiftiType, ReaderOptions,
};
use rayon::prelude::*;
use std::io::{Read, Write};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VolumeError {
    #[error("nifti error: {0}")]
    Nifti(#[from] nifti::error::NiftiError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("{0}")]
    Unsupported(String),
}

/// Lazily built rendering acceleration data (macro-grid of per-block
/// maxima for empty-space skipping). Invalidated whenever voxels change.
#[derive(Default)]
pub struct AccelCache {
    grid: std::sync::OnceLock<MacroGrid>,
    /// Memoized `percentile` results, keyed by the pct bits. Each render
    /// asks for p1 and p99; recomputing them cost ~3.5ms apiece.
    percentiles: std::sync::Mutex<Vec<(u64, f32)>>,
}

pub struct MacroGrid {
    /// Edge length of a cubic block, voxels.
    pub block: usize,
    /// Grid dimensions (ceil(n/block) per axis).
    pub dims: [usize; 3],
    /// Per-block maximum intensity.
    pub max: Vec<f32>,
}

impl AccelCache {
    pub fn invalidate(&mut self) {
        self.grid = std::sync::OnceLock::new();
        self.percentiles
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

pub struct Volume {
    /// Scaled intensities (scl_slope/scl_inter applied), shape (nx, ny, nz),
    /// index `[i, j, k]` = voxel (i, j, k).
    pub data: Array3<f32>,
    /// Voxel (i,j,k) -> world RAS+ (mm).
    pub affine: Mat4,
    /// World RAS+ (mm) -> voxel (i,j,k).
    pub world_to_voxel: Mat4,
    /// Voxel spacing in mm per voxel axis.
    pub spacing: [f64; 3],
    pub header: NiftiHeader,
    pub datatype: NiftiType,
    pub scl_slope: f32,
    pub scl_inter: f32,
    pub accel: AccelCache,
}

fn affine_from_header(h: &NiftiHeader) -> Mat4 {
    if h.sform_code > 0 {
        let sx = h.srow_x;
        let sy = h.srow_y;
        let sz = h.srow_z;
        return [
            [sx[0] as f64, sx[1] as f64, sx[2] as f64, sx[3] as f64],
            [sy[0] as f64, sy[1] as f64, sy[2] as f64, sy[3] as f64],
            [sz[0] as f64, sz[1] as f64, sz[2] as f64, sz[3] as f64],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }
    let dx = h.pixdim[1].abs() as f64;
    let dy = h.pixdim[2].abs() as f64;
    let dz = h.pixdim[3].abs() as f64;
    if h.qform_code > 0 {
        let b = h.quatern_b as f64;
        let c = h.quatern_c as f64;
        let d = h.quatern_d as f64;
        let a = (1.0 - b * b - c * c - d * d).max(0.0).sqrt();
        let qfac = if (h.pixdim[0] as f64) < 0.0 {
            -1.0
        } else {
            1.0
        };
        let r = [
            [
                a * a + b * b - c * c - d * d,
                2.0 * (b * c - a * d),
                2.0 * (b * d + a * c),
            ],
            [
                2.0 * (b * c + a * d),
                a * a + c * c - b * b - d * d,
                2.0 * (c * d - a * b),
            ],
            [
                2.0 * (b * d - a * c),
                2.0 * (c * d + a * b),
                a * a + d * d - b * b - c * c,
            ],
        ];
        return [
            [
                r[0][0] * dx,
                r[0][1] * dy,
                r[0][2] * dz * qfac,
                h.quatern_x as f64,
            ],
            [
                r[1][0] * dx,
                r[1][1] * dy,
                r[1][2] * dz * qfac,
                h.quatern_y as f64,
            ],
            [
                r[2][0] * dx,
                r[2][1] * dy,
                r[2][2] * dz * qfac,
                h.quatern_z as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }
    // Last resort: scale-only (treated as already RAS-ish).
    [
        [dx, 0.0, 0.0, 0.0],
        [0.0, dy, 0.0, 0.0],
        [0.0, 0.0, dz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

impl Volume {
    pub fn load(path: impl AsRef<Path>) -> Result<Volume, VolumeError> {
        Self::from_object(ReaderOptions::new().read_file(path.as_ref())?)
    }

    /// Parse a NIfTI-1 volume (.nii, optionally gzipped) from memory, e.g.
    /// an HTTP upload or an object-store blob. Same result as [`load`](Self::load).
    pub fn from_bytes(bytes: &[u8]) -> Result<Volume, VolumeError> {
        let raw;
        let bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(bytes).read_to_end(&mut out)?;
            raw = out;
            &raw[..]
        } else {
            bytes
        };
        Self::from_object(nifti::InMemNiftiObject::from_reader(bytes)?)
    }

    fn from_object(obj: nifti::InMemNiftiObject) -> Result<Volume, VolumeError> {
        let header = obj.header().clone();
        let datatype = header.data_type()?;
        let scl_slope = if header.scl_slope == 0.0 || !header.scl_slope.is_finite() {
            1.0
        } else {
            header.scl_slope
        };
        let scl_inter = if header.scl_inter.is_finite() {
            header.scl_inter
        } else {
            0.0
        };
        let arr: ArrayD<f32> = obj.into_volume().into_ndarray::<f32>()?;
        // Squeeze trailing singleton axes (e.g. 4D with a single timepoint).
        let mut arr = arr;
        while arr.ndim() > 3 {
            let last = arr.ndim() - 1;
            if arr.shape()[last] == 1 {
                arr = arr.index_axis_move(ndarray::Axis(last), 0);
            } else {
                return Err(VolumeError::Unsupported(format!(
                    "only 3D volumes are supported (got shape {:?}); split 4D series first",
                    arr.shape()
                )));
            }
        }
        if arr.ndim() < 3 {
            return Err(VolumeError::Unsupported(format!(
                "expected a 3D volume, got shape {:?}",
                arr.shape()
            )));
        }
        let data = arr.into_dimensionality::<Ix3>()?;
        let affine = affine_from_header(&header);
        let world_to_voxel = affine::invert(&affine);
        let spacing = [
            affine::norm([affine[0][0], affine[1][0], affine[2][0]]),
            affine::norm([affine[0][1], affine[1][1], affine[2][1]]),
            affine::norm([affine[0][2], affine[1][2], affine[2][2]]),
        ];
        Ok(Volume {
            data,
            affine,
            world_to_voxel,
            spacing,
            header,
            datatype,
            scl_slope,
            scl_inter,
            accel: AccelCache::default(),
        })
    }

    /// Build a volume from an in-memory array and voxel->world affine
    /// (RAS+, mm). This is the entry point for non-NIfTI sources: DICOM
    /// series, research raw formats, or arrays from other toolkits --
    /// anything that can produce voxels plus an affine. Saved output is
    /// float32 NIfTI with the given affine as sform.
    pub fn from_array(data: Array3<f32>, affine: Mat4) -> Volume {
        let world_to_voxel = affine::invert(&affine);
        let spacing = [
            affine::norm([affine[0][0], affine[1][0], affine[2][0]]),
            affine::norm([affine[0][1], affine[1][1], affine[2][1]]),
            affine::norm([affine[0][2], affine[1][2], affine[2][2]]),
        ];
        let mut header = NiftiHeader {
            sform_code: 1,
            scl_slope: 1.0,
            scl_inter: 0.0,
            ..NiftiHeader::default()
        };
        #[allow(clippy::needless_range_loop)]
        for c in 0..4 {
            header.srow_x[c] = affine[0][c] as f32;
            header.srow_y[c] = affine[1][c] as f32;
            header.srow_z[c] = affine[2][c] as f32;
        }
        header.pixdim = [
            1.0,
            spacing[0] as f32,
            spacing[1] as f32,
            spacing[2] as f32,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        Volume {
            data,
            affine,
            world_to_voxel,
            spacing,
            header,
            datatype: NiftiType::Float32,
            scl_slope: 1.0,
            scl_inter: 0.0,
            accel: AccelCache::default(),
        }
    }

    pub fn shape(&self) -> [usize; 3] {
        let s = self.data.shape();
        [s[0], s[1], s[2]]
    }

    /// Orientation codes of the voxel axes (e.g. RAS, LPS, PIR...).
    pub fn orientation(&self) -> String {
        affine::axcodes(&self.affine).iter().collect()
    }

    /// Trilinear sample at world position (mm, RAS). Outside the volume -> 0.
    #[inline]
    pub fn sample_world(&self, p: affine::Vec3) -> f32 {
        let v = affine::apply(&self.world_to_voxel, p);
        self.sample_voxel(v)
    }

    #[inline]
    pub fn sample_voxel(&self, v: affine::Vec3) -> f32 {
        let [nx, ny, nz] = self.shape();
        let x = v[0];
        let y = v[1];
        let z = v[2];
        if x < 0.0 || y < 0.0 || z < 0.0 {
            // allow the -0.5..0 border to sample as edge
            if x < -0.5 || y < -0.5 || z < -0.5 {
                return 0.0;
            }
        }
        let x0 = x.floor();
        let y0 = y.floor();
        let z0 = z.floor();
        let fx = (x - x0) as f32;
        let fy = (y - y0) as f32;
        let fz = (z - z0) as f32;
        let xi = x0 as isize;
        let yi = y0 as isize;
        let zi = z0 as isize;
        if xi < -1 || yi < -1 || zi < -1 {
            return 0.0;
        }
        if xi >= nx as isize || yi >= ny as isize || zi >= nz as isize {
            return 0.0;
        }
        if xi >= 0
            && yi >= 0
            && zi >= 0
            && xi + 1 < nx as isize
            && yi + 1 < ny as isize
            && zi + 1 < nz as isize
        {
            // Interior fast path: all 8 corners in bounds, read them
            // straight off the strides without per-corner checks.
            let st = self.data.strides();
            let base = xi * st[0] + yi * st[1] + zi * st[2];
            let p = self.data.as_ptr();
            // SAFETY: every offset below addresses a voxel with indices
            // in [0, n-1] on each axis, checked just above.
            let g = |o: isize| unsafe { *p.offset(base + o) };
            let (sx, sy, sz) = (st[0], st[1], st[2]);
            let c00 = g(0) + (g(sx) - g(0)) * fx;
            let c10 = g(sy) + (g(sx + sy) - g(sy)) * fx;
            let c01 = g(sz) + (g(sx + sz) - g(sz)) * fx;
            let c11 = g(sy + sz) + (g(sx + sy + sz) - g(sy + sz)) * fx;
            let c0 = c00 + (c10 - c00) * fy;
            let c1 = c01 + (c11 - c01) * fy;
            return c0 + (c1 - c0) * fz;
        }
        let get = |i: isize, j: isize, k: isize| -> f32 {
            if i < 0 || j < 0 || k < 0 || i >= nx as isize || j >= ny as isize || k >= nz as isize {
                0.0
            } else {
                // SAFETY: bounds checked above.
                unsafe { *self.data.uget([i as usize, j as usize, k as usize]) }
            }
        };
        let c000 = get(xi, yi, zi);
        let c100 = get(xi + 1, yi, zi);
        let c010 = get(xi, yi + 1, zi);
        let c110 = get(xi + 1, yi + 1, zi);
        let c001 = get(xi, yi, zi + 1);
        let c101 = get(xi + 1, yi, zi + 1);
        let c011 = get(xi, yi + 1, zi + 1);
        let c111 = get(xi + 1, yi + 1, zi + 1);
        let c00 = c000 + (c100 - c000) * fx;
        let c10 = c010 + (c110 - c010) * fx;
        let c01 = c001 + (c101 - c001) * fx;
        let c11 = c011 + (c111 - c011) * fx;
        let c0 = c00 + (c10 - c00) * fy;
        let c1 = c01 + (c11 - c01) * fy;
        c0 + (c1 - c0) * fz
    }

    /// Macro-grid of per-8^3-block maxima for empty-space skipping,
    /// built lazily on first use.
    pub fn macro_grid(&self) -> &MacroGrid {
        self.accel.grid.get_or_init(|| {
            const B: usize = 8;
            let [nx, ny, nz] = self.shape();
            let dims = [nx.div_ceil(B), ny.div_ceil(B), nz.div_ceil(B)];
            // One slab of blocks per i-block row, filled in parallel.
            let slab = dims[1] * dims[2];
            let mut max = vec![f32::NEG_INFINITY; dims[0] * slab];
            max.par_chunks_mut(slab).enumerate().for_each(|(bi, out)| {
                for i in bi * B..((bi + 1) * B).min(nx) {
                    for j in 0..ny {
                        let row = &mut out[(j / B) * dims[2]..(j / B + 1) * dims[2]];
                        for k in 0..nz {
                            // SAFETY: i < nx, j < ny, k < nz.
                            let v = unsafe { *self.data.uget([i, j, k]) };
                            let m = &mut row[k / B];
                            if v > *m {
                                *m = v;
                            }
                        }
                    }
                }
            });
            MacroGrid {
                block: B,
                dims,
                max,
            }
        })
    }

    /// Approximate intensity percentile (0..=100) over a subsample.
    pub fn percentile(&self, pct: f64) -> f32 {
        let key = pct.to_bits();
        let lock = || {
            self.accel
                .percentiles
                .lock()
                .unwrap_or_else(|e| e.into_inner())
        };
        if let Some(&(_, v)) = lock().iter().find(|(k, _)| *k == key) {
            return v;
        }
        let v = self.percentile_uncached(pct);
        lock().push((key, v));
        v
    }

    fn percentile_uncached(&self, pct: f64) -> f32 {
        let n = self.data.len();
        let stride = (n / 2_000_000).max(1);
        let mut vals: Vec<f32> = self
            .data
            .as_slice_memory_order()
            .map(|s| s.iter().step_by(stride).copied().collect())
            .unwrap_or_else(|| self.data.iter().step_by(stride).copied().collect());
        vals.retain(|v| v.is_finite());
        if vals.is_empty() {
            return 0.0;
        }
        let idx = ((pct / 100.0) * (vals.len() - 1) as f64).round() as usize;
        let idx = idx.min(vals.len() - 1);
        let (_, v, _) = vals.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
        *v
    }

    /// World-space corners of the voxel bounding box.
    pub fn world_bounds(&self) -> ([f64; 3], [f64; 3]) {
        let [nx, ny, nz] = self.shape();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &i in &[0.0, (nx - 1) as f64] {
            for &j in &[0.0, (ny - 1) as f64] {
                for &k in &[0.0, (nz - 1) as f64] {
                    let w = affine::apply(&self.affine, [i, j, k]);
                    for a in 0..3 {
                        lo[a] = lo[a].min(w[a]);
                        hi[a] = hi[a].max(w[a]);
                    }
                }
            }
        }
        (lo, hi)
    }

    /// Save with the original header, datatype and scaling. `path` may end in
    /// .nii or .nii.gz. Values are mapped back through the inverse of
    /// scl_slope/scl_inter, so untouched voxels round-trip exactly for
    /// integer datatypes.
    /// Save with the default gzip level 1: on int16 medical data it is
    /// ~5x faster than level 6 for ~3% larger files.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), VolumeError> {
        self.save_with_compression(path, 1)
    }

    /// Like [`save`](Self::save) with an explicit gzip level (0-9; ignored
    /// for uncompressed .nii output).
    pub fn save_with_compression(
        &self,
        path: impl AsRef<Path>,
        gzip_level: u32,
    ) -> Result<(), VolumeError> {
        let path = path.as_ref();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let gz = name.ends_with(".gz");
        let bytes = self.encode(path, gz.then_some(gzip_level))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Encode as NIfTI-1 bytes, gzipped when `gzip_level` is given. Same
    /// bytes as [`save_with_compression`](Self::save_with_compression)
    /// writes (a private temporary file is used internally, because the
    /// NIfTI writer only targets paths).
    pub fn to_bytes(&self, gzip_level: Option<u32>) -> Result<Vec<u8>, VolumeError> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("radveil-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let out = self.encode(&dir.join("volume.nii"), gzip_level);
        std::fs::remove_dir_all(&dir).ok();
        out
    }

    /// Serialize next to `path` (only its directory and stem are used for
    /// the temporary file) and return the final bytes.
    fn encode(&self, path: &Path, gzip_level: Option<u32>) -> Result<Vec<u8>, VolumeError> {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        // Write uncompressed first; nifti-rs resets scl_slope/inter on write,
        // so we patch those two header fields afterwards, then gzip ourselves.
        // NOTE: nifti-rs rewrites the last path extension, so the temp stem
        // must not contain extra dots.
        let stem: String = name
            .trim_end_matches(".gz")
            .trim_end_matches(".nii")
            .chars()
            .map(|c| if c == '.' { '_' } else { c })
            .collect();
        let tmp = path.with_file_name(format!(".{stem}_radveiltmp.nii"));
        let slope = self.scl_slope as f64;
        let inter = self.scl_inter as f64;
        let unscale = |v: f32| (v as f64 - inter) / slope;
        {
            let w = WriterOptions::new(&tmp).reference_header(&self.header);
            macro_rules! write_int {
                ($t:ty) => {{
                    let raw = self.data.mapv(|v| {
                        unscale(v).round().clamp(<$t>::MIN as f64, <$t>::MAX as f64) as $t
                    });
                    w.write_nifti(&raw)?
                }};
            }
            match self.datatype {
                NiftiType::Uint8 => write_int!(u8),
                NiftiType::Int8 => write_int!(i8),
                NiftiType::Uint16 => write_int!(u16),
                NiftiType::Int16 => write_int!(i16),
                NiftiType::Uint32 => write_int!(u32),
                NiftiType::Int32 => write_int!(i32),
                NiftiType::Uint64 => write_int!(u64),
                NiftiType::Int64 => write_int!(i64),
                NiftiType::Float32 => {
                    let raw = self.data.mapv(|v| unscale(v) as f32);
                    w.write_nifti(&raw)?
                }
                NiftiType::Float64 => {
                    let raw = self.data.mapv(unscale);
                    w.write_nifti(&raw)?
                }
                other => {
                    return Err(VolumeError::Unsupported(format!(
                        "unsupported datatype for writing: {:?}",
                        other
                    )))
                }
            }
        }
        // Patch scl_slope (offset 112) and scl_inter (offset 116) back in.
        let mut bytes = std::fs::read(&tmp)?;
        if bytes.len() >= 120 {
            let le = matches!(self.header.endianness, nifti::Endianness::Little);
            let put = |buf: &mut [u8], val: f32| {
                let b = if le {
                    val.to_le_bytes()
                } else {
                    val.to_be_bytes()
                };
                buf.copy_from_slice(&b);
            };
            put(&mut bytes[112..116], self.header.scl_slope);
            put(&mut bytes[116..120], self.header.scl_inter);
        }
        std::fs::remove_file(&tmp).ok();
        match gzip_level {
            Some(level) => {
                let mut enc = flate2::write::GzEncoder::new(
                    Vec::with_capacity(bytes.len() / 2),
                    flate2::Compression::new(level.min(9)),
                );
                enc.write_all(&bytes)?;
                Ok(enc.finish()?)
            }
            None => Ok(bytes),
        }
    }
}

impl Volume {
    /// Serialize the voxels into a copy of `original`: the bytes of the
    /// single-file NIfTI this volume was loaded from (gzipped or not).
    /// Header, extensions, datatype, byte order and scaling are kept
    /// byte-for-byte; only the voxel block is rewritten, so untouched
    /// integer voxels round-trip exactly. Returns uncompressed `.nii`
    /// bytes. Filesystem-free counterpart of [`save`](Self::save).
    pub fn encode_like(&self, original: &[u8]) -> Result<Vec<u8>, VolumeError> {
        let mut out = gunzip_if_needed(original)?;
        let h = &self.header;
        let le = matches!(h.endianness, nifti::Endianness::Little);
        let offset = h.vox_offset as usize;
        let width = match self.datatype {
            NiftiType::Uint8 | NiftiType::Int8 => 1,
            NiftiType::Uint16 | NiftiType::Int16 => 2,
            NiftiType::Uint32 | NiftiType::Int32 | NiftiType::Float32 => 4,
            NiftiType::Uint64 | NiftiType::Int64 | NiftiType::Float64 => 8,
            other => {
                return Err(VolumeError::Unsupported(format!(
                    "unsupported datatype for writing: {other:?}"
                )))
            }
        };
        let end = offset + self.data.len() * width;
        if end > out.len() {
            return Err(VolumeError::Unsupported(
                "original bytes do not match this volume".into(),
            ));
        }
        let slope = self.scl_slope as f64;
        let inter = self.scl_inter as f64;
        let unscale = |v: f32| (v as f64 - inter) / slope;
        // NIfTI stores x fastest: iterate the transposed view in logical order.
        let dst = out[offset..end].chunks_exact_mut(width);
        macro_rules! put {
            ($conv:expr) => {
                for (d, &v) in dst.zip(self.data.t().iter()) {
                    let b = $conv(v);
                    d.copy_from_slice(if le { &b.0[..] } else { &b.1[..] });
                }
            };
        }
        macro_rules! int {
            ($t:ty) => {
                put!(|v: f32| {
                    let x = unscale(v).round().clamp(<$t>::MIN as f64, <$t>::MAX as f64) as $t;
                    (x.to_le_bytes(), x.to_be_bytes())
                })
            };
        }
        match self.datatype {
            NiftiType::Uint8 => int!(u8),
            NiftiType::Int8 => int!(i8),
            NiftiType::Uint16 => int!(u16),
            NiftiType::Int16 => int!(i16),
            NiftiType::Uint32 => int!(u32),
            NiftiType::Int32 => int!(i32),
            NiftiType::Uint64 => int!(u64),
            NiftiType::Int64 => int!(i64),
            NiftiType::Float32 => put!(|v: f32| {
                let x = unscale(v) as f32;
                (x.to_le_bytes(), x.to_be_bytes())
            }),
            NiftiType::Float64 => put!(|v: f32| {
                let x = unscale(v);
                (x.to_le_bytes(), x.to_be_bytes())
            }),
            _ => unreachable!(),
        }
        Ok(out)
    }
}

/// Read a possibly-gzipped nifti file fully into memory (for tests/diffing).
pub fn read_raw(path: impl AsRef<Path>) -> Result<Vec<u8>, VolumeError> {
    gunzip_if_needed(&std::fs::read(path.as_ref())?)
}

fn gunzip_if_needed(bytes: &[u8]) -> Result<Vec<u8>, VolumeError> {
    if bytes.len() > 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(bytes).read_to_end(&mut out)?;
        Ok(out)
    } else {
        Ok(bytes.to_vec())
    }
}
