// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! SynthStrip brain extraction in pure Rust.
//!
//! Runs the SynthStrip network (Hoopes et al., "SynthStrip: skull-stripping
//! for any brain image", NeuroImage 2022; weights by the SynthStrip authors,
//! used under the MIT license — see `models/LICENSE-synthstrip`) without
//! PyTorch or FreeSurfer:
//!
//! 1. Conform exactly like SynthStrip's surfa preprocessing ([`crate::conform`]):
//!    reorient the voxel axes to LIA, resize to 1 mm (nearest neighbour),
//!    crop to the bounding box of positive voxels, centre-pad/crop to a
//!    multiple of 64 per axis (192..=320).
//! 2. Shift to a zero minimum, divide by the 99th percentile, clip to 0..1.
//! 3. The 7-level 3D U-Net predicts a signed distance (mm) to the brain
//!    boundary; the mask is `distance < border` (1 mm), resampled back to
//!    the input grid (trilinear), largest component, holes filled.
//!
//! The network is 27 3x3x3 convolutions with LeakyReLU(0.2), max pooling
//! and nearest upsampling, run on the shared engine in [`crate::unet`]:
//! convolutions are 27 shifted GEMMs over a zero-padded channels-last
//! volume (no im2col buffer) and U-Net skip concatenations are folded into
//! split weight blocks. Here, on top of that, the two
//! full-resolution stages run in overlapping slabs and their results are
//! kept as bfloat16, so peak memory stays near one 16-channel
//! full-resolution tensor.

use crate::affine;
use crate::conform::{conform, ConformSpec};
use crate::deskull::{BrainExtractor, Modality};
use crate::unet::{
    bf16_to_f32, conv, f16_to_f32, f32_to_bf16, maxpool2, upsample2, Act, Layer, Tensor, LEAK,
};
use crate::volume::Volume;
use ndarray::Array3;
use rayon::prelude::*;
use std::sync::OnceLock;

const MODEL: &[u8] = include_bytes!("../models/synthstrip.1.rvss");

/// SynthStrip's grid: 1 mm, a multiple of 64 per axis (six pooling steps),
/// 192..=320 voxels.
const SPEC: ConformSpec = ConformSpec {
    voxel_mm: 1.0,
    multiple: 64,
    min: 192,
    max: 320,
};

/// Full-resolution activation kept between the encoder and the tail,
/// stored as bfloat16: it is the largest live buffer (16 channels over the
/// whole 1 mm grid) and is only read back slab by slab.
struct HalfTensor {
    d: usize,
    h: usize,
    w: usize,
    c: usize,
    data: Vec<u16>,
}

impl HalfTensor {
    fn zeros(d: usize, h: usize, w: usize, c: usize) -> HalfTensor {
        HalfTensor {
            d,
            h,
            w,
            c,
            data: vec![0; (d + 2) * (h + 2) * (w + 2) * c],
        }
    }

    fn from_tensor(t: &Tensor) -> HalfTensor {
        HalfTensor {
            d: t.d,
            h: t.h,
            w: t.w,
            c: t.c,
            data: t.data.par_iter().map(|&v| f32_to_bf16(v)).collect(),
        }
    }

    /// 2x max pooling straight out of bfloat16 storage (feeding the whole
    /// tensor through `planes` would rebuild it in float32).
    fn maxpool2(&self) -> Tensor {
        let (d, h, w, c) = (self.d / 2, self.h / 2, self.w / 2, self.c);
        let mut out = Tensor::zeros(d, h, w, c);
        let stride_i = (h + 2) * (w + 2) * c;
        let at = |i: usize, j: usize, k: usize| {
            ((i + 1) * (self.h + 2) + (j + 1)) * (self.w + 2) + (k + 1)
        };
        out.data
            .par_chunks_mut(stride_i)
            .enumerate()
            .for_each(|(pi, plane)| {
                if pi == 0 || pi == d + 1 {
                    return;
                }
                let i = pi - 1;
                for j in 0..h {
                    for k in 0..w {
                        let o = ((j + 1) * (w + 2) + (k + 1)) * c;
                        let dst = &mut plane[o..o + c];
                        dst.fill(f32::NEG_INFINITY);
                        for (a, b, e) in (0..8).map(|n| (n >> 2, (n >> 1) & 1, n & 1)) {
                            let s = at(2 * i + a, 2 * j + b, 2 * k + e) * c;
                            for (x, &y) in dst.iter_mut().zip(&self.data[s..s + c]) {
                                *x = x.max(bf16_to_f32(y));
                            }
                        }
                    }
                }
            });
        out
    }

    /// Like [`HalfTensor::planes`], but nearest-upsampled 2x: plane `i0 + p`
    /// of the doubled grid comes from plane `(i0 + p) / 2` here.
    fn planes_up(&self, i0: isize, n: usize) -> Tensor {
        let (h, w, c) = (self.h * 2, self.w * 2, self.c);
        let mut out = Tensor::zeros(n, h, w, c);
        let plane = (h + 2) * (w + 2) * c;
        out.data
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(pi, dst)| {
                let gi = i0 + pi as isize - 1;
                if pi == 0 || pi == n + 1 || gi < 0 || gi >= (self.d * 2) as isize {
                    return;
                }
                let si = gi as usize / 2;
                for j in 0..h {
                    for k in 0..w {
                        let o = ((j + 1) * (w + 2) + (k + 1)) * c;
                        let s = (((si + 1) * (self.h + 2) + (j / 2 + 1)) * (self.w + 2)
                            + (k / 2 + 1))
                            * c;
                        for (d, &v) in dst[o..o + c].iter_mut().zip(&self.data[s..s + c]) {
                            *d = bf16_to_f32(v);
                        }
                    }
                }
            });
        out
    }

    /// Planes `i0 .. i0 + n` as a float32 tensor (zero outside the volume).
    fn planes(&self, i0: isize, n: usize) -> Tensor {
        let (h, w, c) = (self.h, self.w, self.c);
        let mut out = Tensor::zeros(n, h, w, c);
        let plane = (h + 2) * (w + 2) * c;
        out.data
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(pi, dst)| {
                let gi = i0 + pi as isize - 1;
                if pi == 0 || pi == n + 1 || gi < 0 || gi >= self.d as isize {
                    return;
                }
                let s = ((gi + 1) as usize) * plane;
                for (d, &v) in dst.iter_mut().zip(&self.data[s..s + plane]) {
                    *d = bf16_to_f32(v);
                }
            });
        out
    }
}

fn parse(bytes: &[u8]) -> Result<Vec<Layer>, String> {
    let mut pos = 0usize;
    let mut take = |n: usize| -> Result<&[u8], String> {
        let s = bytes
            .get(pos..pos + n)
            .ok_or("truncated SynthStrip model")?;
        pos += n;
        Ok(s)
    };
    if take(4)? != b"RVS1" {
        return Err("not a SynthStrip .rvss model".into());
    }
    let u32le = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize;
    let n = u32le(take(4)?);
    let mut layers = Vec::with_capacity(n);
    for _ in 0..n {
        let cin = u32le(take(4)?);
        let cout = u32le(take(4)?);
        let w = take(27 * cin * cout * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let b = take(cout * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        layers.push(Layer { cin, cout, w, b });
    }
    if layers.len() != 27 {
        return Err(format!("expected 27 layers, got {}", layers.len()));
    }
    Ok(layers)
}

fn model() -> Result<&'static [Layer], String> {
    static M: OnceLock<Result<Vec<Layer>, String>> = OnceLock::new();
    M.get_or_init(|| parse(MODEL))
        .as_deref()
        .map_err(Clone::clone)
}

/// Output planes per full-resolution slab.
const SLAB: usize = 32;

/// Planes `i0 .. i0 + n` (along the first axis; may extend past the
/// volume, where they are zero) of `t`, as a standalone tensor. With
/// `up`, `t` is at half resolution and is nearest-upsampled on the fly.
fn planes(t: &Tensor, i0: isize, n: usize, up: bool) -> Tensor {
    let f = if up { 2 } else { 1 };
    let (d, h, w, c) = (t.d * f, t.h * f, t.w * f, t.c);
    let mut out = Tensor::zeros(n, h, w, c);
    let plane = (h + 2) * (w + 2) * c;
    out.data
        .par_chunks_mut(plane)
        .enumerate()
        .for_each(|(pi, dst)| {
            let gi = i0 + pi as isize - 1;
            if pi == 0 || pi == n + 1 || gi < 0 || gi >= d as isize {
                return;
            }
            let gi = gi as usize;
            if !up {
                let s = (gi + 1) * plane;
                dst.copy_from_slice(&t.data[s..s + plane]);
                return;
            }
            for j in 0..h {
                for k in 0..w {
                    let o = ((j + 1) * (w + 2) + (k + 1)) * c;
                    let s = t.at(gi / 2, j / 2, k / 2) * c;
                    dst[o..o + c].copy_from_slice(&t.data[s..s + c]);
                }
            }
        });
    out
}

/// Zero the planes of a slab tensor that lie outside the volume
/// (`0..d` globally), so later convolutions see "same" zero padding.
fn zero_outside(t: &mut Tensor, i0: isize, d: usize) {
    let plane = (t.h + 2) * (t.w + 2) * t.c;
    for pi in 1..=t.d {
        let gi = i0 + pi as isize - 1;
        if gi < 0 || gi >= d as isize {
            t.data[pi * plane..(pi + 1) * plane].fill(0.0);
        }
    }
}

/// Run a chain of full-resolution convolutions slab by slab so no
/// full-size intermediate (or concatenated/upsampled input) is ever
/// materialised. `inputs(i0, n)` returns the chain's input tensors for
/// global planes `i0 .. i0 + n`.
fn slab_loop(
    [d, _, _]: [usize; 3],
    inputs: &dyn Fn(isize, usize) -> Vec<Tensor>,
    chain: &[(&Layer, Act)],
    mut write: impl FnMut(usize, usize, &Tensor),
) {
    let halo = chain.len();
    let mut o0 = 0usize;
    while o0 < d {
        let n = SLAB.min(d - o0);
        let i0 = o0 as isize - halo as isize;
        let ins = inputs(i0, n + 2 * halo);
        let refs: Vec<&Tensor> = ins.iter().collect();
        let mut x = conv(&refs, chain[0].0, chain[0].1);
        drop(ins);
        zero_outside(&mut x, i0, d);
        for &(layer, act) in &chain[1..] {
            x = conv(&[&x], layer, act);
            zero_outside(&mut x, i0, d);
        }
        write(o0, n, &x);
        o0 += n;
    }
}

/// Slab-wise chain whose result is kept as bfloat16 (for the big
/// full-resolution and half-resolution activations).
fn conv_chain_slabs(
    dims: [usize; 3],
    inputs: &dyn Fn(isize, usize) -> Vec<Tensor>,
    chain: &[(&Layer, Act)],
) -> HalfTensor {
    let [d, h, w] = dims;
    let halo = chain.len();
    let cout = chain.last().unwrap().0.cout;
    let mut out = HalfTensor::zeros(d, h, w, cout);
    let plane = (h + 2) * (w + 2) * cout;
    slab_loop(dims, inputs, chain, |o0, n, x| {
        let src = &x.data[(halo + 1) * plane..(halo + 1 + n) * plane];
        out.data[(o0 + 1) * plane..(o0 + 1 + n) * plane]
            .par_iter_mut()
            .zip(src.par_iter())
            .for_each(|(d, &v)| *d = f32_to_bf16(v));
    });
    out
}

/// Slab-wise chain whose result stays float32: used for the single-channel
/// output head, where bfloat16 would coarsen the predicted distance.
fn conv_chain_slabs_f32(
    dims: [usize; 3],
    inputs: &dyn Fn(isize, usize) -> Vec<Tensor>,
    chain: &[(&Layer, Act)],
) -> Tensor {
    let [d, h, w] = dims;
    let halo = chain.len();
    let cout = chain.last().unwrap().0.cout;
    let mut out = Tensor::zeros(d, h, w, cout);
    let plane = (h + 2) * (w + 2) * cout;
    slab_loop(dims, inputs, chain, |o0, n, x| {
        out.data[(o0 + 1) * plane..(o0 + 1 + n) * plane]
            .copy_from_slice(&x.data[(halo + 1) * plane..(halo + 1 + n) * plane]);
    });
    out
}

/// Run the U-Net on a `d x h x w` input (each a multiple of 64, C-order
/// with the last axis fastest). Returns the signed distance in mm.
fn forward(input: &[f32], [d, h, w]: [usize; 3], low_memory: bool) -> Result<Vec<f32>, String> {
    if d % 64 != 0 || h % 64 != 0 || w % 64 != 0 {
        return Err(format!(
            "SynthStrip input {d}x{h}x{w} is not a multiple of 64"
        ));
    }
    let layers = model()?;
    let mut x = Tensor::zeros(d, h, w, 1);
    for i in 0..d {
        for j in 0..h {
            let o = x.at(i, j, 0);
            x.data[o..o + w].copy_from_slice(&input[(i * h + j) * w..(i * h + j + 1) * w]);
        }
    }
    // Full-resolution encoder level, slab-wise; its output is the first
    // skip connection.
    let skip0 = conv_chain_slabs(
        [d, h, w],
        &|i0, n| vec![planes(&x, i0, n, false)],
        &[(&layers[0], Some(LEAK)), (&layers[1], Some(LEAK))],
    );
    drop(x);
    let mut x = skip0.maxpool2();
    let mut hist = Vec::with_capacity(5);
    let mut li = 2usize;
    for _ in 1..6 {
        let a = conv(&[&x], &layers[li], Some(LEAK));
        let b = conv(&[&a], &layers[li + 1], Some(LEAK));
        li += 2;
        x = maxpool2(&b);
        hist.push(b);
    }
    // Decoder: each level's output is upsampled and concatenated with the
    // matching skip (split weights, no copy). In low-memory mode the
    // half-resolution level (the largest intermediate) also runs in slabs,
    // upsampling per slab instead of building the full tensor: ~0.4 GB
    // less, ~20% slower.
    let levels = if low_memory { 5 } else { 6 };
    let mut pending: Option<(Tensor, Tensor)> = None;
    let mut last_half = None;
    for level in 0..levels {
        let a = match &pending {
            None => conv(&[&x], &layers[li], Some(LEAK)),
            Some((u, s)) => conv(&[u, s], &layers[li], Some(LEAK)),
        };
        let b = conv(&[&a], &layers[li + 1], Some(LEAK));
        li += 2;
        if level + 1 < levels {
            pending = Some((upsample2(&b), hist.pop().expect("skip")));
        } else if low_memory {
            // Leave `b` at quarter resolution: the slab pass below
            // upsamples it plane by plane.
            pending = Some((b, hist.pop().expect("skip")));
        } else {
            pending = None;
            last_half = Some(b);
        }
    }
    let half = match last_half {
        Some(b) => HalfTensor::from_tensor(&b),
        None => {
            let (coarse, skip1) = pending.expect("decoder ran");
            let out = conv_chain_slabs(
                [skip1.d, skip1.h, skip1.w],
                &|i0, n| vec![planes(&coarse, i0, n, true), planes(&skip1, i0, n, false)],
                &[(&layers[li], Some(LEAK)), (&layers[li + 1], Some(LEAK))],
            );
            li += 2;
            out
        }
    };
    // Full-resolution tail: upsample(half) ++ skip0 -> 3 convs, slab-wise.
    let y = conv_chain_slabs_f32(
        [d, h, w],
        &|i0, n| vec![half.planes_up(i0, n), skip0.planes(i0, n)],
        &[
            (&layers[li], Some(LEAK)),
            (&layers[li + 1], Some(LEAK)),
            (&layers[li + 2], None),
        ],
    );
    let mut out = vec![0.0f32; d * h * w];
    for i in 0..d {
        for j in 0..h {
            let o = y.at(i, j, 0);
            out[(i * h + j) * w..(i * h + j + 1) * w].copy_from_slice(&y.data[o..o + w]);
        }
    }
    Ok(out)
}

/// SynthStrip settings.
#[derive(Clone, Debug)]
pub struct SynthStripParams {
    /// Mask boundary: voxels with predicted distance < border (mm) are brain.
    pub border_mm: f32,
    /// Trade speed for peak memory (~0.4 GB less, ~20% slower): also run
    /// the half-resolution decoder level in slabs. Outputs are unchanged.
    pub low_memory: bool,
}

impl Default for SynthStripParams {
    fn default() -> Self {
        SynthStripParams {
            border_mm: 1.0,
            low_memory: false,
        }
    }
}

/// The SynthStrip brain extractor (any MRI contrast, CT, PET).
#[derive(Default)]
pub struct SynthStrip {
    pub params: SynthStripParams,
}

impl SynthStrip {
    /// Predicted signed distance to the brain boundary (mm, negative
    /// inside), resampled onto the volume's grid; 100 outside the network
    /// field of view.
    pub fn distance(&self, vol: &Volume) -> Result<Array3<f32>, String> {
        let c = conform(vol, &SPEC);
        let sdt = forward(&c.data, c.shape, self.params.low_memory)?;
        let [d, h, w] = c.shape;
        let to_grid = affine::mul(&affine::invert(&c.affine), &vol.affine);
        let [nx, ny, nz] = vol.shape();
        let mut out = Array3::<f32>::zeros((nx, ny, nz));
        ndarray::Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
            let g = affine::apply(&to_grid, [i as f64, j as f64, k as f64]);
            *o = crate::conform::trilinear(&sdt, [d, h, w], g).unwrap_or(100.0);
        });
        Ok(out)
    }

    pub fn mask(&self, vol: &Volume) -> Result<Array3<u8>, String> {
        let dist = self.distance(vol)?;
        let [nx, ny, nz] = vol.shape();
        let inside: Vec<bool> = dist.iter().map(|&s| s < self.params.border_mm).collect();
        let brain = crate::deskull::largest_component(&inside, [nx, ny, nz]);
        let outside = crate::deskull::flood_from_border(&brain, [nx, ny, nz]);
        Ok(Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            let x = (i * ny + j) * nz + k;
            u8::from(brain[x] || !outside[x])
        }))
    }
}
impl BrainExtractor for SynthStrip {
    fn name(&self) -> &str {
        "synthstrip"
    }

    fn extract(&self, volume: &Volume, _modality: Modality) -> Result<Array3<u8>, String> {
        self.mask(volume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_decoding() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333_251_95).abs() < 1e-7);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
    }

    #[test]
    fn model_parses() {
        let m = model().unwrap();
        assert_eq!(
            m.iter().map(|l| l.w.len() + l.b.len()).sum::<usize>(),
            2_566_561
        );
        assert_eq!((m[24].cin, m[24].cout), (48, 16));
    }

    /// Low-memory mode must not change the result.
    #[test]
    fn low_memory_matches() {
        let Some(bytes) = crate::golden_data("synthstrip_golden.bin") else {
            return;
        };
        let f: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n = 64 * 64 * 64;
        let a = forward(&f[..n], [64, 64, 64], false).unwrap();
        let b = forward(&f[..n], [64, 64, 64], true).unwrap();
        // Not bit-identical: the slab pass blocks the GEMMs differently.
        let max_err = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "low-memory mode differs by {max_err}");
    }

    /// Numerical parity with the PyTorch model (fp16-rounded weights) on a
    /// seeded random 64^3 input; see tools/export_synthstrip.py --golden.
    /// The predicted distance is in millimetres and thresholded at 1 mm,
    /// so the tolerance is set well below that: the bfloat16 activation
    /// store costs ~0.006 mm, which moves ~0.006% of mask voxels on real
    /// scans.
    #[test]
    fn matches_pytorch_golden() {
        let Some(bytes) = crate::golden_data("synthstrip_golden.bin") else {
            return;
        };
        let f: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let n = 64 * 64 * 64;
        let y = forward(&f[..n], [64, 64, 64], false).unwrap();
        let max_err = y
            .iter()
            .zip(&f[n..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.02, "max abs error {max_err}");
        // What matters is the thresholded decision.
        let flipped = y
            .iter()
            .zip(&f[n..])
            .filter(|(a, b)| (**a < 1.0) != (**b < 1.0))
            .count();
        assert!(
            flipped * 10000 < n,
            "{flipped} of {n} mask decisions differ"
        );
    }
}
