// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared inference engine for RadVeil's built-in 3D U-Nets.
//!
//! Both brain-extraction networks ([`crate::synthstrip`] and
//! [`crate::rvbrain`]) are the same kind of graph: 3x3x3 "same"
//! convolutions with LeakyReLU, 2x max pooling, 2x nearest upsampling and
//! skip concatenations, with no normalisation layers. This module holds the
//! pieces they share.
//!
//! Activations are channels-last with a one-voxel zero border, so a "same"
//! convolution is 27 shifted GEMMs straight over the buffer: no im2col
//! copy, and a channel concatenation is expressed by passing several input
//! tensors and slicing the weight matrix, so skips are never materialised.

use rayon::prelude::*;

/// One 3x3x3 convolution: `cout` filters over `cin` input channels.
#[derive(Debug)]
pub(crate) struct Layer {
    pub(crate) cin: usize,
    pub(crate) cout: usize,
    /// [tap][cin][cout], tap = (kd*3 + kh)*3 + kw.
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
}

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Round an f32 to bfloat16 (ties to even): the top 16 bits of the f32,
/// which keeps the exponent range and costs a few instructions.
#[cfg(feature = "synthstrip")]
#[inline]
pub(crate) fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[cfg(feature = "synthstrip")]
#[inline]
pub(crate) fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
/// Channels-last activation over a `d x h x w` grid, stored with a one-voxel
/// zero border: storage index of interior voxel (i, j, k) is
/// `((i + 1) * (h + 2) + (j + 1)) * (w + 2) + (k + 1)`.
pub(crate) struct Tensor {
    pub(crate) d: usize,
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) c: usize,
    pub(crate) data: Vec<f32>,
}

impl Tensor {
    pub(crate) fn zeros(d: usize, h: usize, w: usize, c: usize) -> Tensor {
        Tensor {
            d,
            h,
            w,
            c,
            data: vec![0.0; (d + 2) * (h + 2) * (w + 2) * c],
        }
    }

    #[inline]
    pub(crate) fn at(&self, i: usize, j: usize, k: usize) -> usize {
        ((i + 1) * (self.h + 2) + (j + 1)) * (self.w + 2) + (k + 1)
    }
}

#[cfg(feature = "synthstrip")]
pub(crate) const LEAK: f32 = 0.2;

/// A layer's activation: the LeakyReLU negative slope, or `None` for a
/// linear layer (the networks' output heads predict a signed distance).
pub(crate) type Act = Option<f32>;

/// Output positions per GEMM block (rows of the flattened volume).
const BLOCK: usize = 4096;

/// 3x3x3 "same" convolution of the channel-concatenation of `inputs`.
pub(crate) fn conv(inputs: &[&Tensor], layer: &Layer, act: Act) -> Tensor {
    if layer.cout == 1 && inputs.len() == 1 {
        return conv_to_one(inputs[0], layer, act);
    }
    let t0 = inputs[0];
    let (d, h, w) = (t0.d, t0.h, t0.w);
    debug_assert_eq!(inputs.iter().map(|t| t.c).sum::<usize>(), layer.cin);
    let cout = layer.cout;
    let mut out = Tensor::zeros(d, h, w, cout);
    let (hp, wp) = (h + 2, w + 2);
    let first = out.at(0, 0, 0);
    let last = out.at(d - 1, h - 1, w - 1) + 1;
    let offs: Vec<isize> = (0..27)
        .map(|t| {
            let (a, b, c) = (t / 9, (t / 3) % 3, t % 3);
            ((a as isize - 1) * hp as isize + (b as isize - 1)) * wp as isize + (c as isize - 1)
        })
        .collect();
    let region = &mut out.data[first * cout..last * cout];
    region
        .par_chunks_mut(BLOCK * cout)
        .enumerate()
        .for_each(|(bi, chunk)| {
            let p0 = first + bi * BLOCK;
            let rows = chunk.len() / cout;
            let mut ci0 = 0usize;
            for t in inputs {
                let cin = t.c;
                for (tap, &off) in offs.iter().enumerate() {
                    let a_start = ((p0 as isize + off) as usize) * cin;
                    let wblk = &layer.w[(tap * layer.cin + ci0) * cout..];
                    // SAFETY: rows of A lie inside the padded input (the
                    // first interior voxel minus the largest offset is 0,
                    // the last plus it is the final voxel); B holds cin rows
                    // of cout; C is this chunk.
                    unsafe {
                        matrixmultiply::sgemm(
                            rows,
                            cin,
                            cout,
                            1.0,
                            t.data.as_ptr().add(a_start),
                            cin as isize,
                            1,
                            wblk.as_ptr(),
                            cout as isize,
                            1,
                            1.0,
                            chunk.as_mut_ptr(),
                            cout as isize,
                            1,
                        );
                    }
                }
                ci0 += cin;
            }
            // Bias, activation, and re-zero border positions (the flat
            // range also covers the padding columns/rows between
            // interior rows).
            for r in 0..rows {
                let p = p0 + r;
                let k = p % wp;
                let j = (p / wp) % hp;
                let o = &mut chunk[r * cout..(r + 1) * cout];
                if k == 0 || k == wp - 1 || j == 0 || j == hp - 1 {
                    o.fill(0.0);
                    continue;
                }
                for (v, &b) in o.iter_mut().zip(&layer.b) {
                    let x = *v + b;
                    *v = match act {
                        Some(slope) if x < 0.0 => x * slope,
                        _ => x,
                    };
                }
            }
        });
    out
}

/// Single-output-channel convolution (the network's last layer): a GEMM
/// with one output column is inefficient, so this is a direct dot product
/// over the 27 taps x `cin` channels per voxel.
fn conv_to_one(t: &Tensor, layer: &Layer, act: Act) -> Tensor {
    let (d, h, w, cin) = (t.d, t.h, t.w, t.c);
    let mut out = Tensor::zeros(d, h, w, 1);
    let (hp, wp) = (h + 2, w + 2);
    let offs: Vec<isize> = (0..27)
        .map(|k| {
            let (a, b, c) = (k / 9, (k / 3) % 3, k % 3);
            ((a as isize - 1) * hp as isize + (b as isize - 1)) * wp as isize + (c as isize - 1)
        })
        .collect();
    let bias = layer.b[0];
    let plane = hp * wp;
    out.data
        .par_chunks_mut(plane)
        .enumerate()
        .for_each(|(pi, dst)| {
            if pi == 0 || pi == d + 1 {
                return;
            }
            for j in 1..=h {
                for k in 1..=w {
                    let p = (pi * hp + j) * wp + k;
                    let mut acc = bias;
                    for (tap, &off) in offs.iter().enumerate() {
                        let x = &t.data[((p as isize + off) as usize) * cin..][..cin];
                        let wt = &layer.w[tap * cin..][..cin];
                        acc += x.iter().zip(wt).map(|(a, b)| a * b).sum::<f32>();
                    }
                    dst[j * wp + k] = match act {
                        Some(slope) if acc < 0.0 => acc * slope,
                        _ => acc,
                    };
                }
            }
        });
    out
}

pub(crate) fn maxpool2(t: &Tensor) -> Tensor {
    let (d, h, w, c) = (t.d / 2, t.h / 2, t.w / 2, t.c);
    let mut out = Tensor::zeros(d, h, w, c);
    let stride_i = (h + 2) * (w + 2) * c;
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
                        let s = t.at(2 * i + a, 2 * j + b, 2 * k + e) * c;
                        for (x, &y) in dst.iter_mut().zip(&t.data[s..s + c]) {
                            *x = x.max(y);
                        }
                    }
                }
            }
        });
    out
}

pub(crate) fn upsample2(t: &Tensor) -> Tensor {
    let (d, h, w, c) = (t.d * 2, t.h * 2, t.w * 2, t.c);
    let mut out = Tensor::zeros(d, h, w, c);
    let stride_i = (h + 2) * (w + 2) * c;
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
                    let s = t.at(i / 2, j / 2, k / 2) * c;
                    plane[o..o + c].copy_from_slice(&t.data[s..s + c]);
                }
            }
        });
    out
}
