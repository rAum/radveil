// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU compute kernels for the blur/adaptive defacing paths.
//!
//! Design: every hot inner loop is a contiguous element-wise pass over
//! slices, so LLVM auto-vectorizes it on any target (NEON is baseline on
//! aarch64, SSE2 on x86-64), and the leaf kernels are additionally
//! compiled in several SIMD variants with runtime dispatch via
//! `multiversion` — AVX2+FMA is selected automatically on x86-64 CPUs
//! that have it. No unsafe intrinsics; fully portable.
//!
//! The box-filter running sum is serial *along* a line, so vectorization
//! happens *across* lines: passes are laid out so the innermost loop
//! walks the contiguous last axis while the recurrence advances along the
//! blur axis.

use multiversion::multiversion;
use rayon::prelude::*;

/// One recurrence step of a box filter for a whole row of independent
/// running sums: acc += add - sub; out = acc * norm.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1"))]
fn row_box_step(acc: &mut [f32], add: &[f32], sub: &[f32], out: &mut [f32], norm: f32) {
    for (((a, &ad), &su), o) in acc
        .iter_mut()
        .zip(add.iter())
        .zip(sub.iter())
        .zip(out.iter_mut())
    {
        *a += ad - su;
        *o = *a * norm;
    }
}

#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1"))]
fn row_add(acc: &mut [f32], row: &[f32]) {
    for (a, &v) in acc.iter_mut().zip(row.iter()) {
        *a += v;
    }
}

/// Interior gradient-magnitude row: central differences along all three
/// axes; dz uses the row itself shifted by one.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1"))]
#[allow(clippy::too_many_arguments)]
fn row_grad_interior(
    out: &mut [f32],
    xp: &[f32],
    xm: &[f32],
    yp: &[f32],
    ym: &[f32],
    row: &[f32],
    inv2d: [f32; 3],
) {
    let n = out.len();
    // k runs over 1..n-1 of the original row; slices are pre-offset.
    for k in 0..n {
        let dx = (xp[k] - xm[k]) * inv2d[0];
        let dy = (yp[k] - ym[k]) * inv2d[1];
        let dz = (row[k + 2] - row[k]) * inv2d[2];
        out[k] = (dx * dx + dy * dy + dz * dz).sqrt();
    }
}

/// out = out * (1 - w) + heavy * w
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse4.1"))]
fn row_mix(out: &mut [f32], heavy: &[f32], w: &[f32]) {
    for ((o, &h), &wv) in out.iter_mut().zip(heavy.iter()).zip(w.iter()) {
        *o += (h - *o) * wv;
    }
}

/// Scalar box pass along a contiguous line (recurrence is serial here).
fn line_pass(dst: &mut [f32], src: &[f32], r: usize, norm: f32) {
    let n = dst.len();
    let mut acc = (r + 1) as f32 * src[0];
    for v in src.iter().take(r.min(n)) {
        acc += *v;
    }
    for i in 0..n {
        acc += src[(i + r).min(n - 1)];
        acc -= if i > r { src[i - r - 1] } else { src[0] };
        dst[i] = acc * norm;
    }
}

/// Shared-pointer wrapper for disjoint strided writes from parallel tasks.
/// SAFETY: users must guarantee tasks write non-overlapping index ranges.
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// One box-filter pass of radius `r` along `axis` over a C-order
/// (nx, ny, nz) volume. `src` is scratch and receives a copy of `data`.
pub(crate) fn box_pass_axis(
    data: &mut [f32],
    src: &mut Vec<f32>,
    dims: [usize; 3],
    axis: usize,
    r: usize,
) {
    let [nx, ny, nz] = dims;
    let n_axis = dims[axis];
    if r == 0 || n_axis == 0 {
        return;
    }
    let norm = 1.0 / (2 * r + 1) as f32;
    let nynz = ny * nz;
    if r >= n_axis {
        // Degenerate: window covers the whole extent everywhere (with
        // clamped edges the result is not exactly the mean, but this only
        // occurs when the kernel dwarfs the volume; the mean is fine).
        mean_fill_axis(data, dims, axis);
        return;
    }
    src.clear();
    src.extend_from_slice(data);
    let src = &src[..];

    match axis {
        // Blur along y: per x-plane, rows are contiguous z-lines; the
        // recurrence advances j, the inner loop vectorizes over z.
        1 => {
            data.par_chunks_mut(nynz)
                .zip(src.par_chunks(nynz))
                .for_each(|(dplane, splane)| {
                    let mut acc = vec![0.0f32; nz];
                    for (a, &v) in acc.iter_mut().zip(&splane[..nz]) {
                        *a = (r + 1) as f32 * v;
                    }
                    for j in 0..r {
                        row_add(&mut acc, &splane[j.min(ny - 1) * nz..][..nz]);
                    }
                    for j in 0..ny {
                        let add = &splane[(j + r).min(ny - 1) * nz..][..nz];
                        let subi = if j > r { j - r - 1 } else { 0 };
                        let sub = &splane[subi * nz..][..nz];
                        row_box_step(&mut acc, add, sub, &mut dplane[j * nz..][..nz], norm);
                    }
                });
        }
        // Blur along z (the contiguous axis): serial recurrence per line.
        2 => {
            data.par_chunks_mut(nynz)
                .zip(src.par_chunks(nynz))
                .for_each(|(dplane, splane)| {
                    for j in 0..ny {
                        line_pass(
                            &mut dplane[j * nz..][..nz],
                            &splane[j * nz..][..nz],
                            r,
                            norm,
                        );
                    }
                });
        }
        // Blur along x: split the (y,z) plane into contiguous column
        // blocks; each task owns its block's running sums across all x.
        _ => {
            let nblocks = (rayon::current_num_threads() * 4).max(1);
            let block = nynz.div_ceil(nblocks).max(64);
            let dptr = SendPtr(data.as_mut_ptr());
            let dptr = &dptr;
            (0..nynz.div_ceil(block)).into_par_iter().for_each(|b| {
                let start = b * block;
                let width = block.min(nynz - start);
                let mut acc = vec![0.0f32; width];
                for (a, &v) in acc.iter_mut().zip(&src[start..start + width]) {
                    *a = (r + 1) as f32 * v;
                }
                for i in 0..r {
                    row_add(&mut acc, &src[i.min(nx - 1) * nynz + start..][..width]);
                }
                for i in 0..nx {
                    let add = &src[(i + r).min(nx - 1) * nynz + start..][..width];
                    let subi = if i > r { i - r - 1 } else { 0 };
                    let sub = &src[subi * nynz + start..][..width];
                    // SAFETY: blocks cover disjoint [start, start+width)
                    // column ranges, so writes never overlap across tasks.
                    let out = unsafe {
                        std::slice::from_raw_parts_mut(dptr.0.add(i * nynz + start), width)
                    };
                    row_box_step(&mut acc, add, sub, out, norm);
                }
            });
        }
    }
}

fn mean_fill_axis(data: &mut [f32], dims: [usize; 3], axis: usize) {
    let [nx, ny, nz] = dims;
    let nynz = ny * nz;
    match axis {
        0 => {
            let mut mean = vec![0.0f32; nynz];
            for i in 0..nx {
                row_add(&mut mean, &data[i * nynz..][..nynz]);
            }
            let inv = 1.0 / nx as f32;
            for m in mean.iter_mut() {
                *m *= inv;
            }
            for i in 0..nx {
                data[i * nynz..][..nynz].copy_from_slice(&mean);
            }
        }
        1 => {
            for plane in data.chunks_mut(nynz) {
                let mut mean = vec![0.0f32; nz];
                for j in 0..ny {
                    row_add(&mut mean, &plane[j * nz..][..nz]);
                }
                let inv = 1.0 / ny as f32;
                for m in mean.iter_mut() {
                    *m *= inv;
                }
                for j in 0..ny {
                    plane[j * nz..][..nz].copy_from_slice(&mean);
                }
            }
        }
        _ => {
            for line in data.chunks_mut(nz) {
                let mean = line.iter().sum::<f32>() / nz as f32;
                line.fill(mean);
            }
        }
    }
}

/// Gradient magnitude of a C-order (nx, ny, nz) field, spacing in mm.
pub(crate) fn gradient_mag(field: &[f32], dims: [usize; 3], spacing: [f64; 3], out: &mut [f32]) {
    let [nx, ny, nz] = dims;
    let nynz = ny * nz;
    let inv2d = [
        (0.5 / spacing[0]) as f32,
        (0.5 / spacing[1]) as f32,
        (0.5 / spacing[2]) as f32,
    ];
    out.par_chunks_mut(nynz)
        .enumerate()
        .for_each(|(i, oplane)| {
            let xp = &field[(i + 1).min(nx - 1) * nynz..][..nynz];
            let xm = &field[i.saturating_sub(1) * nynz..][..nynz];
            let plane = &field[i * nynz..][..nynz];
            for j in 0..ny {
                let row = &plane[j * nz..][..nz];
                let yp = &plane[(j + 1).min(ny - 1) * nz..][..nz];
                let ym = &plane[j.saturating_sub(1) * nz..][..nz];
                let xpr = &xp[j * nz..][..nz];
                let xmr = &xm[j * nz..][..nz];
                let orow = &mut oplane[j * nz..][..nz];
                if nz >= 3 {
                    let (head, rest) = orow.split_at_mut(1);
                    let (mid, tail) = rest.split_at_mut(nz - 2);
                    row_grad_interior(
                        mid,
                        &xpr[1..nz - 1],
                        &xmr[1..nz - 1],
                        &yp[1..nz - 1],
                        &ym[1..nz - 1],
                        row,
                        inv2d,
                    );
                    // clamped edges, scalar
                    for (k, o) in [(0usize, head), (nz - 1, tail)] {
                        let dz = (row[(k + 1).min(nz - 1)] - row[k.saturating_sub(1)]) * inv2d[2];
                        let dx = (xpr[k] - xmr[k]) * inv2d[0];
                        let dy = (yp[k] - ym[k]) * inv2d[1];
                        o[0] = (dx * dx + dy * dy + dz * dz).sqrt();
                    }
                } else {
                    for k in 0..nz {
                        let dz = (row[(k + 1).min(nz - 1)] - row[k.saturating_sub(1)]) * inv2d[2];
                        let dx = (xpr[k] - xmr[k]) * inv2d[0];
                        let dy = (yp[k] - ym[k]) * inv2d[1];
                        orow[k] = (dx * dx + dy * dy + dz * dz).sqrt();
                    }
                }
            }
        });
}

/// out = out*(1-w) + heavy*w over whole buffers, in parallel.
pub(crate) fn mix_buffers(out: &mut [f32], heavy: &[f32], w: &[f32]) {
    const CHUNK: usize = 1 << 15;
    out.par_chunks_mut(CHUNK)
        .zip(heavy.par_chunks(CHUNK))
        .zip(w.par_chunks(CHUNK))
        .for_each(|((o, h), wv)| row_mix(o, h, wv));
}

/// Clamp-normalize in place: v = min(v / scale, 1).
pub(crate) fn clamp_scale(buf: &mut [f32], scale: f32) {
    let inv = 1.0 / scale;
    buf.par_chunks_mut(1 << 15).for_each(|c| {
        for v in c.iter_mut() {
            *v = (*v * inv).min(1.0);
        }
    });
}
