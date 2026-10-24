// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! RadVeil's own brain-extraction network: a small U-Net trained here, run
//! from a `.rvb` file.
//!
//! Same idea as [`crate::synthstrip`] — predict the signed distance (mm) to
//! the brain boundary and threshold it — but the network is chosen to be
//! cheap on a CPU rather than to be SynthStrip:
//!
//! * it runs on a **coarser isotropic grid** (2 mm by default, against
//!   SynthStrip's 1 mm), which is where nearly all of the speedup comes
//!   from: eight times fewer voxels at every level. The prediction is a
//!   smooth distance field, so it survives being resampled back onto the
//!   input grid much better than a probability map would;
//! * it is **narrower and shallower**, and has no normalisation layers, so
//!   the weights fit in a couple of megabytes;
//! * the predicted field is resampled back with cubic (Catmull-Rom)
//!   interpolation by default, which costs little and recovers most of the
//!   boundary detail a coarse grid gives up.
//!
//! Everything downstream is shared with SynthStrip: threshold at
//! `border_mm`, keep the largest connected component, fill holes.
//!
//! The architecture is *not* hardcoded here. A `.rvb` file carries the grid
//! spacing, the level count and every layer's shape, so retraining at a
//! different width, depth or resolution is an export away and needs no
//! change to this file. See `training/README.md`.
//!
//! # Format (`.rvb`, little endian)
//!
//! ```text
//! magic     b"RVB1"
//! u32       flags (0)
//! f32       voxel_mm        grid spacing the network runs at
//! u16       grid_multiple   conformed shape is a multiple of this (2^levels)
//! u16       grid_min        per-axis clamp, voxels
//! u16       grid_max
//! u8        levels          encoder levels == decoder levels
//! u8        tail            convolutions after the last decoder level
//! f32       leak            LeakyReLU negative slope
//! f32       border_mm       default mask threshold
//! u32       n_layers        == 4 * levels + tail
//! layers    u32 cin, u32 cout,
//!           f16 w[27][cin][cout]   tap = (kd * 3 + kh) * 3 + kw
//!           f32 b[cout]
//! ```
//!
//! Layers come in forward order: the two convolutions of encoder level 0,
//! ... of level `levels - 1`, then the two of each decoder level, then the
//! tail. Every layer has LeakyReLU except the last.

use crate::affine;
use crate::conform::{conform, ConformSpec, Conformed};
use crate::deskull::{BrainExtractor, Modality};
use crate::unet::{conv, f16_to_f32, maxpool2, upsample2, Act, Layer, Tensor};
use crate::volume::Volume;
use ndarray::Array3;
use std::path::Path;
use std::sync::OnceLock;

/// A model compiled into the binary. `None` for now: the crate ships no
/// `.rvb`, so [`RvBrain::load`] is the only way in. To embed a trained model
/// once one is good enough, drop it in `models/rvbrain.rvb` and make this
///
/// ```ignore
/// const EMBEDDED: Option<&[u8]> = Some(include_bytes!("../models/rvbrain.rvb"));
/// ```
///
/// (`models/*.rvb` is already in the crate's `include` list). Everything
/// else, including [`RvBrain::embedded`], then works unchanged.
const EMBEDDED: Option<&[u8]> = None;

/// Header of a `.rvb` file: everything about the graph except the weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arch {
    pub voxel_mm: f64,
    pub grid_multiple: usize,
    pub grid_min: usize,
    pub grid_max: usize,
    pub levels: usize,
    pub tail: usize,
    pub leak: f32,
    pub border_mm: f32,
}

/// A parsed `.rvb` model: architecture plus weights.
#[derive(Debug)]
pub struct Model {
    pub arch: Arch,
    layers: Vec<Layer>,
}

const HEADER: usize = 32;

impl Model {
    /// Parse a `.rvb` file's bytes.
    pub fn parse(bytes: &[u8]) -> Result<Model, String> {
        if bytes.len() < HEADER || &bytes[0..4] != b"RVB1" {
            return Err("not a RadVeil brain model (.rvb, magic RVB1)".into());
        }
        let u16at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]) as usize;
        let u32at = |o: usize| {
            u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize
        };
        let f32at =
            |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        let flags = u32at(4);
        if flags != 0 {
            return Err(format!("unknown .rvb flags {flags:#x}"));
        }
        let arch = Arch {
            voxel_mm: f32at(8) as f64,
            grid_multiple: u16at(12),
            grid_min: u16at(14),
            grid_max: u16at(16),
            levels: bytes[18] as usize,
            tail: bytes[19] as usize,
            leak: f32at(20),
            border_mm: f32at(24),
        };
        let n = u32at(28);
        if arch.levels == 0 || arch.tail == 0 {
            return Err("a .rvb model needs at least one level and one tail layer".into());
        }
        if arch.grid_multiple != 1 << arch.levels {
            return Err(format!(
                "grid multiple {} does not match {} pooling steps",
                arch.grid_multiple, arch.levels
            ));
        }
        if arch.grid_min == 0
            || arch.grid_min > arch.grid_max
            || !arch.grid_min.is_multiple_of(arch.grid_multiple)
            || !arch.grid_max.is_multiple_of(arch.grid_multiple)
        {
            return Err("a .rvb grid clamp must be an ordered pair of multiples".into());
        }
        if !(arch.voxel_mm.is_finite() && arch.voxel_mm > 0.0) {
            return Err(format!("bad .rvb voxel size {}", arch.voxel_mm));
        }
        if n != 4 * arch.levels + arch.tail {
            return Err(format!(
                "{n} layers, but {} levels and a {}-layer tail need {}",
                arch.levels,
                arch.tail,
                4 * arch.levels + arch.tail
            ));
        }
        let mut pos = HEADER;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            if bytes.len() < pos + 8 {
                return Err(format!("truncated .rvb at layer {i}"));
            }
            let (cin, cout) = (u32at(pos), u32at(pos + 4));
            pos += 8;
            if cin == 0 || cout == 0 {
                return Err(format!("layer {i} has a zero channel count"));
            }
            let wn = 27 * cin * cout;
            let w: Vec<f32> = bytes
                .get(pos..pos + wn * 2)
                .ok_or_else(|| format!("truncated .rvb weights at layer {i}"))?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            pos += wn * 2;
            let b: Vec<f32> = bytes
                .get(pos..pos + cout * 4)
                .ok_or_else(|| format!("truncated .rvb bias at layer {i}"))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            pos += cout * 4;
            layers.push(Layer { cin, cout, w, b });
        }
        check_topology(&arch, &layers)?;
        Ok(Model { arch, layers })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Model, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        Model::parse(&bytes)
    }

    /// The model compiled into this build, if any.
    pub fn embedded() -> Option<Result<&'static Model, String>> {
        static M: OnceLock<Result<Model, String>> = OnceLock::new();
        let bytes = EMBEDDED?;
        Some(
            M.get_or_init(|| Model::parse(bytes))
                .as_ref()
                .map_err(Clone::clone),
        )
    }

    /// Total number of weights and biases.
    pub fn parameters(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }
}

/// Check that the layer shapes really form the U-Net the header describes:
/// each convolution must consume what the previous one produced, plus the
/// skip that is concatenated at that level.
fn check_topology(arch: &Arch, layers: &[Layer]) -> Result<(), String> {
    let l = arch.levels;
    let ch: Vec<usize> = (0..l).map(|i| layers[2 * i + 1].cout).collect();
    let want = |i: usize, cin: usize| -> Result<(), String> {
        if layers[i].cin != cin {
            return Err(format!(
                "layer {i} takes {} channels, the graph gives it {cin}",
                layers[i].cin
            ));
        }
        Ok(())
    };
    for i in 0..l {
        want(2 * i, if i == 0 { 1 } else { ch[i - 1] })?;
        want(2 * i + 1, ch[i])?;
    }
    for i in 0..l {
        let base = 2 * l + 2 * i;
        let cin = if i == 0 {
            ch[l - 1]
        } else {
            layers[base - 1].cout + ch[l - i]
        };
        want(base, cin)?;
        want(base + 1, layers[base].cout)?;
    }
    let tail0 = 4 * l;
    want(tail0, layers[tail0 - 1].cout + ch[0])?;
    for i in 1..arch.tail {
        want(tail0 + i, layers[tail0 + i - 1].cout)?;
    }
    let last = layers.len() - 1;
    if layers[last].cout != 1 {
        return Err(format!(
            "the output layer produces {} channels, expected 1 (a distance in mm)",
            layers[last].cout
        ));
    }
    Ok(())
}

/// Run the U-Net over a `d x h x w` grid (each a multiple of
/// `arch.grid_multiple`). Returns the predicted distance in mm.
fn forward(m: &Model, input: &[f32], [d, h, w]: [usize; 3]) -> Result<Vec<f32>, String> {
    let mult = m.arch.grid_multiple;
    if !d.is_multiple_of(mult) || !h.is_multiple_of(mult) || !w.is_multiple_of(mult) {
        return Err(format!("input {d}x{h}x{w} is not a multiple of {mult}"));
    }
    let leak: Act = Some(m.arch.leak);
    let layers = &m.layers;
    let mut x = Tensor::zeros(d, h, w, 1);
    for i in 0..d {
        for j in 0..h {
            let o = x.at(i, j, 0);
            x.data[o..o + w].copy_from_slice(&input[(i * h + j) * w..(i * h + j + 1) * w]);
        }
    }
    // Encoder: two convolutions per level, then pool; keep each level's
    // output for the matching skip.
    let mut skips: Vec<Tensor> = Vec::with_capacity(m.arch.levels);
    let mut li = 0usize;
    for _ in 0..m.arch.levels {
        let a = conv(&[&x], &layers[li], leak);
        drop(x);
        let b = conv(&[&a], &layers[li + 1], leak);
        drop(a);
        li += 2;
        x = maxpool2(&b);
        skips.push(b);
    }
    // Decoder: each level takes the previous level upsampled, concatenated
    // with the matching skip. `cur` holds those inputs, so the concatenation
    // is only ever a pair of pointers handed to `conv`.
    let mut cur: Vec<Tensor> = vec![x];
    for _ in 0..m.arch.levels {
        let a = {
            let refs: Vec<&Tensor> = cur.iter().collect();
            conv(&refs, &layers[li], leak)
        };
        cur.clear();
        let b = conv(&[&a], &layers[li + 1], leak);
        drop(a);
        li += 2;
        let up = upsample2(&b);
        drop(b);
        cur = vec![up, skips.pop().expect("one skip per decoder level")];
    }
    // Tail, at the network's own resolution: the last layer has no
    // activation because its output is a signed distance.
    let mut y = {
        let refs: Vec<&Tensor> = cur.iter().collect();
        conv(
            &refs,
            &layers[li],
            if m.arch.tail > 1 { leak } else { None },
        )
    };
    drop(cur);
    li += 1;
    for i in 1..m.arch.tail {
        let last = i == m.arch.tail - 1;
        y = conv(&[&y], &layers[li], if last { None } else { leak });
        li += 1;
    }
    let mut out = vec![0.0f32; d * h * w];
    for i in 0..d {
        for j in 0..h {
            let o = y.at(i, j, 0);
            out[(i * h + j) * w..(i * h + j + 1) * w].copy_from_slice(&y.data[o..o + w]);
        }
    }
    Ok(out)
}

/// Settings for [`RvBrain`].
#[derive(Clone, Debug)]
pub struct RvBrainParams {
    /// Mask boundary: voxels with predicted distance < border (mm) are brain.
    /// `None` takes the value the model file was exported with.
    pub border_mm: Option<f32>,
    /// Resample the predicted distance back to the input grid with cubic
    /// (Catmull-Rom) interpolation instead of trilinear. The field is
    /// smooth, so this recovers boundary detail lost to the coarse grid at
    /// a cost that does not scale with the network.
    pub cubic: bool,
}

impl Default for RvBrainParams {
    fn default() -> Self {
        RvBrainParams {
            border_mm: None,
            cubic: true,
        }
    }
}

/// RadVeil's own brain extractor. Needs a `.rvb` model: either the one
/// compiled into this build or one loaded from a path.
pub struct RvBrain {
    pub model: ModelRef,
    pub params: RvBrainParams,
}

/// A model this extractor runs: borrowed from the build, or owned.
pub enum ModelRef {
    Embedded(&'static Model),
    Owned(Box<Model>),
}

impl std::ops::Deref for ModelRef {
    type Target = Model;
    fn deref(&self) -> &Model {
        match self {
            ModelRef::Embedded(m) => m,
            ModelRef::Owned(m) => m,
        }
    }
}

impl RvBrain {
    /// The model compiled into this build.
    pub fn embedded() -> Result<RvBrain, String> {
        let m = Model::embedded().ok_or(
            "this build has no embedded RadVeil brain model: pass one with \
             --model path/to.rvb (or set $RADVEIL_RVBRAIN_MODEL). \
             training/README.md covers training and embedding one.",
        )??;
        Ok(RvBrain {
            model: ModelRef::Embedded(m),
            params: RvBrainParams::default(),
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<RvBrain, String> {
        Ok(RvBrain {
            model: ModelRef::Owned(Box::new(Model::load(path)?)),
            params: RvBrainParams::default(),
        })
    }

    fn spec(&self) -> ConformSpec {
        let a = self.model.arch;
        ConformSpec {
            voxel_mm: a.voxel_mm,
            multiple: a.grid_multiple,
            min: a.grid_min,
            max: a.grid_max,
        }
    }

    /// Predicted signed distance to the brain boundary (mm, negative
    /// inside), resampled onto the volume's grid; 100 outside the network
    /// field of view.
    pub fn distance(&self, vol: &Volume) -> Result<Array3<f32>, String> {
        let c: Conformed = conform(vol, &self.spec());
        let sdt = forward(&self.model, &c.data, c.shape)?;
        let [d, h, w] = c.shape;
        let to_grid = affine::mul(&affine::invert(&c.affine), &vol.affine);
        let [nx, ny, nz] = vol.shape();
        let cubic = self.params.cubic;
        let mut out = Array3::<f32>::zeros((nx, ny, nz));
        ndarray::Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
            let g = affine::apply(&to_grid, [i as f64, j as f64, k as f64]);
            *o = if cubic {
                tricubic(&sdt, [d, h, w], g)
            } else {
                crate::conform::trilinear(&sdt, [d, h, w], g)
            }
            .unwrap_or(100.0);
        });
        Ok(out)
    }

    pub fn mask(&self, vol: &Volume) -> Result<Array3<u8>, String> {
        let border = self.params.border_mm.unwrap_or(self.model.arch.border_mm);
        let dist = self.distance(vol)?;
        let [nx, ny, nz] = vol.shape();
        let inside: Vec<bool> = dist.iter().map(|&s| s < border).collect();
        let brain = crate::deskull::largest_component(&inside, [nx, ny, nz]);
        let outside = crate::deskull::flood_from_border(&brain, [nx, ny, nz]);
        Ok(Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            let x = (i * ny + j) * nz + k;
            u8::from(brain[x] || !outside[x])
        }))
    }
}

/// Catmull-Rom cubic interpolation of a scalar field. Falls back to
/// trilinear within one voxel of the border, where the 4-tap stencil does
/// not fit.
fn tricubic(v: &[f32], [d, h, w]: [usize; 3], p: [f64; 3]) -> Option<f32> {
    let lo = [1.0, 1.0, 1.0];
    let hi = [
        d.saturating_sub(3) as f64,
        h.saturating_sub(3) as f64,
        w.saturating_sub(3) as f64,
    ];
    if (0..3).any(|a| p[a] < lo[a] || p[a] > hi[a]) {
        return crate::conform::trilinear(v, [d, h, w], p);
    }
    let base = [0, 1, 2].map(|a| p[a].floor() as usize);
    let frac = [0, 1, 2].map(|a| (p[a] - base[a] as f64) as f32);
    // Catmull-Rom basis at t for taps -1, 0, 1, 2.
    let taps = |t: f32| {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            0.5 * (-t3 + 2.0 * t2 - t),
            0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
            0.5 * (-3.0 * t3 + 4.0 * t2 + t),
            0.5 * (t3 - t2),
        ]
    };
    let (bx, by, bz) = (taps(frac[0]), taps(frac[1]), taps(frac[2]));
    let mut acc = 0.0f32;
    for (a, &wx) in bx.iter().enumerate() {
        let i = base[0] + a - 1;
        for (b, &wy) in by.iter().enumerate() {
            let j = base[1] + b - 1;
            let row = (i * h + j) * w + base[2] - 1;
            let mut s = 0.0f32;
            for (c, &wz) in bz.iter().enumerate() {
                s += wz * v[row + c];
            }
            acc += wx * wy * s;
        }
    }
    Some(acc)
}

impl BrainExtractor for RvBrain {
    fn name(&self) -> &str {
        "rvbrain"
    }

    fn extract(&self, volume: &Volume, _modality: Modality) -> Result<Array3<u8>, String> {
        self.mask(volume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::golden_data;

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn golden_model_parses() {
        let Some(model) = golden_data("rvbrain_golden.rvb") else {
            return;
        };
        let m = Model::parse(&model).unwrap();
        assert_eq!(m.arch.levels, 3);
        assert_eq!(m.arch.grid_multiple, 8);
        assert_eq!(m.arch.tail, 2);
        assert_eq!(m.arch.voxel_mm, 2.0);
        assert_eq!(m.parameters(), 15_658);
    }

    /// The Rust engine must reproduce the numpy reference implementation
    /// (`training/rvbrain/numpy_ref.py`) on a seeded random input. Both read
    /// the same float16 weights, so the only difference is the order of the
    /// float32 accumulations.
    #[test]
    fn matches_numpy_reference() {
        let (Some(model), Some(io)) = (
            golden_data("rvbrain_golden.rvb"),
            golden_data("rvbrain_golden.bin"),
        ) else {
            return;
        };
        let m = Model::parse(&model).unwrap();
        let io = floats(&io);
        let n = 32 * 32 * 32;
        let y = forward(&m, &io[..n], [32, 32, 32]).unwrap();
        let max_err = y
            .iter()
            .zip(&io[n..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = io[n..].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            max_err < 1e-4 * scale.max(1.0),
            "max abs error {max_err} over a range of {scale}"
        );
    }

    /// Build a `.rvb` with the given header and layer shapes (zero
    /// weights), to exercise the structural checks.
    fn synth(levels: u8, tail: u8, mult: u16, shapes: &[(u32, u32)]) -> Vec<u8> {
        let mut v = Vec::from(*b"RVB1");
        v.extend_from_slice(&0u32.to_le_bytes()); // flags
        v.extend_from_slice(&2.0f32.to_le_bytes()); // voxel_mm
        v.extend_from_slice(&mult.to_le_bytes());
        v.extend_from_slice(&mult.to_le_bytes()); // grid_min
        v.extend_from_slice(&(mult * 8).to_le_bytes()); // grid_max
        v.push(levels);
        v.push(tail);
        v.extend_from_slice(&0.2f32.to_le_bytes()); // leak
        v.extend_from_slice(&1.0f32.to_le_bytes()); // border_mm
        v.extend_from_slice(&(shapes.len() as u32).to_le_bytes());
        assert_eq!(v.len(), HEADER);
        for &(cin, cout) in shapes {
            v.extend_from_slice(&cin.to_le_bytes());
            v.extend_from_slice(&cout.to_le_bytes());
            v.extend(std::iter::repeat_n(
                0u8,
                27 * cin as usize * cout as usize * 2,
            ));
            v.extend(std::iter::repeat_n(0u8, cout as usize * 4));
        }
        v
    }

    /// A one-level model: enc(1->4, 4->4), dec(4->4, 4->4), tail(8->1).
    fn one_level() -> Vec<(u32, u32)> {
        vec![(1, 4), (4, 4), (4, 4), (4, 4), (8, 1)]
    }

    /// A file whose layer shapes do not form the U-Net its header describes
    /// must be rejected rather than read as garbage.
    #[test]
    fn topology_is_checked() {
        assert!(Model::parse(&synth(1, 1, 2, &one_level())).is_ok());

        // The tail's input is the last decoder output plus skip 0: 4 + 4.
        let mut bad = one_level();
        bad[4].0 = 4;
        let err = Model::parse(&synth(1, 1, 2, &bad)).unwrap_err();
        assert!(err.contains("channels"), "{err}");

        // The output head must produce exactly one channel: a distance.
        let mut bad = one_level();
        bad[4].1 = 2;
        let err = Model::parse(&synth(1, 1, 2, &bad)).unwrap_err();
        assert!(err.contains("expected 1"), "{err}");

        // The grid multiple has to be 2^levels.
        let err = Model::parse(&synth(1, 1, 8, &one_level())).unwrap_err();
        assert!(err.contains("pooling steps"), "{err}");

        // Layer count has to match the declared shape of the graph.
        let err = Model::parse(&synth(2, 1, 4, &one_level())).unwrap_err();
        assert!(err.contains("layers"), "{err}");

        // Truncation is an error, not a panic.
        let full = synth(1, 1, 2, &one_level());
        for cut in [HEADER + 4, full.len() - 1] {
            assert!(Model::parse(&full[..cut]).is_err());
        }
    }

    /// Cubic resampling of a linear field is exact, and agrees with
    /// trilinear; that is the property that makes it safe for a distance.
    #[test]
    fn tricubic_is_exact_on_a_linear_field() {
        let (d, h, w) = (8usize, 8, 8);
        let f = |i: f64, j: f64, k: f64| (2.0 * i - 3.0 * j + 0.5 * k + 1.0) as f32;
        let v: Vec<f32> = (0..d * h * w)
            .map(|x| f((x / (h * w)) as f64, ((x / w) % h) as f64, (x % w) as f64))
            .collect();
        for p in [[3.25, 4.5, 2.75], [1.0, 1.0, 1.0], [4.5, 3.5, 4.5]] {
            let got = tricubic(&v, [d, h, w], p).unwrap();
            assert!(
                (got - f(p[0], p[1], p[2])).abs() < 2e-3,
                "cubic at {p:?}: {got} != {}",
                f(p[0], p[1], p[2])
            );
        }
        // Outside the 4-tap stencil it must fall back, not panic.
        assert!(tricubic(&v, [d, h, w], [0.5, 0.5, 0.5]).is_some());
        assert!(tricubic(&v, [d, h, w], [7.5, 0.5, 0.5]).is_none());
    }
}
