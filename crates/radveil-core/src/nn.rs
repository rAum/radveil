// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal inference engine for the embedded face networks.
//!
//! Executes `.rvnn` files (see `tools/export_rvnn.py`): flat, pre-resolved
//! op lists for small NHWC convolutional networks (MediaPipe BlazeFace and
//! Face Mesh). It supports exactly what those networks use (convolution,
//! depthwise convolution, residual add, max-pool, channel padding,
//! concatenation of output heads, logistic) with bias and ReLU/PReLU fused
//! into the producing op. Point-wise and strided convolutions run as
//! single-precision GEMMs through `matrixmultiply` (runtime-dispatched
//! AVX2/FMA, AVX-512 or NEON kernels); depthwise convolutions and the
//! element-wise passes are hand-written loops compiled per SIMD target with
//! `multiversion`. Activations live in a per-network arena whose buffers
//! are reused by liveness, allocated once per [`Workspace`].

use multiversion::multiversion;

/// Upper bound on one tensor's element count (256 Mi floats = 1 GiB).
const MAX_TENSOR_ELEMS: usize = 1 << 28;

const OP_CONV: u8 = 1;
const OP_DWCONV: u8 = 2;
const OP_ADD: u8 = 3;
const OP_MAXPOOL: u8 = 4;
const OP_PADC: u8 = 5;
const OP_CONCAT: u8 = 6;
const OP_LOGISTIC: u8 = 7;
const OP_ALIAS: u8 = 8;

#[derive(Clone, Debug)]
enum Act {
    None,
    Relu,
    Prelu(Vec<f32>),
}

#[derive(Clone, Debug)]
struct ConvGeom {
    kh: usize,
    kw: usize,
    cin: usize,
    cout: usize,
    sh: usize,
    sw: usize,
    pt: usize,
    pl: usize,
}

#[derive(Clone, Debug)]
enum Op {
    /// Dense convolution. `w` is the GEMM right-hand side:
    /// [(ky, kx, ci)][co], row-major. `res` is a residual tensor added
    /// before the activation (a following Add fused at load time).
    Conv {
        i: usize,
        o: usize,
        g: ConvGeom,
        w: Vec<f32>,
        b: Vec<f32>,
        act: Act,
        res: Option<usize>,
    },
    /// Depthwise convolution (multiplier 1); `w` is [ky][kx][c]. For
    /// stride 1, `tiled` holds each tap's weights repeated across an
    /// output row ([tap][ox][c]) and `btiled` the bias likewise, so a
    /// whole row is one contiguous multiply-add per tap.
    DwConv {
        i: usize,
        o: usize,
        g: ConvGeom,
        w: Vec<f32>,
        b: Vec<f32>,
        act: Act,
        tiled: Vec<f32>,
        btiled: Vec<f32>,
    },
    Add {
        a: usize,
        b: usize,
        o: usize,
        act: Act,
    },
    MaxPool {
        i: usize,
        o: usize,
        k: usize,
        s: usize,
        pt: usize,
        pl: usize,
    },
    PadC {
        i: usize,
        o: usize,
        before: usize,
        after: usize,
    },
    Concat {
        ins: Vec<usize>,
        o: usize,
    },
    Logistic {
        i: usize,
        o: usize,
    },
}

/// A loaded network. Immutable and shareable across threads; run it with
/// a [`Workspace`].
#[derive(Debug)]
pub struct Net {
    shapes: Vec<[usize; 3]>,
    input: usize,
    outputs: Vec<usize>,
    ops: Vec<Op>,
    /// tensor -> arena slot
    slot: Vec<usize>,
    slot_len: Vec<usize>,
    im2col_len: usize,
}

/// Mutable activation buffers for one [`Net`] (one per thread).
#[derive(Debug)]
pub struct Workspace {
    slots: Vec<Vec<f32>>,
    im2col: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Parsing

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl Reader<'_> {
    fn bytes(&mut self, n: usize) -> Result<&[u8], String> {
        let s = self
            .b
            .get(self.p..self.p + n)
            .ok_or("truncated network file")?;
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.bytes(1)?[0])
    }
    fn u32(&mut self) -> Result<usize, String> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    }
    fn f16s(&mut self) -> Result<Vec<f32>, String> {
        let n = self.u32()?;
        let b = self.bytes(2 * n)?;
        Ok(b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: normalize
            let mut e = 113u32; // 127 - 15 + 1
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (31, 0) => sign | 0x7f80_0000,
        (31, _) => sign | 0x7fc0_0000,
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// The hand-written kernel set in use (`"neon"`, `"avx2+fma"` or a
/// portable fallback), for logs and bug reports.
pub fn simd_level() -> &'static str {
    simd::level()
}

impl Net {
    /// Parse an `.rvnn` network.
    pub fn from_bytes(bytes: &[u8]) -> Result<Net, String> {
        let mut r = Reader { b: bytes, p: 0 };
        if r.bytes(4)? != b"RVNN" || r.u32()? != 1 {
            return Err("not an RVNN v1 network".into());
        }
        let n_t = r.u32()?;
        if n_t > 100_000 {
            return Err("implausible tensor count".into());
        }
        let mut shapes = Vec::with_capacity(n_t);
        for _ in 0..n_t {
            let s = [r.u32()?, r.u32()?, r.u32()?];
            // Refuse absurd sizes before anything is allocated from them.
            let numel = s.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
            if !numel.is_some_and(|n| (1..=MAX_TENSOR_ELEMS).contains(&n)) {
                return Err(format!("invalid tensor shape {s:?}"));
            }
            shapes.push(s);
        }
        let input = r.u32()?;
        let n_out = r.u32()?;
        let outputs = (0..n_out).map(|_| r.u32()).collect::<Result<Vec<_>, _>>()?;
        let n_ops = r.u32()?;
        if n_ops > 100_000 {
            return Err("implausible op count".into());
        }
        let mut ops = Vec::with_capacity(n_ops);
        // alias[t] = tensor whose storage t shares
        let mut alias: Vec<usize> = (0..n_t).collect();
        let act = |r: &mut Reader, code: u8| -> Result<Act, String> {
            Ok(match code {
                0 => Act::None,
                1 => Act::Relu,
                2 => Act::Prelu(r.f16s()?),
                c => return Err(format!("unknown activation {c}")),
            })
        };
        for _ in 0..n_ops {
            match r.u8()? {
                k @ (OP_CONV | OP_DWCONV) => {
                    let (i, o) = (r.u32()?, r.u32()?);
                    let g = ConvGeom {
                        kh: r.u32()?,
                        kw: r.u32()?,
                        cin: r.u32()?,
                        cout: r.u32()?,
                        sh: r.u32()?,
                        sw: r.u32()?,
                        pt: r.u32()?,
                        pl: r.u32()?,
                    };
                    let a = r.u8()?;
                    let w = r.f16s()?;
                    let b = r.f16s()?;
                    let act = act(&mut r, a)?;
                    if k == OP_CONV {
                        // [co][ky][kx][ci] -> [(ky,kx,ci)][co]
                        let kk = g.kh * g.kw * g.cin;
                        if w.len() != kk * g.cout {
                            return Err("conv weight size".into());
                        }
                        let mut t = vec![0f32; w.len()];
                        for co in 0..g.cout {
                            for j in 0..kk {
                                t[j * g.cout + co] = w[co * kk + j];
                            }
                        }
                        ops.push(Op::Conv {
                            i,
                            o,
                            g,
                            w: t,
                            b,
                            act,
                            res: None,
                        });
                    } else {
                        if w.len() != g.kh * g.kw * g.cout {
                            return Err("dwconv weight size".into());
                        }
                        let (mut tiled, mut btiled) = (Vec::new(), Vec::new());
                        if g.sh == 1 && g.sw == 1 {
                            let ow = shapes.get(o).ok_or("tensor id")?[1];
                            let c = g.cout;
                            tiled.reserve(g.kh * g.kw * ow * c);
                            for tap in w.chunks_exact(c) {
                                for _ in 0..ow {
                                    tiled.extend_from_slice(tap);
                                }
                            }
                            for _ in 0..ow {
                                btiled.extend_from_slice(&b);
                            }
                        }
                        ops.push(Op::DwConv {
                            i,
                            o,
                            g,
                            w,
                            b,
                            act,
                            tiled,
                            btiled,
                        });
                    }
                }
                OP_ADD => {
                    let (a, b, o) = (r.u32()?, r.u32()?, r.u32()?);
                    let code = r.u8()?;
                    let act = act(&mut r, code)?;
                    ops.push(Op::Add { a, b, o, act });
                }
                OP_MAXPOOL => ops.push(Op::MaxPool {
                    i: r.u32()?,
                    o: r.u32()?,
                    k: r.u32()?,
                    s: r.u32()?,
                    pt: r.u32()?,
                    pl: r.u32()?,
                }),
                OP_PADC => ops.push(Op::PadC {
                    i: r.u32()?,
                    o: r.u32()?,
                    before: r.u32()?,
                    after: r.u32()?,
                }),
                OP_CONCAT => {
                    let n = r.u32()?;
                    let ins = (0..n).map(|_| r.u32()).collect::<Result<Vec<_>, _>>()?;
                    ops.push(Op::Concat { ins, o: r.u32()? });
                }
                OP_LOGISTIC => ops.push(Op::Logistic {
                    i: r.u32()?,
                    o: r.u32()?,
                }),
                OP_ALIAS => {
                    let (i, o) = (r.u32()?, r.u32()?);
                    if i >= n_t || o >= n_t {
                        return Err("tensor id out of range".into());
                    }
                    alias[o] = alias[i];
                }
                k => return Err(format!("unknown op {k}")),
            }
        }
        if r.p != bytes.len() {
            return Err("trailing bytes in network file".into());
        }
        let in_range = |t: &usize| *t < n_t;
        if !in_range(&input)
            || !outputs.iter().all(in_range)
            || !ops
                .iter()
                .all(|op| op.inputs().iter().all(in_range) && in_range(&op.output()))
        {
            return Err("tensor id out of range".into());
        }

        let numel = |t: usize| shapes[t][0] * shapes[t][1] * shapes[t][2];
        for (t, &a) in alias.iter().enumerate() {
            if numel(t) != numel(a) {
                return Err("reshape changes element count".into());
            }
        }
        // Resolve aliases in ops/outputs, then allocate arena slots by
        // liveness (a buffer is reused once its last reader has run).
        let root = |t: usize| alias[t];
        for op in &mut ops {
            op.remap(&root);
        }
        let input = root(input);
        let outputs: Vec<usize> = outputs.into_iter().map(root).collect();
        // Every op must read tensors that exist by then (the input or an
        // earlier op's output, reshapes resolved), and every output must
        // be produced.
        let mut produced = vec![false; n_t];
        produced[input] = true;
        for (k, op) in ops.iter().enumerate() {
            if let Some(t) = op.inputs().into_iter().find(|&t| !produced[t]) {
                return Err(format!("op {k} reads tensor {t} before it is produced"));
            }
            produced[op.output()] = true;
        }
        if outputs.iter().any(|&t| !produced[t]) {
            return Err("network output is never produced".into());
        }
        let ops = fuse_residual_adds(ops, input, &outputs, n_t);
        let mut last_use = vec![0usize; n_t];
        for (k, op) in ops.iter().enumerate() {
            for t in op.inputs() {
                last_use[t] = k;
            }
        }
        for &t in &outputs {
            last_use[t] = usize::MAX;
        }
        let mut slot = vec![usize::MAX; n_t];
        let mut slot_len: Vec<usize> = Vec::new();
        let mut free: Vec<usize> = Vec::new();
        take_slot(numel(input), input, &mut slot, &mut slot_len, &mut free);
        let mut im2col_len = 0;
        for (k, op) in ops.iter().enumerate() {
            let o = op.output();
            if slot[o] == usize::MAX {
                take_slot(numel(o), o, &mut slot, &mut slot_len, &mut free);
            }
            if let Op::Conv { g, o, .. } = op {
                if !g.is_pointwise() {
                    let ow = shapes[*o][1];
                    im2col_len = im2col_len.max(ow * g.kh * g.kw * g.cin);
                }
            }
            for t in op.inputs() {
                if last_use[t] == k && !free.contains(&slot[t]) {
                    free.push(slot[t]);
                }
            }
        }
        let net = Net {
            shapes,
            input,
            outputs,
            ops,
            slot,
            slot_len,
            im2col_len,
        };
        net.check()?;
        Ok(net)
    }

    fn check(&self) -> Result<(), String> {
        for op in &self.ops {
            let sh = |t: usize| self.shapes[t];
            let ok = match op {
                Op::Conv {
                    i, o, g, b, res, ..
                } if res.is_some_and(|r| sh(r) != sh(*o)) => {
                    let _ = (i, g, b);
                    false
                }
                Op::Conv { i, o, g, b, .. } | Op::DwConv { i, o, g, b, .. } => {
                    let [h, w, c] = sh(*i);
                    let [oh, ow, oc] = sh(*o);
                    c == g.cin
                        && oc == g.cout
                        && b.len() == g.cout
                        && (oh - 1) * g.sh + g.kh <= h + g.pt + g.kh
                        && (ow - 1) * g.sw + g.kw <= w + g.pl + g.kw
                }
                Op::Add { a, b, o, .. } => sh(*a) == sh(*b) && sh(*a) == sh(*o),
                Op::MaxPool { i, o, .. } => sh(*i)[2] == sh(*o)[2],
                Op::PadC {
                    i,
                    o,
                    before,
                    after,
                } => {
                    let ([h, w, c], [oh, ow, oc]) = (sh(*i), sh(*o));
                    h == oh && w == ow && c + before + after == oc
                }
                Op::Concat { ins, o } => {
                    let n: usize = ins.iter().map(|&t| sh(t).iter().product::<usize>()).sum();
                    n == sh(*o).iter().product::<usize>()
                }
                Op::Logistic { i, o } => sh(*i) == sh(*o),
            };
            if !ok {
                return Err(format!("inconsistent shapes in {op:?}")
                    .chars()
                    .take(200)
                    .collect());
            }
            if let Op::Add {
                act: Act::Prelu(s),
                o,
                ..
            }
            | Op::Conv {
                act: Act::Prelu(s),
                o,
                ..
            }
            | Op::DwConv {
                act: Act::Prelu(s),
                o,
                ..
            } = op
            {
                if s.len() != sh(*o)[2] {
                    return Err("prelu slope count".into());
                }
            }
        }
        Ok(())
    }

    /// Input shape (h, w, c).
    pub fn input_shape(&self) -> [usize; 3] {
        self.shapes[self.input]
    }

    /// Bytes one [`Workspace`] allocates for this network.
    pub fn workspace_bytes(&self) -> usize {
        4 * (self.slot_len.iter().sum::<usize>() + self.im2col_len)
    }

    pub fn workspace(&self) -> Workspace {
        Workspace {
            slots: self.slot_len.iter().map(|&n| vec![0f32; n]).collect(),
            im2col: vec![0f32; self.im2col_len],
        }
    }

    /// Run the network on an NHWC input (h*w*c floats). Returns the
    /// outputs in the model's output order, flattened.
    pub fn run(&self, ws: &mut Workspace, input: &[f32]) -> Vec<Vec<f32>> {
        let n_in = self.input_shape().iter().product::<usize>();
        assert_eq!(input.len(), n_in, "input size");
        ws.slots[self.slot[self.input]][..n_in].copy_from_slice(input);
        for op in &self.ops {
            self.exec(op, ws);
        }
        self.outputs
            .iter()
            .map(|&t| {
                let n = self.shapes[t].iter().product::<usize>();
                ws.slots[self.slot[t]][..n].to_vec()
            })
            .collect()
    }

    /// Like [`run`](Self::run), returning per-op wall time (label, ms)
    /// instead of outputs. For profiling.
    #[doc(hidden)]
    pub fn profile(&self, ws: &mut Workspace, input: &[f32]) -> Vec<(String, f64)> {
        let n_in = self.input_shape().iter().product::<usize>();
        ws.slots[self.slot[self.input]][..n_in].copy_from_slice(input);
        self.ops
            .iter()
            .map(|op| {
                let t = std::time::Instant::now();
                self.exec(op, ws);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let o = self.shapes[op.output()];
                let label = match op {
                    Op::Conv { g, .. } => format!(
                        "conv{}x{}/{} {}->{} @{}x{}",
                        g.kh, g.kw, g.sh, g.cin, g.cout, o[0], o[1]
                    ),
                    Op::DwConv { g, .. } => {
                        format!("dw{}x{}/{} c{} @{}x{}", g.kh, g.kw, g.sh, g.cin, o[0], o[1])
                    }
                    Op::Add { .. } => format!("add c{} @{}x{}", o[2], o[0], o[1]),
                    Op::MaxPool { .. } => format!("maxpool c{} @{}x{}", o[2], o[0], o[1]),
                    Op::PadC { .. } => format!("padc c{} @{}x{}", o[2], o[0], o[1]),
                    Op::Concat { .. } => "concat".into(),
                    Op::Logistic { .. } => "logistic".into(),
                };
                (label, ms)
            })
            .collect()
    }

    fn exec(&self, op: &Op, ws: &mut Workspace) {
        let numel = |t: usize| self.shapes[t].iter().product::<usize>();
        let o = op.output();
        // Take the output buffer out of the arena so inputs can be
        // borrowed alongside it (slots never alias within one op).
        let mut out = std::mem::take(&mut ws.slots[self.slot[o]]);
        {
            let buf = |t: usize| &ws.slots[self.slot[t]][..numel(t)];
            let y = &mut out[..numel(o)];
            match op {
                Op::Conv {
                    i,
                    g,
                    w,
                    b,
                    act,
                    res,
                    ..
                } => {
                    let [h, wd, _] = self.shapes[*i];
                    let [oh, ow, _] = self.shapes[o];
                    let x = buf(*i);
                    if g.is_pointwise() {
                        matmul_res(x, oh * ow, g.cin, w, g.cout, b, res.map(buf), y);
                    } else {
                        // im2col one output row at a time: the unfolded
                        // block stays in L1 between unfolding and GEMM.
                        let k = g.kh * g.kw * g.cin;
                        let col = &mut ws.im2col[..ow * k];
                        for (oy, yrow) in y.chunks_exact_mut(ow * g.cout).enumerate() {
                            im2col_row(x, h, wd, g, oy, ow, col);
                            matmul(col, ow, k, w, g.cout, b, yrow);
                        }
                        if let Some(r) = res {
                            add_assign(y, buf(*r));
                        }
                    }
                    apply_act(y, g.cout, act);
                }
                Op::DwConv {
                    i,
                    g,
                    w,
                    b,
                    act,
                    tiled,
                    btiled,
                    ..
                } => {
                    let [h, wd, c] = self.shapes[*i];
                    let [oh, ow, _] = self.shapes[o];
                    if tiled.is_empty() {
                        dwconv(buf(*i), h, wd, c, g, w, b, oh, ow, y);
                    } else {
                        dwconv_s1(buf(*i), h, wd, c, g, tiled, btiled, oh, ow, y);
                    }
                    apply_act(y, c, act);
                }
                Op::Add { a, b, act, .. } => {
                    add(buf(*a), buf(*b), y);
                    apply_act(y, self.shapes[o][2], act);
                }
                Op::MaxPool {
                    i, k, s, pt, pl, ..
                } => {
                    let [h, wd, c] = self.shapes[*i];
                    let [oh, ow, _] = self.shapes[o];
                    maxpool(buf(*i), h, wd, c, *k, *s, *pt, *pl, oh, ow, y);
                }
                Op::PadC {
                    i, before, after, ..
                } => {
                    let c = self.shapes[*i][2];
                    let oc = c + before + after;
                    for (src, dst) in buf(*i).chunks_exact(c).zip(y.chunks_exact_mut(oc)) {
                        dst[..*before].fill(0.0);
                        dst[*before..*before + c].copy_from_slice(src);
                        dst[*before + c..].fill(0.0);
                    }
                }
                Op::Concat { ins, .. } => {
                    let mut at = 0;
                    for &t in ins {
                        let s = buf(t);
                        y[at..at + s.len()].copy_from_slice(s);
                        at += s.len();
                    }
                }
                Op::Logistic { i, .. } => {
                    for (d, &v) in y.iter_mut().zip(buf(*i)) {
                        *d = 1.0 / (1.0 + (-v).exp());
                    }
                }
            }
        }
        ws.slots[self.slot[o]] = out;
    }
}

/// Fuse `Add(conv_out, skip)` into the convolution when the conv output
/// feeds only that add and the skip tensor already exists when the conv
/// runs: the residual is then added while the conv output is still in
/// cache, and the add's separate pass over memory disappears.
fn fuse_residual_adds(mut ops: Vec<Op>, input: usize, outputs: &[usize], n_t: usize) -> Vec<Op> {
    let mut readers = vec![0usize; n_t];
    for op in &ops {
        for t in op.inputs() {
            readers[t] += 1;
        }
    }
    for &t in outputs {
        readers[t] += 1;
    }
    let mut producer = vec![usize::MAX; n_t];
    for (k, op) in ops.iter().enumerate() {
        producer[op.output()] = k;
    }
    let ready_before = |t: usize, k: usize, producer: &[usize]| t == input || producer[t] < k;
    let mut drop = vec![false; ops.len()];
    for k in 0..ops.len() {
        let Op::Add { a, b, o, act } = ops[k].clone() else {
            continue;
        };
        for (conv_out, skip) in [(a, b), (b, a)] {
            let j = producer[conv_out];
            if j == usize::MAX || readers[conv_out] != 1 || !ready_before(skip, j, &producer) {
                continue;
            }
            if let Op::Conv {
                o: co,
                act: cact @ Act::None,
                res: res @ None,
                ..
            } = &mut ops[j]
            {
                *co = o;
                *cact = act;
                *res = Some(skip);
                producer[o] = j;
                drop[k] = true;
                break;
            }
        }
    }
    ops.into_iter()
        .zip(drop)
        .filter_map(|(op, d)| (!d).then_some(op))
        .collect()
}

/// Assign tensor `t` (needing `need` floats) an arena slot: the smallest
/// free slot that fits, else the largest free one (grown), else a new one.
fn take_slot(
    need: usize,
    t: usize,
    slot: &mut [usize],
    slot_len: &mut Vec<usize>,
    free: &mut Vec<usize>,
) {
    let fits = free
        .iter()
        .enumerate()
        .filter(|(_, &s)| slot_len[s] >= need)
        .min_by_key(|(_, &s)| slot_len[s])
        .map(|(j, _)| j);
    let pick = fits.or_else(|| {
        free.iter()
            .enumerate()
            .max_by_key(|(_, &s)| slot_len[s])
            .map(|(j, _)| j)
    });
    let s = match pick {
        Some(j) => free.swap_remove(j),
        None => {
            slot_len.push(0);
            slot_len.len() - 1
        }
    };
    slot_len[s] = slot_len[s].max(need);
    slot[t] = s;
}

impl ConvGeom {
    fn is_pointwise(&self) -> bool {
        self.kh == 1 && self.kw == 1 && self.sh == 1 && self.sw == 1 && self.pt == 0 && self.pl == 0
    }
}

impl Op {
    fn output(&self) -> usize {
        match self {
            Op::Conv { o, .. }
            | Op::DwConv { o, .. }
            | Op::Add { o, .. }
            | Op::MaxPool { o, .. }
            | Op::PadC { o, .. }
            | Op::Concat { o, .. }
            | Op::Logistic { o, .. } => *o,
        }
    }
    fn inputs(&self) -> Vec<usize> {
        match self {
            Op::Conv { i, res, .. } => std::iter::once(*i).chain(*res).collect(),
            Op::DwConv { i, .. }
            | Op::MaxPool { i, .. }
            | Op::PadC { i, .. }
            | Op::Logistic { i, .. } => vec![*i],
            Op::Add { a, b, .. } => vec![*a, *b],
            Op::Concat { ins, .. } => ins.clone(),
        }
    }
    fn remap(&mut self, f: &dyn Fn(usize) -> usize) {
        match self {
            Op::Conv { i, o, res, .. } => {
                *i = f(*i);
                *o = f(*o);
                if let Some(r) = res {
                    *r = f(*r);
                }
            }
            Op::DwConv { i, o, .. }
            | Op::MaxPool { i, o, .. }
            | Op::PadC { i, o, .. }
            | Op::Logistic { i, o } => {
                *i = f(*i);
                *o = f(*o);
            }
            Op::Add { a, b, o, .. } => {
                *a = f(*a);
                *b = f(*b);
                *o = f(*o);
            }
            Op::Concat { ins, o } => {
                for t in ins.iter_mut() {
                    *t = f(*t);
                }
                *o = f(*o);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels

/// y[m, n] = a[m, k] @ w[k, n] + bias[n], picking the kernel by shape:
/// matrix-vector for a single row, a register-blocked kernel for narrow
/// outputs (the high-resolution layers of these networks have 8-32
/// channels, where a general GEMM spends its time packing), and
/// `matrixmultiply`'s SGEMM otherwise.
fn matmul(a: &[f32], m: usize, k: usize, w: &[f32], n: usize, bias: &[f32], y: &mut [f32]) {
    matmul_res(a, m, k, w, n, bias, None, y)
}

/// [`matmul`] plus an optional residual `res[m, n]` added to the result
/// (inside the SIMD kernels' accumulators when possible).
#[allow(clippy::too_many_arguments)]
fn matmul_res(
    a: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    bias: &[f32],
    res: Option<&[f32]>,
    y: &mut [f32],
) {
    if simd::enabled() && simd::narrow(a, k, w, n, bias, res, y) {
        return;
    }
    matmul_plain(a, m, k, w, n, bias, y);
    if let Some(r) = res {
        add_assign(y, r);
    }
}

fn matmul_plain(a: &[f32], m: usize, k: usize, w: &[f32], n: usize, bias: &[f32], y: &mut [f32]) {
    if m == 1 {
        y.copy_from_slice(bias);
        for (kk, &av) in a.iter().enumerate() {
            fma_acc(y, av, &w[kk * n..(kk + 1) * n]);
        }
        return;
    }
    match n {
        8 => narrow_8(a, k, w, bias, y),
        16 => narrow_16(a, k, w, bias, y),
        24 => narrow_24(a, k, w, bias, y),
        32 => narrow_32(a, k, w, bias, y),
        _ => conv_gemm(a, m, k, w, n, bias, y),
    }
}

/// Explicit-SIMD narrow-output kernels: each block of `R` rows keeps its
/// `R x N` outputs in vector registers while streaming over `k`, with one
/// weight-row load shared by the `R` rows. `narrow` returns false when no
/// kernel applies (unsupported width or CPU) so the caller falls back.
mod simd {
    /// Hand-written intrinsics can be switched off with `RADVEIL_SIMD=off`
    /// (an escape hatch, and how CI tests the portable path). Portable
    /// kernels are still auto-vectorized for the CPU found at run time.
    pub fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            !matches!(
                std::env::var("RADVEIL_SIMD").as_deref(),
                Ok("off" | "0" | "false" | "no")
            )
        })
    }

    /// Which hand-written kernel set is active on this machine.
    pub fn level() -> &'static str {
        if !enabled() {
            return "portable (RADVEIL_SIMD=off)";
        }
        #[cfg(target_arch = "aarch64")]
        return "neon";
        #[cfg(target_arch = "x86_64")]
        return if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            "avx2+fma"
        } else {
            "portable (no AVX2/FMA)"
        };
        #[allow(unreachable_code)]
        "portable"
    }

    #[cfg(target_arch = "aarch64")]
    pub fn narrow(
        a: &[f32],
        k: usize,
        w: &[f32],
        n: usize,
        b: &[f32],
        res: Option<&[f32]>,
        y: &mut [f32],
    ) -> bool {
        // SAFETY: NEON is part of the aarch64 baseline; the kernels only
        // touch a[..m*k], w[..k*n], b[..n], res/y[..m*n] (checked below).
        let m = a.len() / k;
        assert!(a.len() == m * k && y.len() >= m * n && w.len() >= k * n && b.len() >= n);
        assert!(res.is_none_or(|r| r.len() >= m * n));
        let r = res.map_or(std::ptr::null(), |r| r.as_ptr());
        unsafe {
            match n {
                8 => neon::<2, 4>(a, k, w, b, r, y),
                16 => neon::<4, 4>(a, k, w, b, r, y),
                24 => neon::<6, 2>(a, k, w, b, r, y),
                32 => neon::<8, 2>(a, k, w, b, r, y),
                _ => return false,
            }
        }
        true
    }

    /// V = N/4 NEON vectors per output row, R rows per block.
    #[cfg(target_arch = "aarch64")]
    unsafe fn neon<const V: usize, const R: usize>(
        a: &[f32],
        k: usize,
        w: &[f32],
        b: &[f32],
        res: *const f32,
        y: &mut [f32],
    ) {
        use std::arch::aarch64::*;
        let n = 4 * V;
        let m = a.len() / k;
        let (ap, wp, yp) = (a.as_ptr(), w.as_ptr(), y.as_mut_ptr());
        let mut bias = [vdupq_n_f32(0.0); V];
        for (j, v) in bias.iter_mut().enumerate() {
            *v = vld1q_f32(b.as_ptr().add(4 * j));
        }
        let block = |r: usize, rows: usize| {
            let mut acc = [bias; R];
            if !res.is_null() {
                for (i, acc) in acc.iter_mut().enumerate().take(rows) {
                    for (j, v) in acc.iter_mut().enumerate() {
                        *v = vaddq_f32(*v, vld1q_f32(res.add((r + i) * n + 4 * j)));
                    }
                }
            }
            for kk in 0..k {
                let mut wv = [vdupq_n_f32(0.0); V];
                for (j, v) in wv.iter_mut().enumerate() {
                    *v = vld1q_f32(wp.add(kk * n + 4 * j));
                }
                for (i, acc) in acc.iter_mut().enumerate().take(rows) {
                    let s = *ap.add((r + i) * k + kk);
                    for j in 0..V {
                        acc[j] = vfmaq_n_f32(acc[j], wv[j], s);
                    }
                }
            }
            for (i, acc) in acc.iter().enumerate().take(rows) {
                for (j, v) in acc.iter().enumerate() {
                    vst1q_f32(yp.add((r + i) * n + 4 * j), *v);
                }
            }
        };
        let full = m / R * R;
        for r in (0..full).step_by(R) {
            block(r, R);
        }
        if full < m {
            block(full, m - full);
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub fn narrow(
        a: &[f32],
        k: usize,
        w: &[f32],
        n: usize,
        b: &[f32],
        res: Option<&[f32]>,
        y: &mut [f32],
    ) -> bool {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return false;
        }
        let m = a.len() / k;
        assert!(a.len() == m * k && y.len() >= m * n && w.len() >= k * n && b.len() >= n);
        assert!(res.is_none_or(|r| r.len() >= m * n));
        let r = res.map_or(std::ptr::null(), |r| r.as_ptr());
        // SAFETY: AVX2+FMA checked above; bounds checked above.
        unsafe {
            match n {
                8 => avx::<1, 4>(a, k, w, b, r, y),
                16 => avx::<2, 4>(a, k, w, b, r, y),
                24 => avx::<3, 3>(a, k, w, b, r, y),
                32 => avx::<4, 2>(a, k, w, b, r, y),
                _ => return false,
            }
        }
        true
    }

    /// V = N/8 AVX vectors per output row, R rows per block.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx<const V: usize, const R: usize>(
        a: &[f32],
        k: usize,
        w: &[f32],
        b: &[f32],
        res: *const f32,
        y: &mut [f32],
    ) {
        use std::arch::x86_64::*;
        let n = 8 * V;
        let m = a.len() / k;
        let (ap, wp, yp) = (a.as_ptr(), w.as_ptr(), y.as_mut_ptr());
        let mut bias = [_mm256_setzero_ps(); V];
        for (j, v) in bias.iter_mut().enumerate() {
            *v = _mm256_loadu_ps(b.as_ptr().add(8 * j));
        }
        let full = m / R * R;
        let mut r = 0;
        while r < m {
            let rows = if r < full { R } else { m - full };
            let mut acc = [bias; R];
            if !res.is_null() {
                for (i, acc) in acc.iter_mut().enumerate().take(rows) {
                    for (j, v) in acc.iter_mut().enumerate() {
                        *v = _mm256_add_ps(*v, _mm256_loadu_ps(res.add((r + i) * n + 8 * j)));
                    }
                }
            }
            for kk in 0..k {
                let mut wv = [_mm256_setzero_ps(); V];
                for (j, v) in wv.iter_mut().enumerate() {
                    *v = _mm256_loadu_ps(wp.add(kk * n + 8 * j));
                }
                for (i, acc) in acc.iter_mut().enumerate().take(rows) {
                    let s = _mm256_set1_ps(*ap.add((r + i) * k + kk));
                    for j in 0..V {
                        acc[j] = _mm256_fmadd_ps(wv[j], s, acc[j]);
                    }
                }
            }
            for (i, acc) in acc.iter().enumerate().take(rows) {
                for (j, v) in acc.iter().enumerate() {
                    _mm256_storeu_ps(yp.add((r + i) * n + 8 * j), *v);
                }
            }
            r += rows;
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    pub fn narrow(
        _: &[f32],
        _: usize,
        _: &[f32],
        _: usize,
        _: &[f32],
        _: Option<&[f32]>,
        _: &mut [f32],
    ) -> bool {
        false
    }
}

macro_rules! narrow_kernel {
    ($name:ident, $n:literal) => {
        /// y = a @ w + b for a narrow output: whole output rows live in
        /// registers, four rows at a time so four independent
        /// multiply-add chains are in flight.
        #[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "aarch64+neon"))]
        fn $name(a: &[f32], k: usize, w: &[f32], bias: &[f32], y: &mut [f32]) {
            let w = &w[..k * $n];
            let bias: &[f32; $n] = bias[..$n].try_into().unwrap();
            let mut rows = a.chunks_exact(4 * k).zip(y.chunks_exact_mut(4 * $n));
            for (a4, y4) in &mut rows {
                let (r0, rest) = a4.split_at(k);
                let (r1, rest) = rest.split_at(k);
                let (r2, r3) = rest.split_at(k);
                let (mut c0, mut c1, mut c2, mut c3) = (*bias, *bias, *bias, *bias);
                for (kk, wr) in w.chunks_exact($n).enumerate() {
                    let wr: &[f32; $n] = wr.try_into().unwrap();
                    let (v0, v1, v2, v3) = (r0[kk], r1[kk], r2[kk], r3[kk]);
                    for j in 0..$n {
                        c0[j] += v0 * wr[j];
                        c1[j] += v1 * wr[j];
                        c2[j] += v2 * wr[j];
                        c3[j] += v3 * wr[j];
                    }
                }
                for (dst, acc) in y4.chunks_exact_mut($n).zip([c0, c1, c2, c3]) {
                    dst.copy_from_slice(&acc);
                }
            }
            let done = (a.len() / k) / 4 * 4;
            for (row, out) in a[done * k..]
                .chunks_exact(k)
                .zip(y[done * $n..].chunks_exact_mut($n))
            {
                let mut acc = *bias;
                for (&v, wr) in row.iter().zip(w.chunks_exact($n)) {
                    let wr: &[f32; $n] = wr.try_into().unwrap();
                    for j in 0..$n {
                        acc[j] += v * wr[j];
                    }
                }
                out.copy_from_slice(&acc);
            }
        }
    };
}
narrow_kernel!(narrow_8, 8);
narrow_kernel!(narrow_16, 16);
narrow_kernel!(narrow_24, 24);
narrow_kernel!(narrow_32, 32);

/// acc += s * x
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "aarch64+neon"))]
fn fma_acc(acc: &mut [f32], s: f32, x: &[f32]) {
    for (a, &v) in acc.iter_mut().zip(x) {
        *a += s * v;
    }
}

/// acc += x * w (element-wise)
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "aarch64+neon"))]
fn mul_acc(acc: &mut [f32], x: &[f32], w: &[f32]) {
    for ((a, &v), &k) in acc.iter_mut().zip(x).zip(w) {
        *a += v * k;
    }
}

/// Stride-1 depthwise convolution, one output row at a time: for each
/// tap, the valid span of the row is a single contiguous multiply-add of
/// the shifted input row with the pre-tiled tap weights.
#[allow(clippy::too_many_arguments)]
fn dwconv_s1(
    x: &[f32],
    h: usize,
    w: usize,
    c: usize,
    g: &ConvGeom,
    tiled: &[f32],
    btiled: &[f32],
    oh: usize,
    ow: usize,
    y: &mut [f32],
) {
    let row = ow * c;
    for (oy, yrow) in y.chunks_exact_mut(row).take(oh).enumerate() {
        yrow.copy_from_slice(btiled);
        for ky in 0..g.kh {
            let iy = (oy + ky) as isize - g.pt as isize;
            if iy < 0 || iy >= h as isize {
                continue;
            }
            let xrow = &x[iy as usize * w * c..][..w * c];
            for kx in 0..g.kw {
                let off = kx as isize - g.pl as isize;
                let ox0 = (-off).max(0) as usize;
                let ox1 = (ow as isize).min(w as isize - off).max(ox0 as isize) as usize;
                let n = (ox1 - ox0) * c;
                if n == 0 {
                    continue;
                }
                let ix0 = (ox0 as isize + off) as usize;
                let tap = &tiled[(ky * g.kw + kx) * row..][..row];
                mul_acc(
                    &mut yrow[ox0 * c..ox0 * c + n],
                    &xrow[ix0 * c..ix0 * c + n],
                    &tap[ox0 * c..ox0 * c + n],
                );
            }
        }
    }
}

/// y[m, n] = a[m, k] @ w[k, n] + bias[n] via SGEMM.
fn conv_gemm(a: &[f32], m: usize, k: usize, w: &[f32], n: usize, bias: &[f32], y: &mut [f32]) {
    for row in y.chunks_exact_mut(n) {
        row.copy_from_slice(bias);
    }
    // SAFETY: slices are sized m*k, k*n, m*n with the row-major strides
    // passed below.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            w.as_ptr(),
            n as isize,
            1,
            1.0,
            y.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// Unfold the kernel patches of output row `oy` into `col`:
/// [ox][(ky, kx, ci)], zero-padded.
fn im2col_row(x: &[f32], h: usize, w: usize, g: &ConvGeom, oy: usize, ow: usize, col: &mut [f32]) {
    let c = g.cin;
    let span = g.kw * c; // one kernel row: contiguous in both input and col
    let row_len = g.kh * span;
    for ox in 0..ow {
        let row = &mut col[ox * row_len..][..row_len];
        let ix0 = (ox * g.sw) as isize - g.pl as isize;
        let inside_x = ix0 >= 0 && ix0 + g.kw as isize <= w as isize;
        for ky in 0..g.kh {
            let iy = (oy * g.sh + ky) as isize - g.pt as isize;
            let dst = &mut row[ky * span..][..span];
            if iy < 0 || iy >= h as isize {
                dst.fill(0.0);
            } else if inside_x {
                dst.copy_from_slice(&x[(iy as usize * w + ix0 as usize) * c..][..span]);
            } else {
                for (kx, d) in dst.chunks_exact_mut(c).enumerate() {
                    let ix = ix0 + kx as isize;
                    if ix < 0 || ix >= w as isize {
                        d.fill(0.0);
                    } else {
                        d.copy_from_slice(&x[(iy as usize * w + ix as usize) * c..][..c]);
                    }
                }
            }
        }
    }
}

#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "aarch64+neon"))]
#[allow(clippy::too_many_arguments)]
fn dwconv(
    x: &[f32],
    h: usize,
    w: usize,
    c: usize,
    g: &ConvGeom,
    wt: &[f32],
    b: &[f32],
    oh: usize,
    ow: usize,
    y: &mut [f32],
) {
    let (kh, kw, sh, sw) = (g.kh, g.kw, g.sh, g.sw);
    let (pt, pl) = (g.pt as isize, g.pl as isize);
    for oy in 0..oh {
        let iy0 = (oy * sh) as isize - pt;
        for ox in 0..ow {
            let ix0 = (ox * sw) as isize - pl;
            let acc = &mut y[(oy * ow + ox) * c..][..c];
            acc.copy_from_slice(b);
            let interior = iy0 >= 0
                && ix0 >= 0
                && iy0 + kh as isize <= h as isize
                && ix0 + kw as isize <= w as isize;
            for ky in 0..kh {
                let iy = iy0 + ky as isize;
                if !interior && (iy < 0 || iy >= h as isize) {
                    continue;
                }
                for kx in 0..kw {
                    let ix = ix0 + kx as isize;
                    if !interior && (ix < 0 || ix >= w as isize) {
                        continue;
                    }
                    let src = &x[(iy as usize * w + ix as usize) * c..][..c];
                    let k = &wt[(ky * kw + kx) * c..][..c];
                    for ((a, &s), &kv) in acc.iter_mut().zip(src).zip(k) {
                        *a += s * kv;
                    }
                }
            }
        }
    }
}

#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "aarch64+neon"))]
fn add_assign(y: &mut [f32], b: &[f32]) {
    for (d, &q) in y.iter_mut().zip(b) {
        *d += q;
    }
}

#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "aarch64+neon"))]
fn add(a: &[f32], b: &[f32], y: &mut [f32]) {
    for ((d, &p), &q) in y.iter_mut().zip(a).zip(b) {
        *d = p + q;
    }
}

#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "aarch64+neon"))]
fn apply_act(y: &mut [f32], c: usize, act: &Act) {
    match act {
        Act::None => {}
        Act::Relu => {
            for v in y.iter_mut() {
                *v = v.max(0.0);
            }
        }
        Act::Prelu(s) => {
            for row in y.chunks_exact_mut(c) {
                for (v, &a) in row.iter_mut().zip(s) {
                    // branch-free: v >= 0 ? v : a * v
                    *v = v.max(0.0) + a * v.min(0.0);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn maxpool(
    x: &[f32],
    h: usize,
    w: usize,
    c: usize,
    k: usize,
    s: usize,
    pt: usize,
    pl: usize,
    oh: usize,
    ow: usize,
    y: &mut [f32],
) {
    for oy in 0..oh {
        for ox in 0..ow {
            let d = &mut y[(oy * ow + ox) * c..][..c];
            d.fill(f32::NEG_INFINITY);
            for ky in 0..k {
                let iy = (oy * s + ky) as isize - pt as isize;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                for kx in 0..k {
                    let ix = (ox * s + kx) as isize - pl as isize;
                    if ix < 0 || ix >= w as isize {
                        continue;
                    }
                    let src = &x[(iy as usize * w + ix as usize) * c..][..c];
                    for (a, &v) in d.iter_mut().zip(src) {
                        *a = a.max(v);
                    }
                }
            }
        }
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
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert!(f16_to_f32(0x7c00).is_infinite());
    }

    fn conv_net() -> Vec<u8> {
        // input 4x4x2 -> 3x3 conv (SAME, stride 1) to 3 channels, relu
        let mut b = b"RVNN".to_vec();
        let u = |v: u32| v.to_le_bytes();
        b.extend(u(1));
        b.extend(u(2));
        for s in [[4, 4, 2], [4, 4, 3]] {
            for v in s {
                b.extend(u(v));
            }
        }
        b.extend(u(0));
        b.extend(u(1));
        b.extend(u(1));
        b.extend(u(1));
        b.push(OP_CONV);
        for v in [0, 1, 3, 3, 2, 3, 1, 1, 1, 1] {
            b.extend(u(v));
        }
        b.push(1);
        // weights [co][ky][kx][ci]: co-th output sums ci=co%2 over the 3x3
        b.extend(u(54));
        for co in 0..3 {
            for _ in 0..9 {
                for ci in 0..2 {
                    let v: u16 = if ci == co % 2 { 0x3c00 } else { 0 };
                    b.extend(v.to_le_bytes());
                }
            }
        }
        b.extend(u(3));
        for v in [0u16, 0, 0xbc00] {
            b.extend(v.to_le_bytes()); // bias 0, 0, -1
        }
        b
    }

    /// Every matmul path (explicit SIMD, portable narrow kernels, SGEMM,
    /// matrix-vector) agrees on odd shapes, with and without residual.
    #[test]
    fn matmul_paths_agree() {
        let val = |i: usize| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0;
        for &n in &[8usize, 16, 24, 32, 40] {
            for &(m, k) in &[(1usize, 7usize), (5, 3), (13, 27), (64, 16)] {
                let a: Vec<f32> = (0..m * k).map(val).collect();
                let w: Vec<f32> = (0..k * n).map(|i| val(i + 7)).collect();
                let b: Vec<f32> = (0..n).map(|i| val(i + 11)).collect();
                let r: Vec<f32> = (0..m * n).map(|i| val(i + 13)).collect();
                let mut want = vec![0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut s = b[j] + r[i * n + j];
                        for kk in 0..k {
                            s += a[i * k + kk] * w[kk * n + j];
                        }
                        want[i * n + j] = s;
                    }
                }
                let close = |got: &[f32], what: &str| {
                    for (g, e) in got.iter().zip(&want) {
                        assert!((g - e).abs() < 1e-4, "{what} m{m} k{k} n{n}: {g} vs {e}");
                    }
                };
                let mut y = vec![0f32; m * n];
                matmul_res(&a, m, k, &w, n, &b, Some(&r), &mut y);
                close(&y, "dispatch");
                let mut y = vec![0f32; m * n];
                conv_gemm(&a, m, k, &w, n, &b, &mut y);
                add_assign(&mut y, &r);
                close(&y, "sgemm");
                let mut y = vec![0f32; m * n];
                let portable = match n {
                    8 => Some(narrow_8 as fn(&[f32], usize, &[f32], &[f32], &mut [f32])),
                    16 => Some(narrow_16 as fn(&[f32], usize, &[f32], &[f32], &mut [f32])),
                    24 => Some(narrow_24 as fn(&[f32], usize, &[f32], &[f32], &mut [f32])),
                    32 => Some(narrow_32 as fn(&[f32], usize, &[f32], &[f32], &mut [f32])),
                    _ => None,
                };
                if let Some(f) = portable {
                    f(&a, k, &w, &b, &mut y);
                    add_assign(&mut y, &r);
                    close(&y, "portable");
                }
            }
        }
    }

    /// Corrupted network files must be rejected with an error, never a
    /// panic or a huge allocation.
    #[test]
    fn malformed_networks_are_rejected() {
        let good = conv_net();
        assert!(Net::from_bytes(&good).is_ok());
        for cut in 0..good.len() {
            assert!(Net::from_bytes(&good[..cut]).is_err(), "truncated at {cut}");
        }
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..2000 {
            let mut b = good.clone();
            for _ in 0..3 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let at = (seed as usize) % b.len();
                b[at] = (seed >> 32) as u8;
            }
            // must not panic; a flip inside float payloads may still parse
            let _ = std::panic::catch_unwind(|| Net::from_bytes(&b)).expect("parser panicked");
        }
        let mut huge = good.clone();
        huge[12..16].copy_from_slice(&u32::MAX.to_le_bytes()); // tensor 0: h
        assert!(Net::from_bytes(&huge).is_err());
    }

    #[test]
    fn conv_same_padding_bias_relu() {
        let net = Net::from_bytes(&conv_net()).unwrap();
        let mut ws = net.workspace();
        let x: Vec<f32> = (0..32)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let out = &net.run(&mut ws, &x)[0];
        // channel 0 = number of in-bounds taps (ci=0 is all ones)
        let taps = [4.0, 6.0, 6.0, 4.0, 6.0, 9.0, 9.0, 6.0];
        for (p, &t) in taps.iter().enumerate() {
            assert_eq!(out[p * 3], t);
            assert_eq!(out[p * 3 + 1], 0.0);
            assert_eq!(out[p * 3 + 2], t - 1.0); // relu(t - 1)
        }
    }
}

#[cfg(test)]
mod golden {
    //! The embedded networks against TensorFlow Lite's own outputs
    //! (reference interpreter, XNNPACK disabled) on fixed inputs
    //! `x_i = lo + (hi - lo) * ((i * 7919) mod 255) / 255`.
    use super::Net;

    fn read_f32(b: &[u8], p: &mut usize) -> f32 {
        let v = f32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    }
    fn read_u32(b: &[u8], p: &mut usize) -> usize {
        let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap()) as usize;
        *p += 4;
        v
    }

    #[test]
    fn matches_tflite_reference() {
        let Some(golden) = crate::golden_data("nn_golden.bin") else {
            return;
        };
        let golden = &golden[..];
        assert_eq!(&golden[..4], b"RVGD");
        let mut p = 4;
        let nets: [&[u8]; 2] = [
            include_bytes!("../models/blazeface_short_range.rvnn"),
            include_bytes!("../models/face_mesh_v2.rvnn"),
        ];
        for bytes in nets {
            let net = Net::from_bytes(bytes).unwrap();
            let (lo, hi) = (read_f32(golden, &mut p), read_f32(golden, &mut p));
            let n_in: usize = net.input_shape().iter().product();
            let x: Vec<f32> = (0..n_in)
                .map(|i| lo + (hi - lo) * ((i * 7919) % 255) as f32 / 255.0)
                .collect();
            let outs = net.run(&mut net.workspace(), &x);
            assert_eq!(outs.len(), read_u32(golden, &mut p));
            for got in outs {
                let n = read_u32(golden, &mut p);
                assert_eq!(got.len(), n);
                let want: Vec<f32> = (0..n).map(|_| read_f32(golden, &mut p)).collect();
                let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
                let err = got
                    .iter()
                    .zip(&want)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    err / scale < 1e-4,
                    "relative error {} vs TFLite",
                    err / scale
                );
            }
        }
        assert_eq!(p, golden.len());
    }
}
