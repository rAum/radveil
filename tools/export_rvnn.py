# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a MediaPipe TFLite model to RadVeil's `.rvnn` network format.

    pip install tflite numpy
    python tools/export_rvnn.py face_detector.tflite blazeface_short_range.rvnn
    python tools/export_rvnn.py --verify face_detector.tflite blazeface_short_range.rvnn

`.rvnn` is a flat, pre-resolved op list for the small NHWC convolutional
networks RadVeil embeds (BlazeFace, Face Mesh): padding is resolved to
explicit offsets, ReLU/PReLU are fused into the op that produces their
input, reshapes of the output heads become aliases, and weights stay in
float16 (as published). `crates/radveil-core/src/nn.rs` executes it.

`--verify` runs the exported network with the NumPy reference interpreter
in this file and compares it against TensorFlow Lite's interpreter on
random inputs (needs `tensorflow` or `ai_edge_litert`).

Format (little-endian):
    b"RVNN" u32 version=1
    u32 n_tensors, then per tensor: u32 h, w, c      (batch is always 1)
    u32 input tensor
    u32 n_outputs, then u32 tensor ids
    u32 n_ops, then per op: u8 kind + fields (see OP_* below)
    f16 payloads are u32 element count + data
"""

import argparse
import struct
import sys

import numpy as np

OP_CONV, OP_DWCONV, OP_ADD, OP_MAXPOOL, OP_PADC, OP_CONCAT, OP_LOGISTIC, OP_ALIAS = range(1, 9)
ACT_NONE, ACT_RELU, ACT_PRELU = 0, 1, 2


# --------------------------------------------------------------------------
# TFLite parsing


def _builtin_names():
    import tflite

    return {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith("_")}


def _opts(op, cls):
    o = cls()
    t = op.BuiltinOptions()
    o.Init(t.Bytes, t.Pos)
    return o


def _same_pad(n, k, s):
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return out, total // 2


def parse_tflite(path):
    import tflite

    buf = open(path, "rb").read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    g = m.Subgraphs(0)
    names = _builtin_names()

    def tensor_shape(i):
        return [int(x) for x in g.Tensors(i).ShapeAsNumpy()]

    def const(i):
        t = g.Tensors(i)
        b = m.Buffers(t.Buffer()).DataAsNumpy()
        if isinstance(b, int) or b is None or len(b) == 0:
            return None
        dt = {tflite.TensorType.FLOAT16: np.float16, tflite.TensorType.FLOAT32: np.float32,
              tflite.TensorType.INT32: np.int32}[t.Type()]
        return np.frombuffer(b.tobytes(), dt).reshape(tensor_shape(i))

    consts = {}  # tensor id -> ndarray (after DEQUANTIZE resolution)
    ops = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        code = m.OperatorCodes(op.OpcodeIndex())
        name = names[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        outs = [op.Outputs(j) for j in range(op.OutputsLength())]
        if name == "DEQUANTIZE":
            consts[outs[0]] = const(ins[0]).astype(np.float16)
            continue

        def w(j, ins=ins):  # bind this op's inputs (closures are late-binding)
            t = ins[j]
            return consts[t] if t in consts else const(t)

        ops.append(dict(name=name, ins=ins, outs=outs, op=op, w=w))
    inputs = [g.Inputs(i) for i in range(g.InputsLength())]
    outputs = [g.Outputs(i) for i in range(g.OutputsLength())]
    return g, ops, inputs, outputs, tensor_shape, _opts


# --------------------------------------------------------------------------
# Lowering to rvnn ops


def lower(path):
    import tflite

    g, ops, inputs, outputs, shape, opts = parse_tflite(path)
    assert len(inputs) == 1
    act_code = {0: ACT_NONE, 1: ACT_RELU}
    consumers = {}
    for o in ops:
        for t in o["ins"]:
            consumers.setdefault(t, []).append(o)

    prog = []  # dicts
    for o in ops:
        n = o["name"]
        i0, out = o["ins"][0], o["outs"][0]
        if n in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            conv = n == "CONV_2D"
            opt = opts(o["op"], tflite.Conv2DOptions if conv else tflite.DepthwiseConv2DOptions)
            assert opt.DilationHFactor() == 1 and opt.DilationWFactor() == 1
            assert opt.FusedActivationFunction() in act_code
            wt = o["w"](1).astype(np.float16)
            b = o["w"](2).astype(np.float16)
            _, h, wd, cin = shape(i0)
            if conv:
                cout, kh, kw, _ = wt.shape
            else:
                assert opt.DepthMultiplier() == 1
                _, kh, kw, cout = wt.shape
                wt = wt[0]  # (kh, kw, c)
            sh, sw = opt.StrideH(), opt.StrideW()
            if opt.Padding() == tflite.Padding.SAME:
                oh, pt = _same_pad(h, kh, sh)
                ow, pl = _same_pad(wd, kw, sw)
            else:
                oh, ow, pt, pl = (h - kh) // sh + 1, (wd - kw) // sw + 1, 0, 0
            assert [1, oh, ow, cout] == shape(out), (shape(out), oh, ow)
            prog.append(dict(kind=OP_CONV if conv else OP_DWCONV, ins=[i0], out=out,
                             p=[kh, kw, cin, cout, sh, sw, pt, pl],
                             act=act_code[opt.FusedActivationFunction()], w=wt, b=b, slope=None))
        elif n == "ADD":
            opt = opts(o["op"], tflite.AddOptions)
            assert opt.FusedActivationFunction() in act_code
            assert shape(o["ins"][0]) == shape(o["ins"][1])
            prog.append(dict(kind=OP_ADD, ins=o["ins"][:2], out=out, p=[],
                             act=act_code[opt.FusedActivationFunction()], slope=None))
        elif n in ("RELU", "PRELU"):
            # fuse into the producer when it is the only consumer of its input
            prod = next((p for p in prog if p["out"] == i0), None)
            fusable = (prod is not None and prod["kind"] in (OP_CONV, OP_DWCONV, OP_ADD)
                       and prod["act"] == ACT_NONE and len(consumers.get(i0, [])) == 1
                       and i0 not in outputs)
            assert fusable, f"unfusable {n}"
            prod["out"] = out
            if n == "RELU":
                prod["act"] = ACT_RELU
            else:
                s = o["w"](1).astype(np.float16).reshape(-1)
                prod["act"], prod["slope"] = ACT_PRELU, s
        elif n == "MAX_POOL_2D":
            opt = opts(o["op"], tflite.Pool2DOptions)
            k, s = opt.FilterHeight(), opt.StrideH()
            assert opt.FilterWidth() == k and opt.StrideW() == s
            _, h, wd, c = shape(i0)
            if opt.Padding() == tflite.Padding.SAME:
                _, pt = _same_pad(h, k, s)
                _, pl = _same_pad(wd, k, s)
            else:
                pt = pl = 0
            prog.append(dict(kind=OP_MAXPOOL, ins=[i0], out=out, p=[k, s, pt, pl]))
        elif n == "PAD":
            pads = o["w"](1).astype(np.int64).reshape(4, 2)
            assert (pads[:3] == 0).all(), "only channel padding is supported"
            prog.append(dict(kind=OP_PADC, ins=[i0], out=out, p=[int(pads[3, 0]), int(pads[3, 1])]))
        elif n == "RESHAPE":
            prog.append(dict(kind=OP_ALIAS, ins=[i0], out=out, p=[]))
        elif n == "CONCATENATION":
            opt = opts(o["op"], tflite.ConcatenationOptions)
            sh0 = shape(o["ins"][0])
            ax = opt.Axis() % len(sh0)
            assert all(d == 1 for d in sh0[:ax]), "concat must be over the outermost axis"
            assert opt.FusedActivationFunction() == 0
            prog.append(dict(kind=OP_CONCAT, ins=list(o["ins"]), out=out, p=[]))
        elif n == "LOGISTIC":
            prog.append(dict(kind=OP_LOGISTIC, ins=[i0], out=out, p=[]))
        else:
            raise SystemExit(f"unsupported op {n}")

    # compact tensor ids
    used = [inputs[0]]
    for p in prog:
        for t in p["ins"] + [p["out"]]:
            if t not in used:
                used.append(t)
    ids = {t: i for i, t in enumerate(used)}

    def hwc(t):
        s = shape(t)
        s = [1] * (4 - len(s)) + s
        assert s[0] == 1
        return s[1:]

    tensors = [hwc(t) for t in used]
    for p in prog:
        p["ins"] = [ids[t] for t in p["ins"]]
        p["out"] = ids[p["out"]]
    return tensors, ids[inputs[0]], [ids[t] for t in outputs], prog


# --------------------------------------------------------------------------
# Serialization


def _f16(a):
    a = np.ascontiguousarray(a, np.float16).reshape(-1)
    return struct.pack("<I", a.size) + a.tobytes()


def write(path, tensors, inp, outs, prog):
    b = bytearray(b"RVNN" + struct.pack("<I", 1))
    b += struct.pack("<I", len(tensors))
    for t in tensors:
        b += struct.pack("<3I", *t)
    b += struct.pack("<I", inp)
    b += struct.pack("<I", len(outs)) + struct.pack(f"<{len(outs)}I", *outs)
    b += struct.pack("<I", len(prog))
    for p in prog:
        k = p["kind"]
        b += struct.pack("<B", k)
        if k in (OP_CONV, OP_DWCONV):
            b += struct.pack("<2I8IB", p["ins"][0], p["out"], *p["p"], p["act"])
            b += _f16(p["w"]) + _f16(p["b"])
            if p["act"] == ACT_PRELU:
                b += _f16(p["slope"])
        elif k == OP_ADD:
            b += struct.pack("<3IB", p["ins"][0], p["ins"][1], p["out"], p["act"])
            if p["act"] == ACT_PRELU:
                b += _f16(p["slope"])
        elif k == OP_MAXPOOL:
            b += struct.pack("<6I", p["ins"][0], p["out"], *p["p"])
        elif k == OP_PADC:
            b += struct.pack("<4I", p["ins"][0], p["out"], *p["p"])
        elif k == OP_CONCAT:
            b += struct.pack("<I", len(p["ins"])) + struct.pack(f"<{len(p['ins'])}I", *p["ins"])
            b += struct.pack("<I", p["out"])
        elif k in (OP_LOGISTIC, OP_ALIAS):
            b += struct.pack("<2I", p["ins"][0], p["out"])
    open(path, "wb").write(bytes(b))
    return len(b)


# --------------------------------------------------------------------------
# NumPy reference interpreter (semantics of nn.rs)


def run_reference(tensors, inp, outs, prog, x):
    v = {inp: x.reshape(tensors[inp]).astype(np.float32)}

    def act(y, p):
        if p["act"] == ACT_RELU:
            return np.maximum(y, 0)
        if p["act"] == ACT_PRELU:
            s = p["slope"].astype(np.float32)
            return np.where(y >= 0, y, y * s)
        return y

    for p in prog:
        k, o = p["kind"], p["out"]
        if k in (OP_CONV, OP_DWCONV):
            kh, kw, cin, cout, sh, sw, pt, pl = p["p"]
            a = v[p["ins"][0]]
            oh, ow, _ = tensors[o]
            h, wd, _ = a.shape
            pb, pr = max((oh - 1) * sh + kh - h - pt, 0), max((ow - 1) * sw + kw - wd - pl, 0)
            ap = np.pad(a, ((pt, pb), (pl, pr), (0, 0)))
            w = p["w"].astype(np.float32)
            y = np.zeros((oh, ow, cout), np.float32)
            for dy in range(kh):
                for dx in range(kw):
                    patch = ap[dy:dy + (oh - 1) * sh + 1:sh, dx:dx + (ow - 1) * sw + 1:sw]
                    if k == OP_CONV:
                        y += patch @ w[:, dy, dx, :].T
                    else:
                        y += patch * w[dy, dx]
            v[o] = act(y + p["b"].astype(np.float32), p)
        elif k == OP_ADD:
            v[o] = act(v[p["ins"][0]] + v[p["ins"][1]], p)
        elif k == OP_MAXPOOL:
            ks, s, pt, pl = p["p"]
            a = v[p["ins"][0]]
            oh, ow, c = tensors[o]
            ap = np.pad(a, ((pt, ks), (pl, ks), (0, 0)), constant_values=-np.inf)
            y = np.full((oh, ow, c), -np.inf, np.float32)
            for dy in range(ks):
                for dx in range(ks):
                    y = np.maximum(y, ap[dy:dy + (oh - 1) * s + 1:s, dx:dx + (ow - 1) * s + 1:s])
            v[o] = y
        elif k == OP_PADC:
            v[o] = np.pad(v[p["ins"][0]], ((0, 0), (0, 0), tuple(p["p"])))
        elif k == OP_ALIAS:
            v[o] = v[p["ins"][0]].reshape(tensors[o])
        elif k == OP_CONCAT:
            v[o] = np.concatenate([v[i].reshape(-1) for i in p["ins"]]).reshape(tensors[o])
        elif k == OP_LOGISTIC:
            v[o] = 1 / (1 + np.exp(-v[p["ins"][0]]))
    return [v[i] for i in outs]


def verify(tflite_path, net, trials=3):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter, OpResolverType

        def Interpreter(model_path, _I=Interpreter):  # noqa: N802
            return _I(model_path=model_path,
                      experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it = Interpreter(model_path=tflite_path)
    it.allocate_tensors()
    ind = it.get_input_details()[0]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        x = rng.uniform(-1, 1, ind["shape"]).astype(np.float32)
        it.set_tensor(ind["index"], x)
        it.invoke()
        want = [it.get_tensor(d["index"]) for d in it.get_output_details()]
        got = run_reference(*net, x[0])
        for a, b in zip(want, got):
            err = float(np.abs(a.reshape(-1) - b.reshape(-1)).max() / (np.abs(a).max() + 1e-6))
            worst = max(worst, err)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite")
    ap.add_argument("out")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    net = lower(a.tflite)
    n = write(a.out, *net)
    kinds = {}
    for p in net[3]:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"{a.out}: {n} bytes, {len(net[3])} ops {kinds}, {len(net[0])} tensors")
    if a.verify:
        err = verify(a.tflite, net)
        print(f"reference vs TFLite: max relative error {err:.2e}")
        if err > 1e-4:
            sys.exit("mismatch")


if __name__ == "__main__":
    main()
