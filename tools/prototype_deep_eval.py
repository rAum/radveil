#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Prototype evaluation of the `deep` / `reshape` options (docs/prototype-deep-anonymize.md).

    export RADVEIL_PAPER=<paper repo> DATA_DIR=<its data directory>
    prototype_deep_eval.py select --subjects subjects.json --out DIR   # 60 + 60 subjects, seed 0
    prototype_deep_eval.py run   --subjects subjects.json --out DIR [--group eval] [--jobs 4]
    prototype_deep_eval.py brain --out DIR [--jobs 4]          # SynthStrip set, expert masks
    prototype_deep_eval.py time  --subjects subjects.json --out DIR
    prototype_deep_eval.py table --subjects subjects.json --out DIR

Uses the evaluation paper's attack code (bench/attack.py, analysis/reid.py)
from PAPER and its clean gallery. Everything written to --out is local
working data (embeddings are biometric): never commit it.
"""
import argparse, csv, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

PAPER = Path(os.environ["RADVEIL_PAPER"])  # checkout of the evaluation paper's repository
DATA = Path(os.environ["DATA_DIR"])  # its data directory (ixi/, synthstrip/, models/)
sys.path.insert(0, str(PAPER / "bench")); sys.path.insert(0, str(PAPER / "analysis"))

VARIANTS = {
    "adaptive": dict(mode="adaptive"),
    "block": dict(mode="block"),
    # deep_guard_mm defaults to 5 mm; the report's "3 mm guard" rows are deep3*
    "deep3": dict(mode="adaptive", deep=True, deep_guard_mm=3.0),
    "deep3_reshape": dict(mode="adaptive", deep=True, reshape=True, deep_guard_mm=3.0),
    "deep5": dict(mode="adaptive", deep=True),
    "deep5_reshape": dict(mode="adaptive", deep=True, reshape=True),
    # reshape alone: a no-op blur; the self-check is off (the output is a face by design)
    "reshape_only": dict(mode="blur", blur_sigma_mm=0.01, deep=True, reshape=True, deep_guard_mm=3.0,
                         verify=False),
}
MASKS = {"skin": dict(mode="adaptive"), "skin_guard": dict(mode="adaptive", brain_guard=True),
         "deep3": dict(mode="adaptive", deep=True, deep_guard_mm=3.0), "deep5": dict(mode="adaptive", deep=True)}


ONLY = None


def _init(threads, only=None):
    global ONLY
    ONLY = only
    os.environ["RAYON_NUM_THREADS"] = str(threads)


def run_scan(job):
    import attack, radveil
    sid, image, out, variants = job
    models = str(DATA / "models")
    src = str(DATA / image)
    vin = radveil.Volume(src)
    orig = np.array(vin.data())
    tissue = orig > 0.1 * np.percentile(orig, 99)
    vox_ml = float(np.prod(vin.spacing)) / 1000.0
    emb_in, ref = attack.embed_volume(vin, models)
    recs = []
    for name in variants:
        p = Path(out) / "emb" / name / f"{sid}.npz"
        rec = {"id": sid, "variant": name}
        try:
            v = radveil.Volume(src)
            t0 = time.perf_counter()
            rep = radveil.deface_volume(v, return_mask=True, brain_check=True, **VARIANTS[name])
            rec["seconds_with_brain_check"] = round(time.perf_counter() - t0, 2)
            mask = np.asarray(rep["mask"]) > 0
            new = np.array(v.data())
            changed = new != orig
            rec.update(status="ok", mask_ml=round(float(mask.sum()) * vox_ml, 1),
                       tissue_ml=round(float((changed & tissue).sum()) * vox_ml, 1),
                       bit_exact_outside=not bool(changed[~mask].any()),
                       brain_voxels=rep["brain"]["voxels_modified"], brain_margin_mm=rep["brain"].get("margin_mm"),
                       verify_rounds=rep.get("verify_rounds"), residual=rep.get("residual_detections"),
                       escalated=rep.get("escalated_to_block"), spared=rep.get("protected_voxels_spared"))
            e = attack.embed_volume(v, models, ref)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **{k: a.astype(np.float16) for k, a in e.items()})
        except Exception as ex:
            msg = str(ex)
            rec.update(status="refused" if ("face" in msg or "field of view" in msg) else "error", reason=msg[:200])
        recs.append(rec)
    return recs


def cmd_select(a):
    """Subjects whose T1, T2 and PD the released adaptive mode accepted and
    that show a face to the attacker: groups eval (60) and more (60), seed 0."""
    import random

    man = {r["id"]: r for r in csv.DictReader(open(PAPER / "results/manifests/ixi.csv"))}
    ok = {r["id"] for r in map(json.loads, open(PAPER / "results/runs/ixi__adaptive.jsonl")) if r.get("status") == "ok"}
    g = np.load(PAPER / "results/embeddings/ixi__gallery.npz")
    face = {i: bool(np.isfinite(e[1, :, 0].astype(np.float32)).any()) for i, e in zip(g["ids"].tolist(), g["arcface"])}
    by = {}
    for i, r in man.items():
        by.setdefault(r["subject"], {})[r["contrast"]] = i
    full = sorted(s for s, d in by.items() if all(c in d and d[c] in ok and face.get(d[c]) for c in ("T1", "T2", "PD")))
    pick = random.Random(0).sample(full, len(full))
    out = {"eval": sorted(pick[:60]), "more": sorted(pick[60:120]), "scans": {s: by[s] for s in pick[:120]},
           "image": {i: man[i]["image"] for s in pick[:120] for i in by[s].values()}}
    json.dump(out, open(a.subjects, "w"))
    print(len(full), "eligible subjects;", len(out["eval"]), "+", len(out["more"]), "selected")


def cmd_run(a):
    sub = json.load(open(a.subjects))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = out / "runs.jsonl"
    done = set()
    if log.exists():
        done = {(r["id"], r["variant"]) for r in map(json.loads, open(log)) if r["status"] != "error"}
    jobs = []
    for s in sub[a.group]:
        for sid in sub["scans"][s].values():
            todo = [v for v in a.variants if (sid, v) not in done]
            if todo:
                jobs.append((sid, sub["image"][sid], str(out), todo))
    print(len(jobs), "scans to run", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.threads,)) as ex:
        for i, f in enumerate(as_completed([ex.submit(run_scan, j) for j in jobs]), 1):
            with open(log, "a") as fh:
                for r in f.result():
                    fh.write(json.dumps(r) + "\n")
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} {(time.time() - t0) / 60:.1f} min", flush=True)


def brain_scan(row):
    import nibabel as nib, radveil
    out = {"id": row["id"]}
    expert = np.squeeze(np.asarray(nib.load(str(DATA / row["brain_mask"])).dataobj)) > 0
    for name, kw in MASKS.items():
        if ONLY and name not in ONLY:
            continue
        try:
            v = radveil.Volume(str(DATA / row["image"]))
            rep = radveil.anonymization_mask(v, brain_check=True, **kw)
            m = np.asarray(rep["mask"]) > 0
            out[name] = {"expert_voxels": int((m & expert).sum()) if m.shape == expert.shape else None,
                         "rust_voxels": rep["brain"]["voxels_modified"]}
        except Exception as ex:
            out[name] = {"refused": str(ex)[:120]}
    return out


def cmd_brain(a):
    rows = list(csv.DictReader(open(PAPER / "results/manifests/synthstrip.csv")))
    ok = {json.loads(l)["id"] for l in open(PAPER / "results/runs/synthstrip__adaptive.jsonl")
          if json.loads(l).get("status") == "ok"}
    rows = [r for r in rows if r["id"] in ok and r["brain_mask"]]
    out = Path(a.out) / "brain.jsonl"
    out = Path(a.out) / f"brain{a.tag}.jsonl"
    print(len(rows), "scans", flush=True)
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.threads, a.masks)) as ex:
        for i, r in enumerate(ex.map(brain_scan, rows), 1):
            with open(out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            if i % 20 == 0:
                print(" ", i, flush=True)


def cmd_time(a):
    import radveil
    sub = json.load(open(a.subjects))
    scans = [sub["scans"][s][c] for s in sub["eval"][:4] for c in ("T1", "T2", "PD")]
    res = {}
    for name, kw in VARIANTS.items():
        if name not in a.variants:
            continue
        ts = []
        for sid in scans:
            v = radveil.Volume(str(DATA / sub["image"][sid]))
            t0 = time.perf_counter()
            radveil.deface_volume(v, **kw)
            ts.append(time.perf_counter() - t0)
        res[name] = {"median_s": round(float(np.median(ts)), 2), "n": len(ts)}
        print(name, res[name], flush=True)
    json.dump(res, open(Path(a.out) / f"timing{a.tag}.json", "w"))


def cmd_table(a):
    import reid
    sub = json.load(open(a.subjects)); out = Path(a.out)
    man = {r["id"]: r for r in csv.DictReader(open(PAPER / "results/manifests/ixi.csv"))}
    g = np.load(PAPER / "results/embeddings/ixi__gallery.npz")
    gids = g["ids"].tolist(); one = reid.MULTS.index(1.0)
    runs = {}
    for r in map(json.loads, open(out / "runs.jsonl")):
        runs[(r["id"], r["variant"])] = r
    subjects = [s for g_ in a.groups for s in sub[g_]]
    res = {"subjects": len(subjects), "rank1": {}, "ops": {}}
    for rec in ("arcface", "sface"):
        clean = dict(zip(gids, reid._templates(g[rec])))
        nan_t = np.full_like(next(iter(clean.values())), np.nan)
        # galleries over ALL IXI subjects, as in the paper: T1 for T2/PD probes, PD (else T2) for T1 probes
        t1 = {man[i]["subject"]: i for i in gids if man[i]["contrast"] == "T1" and np.isfinite(clean[i][one, 0])}
        g2 = {}
        for i in gids:
            c = man[i]["contrast"]
            if c in ("PD", "T2") and np.isfinite(clean[i][one, 0]) and (man[i]["subject"] not in g2 or c == "PD"):
                g2[man[i]["subject"]] = i
        for direction, gal, contrasts in (("T2/PD probes vs T1 gallery", t1, ("T2", "PD")),
                                          ("T1 probes vs PD/T2 gallery", g2, ("T1",))):
            gsub = sorted(gal); gi = {s: k for k, s in enumerate(gsub)}
            G = np.stack([clean[gal[s]] for s in gsub]); gvalid = np.isfinite(G[..., 0])
            probes = [sub["scans"][s][c] for s in subjects for c in contrasts if s in gi]
            truth = np.array([gi[man[i]["subject"]] for i in probes])

            def table(P, keep):
                S = P
                dv = reid._depth_views(S[keep], truth[keep], gvalid)
                return {k: {"hits": int((m["rank"] == 1).sum()), "n": int(len(m["rank"])), "gallery": gk}
                        for k, (m, _, gk) in dv.items()}

            every = np.ones(len(probes), bool)
            rows = {"clean": table(reid._sim(np.stack([clean[i] for i in probes]), G), every)}
            for name in VARIANTS:
                ok = np.array([runs.get((i, name), {}).get("status") == "ok" for i in probes])
                if not ok.any():
                    continue

                def load(key):
                    P = []
                    for i, o in zip(probes, ok):
                        if o:
                            z = np.load(out / "emb" / name / f"{i}.npz")
                            P.append(reid._templates(z[f"{rec}_{key}"][None])[0])
                        else:
                            P.append(nan_t)
                    return reid._sim(np.stack(P), G)

                S, Sn = load("oracle"), load("near")
                S = np.where(np.isfinite(Sn) & (~np.isfinite(S) | (Sn > S)), Sn, S)
                rows[name] = table(S, ok)
                rows[name]["accepted"] = int(ok.sum())
            res["rank1"].setdefault(rec, {})[direction] = rows
    ids = [i for s in subjects for i in sub["scans"][s].values()]
    for name in VARIANTS:
        rs = [runs[(i, name)] for i in ids if (i, name) in runs]
        ok = [r for r in rs if r["status"] == "ok"]
        if not ok:
            continue
        res["ops"][name] = {
            "run": len(rs), "ok": len(ok), "refused": sum(r["status"] == "refused" for r in rs),
            "error": sum(r["status"] == "error" for r in rs),
            "rust_brain_touched_scans": sum(r["brain_voxels"] > 0 for r in ok),
            "min_brain_margin_mm": min((r["brain_margin_mm"] for r in ok if r["brain_margin_mm"] is not None), default=None),
            "median_tissue_ml": float(np.median([r["tissue_ml"] for r in ok])),
            "median_mask_ml": float(np.median([r["mask_ml"] for r in ok])),
            "bit_exact_outside_all": all(r["bit_exact_outside"] for r in ok),
            "escalated_to_block": sum(bool(r["escalated"]) for r in ok),
            "residual_after_verify": sum((r["residual"] or 0) > 0 for r in ok),
        }
    import glob
    B = {}
    for f in sorted(glob.glob(str(out / "brain*.jsonl"))):
        for b in map(json.loads, open(f)):
            B.setdefault(b["id"], {}).update(b)
    B = list(B.values())
    if B:
        res["synthstrip_expert"] = {}
        for name in MASKS:
            if not all(name in b for b in B):
                continue
            ok = [b[name] for b in B if "refused" not in b[name] and b[name]["expert_voxels"] is not None]
            res["synthstrip_expert"][name] = {
                "scans": len(ok), "refused": sum("refused" in b[name] for b in B),
                "expert_touched_scans": sum(o["expert_voxels"] > 0 for o in ok),
                "expert_max_voxels": max((o["expert_voxels"] for o in ok), default=0),
                "rust_touched_scans": sum(o["rust_voxels"] > 0 for o in ok)}
    res["timing"] = {}
    for f in sorted(glob.glob(str(out / "timing*.json"))):
        res["timing"].update(json.load(open(f)))
    json.dump(res, open(out / f"summary{a.tag}.json", "w"), indent=1)
    for rec, d in res["rank1"].items():
        for direction, rows in d.items():
            print(f"\n{rec}: {direction}")
            for name, t in rows.items():
                print(f"  {name:13s}", "  ".join(f"{k}x {v['hits']}/{v['n']} ({100 * v['hits'] / max(v['n'], 1):.1f}%)"
                                                 for k, v in t.items() if isinstance(v, dict)))
    print(json.dumps({k: res[k] for k in res if k != "rank1"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "run", "brain", "time", "table"])
    ap.add_argument("--subjects"); ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="eval"); ap.add_argument("--groups", nargs="+", default=["eval"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--masks", nargs="+"); ap.add_argument("--tag", default="")
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    {"select": cmd_select, "run": cmd_run, "brain": cmd_brain, "time": cmd_time, "table": cmd_table}[a.cmd](a)
