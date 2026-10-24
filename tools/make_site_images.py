# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the website's before/after images from public data.

    cargo build --release -p radveil-cli
    pip install numpy nibabel pynrrd pillow
    python tools/make_site_images.py --work /tmp/site-images --site site

Downloads 3D Slicer's "MRHead" sample (checksum-verified), which its donor,
the person pictured, gave to the 3D Slicer project "to be used without any
restrictions" (Slicer's SampleData module). It runs `radveil auto` with
default settings in each mode, plus `cut` with verification off (on this
head the plain cut still reads as a face and verification escalates it to a
block, so the site shows the plain cut to illustrate the mode), plus
`--region face+ears --ear-mode cut`, and `radveil deskull` for the brain.
It writes these images to --work and, with `--site site`, copies the ones
site/index.html and site/docs.html reference (img/NAME.png) to site/img/:

- site_<name>.png: frontal surface render, 364x404;
- slices_<name>.png: the voxel data itself, a mid-sagittal slice above an
  axial slice through the orbits, same intensity window for every image,
  anonymization mask outlined;
- side_input.png / side_ears.png: lateral renders before and after the ear
  cut; slices_earcut.png: an axial slice through the ears, mask outlined;
- brain.png: mid-sagittal slice with the extracted brain (blue) and the
  anonymization region (amber) outlined; numbers.json: the figures quoted
  in the captions (voxels changed, brain clearance, ear detection).

The volume and the other outputs stay in --work and are not committed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import urllib.request

import nibabel as nib
import nrrd
import numpy as np
from PIL import Image

SHA256 = "cc211f0dfd9a05ca3841ce1141b292898b2dd2d3f08286affadf823a7e58df93"
URL = f"https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/{SHA256}"
MODES = ("blur", "adaptive", "cut", "block")
SIZE = (364, 404)
SHOWN = ("input", "blur", "adaptive", "cut-noverify", "block")
SAGITTAL_X, AXIAL_Y = 65, 140  # voxel indices in MRHead's PIR grid: midline, orbits
ACCENT = (227, 170, 75)  # site accent colour, for the mask outline
BRAIN = (96, 165, 250)  # brain-mask outline
SIDE_SIZE = (404, 364)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def fetch(work: str) -> str:
    raw, nii = os.path.join(work, "MRHead.nrrd"), os.path.join(work, "MRHead.nii.gz")
    if not os.path.exists(raw):
        urllib.request.urlretrieve(URL, raw)
    with open(raw, "rb") as fh:
        if hashlib.sha256(fh.read()).hexdigest() != SHA256:
            raise SystemExit("MRHead.nrrd checksum mismatch")
    data, hdr = nrrd.read(raw)
    affine = np.eye(4)
    affine[:3, :3] = np.array(hdr["space directions"], float).T
    affine[:3, 3] = hdr["space origin"]
    if hdr["space"] == "left-posterior-superior":
        affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine  # LPS -> RAS
    nib.save(nib.Nifti1Image(data, affine), nii)
    return nii


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="scratch directory")
    ap.add_argument("--cli", default=os.path.join(ROOT, "target", "release", "radveil"))
    ap.add_argument("--site", help="copy the referenced images to SITE/img/")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    vols = {"input": fetch(a.work)}
    runs = [(m, [m]) for m in MODES] + [("cut-noverify", ["cut", "--no-verify"]),
                                        ("ears", ["adaptive", "--region", "face+ears",
                                                  "--ear-mode", "cut"])]
    numbers = {}
    for name, (mode, *extra) in runs:
        out = os.path.join(a.work, f"{name}.nii.gz")
        rep = os.path.join(a.work, f"{name}.json")
        mask = os.path.join(a.work, f"{name}.mask.nii.gz")
        subprocess.check_call([a.cli, "auto", vols["input"], out, "--mode", mode, *extra,
                               "--report", rep, "--mask-out", mask])
        r = json.load(open(rep))
        numbers[name] = dict(voxels_changed=r["deface"]["voxels_changed"],
                             residual=r.get("residual_detections"),
                             escalated=r.get("escalated_to_block"),
                             ears=[{k: o.get(k) for k in ("label", "detected", "protrusion_mm")}
                                   for o in r["deface"]["rois"] if str(o.get("label")).startswith("ear")])
        print(f"{name}: views {r['views_detected']}, residual {r['residual_detections']}, "
              f"escalated to block {r['escalated_to_block']}")
        vols[name] = out

    write_slices(a.work, vols)
    brain = os.path.join(a.work, "brain.mask.nii.gz")
    subprocess.check_call([a.cli, "deskull", vols["input"], "--mask-out", brain],
                          stdout=subprocess.DEVNULL)
    numbers["brain"] = write_brain(a.work, vols["input"], brain,
                                   os.path.join(a.work, "adaptive.mask.nii.gz"))
    write_ear_slice(a.work, vols)
    sides = {}
    for name in ("input", "ears"):
        d = os.path.join(a.work, f"side_{name}")
        subprocess.check_call([a.cli, "render", vols[name], "-o", d, "--yaw", "90"],
                              stdout=subprocess.DEVNULL)
        sides[name] = Image.open(os.path.join(d, "view_00_yaw+90.png")).convert("RGB")
    sbox = head_box(sides["input"], SIDE_SIZE)
    for name, im in sides.items():
        save(im.crop(sbox).resize(SIDE_SIZE, Image.LANCZOS), os.path.join(a.work, f"side_{name}"))
    renders = {}
    for name, vol in vols.items():
        if name not in SHOWN:
            continue
        d = os.path.join(a.work, f"render_{name}")
        subprocess.check_call([a.cli, "render", vol, "-o", d, "--yaw", "0"],
                              stdout=subprocess.DEVNULL)
        renders[name] = Image.open(os.path.join(d, "view_00_yaw+0.png")).convert("RGB")

    box = head_box(renders["input"], SIZE)  # one crop for all frontal images
    for name, im in renders.items():
        save(im.crop(box).resize(SIZE, Image.LANCZOS), os.path.join(a.work, f"site_{name}"))
    with open(os.path.join(a.work, "numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=1)
    print(json.dumps(numbers, indent=1))
    print(f"wrote site_*, side_*, slices_*, brain .png in {a.work}")
    if a.site:
        publish(a.work, a.site)


def head_box(im: Image.Image, size: tuple[int, int]) -> tuple[int, ...]:
    """The head's bounding box plus a margin, at the aspect ratio of `size`."""
    img = np.asarray(im.convert("L")).astype(int)
    ys, xs = np.nonzero(np.abs(img - img[0, 0]) > 6)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    h = max(ys.max() - ys.min(), (xs.max() - xs.min()) * size[1] / size[0]) * 1.12
    w = h * size[0] / size[1]
    return tuple(int(round(v)) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def publish(work: str, site: str) -> None:
    """Copy every image a site page references (img/NAME.png) from `work`."""
    import re
    import shutil

    names = set()
    for page in ("index.html", "docs.html"):
        path = os.path.join(site, page)
        if os.path.exists(path):
            text = open(path, encoding="utf-8").read()
            names |= set(re.findall(r'src="img/([\w-]+)\.png"', text))
    os.makedirs(os.path.join(site, "img"), exist_ok=True)
    for n in sorted(names):
        shutil.copy(os.path.join(work, n + ".png"), os.path.join(site, "img", n + ".png"))
    print(f"copied {len(names)} images to {site}/img")


def outline(m: np.ndarray) -> np.ndarray:
    inner = np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m & ~inner


def window(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v[v > 0], [1, 99.5])
    return float(lo), float(hi)


def gray(sl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = (np.clip((sl - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, -1)


def write_brain(work: str, inp: str, brain: str, face: str) -> dict:
    """Mid-sagittal slice: brain (blue) and anonymization region (amber)."""
    from scipy.ndimage import distance_transform_edt

    v = nib.load(inp)
    x = v.get_fdata()
    b = nib.load(brain).get_fdata() > 0
    f = nib.load(face).get_fdata() > 0
    zooms = v.header.get_zooms()
    clear = float(distance_transform_edt(~b, sampling=zooms)[f].min())
    lo, hi = window(x)
    # Upsample first so the outlines can be 2 px wide at display size.
    up = lambda a: np.kron(a, np.ones((2, 2), a.dtype))
    rgb = gray(up(x[:, :, SAGITTAL_X].T), lo, hi)
    for mk, col in ((b, BRAIN), (f, ACCENT)):
        o = outline(up(mk[:, :, SAGITTAL_X].T))
        rgb[o | np.roll(o, 1, 0) | np.roll(o, 1, 1)] = col
    im = Image.fromarray(rgb).crop((0, 16, 512, 480))
    save(im, os.path.join(work, "brain"))
    return dict(clearance_mm=round(clear, 1), brain_ml=round(float(b.sum() * np.prod(zooms)) / 1000, 0),
                face_voxels=int(f.sum()), overlap_voxels=int((b & f).sum()))


def write_ear_slice(work: str, vols: dict[str, str]) -> None:
    """Axial slice through the ears (the level of most ear-mask voxels)."""
    v = nib.load(vols["ears"]).get_fdata()
    ref = nib.load(vols["input"])
    m = nib.load(os.path.join(work, "ears.mask.nii.gz")).get_fdata() > 0
    face = nib.load(os.path.join(work, "adaptive.mask.nii.gz")).get_fdata() > 0
    # The level of the pinna centre: 0.39 eye distances below the eye line
    # (the calibrated ear prior), from the detected face frame.
    fr = json.load(open(os.path.join(work, "ears.json")))["frame"]
    c = np.array(fr["origin"]) - 0.39 * fr["eye_dist"] * np.array([0.0, 0.0, 1.0])
    y = int(round((np.linalg.inv(ref.affine) @ np.append(c, 1.0))[1]))
    lo, hi = window(ref.get_fdata())
    zooms = ref.header.get_zooms()
    # Crop to the side renders' aspect ratio, centred on the ear regions.
    rows = np.nonzero((m & ~face)[:, y, :].any(axis=1))[0]
    centre = (rows.min() + rows.max()) / 2
    for name, vol, mk in (("earcut_input", ref.get_fdata(), None), ("earcut", v, m)):
        rgb = gray(vol[:, y, ::-1], lo, hi)
        if mk is not None:
            rgb[outline(mk[:, y, ::-1])] = ACCENT
        im = Image.fromarray(rgb)
        im = im.resize((round(im.width * zooms[2] / zooms[0]), im.height), Image.LANCZOS)
        h = im.width * SIDE_SIZE[1] / SIDE_SIZE[0]
        top = int(np.clip(centre - h / 2, 0, im.height - h))
        im = im.crop((0, top, im.width, top + round(h)))
        save(im.resize(SIDE_SIZE, Image.LANCZOS), os.path.join(work, f"slices_{name}"))


def save(im: Image.Image, stem: str) -> None:
    # A 64-colour palette: the renders are one skin tone and the slices are
    # grey plus two outline colours, so this is visually lossless and keeps
    # the page (which embeds every image) small.
    im = im.convert("RGB")
    a = np.asarray(im).astype(int)
    if (np.abs(a - a.mean(-1, keepdims=True)).max(-1) < 3).mean() > 0.9:
        # A slice: grey levels plus the exact outline colours.
        pal = Image.new("P", (1, 1))
        greys = [(v, v, v) for v in np.linspace(0, 255, 62).astype(int)]
        pal.putpalette([c for rgb in greys + [ACCENT, BRAIN] for c in rgb])
        q = im.quantize(palette=pal, dither=Image.Dither.NONE)
    else:
        q = im.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    buf = io.BytesIO()
    q.save(buf, "PNG", optimize=True)
    with open(stem + ".png", "wb") as fh:
        fh.write(buf.getvalue())


def write_slices(work: str, vols: dict[str, str]) -> None:
    """Sagittal over axial slice of each shown volume, input's intensity window."""
    ref = nib.load(vols["input"]).get_fdata()
    lo, hi = np.percentile(ref[ref > 0], [1, 99.5])
    zooms = nib.load(vols["input"]).header.get_zooms()

    def to_rgb(sl: np.ndarray, m: np.ndarray | None) -> Image.Image:
        g = (np.clip((sl - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        rgb = np.repeat(g[..., None], 3, -1)
        if m is not None:
            inner = np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
            rgb[m & ~inner] = ACCENT
        return Image.fromarray(rgb)

    for name in SHOWN:
        v = nib.load(vols[name]).get_fdata()
        m = None
        if name != "input":
            m = nib.load(os.path.join(work, f"{name}.mask.nii.gz")).get_fdata() > 0
        # MRHead is stored PIR: axis 0 anterior->posterior, 1 superior->inferior,
        # 2 left->right. Sagittal: face on the left. Axial: radiological view.
        sag = to_rgb(v[:, :, SAGITTAL_X].T, None if m is None else m[:, :, SAGITTAL_X].T)
        axi = to_rgb(v[:, AXIAL_Y, ::-1], None if m is None else m[:, AXIAL_Y, ::-1])
        axi = axi.resize((round(axi.width * zooms[2] / zooms[0]), axi.height), Image.LANCZOS)
        sag = sag.crop((0, 8, 256, 240))
        axi = axi.resize((256, round(axi.height * 256 / axi.width)), Image.LANCZOS)
        axi = axi.crop((0, 0, 256, 200))
        col = Image.new("RGB", (256, sag.height + 6 + axi.height), (0, 0, 0))
        col.paste(sag, (0, 0))
        col.paste(axi, (0, sag.height + 6))
        save(col, os.path.join(work, f"slices_{name}"))


if __name__ == "__main__":
    main()
