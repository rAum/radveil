# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Python-level tests on a synthetic phantom (no dlib, no patient data).

Set RADVEIL_TEST_T1 to a real T1 NIfTI with a visible face to also run the
end-to-end detector test."""

import os

import numpy as np
import nibabel as nib
import pytest

import radveil


@pytest.fixture(scope="module")
def phantom(tmp_path_factory):
    """Ellipsoid head with nose and ear bumps, stored LPS-ordered,
    int16 + scaling."""
    n = 96
    c = n // 2
    i, j, k = np.mgrid[0:n, 0:n, 0:n].astype(np.float64)
    x, y, z = i - c, j - c, k - c  # LPS voxel axes
    head = (x / 30) ** 2 + (y / 35) ** 2 + (z / 38) ** 2 <= 1.0
    nose = (x / 6) ** 2 + ((y + 40) / 10) ** 2 + ((z + 5) / 8) ** 2 <= 1.0
    # protruding bumps just off the lateral surface (air gap, like a pinna)
    ears = ((np.abs(x) - 33) / 2.5) ** 2 + ((y - 2) / 6) ** 2 + ((z + 4) / 7) ** 2 <= 1.0
    data = np.where(head | nose | ears, 800, 0).astype(np.int16)
    affine = np.array(
        [
            [-2.0, 0, 0, 96.0],
            [0, -2.0, 0, 96.0],
            [0, 0, 2.0, -96.0],
            [0, 0, 0, 1.0],
        ]
    )
    img = nib.Nifti1Image(data, affine)
    img.header.set_slope_inter(0.5, 10.0)
    path = tmp_path_factory.mktemp("data") / "phantom.nii.gz"
    nib.save(img, path)
    return str(path)


LANDMARKS = {
    **{i: (-22.0, 62.0, 10.0) for i in range(36, 42)},  # right eye
    **{i: (22.0, 62.0, 10.0) for i in range(42, 48)},  # left eye
    30: (0.0, 80.0, -10.0),  # nose tip
    **{i: (0.0, 40.0, -35.0) for i in range(48, 68)},  # mouth (on the skin,
    # posterior of the eye surface — matches the tilt of detector-lifted frames)
    8: (0.0, 50.0, -55.0),  # chin
}


def landmarks68():
    return [LANDMARKS.get(i) for i in range(68)]


def test_load_orientation(phantom):
    vol = radveil.Volume(phantom)
    assert vol.orientation == "LPS"
    assert vol.shape == (96, 96, 96)
    aff = vol.affine
    assert aff.shape == (4, 4)
    assert aff[0, 0] == -2.0


def test_render_and_pos_map(phantom):
    vol = radveil.Volume(phantom)
    view = vol.render(yaw_deg=0.0, size=256)
    assert view["rgb"].shape == (256, 256, 3)
    assert view["pos"].shape == (256, 256, 3)
    assert view["hit"].sum() > 2000
    # positions of hits must lie inside the world bbox
    pos = view["pos"][view["hit"]]
    assert np.isfinite(pos).all()
    assert pos[:, 1].max() > 60  # nose bump on the anterior side


def test_deface_modes_bit_exact(phantom):
    orig = np.asanyarray(nib.load(phantom).dataobj)
    for mode in ("blur", "cut", "block", "adaptive"):
        vol = radveil.Volume(phantom)
        rep = vol.deface(landmarks68(), mode=mode, return_mask=True)
        assert rep["voxels_changed"] > 1000
        mask = rep["mask"].astype(bool)
        out_path = phantom.replace(".nii.gz", f"_{mode}.nii.gz")
        vol.save(out_path)
        res_img = nib.load(out_path)
        res = np.asanyarray(res_img.dataobj)
        assert res_img.get_data_dtype() == np.int16
        # on-disk scaling preserved (nibabel exposes it via dataobj)
        assert (res_img.dataobj.slope, res_img.dataobj.inter) == (0.5, 10.0)
        # raw stored voxels outside the mask are bit-exact
        raw_orig = nib.load(phantom).dataobj.get_unscaled()
        raw_res = res_img.dataobj.get_unscaled()
        assert np.array_equal(raw_res[~mask], raw_orig[~mask]), f"{mode}: outside-mask voxels changed"
        assert np.array_equal(res[~mask], orig[~mask])
        assert (res[mask] != orig[mask]).any(), f"{mode}: nothing changed inside mask"
        if mode == "cut":
            # cut zeroes the scaled value; raw = (0 - inter)/slope = -20
            assert (res[mask] == 0).all()


def test_ear_regions(phantom):
    """Ear anonymization: same modes, pinna ROIs found from the face frame."""
    orig = np.asanyarray(nib.load(phantom).dataobj)
    ear_a = (15, 50, 44)  # world (+66, -4, -8) — ear bump center
    ear_b = (81, 50, 44)  # world (-66, -4, -8) — the other ear
    nose = (48, 8, 43)  # world (0, 80, -10) — nose tip

    vol = radveil.Volume(phantom)
    rep = vol.deface(landmarks68(), mode="cut", region="ears", return_mask=True)
    assert rep["region"] == "ears"
    assert [r["label"] for r in rep["rois"]] == ["ear_left", "ear_right"]
    for r in rep["rois"]:
        assert r["voxels"] > 100
        assert r["detected"] is True  # located by surface evidence
        assert r["protrusion_mm"] > 5.0
    assert "roi_u_mm" not in rep  # legacy face keys only when face included
    mask = rep["mask"].astype(bool)
    assert mask[ear_a] and mask[ear_b]
    assert not mask[nose]
    data = vol.data()
    assert (data[mask] == 0).all()  # cut zeroes the scaled value
    assert np.allclose(data[~mask], orig[~mask])  # dataobj is already scaled

    vol2 = radveil.Volume(phantom)
    rep2 = vol2.deface(landmarks68(), mode="cut", region="face+ears", return_mask=True)
    assert [r["label"] for r in rep2["rois"]] == ["face", "ear_left", "ear_right"]
    assert "detected" not in rep2["rois"][0]  # face ROI carries no ear evidence
    assert "roi_u_mm" in rep2
    mask2 = rep2["mask"].astype(bool)
    assert mask2[nose] and mask2[ear_a] and mask2[ear_b]
    assert rep2["voxels_changed"] > rep["voxels_changed"]

    with pytest.raises(ValueError):
        vol2.deface(landmarks68(), region="nose")


def test_per_region_modes(phantom):
    """One pass: blur the face, cut the ears."""
    orig = np.asanyarray(nib.load(phantom).dataobj)
    ear_a = (15, 55, 35)
    nose = (48, 8, 43)

    ears_only = radveil.Volume(phantom).deface(
        landmarks68(), mode="cut", region="ears", return_mask=True
    )["mask"].astype(bool)

    vol = radveil.Volume(phantom)
    rep = vol.deface(
        landmarks68(), mode="blur", ear_mode="cut", region="face+ears", return_mask=True
    )
    assert rep["mode"] == "blur" and rep["ear_mode"] == "cut"
    mask = rep["mask"]
    assert set(np.unique(mask)) <= {0, 1}
    mask = mask.astype(bool)
    assert mask[nose] and mask[ear_a]
    data = vol.data()
    ear_only = ears_only & mask
    assert (data[ear_only] == 0).all()  # ears cut
    assert data[nose] != 0  # face blurred, not cut
    assert np.allclose(data[~mask], orig[~mask])

    assert radveil.Volume(phantom).deface(landmarks68(), mode="cut")["ear_mode"] == "cut"
    with pytest.raises(ValueError):
        radveil.Volume(phantom).deface(landmarks68(), ear_mode="melt")


def test_eye_surface_anchor(phantom):
    """Face depth bounds anchor to the measured skin; opt-out reverts."""
    vol = radveil.Volume(phantom)
    rep = vol.deface(landmarks68(), mode="cut")
    assert rep["eye_bias_mm"] >= 0.0
    vol2 = radveil.Volume(phantom)
    rep2 = vol2.deface(landmarks68(), mode="cut", eye_surface_anchor=False)
    assert rep2["eye_bias_mm"] == 0.0


def test_lift_landmarks(phantom):
    vol = radveil.Volume(phantom)
    views = [vol.render(yaw_deg=y, size=256) for y in (-15.0, 0.0, 15.0)]
    # Project the known nose tip into 2D is hard without camera math here;
    # instead check that lifting the image-center pixel of the frontal view
    # returns the nose region (max anterior surface).
    center = views[1]["pos"][128, 128]
    lm = np.tile([[128.0, 128.0]], (68, 1))
    pts = radveil.lift_landmarks([views[1]], [lm])
    assert pts[30] is not None
    assert abs(pts[30][1] - center[1]) < 3.0


def test_deface_rejects_garbage(phantom):
    vol = radveil.Volume(phantom)
    with pytest.raises(ValueError):
        vol.deface([None] * 68)


def test_from_array_roundtrip(tmp_path):
    """DICOM/raw entry point: array + affine in, valid NIfTI out."""
    n = 48
    arr = np.zeros((n, n, n), np.float32)
    arr[12:36, 12:36, 12:36] = 500.0
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-n, -n, -n]
    vol = radveil.Volume.from_array(arr, aff)
    assert vol.orientation == "RAS"
    out = tmp_path / "arr.nii.gz"
    vol.save(str(out))
    img = nib.load(out)
    assert img.shape == (n, n, n)
    assert np.allclose(img.affine, aff)
    assert np.allclose(np.asanyarray(img.dataobj), arr)


def test_detect_face_blank_image():
    assert radveil.detect_landmarks(np.full((128, 128, 3), 12, np.uint8)) is None
    assert radveil.detect_landmarks(np.zeros((64, 96, 3), np.uint8), try_rotations=True) is None


def test_deface_auto_refuses_faceless_phantom(phantom, tmp_path):
    """The phantom has a nose bump but no face: the built-in detector must
    not find a validated face, and nothing may be written or modified."""
    out = tmp_path / "out.nii.gz"
    with pytest.raises(RuntimeError):
        radveil.deface_auto(phantom, str(out))
    assert not out.exists()


def _phantom_detector(vol, yaws, size):
    """A stand-in 'detector' that knows the phantom's 3D landmarks: on the
    exact renders the pipeline asks about it returns their pixel positions
    (nearest surface hit), and on anything else (e.g. the anonymized
    re-renders of the verify loop) it finds no face."""
    known = {}
    for y in yaws:
        view = vol.render(yaw_deg=y, size=size)
        pos, hit = view["pos"], view["hit"]
        pts = np.full((68, 2), -1000.0)
        for i, p in enumerate(landmarks68()):
            if p is None:
                continue
            d = np.linalg.norm(pos - np.asarray(p), axis=2)
            d[~hit] = np.inf
            yy, xx = np.unravel_index(np.argmin(d), d.shape)
            pts[i] = (xx, yy)
        known[view["rgb"].tobytes()] = pts

    calls = []

    def detect(rgb, try_rotations):
        calls.append(try_rotations)
        pts = known.get(np.ascontiguousarray(rgb).tobytes())
        return None if pts is None else (pts, 0.9)

    return detect, calls


def test_deface_volume_custom_detector(phantom):
    yaws = (-15.0, 0.0, 15.0)
    vol = radveil.Volume(phantom)
    det, calls = _phantom_detector(radveil.Volume(phantom), yaws, 256)
    rep = radveil.deface_volume(vol, mode="blur", yaws=yaws, size=256, detector=det)
    assert rep["detector"] == "custom"
    assert rep["views_detected"] == 3
    assert not rep["orientation_sweep"]
    assert rep["verify_rounds"] == 0 and rep["residual_detections"] == 0
    assert rep["voxels_changed"] > 1000
    assert [v["score"] for v in rep["views"]] == [0.9] * 3
    # 3 detection views + 3 verification re-renders at pitch 0 and 3 at -15,
    # + the default (balanced) 3x3 grid around the best view minus the one
    # angle already rendered
    assert len(calls) == 9 + 8
    calls.clear()
    rep = radveil.deface_volume(
        radveil.Volume(phantom), mode="blur", yaws=yaws, size=256, detector=det,
        verify_effort="fast",
    )
    assert rep["residual_detections"] == 0
    assert len(calls) == 9  # no grid


def test_anonymization_mask_leaves_volume_untouched(phantom, tmp_path):
    yaws = (-15.0, 0.0, 15.0)
    vol = radveil.Volume(phantom)
    before = vol.data().copy()
    det, _ = _phantom_detector(radveil.Volume(phantom), yaws, 256)
    mask_file = tmp_path / "mask.nii.gz"
    rep = radveil.anonymization_mask(
        vol, mode="blur", yaws=yaws, size=256, detector=det, mask_path=str(mask_file)
    )
    mask = rep["mask"]
    assert np.array_equal(vol.data(), before)  # nothing rewritten
    assert "verify_rounds" not in rep  # nothing verified
    assert mask.shape == vol.shape and mask.dtype == np.uint8
    assert int(mask.sum()) == rep["voxels_changed"] > 1000

    # It is exactly the region deface_volume rewrites.
    vol2 = radveil.Volume(phantom)
    det2, _ = _phantom_detector(radveil.Volume(phantom), yaws, 256)
    rep2 = radveil.deface_volume(
        vol2, mode="blur", yaws=yaws, size=256, detector=det2, return_mask=True
    )
    assert np.array_equal(rep2["mask"], mask)

    saved = nib.load(str(mask_file))
    assert np.array_equal(np.asarray(saved.dataobj), mask)
    assert np.allclose(saved.affine, vol.affine)


def test_deface_volume_rejects_unknown_parameter(phantom):
    vol = radveil.Volume(phantom)
    det, _ = _phantom_detector(radveil.Volume(phantom), (-15.0, 0.0, 15.0), 256)
    with pytest.raises(ValueError):
        radveil.deface_volume(vol, yaws=(-15.0, 0.0, 15.0), size=256, detector=det, blur_sigma=3)


@pytest.mark.skipif("RADVEIL_TEST_T1" not in os.environ, reason="needs a real T1 (RADVEIL_TEST_T1)")
def test_deface_auto_real_face(tmp_path):
    out = tmp_path / "t1_defaced.nii.gz"
    rep = radveil.deface_auto(os.environ["RADVEIL_TEST_T1"], str(out), qa_dir=str(tmp_path / "qa"))
    assert rep["detector"] == "mediapipe"
    assert rep["views_detected"] >= 2
    assert rep["residual_detections"] == 0
    assert out.exists()
    assert any(p.name.startswith("after_") for p in (tmp_path / "qa").iterdir())
    # the defaced output no longer has a (validated) face
    with pytest.raises(RuntimeError):
        radveil.deface_auto(str(out), str(tmp_path / "again.nii.gz"))


def test_bytes_roundtrip_matches_files(phantom, tmp_path):
    """Service path: bytes in, bytes out, identical to the file path."""
    raw = open(phantom, "rb").read()
    a, b = radveil.Volume(phantom), radveil.Volume.from_bytes(raw)
    assert a.shape == b.shape and a.orientation == b.orientation
    assert np.array_equal(a.data(), b.data()) and np.allclose(a.affine, b.affine)
    for vol in (a, b):
        vol.deface(landmarks68(), mode="blur")
    out = tmp_path / "saved.nii.gz"
    a.save(str(out))
    assert b.to_bytes() == out.read_bytes()
    plain = tmp_path / "saved.nii"
    a.save(str(plain))
    assert b.to_bytes(gzip_level=None) == plain.read_bytes()
    # (saving rounds to the stored int16 type, so compare with a reload)
    back = radveil.Volume.from_bytes(b.to_bytes())
    assert np.array_equal(back.data(), radveil.Volume(str(out)).data())
    assert np.array_equal(np.asanyarray(nib.load(out).dataobj), np.asanyarray(nib.Nifti1Image.from_bytes(b.to_bytes(gzip_level=None)).dataobj))


def test_fallback_detector_runs_when_primary_finds_nothing(phantom):
    """A refusal leaves the volume untouched, and the fallback detector then
    processes the original data."""
    yaws = (-15.0, 0.0, 15.0)
    vol = radveil.Volume(phantom)
    before = vol.data().copy()
    fallback, _ = _phantom_detector(radveil.Volume(phantom), yaws, 256)
    with pytest.raises(RuntimeError):
        radveil.deface_volume(vol, yaws=yaws, size=256, detector=lambda rgb, rot: None)
    assert np.array_equal(vol.data(), before)
    rep = radveil.deface_volume(
        vol, yaws=yaws, size=256, detector=lambda rgb, rot: None,
        fallback_detector=fallback, fallback_brain_guard=False,
    )
    assert rep["fallback_used"] and rep["detector"] == "custom"
    assert rep["views_detected"] == 3 and rep["voxels_changed"] > 1000
    with pytest.raises(ValueError):
        radveil.deface_volume(vol, fallback_detector="nope")


def test_dlib_five_point_expansion():
    # dlib-5 order: points 0-1 one eye's corners, 2-3 the other's, 4 nose base
    pts = np.array([[160.0, 100.0], [140.0, 100.0], [60.0, 100.0], [80.0, 100.0], [110.0, 150.0]])
    out = radveil._expand_five_to_68(pts)
    assert out.shape == (68, 2)
    assert np.allclose(out[36:42].mean(0), [70.0, 100.0])  # image-left eye
    assert np.allclose(out[42:48].mean(0), [150.0, 100.0])
    assert np.allclose(out[30], [110.0, 150.0])
    # inferred mouth: 0.35 eye distances below the nose base
    assert np.allclose(out[48:68].mean(0), [110.0, 150.0 + 0.35 * 80.0])
    assert (out[0:17] < -1e5).all()  # jaw not measured: dropped by lifting
    with pytest.raises(ValueError):
        radveil._expand_five_to_68(np.zeros((5, 2)))


def test_restricted_68_point_model_never_picked_implicitly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RADVEIL_SHAPE_PREDICTOR", raising=False)
    (tmp_path / "shape_predictor_68_face_landmarks.dat").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="5-point"):
        radveil._predictor_path(None)
    (tmp_path / radveil.DEFAULT_PREDICTOR).write_bytes(b"x")
    assert radveil._predictor_path(None) == radveil.DEFAULT_PREDICTOR
    with pytest.raises(FileNotFoundError):
        radveil._predictor_path(str(tmp_path / "missing.dat"))
