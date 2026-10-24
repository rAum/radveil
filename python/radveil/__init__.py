# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""radveil: CPU-only NIfTI face rendering + anonymization.

Everything runs in Rust (`radveil._radveil`): NIfTI IO with LPS/RAS
orientation handling, volumetric software rendering, face detection (the
built-in MediaPipe BlazeFace + Face Mesh models, no extra dependency), 3D
masking and the verification loop. dlib is available as an optional
alternative or fallback detector (``detector="dlib"`` or
``fallback_detector="dlib"``; ``pip install radveil[dlib]``); its default
landmark model is dlib's CC0-licensed 5-point predictor.

    import radveil
    radveil.deface_auto("t1.nii.gz", "t1_defaced.nii.gz", mode="blur")

or step by step:

    vol = radveil.Volume("t1.nii.gz")
    views = [vol.render(yaw_deg=y) for y in (-20, 0, 20)]
    lms = [radveil.detect_landmarks(v["rgb"]) for v in views]
    pts3d = radveil.lift_landmarks(views, lms)
    vol.deface(pts3d, mode="blur")
    vol.save("t1_defaced.nii.gz")
"""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Optional, Sequence

import numpy as np

from ._radveil import Volume, __version__  # noqa: F401
from ._radveil import detect_face as _detect_face

__all__ = [
    "Volume",
    "fetch_predictor",
    "detect_landmarks",
    "lift_landmarks",
    "deface_volume",
    "anonymization_mask",
    "deface_auto",
    "deface_batch",
    "deskull",
    "__version__",
]

_DETECTOR = None
_PREDICTORS: dict = {}
_PREDICTOR_PARTS: dict = {}
_WARNED_RESTRICTED: set = set()

DEFAULT_PREDICTOR = "shape_predictor_5_face_landmarks.dat"
PREDICTOR_URL = "https://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
PREDICTOR_ARCHIVE_SHA256 = "6e787bbebf5c9efdb793f6cd1f023230c4413306605f24f299f12869f95aa472"
PREDICTOR_SHA256 = "c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888"
PREDICTOR_LICENSE_NOTE = (
    "dlib publishes the 5-point model under CC0-1.0, without the iBUG 300-W "
    "commercial-use restriction of its 68-point models. See docs/legal.md "
    "for the provenance boundary."
)
RESTRICTED_LICENSE_NOTE = (
    "This dlib predictor has 68 points. dlib's 68-point models were trained on "
    "iBUG 300-W, whose license EXCLUDES COMMERCIAL USE; RadVeil never selects "
    "them by default. Obtain permission from Imperial College London (iBUG) "
    "before commercial use. See docs/legal.md."
)


def _predictor_path(explicit: Optional[str]) -> str:
    """The dlib shape predictor to use. An explicit path (or
    RADVEIL_SHAPE_PREDICTOR) wins and must exist; otherwise only the
    commercially licensed 5-point default is looked for. A 68-point model is
    never picked up implicitly."""
    selected = explicit if explicit is not None else os.environ.get("RADVEIL_SHAPE_PREDICTOR")
    if selected is not None:
        if not os.path.isfile(selected):
            raise FileNotFoundError(f"requested shape predictor does not exist: {selected}")
        return os.fspath(selected)
    for c in (
        DEFAULT_PREDICTOR,
        os.path.join(os.path.dirname(__file__), DEFAULT_PREDICTOR),
    ):
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "dlib shape predictor not found. Run radveil.fetch_predictor() to "
        "download the CC0 5-point default (~9 MB), pass predictor_path=..., or "
        "set RADVEIL_SHAPE_PREDICTOR."
    )


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_predictor(dest: str = DEFAULT_PREDICTOR) -> str:
    """Download dlib's CC0-licensed 5-point shape predictor (~9 MB) and
    verify it against pinned SHA-256 checksums (archive and model).

    Returns the path. An existing ``dest`` is accepted only if its checksum
    matches the pinned model."""
    import bz2
    import urllib.request

    if os.path.isfile(dest):
        digest = _sha256_file(dest)
        if digest != PREDICTOR_SHA256:
            raise RuntimeError(
                f"existing model checksum mismatch: got {digest}, expected {PREDICTOR_SHA256}"
            )
        return dest
    if os.path.exists(dest):
        raise RuntimeError(f"predictor destination is not a file: {dest}")
    print(PREDICTOR_LICENSE_NOTE)
    print(f"downloading {PREDICTOR_URL} ...")
    with urllib.request.urlopen(PREDICTOR_URL) as r:
        archive = r.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != PREDICTOR_ARCHIVE_SHA256:
        raise RuntimeError(
            f"download checksum mismatch: got {digest}, expected {PREDICTOR_ARCHIVE_SHA256}"
        )
    data = bz2.decompress(archive)
    digest = hashlib.sha256(data).hexdigest()
    if digest != PREDICTOR_SHA256:
        raise RuntimeError(f"model checksum mismatch: got {digest}, expected {PREDICTOR_SHA256}")
    tmp = f"{dest}.part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    print(f"saved {dest} ({len(data) / 1e6:.0f} MB)")
    return dest


def _expand_five_to_68(points: np.ndarray) -> np.ndarray:
    """Map dlib's 5 points onto the 68-slot layout the 3D geometry uses.

    Points 0-1 and 2-3 are the two eye-corner pairs, point 4 the nose base.
    The mouth centre is an anthropometric prior, 0.35 inter-eye distances
    below the nose along the eye-to-nose direction, not a measurement. Jaw
    slots are placed far outside the image so lifting drops them instead of
    inventing a chin; the remaining slots take the nearest measured centre.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("expected five finite dlib landmarks")
    eye_a = points[0:2].mean(axis=0)
    eye_b = points[2:4].mean(axis=0)
    eye_mid = 0.5 * (eye_a + eye_b)
    eye_dist = float(np.linalg.norm(eye_a - eye_b))
    down = points[4] - eye_mid
    down_norm = float(np.linalg.norm(down))
    if eye_dist < 1.0 or down_norm < 1.0:
        raise ValueError("degenerate 5-point face geometry")
    mouth = points[4] + down / down_norm * (0.35 * eye_dist)
    out = np.repeat(eye_mid[None, :], 68, axis=0)
    # dlib-68: 36-41 is the image-left eye, as in dlib-5 points 2-3.
    out[36:42] = eye_b
    out[42:48] = eye_a
    out[27:36] = points[4]
    out[48:68] = mouth
    out[0:17] = -1.0e6
    return out


def detect_landmarks(
    rgb: np.ndarray,
    predictor_path: Optional[str] = None,
    upsample: int = 1,
    try_rotations: bool = False,
    detector: str = "mediapipe",
) -> Optional[np.ndarray]:
    """Detect a face on a rendered view and return its 68 landmarks as a
    (68, 2) array of pixel coordinates (dlib-68 order), or None.

    ``detector="mediapipe"`` (default) runs the built-in MediaPipe
    BlazeFace + Face Mesh v2 models in Rust (no extra dependency).
    ``detector="dlib"`` runs the dlib HOG detector + a shape predictor
    (``pip install radveil[dlib]``; ``predictor_path``/``upsample`` apply).
    The default predictor is dlib's CC0 5-point model
    (:func:`fetch_predictor`): eyes and nose base are measured, the mouth is
    inferred and the jaw left out (see :func:`_expand_five_to_68`). A
    68-point predictor is used only when passed explicitly; its training
    data excludes commercial use.

    With ``try_rotations=True`` the image is also searched rotated by
    90/180/270 degrees (both detectors are roll-sensitive); returned
    coordinates are always in the original image frame. This makes
    detection robust to volumes whose stored orientation is wrong, e.g.
    Analyze files with a guessed affine."""
    if detector == "mediapipe":
        found = _detect_face(np.ascontiguousarray(rgb, dtype=np.uint8), try_rotations)
        return None if found is None else found[0]
    if detector != "dlib":
        raise ValueError(f"unknown detector {detector!r} (mediapipe|dlib)")
    global _DETECTOR
    import dlib  # deferred: optional dependency

    if _DETECTOR is None:
        _DETECTOR = dlib.get_frontal_face_detector()
    path = os.path.abspath(_predictor_path(predictor_path))
    predictor = _PREDICTORS.get(path)
    if predictor is None:
        predictor = _PREDICTORS[path] = dlib.shape_predictor(path)

    base = np.ascontiguousarray(rgb)
    n_rot = 4 if try_rotations else 1
    # Fast path first: upsample=0 is ~14x cheaper and sufficient for the
    # large faces our renders produce; escalate only when it finds nothing.
    upsamples = [0] if upsample == 0 else list(dict.fromkeys([0, upsample]))
    for up, k in ((u, r) for u in upsamples for r in range(n_rot)):
        img = np.ascontiguousarray(np.rot90(base, k)) if k else base
        dets = _DETECTOR(img, up)
        if len(dets) == 0:
            continue
        det = max(dets, key=lambda d: d.width() * d.height())
        shape = predictor(img, det)
        if shape.num_parts not in (5, 68):
            raise ValueError(
                f"RadVeil supports dlib 5- or 68-point predictors; {path} returns "
                f"{shape.num_parts} landmarks"
            )
        _PREDICTOR_PARTS[path] = shape.num_parts
        if shape.num_parts == 68 and path not in _WARNED_RESTRICTED:
            _WARNED_RESTRICTED.add(path)
            warnings.warn(RESTRICTED_LICENSE_NOTE, stacklevel=2)
        pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)
        # map coordinates back through k CCW rotations (square images)
        h, w = base.shape[:2]
        for _ in range(k):
            # inverse of one np.rot90 (CCW): (x, y)_orig = (N-1-y_r, x_r)
            pts = np.stack([h - 1 - pts[:, 1], pts[:, 0]], axis=1)
        if shape.num_parts == 5:
            pts = _expand_five_to_68(pts)
        return pts
    return None


def _world_at(view: dict, x: float, y: float, max_radius: int = 4, anatomical: bool = False):
    """Nearest surface-hit world position around pixel (x, y).

    With ``anatomical=True``, hits on the volume boundary (scan cut planes)
    are rejected — they are not real anatomy."""
    hit = view["hit"]
    if anatomical and "boundary" in view:
        hit = hit & ~view["boundary"]
    pos = view["pos"]
    h, w = hit.shape
    cx, cy = int(round(x)), int(round(y))
    for r in range(max_radius + 1):
        best = None
        best_d2 = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h and hit[py, px]:
                    d2 = dx * dx + dy * dy
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = pos[py, px]
        if best is not None:
            return best.astype(np.float64)
    return None


def lift_landmarks(
    views: Sequence[dict],
    landmarks_2d: Sequence[Optional[np.ndarray]],
) -> list:
    """Lift per-view 2D landmarks to 3D world space through the views'
    position maps and combine them (coordinate-wise median across views).

    Returns a list of 68 items: (x, y, z) tuples or None, suitable for
    ``Volume.deface``."""
    per_view: list[list] = []
    n_pts = 0
    for view, lm in zip(views, landmarks_2d):
        if lm is None:
            continue
        n_pts = max(n_pts, len(lm))
        # dlib-68: jaw-line points (0-16) must not come from scan cut planes
        per_view.append(
            [
                _world_at(view, float(p[0]), float(p[1]), anatomical=i <= 16)
                for i, p in enumerate(lm)
            ]
        )
    combined = []
    for i in range(n_pts):
        pts = [v[i] for v in per_view if i < len(v) and v[i] is not None]
        if not pts:
            combined.append(None)
        else:
            arr = np.stack(pts)
            med = np.median(arr, axis=0)
            combined.append((float(med[0]), float(med[1]), float(med[2])))
    return combined


def _dlib_detector(predictor_path: Optional[str], upsample: int):
    def detect(rgb, try_rotations):
        return detect_landmarks(
            rgb,
            detector="dlib",
            predictor_path=predictor_path,
            upsample=upsample,
            try_rotations=try_rotations,
        )

    return detect


def deface_volume(
    vol: Volume,
    mode: str = "blur",
    region: str = "face",
    yaws: Sequence[float] = (-20.0, 0.0, 20.0),
    pitch: float = 0.0,
    size: int = 512,
    detector="mediapipe",
    predictor_path: Optional[str] = None,
    fallback_detector=None,
    fallback_brain_guard: bool = True,
    verify: bool = True,
    max_verify_rounds: int = 3,
    verify_pitches: Sequence[float] = (0.0, -15.0),
    verify_effort: str = "balanced",
    escalate_to_block: bool = True,
    brain_check: bool = False,
    brain_guard: bool = False,
    brain_guard_mm: float = 3.0,
    partial_face: bool = False,
    auto_threshold: bool = False,
    threshold_frac: float = 0.14,
    threshold: Optional[float] = None,
    detector_upsample: int = 1,
    return_mask: bool = False,
    **deface_kwargs,
) -> dict:
    """In-memory pipeline: render -> detect -> lift to 3D -> anonymize.

    Operates on an existing :class:`Volume` (from a file, from
    ``Volume.from_array`` — e.g. a DICOM series) and
    modifies it in place; call ``vol.save(...)`` afterwards or read back
    ``vol.data()``. ``deface_auto`` is the file-to-file wrapper. The
    pipeline itself runs in Rust (the same code as ``radveil auto``).

    ``detector`` selects the face detector: ``"mediapipe"`` (default; the
    built-in MediaPipe BlazeFace + Face Mesh models, Apache-2.0, no extra
    dependency or download), ``"dlib"`` (``pip install radveil[dlib]`` plus
    :func:`fetch_predictor` for the CC0 5-point model; ``predictor_path``
    and ``detector_upsample`` apply), or any callable
    ``f(rgb, try_rotations)`` returning None, an (68, 2) array of 68-point
    pixel landmarks, or ``(array, score)``.

    ``fallback_detector`` (default None) takes the same values: when the
    primary detector yields no validated face, the volume (still untouched)
    is processed again with it, e.g. ``fallback_detector="dlib"``. The
    report's ``detector`` names the one that succeeded and
    ``fallback_used`` says whether it was the fallback. A fallback run is
    a scan the default detector rejected (unusual contrast, damaged or
    already-defaced face), so it uses the brain guard unless
    ``fallback_brain_guard=False``: measured on the SynthStrip set, dlib
    also accepts some already-defaced scans and thin-scalp T2s, where the
    skin-following cut alone touched the brain.

    ``mode`` is one of ``blur``, ``cut``, ``block``, ``adaptive``. All
    geometric parameters (ROI margins, blur sigma) default to *auto*: they are
    scaled from the detected inter-eye distance, so the pipeline needs no
    tuning across head sizes, sequences or resolutions.

    ``region`` is ``face`` (default), ``ears`` or ``face+ears``. The pinna
    is a stable biometric identifier, so ear anonymization is offered with
    the same modes as the face. No detector emits ear landmarks; each
    pinna is detected from surface evidence (a protrusion scan in a window
    calibrated on hand-marked IXI pinnae, relative to the detected eyes)
    and the cut follows the measured scalp — face detection still runs
    even for ``ears``. Each ear ROI reports ``detected``.

    Detections are validated before anything is modified: views are grouped
    by *shape consensus* (their lifted 3D landmarks must coincide), and a
    face seen in a single view must be re-found with the same 3D shape from
    most of seven renders around its implied face axis. Volumes without a
    plausible face (e.g. already defaced) are refused. If the upright views
    find nothing, a 6-direction orientation sweep handles volumes whose
    stored orientation is wrong.

    With ``verify=True`` (default) the defaced volume is re-rendered from
    every detected yaw at each pitch in ``verify_pitches`` (default 0 and
    -15 degrees: a cut or heavy blur can expose orbit structure that only
    a downward-tilted view sees) and the face detector is run again.
    ``verify_effort`` adds a grid of views around the face: ``"fast"``
    none, ``"balanced"`` (default) 3 x 3 (yaw -25/0/+25, pitch
    -20/0/+20), ``"thorough"`` 5 x 5 (yaw -45..+45, pitch -30..+30); more
    views catch faces seen only from an unusual angle, at the cost of a
    render and a detection per view and round. For
    the blur-family modes the blur is escalated (sigma x1.6 per round, up to ``max_verify_rounds``) until the
    detector no longer finds a face at the anonymized location. If blurring
    alone cannot defeat re-detection (heavy blur can expose skull structure
    whose orbits read as a face), the region is filled opaquely (`block`)
    as a final guaranteed step unless ``escalate_to_block=False``. The
    report records ``verify_rounds``, ``residual_detections`` and
    ``escalated_to_block``. With ``brain_check=True`` the brain is
    extracted from the input (SynthStrip) and the report's ``brain`` block
    says how many brain voxels the anonymization modified (0 = untouched)
    and how close it came (``margin_mm``). ``partial_face=True`` (opt-in)
    also handles faces cut by the field of view, such as an axial slab that
    ends at the nose. ``auto_threshold=True`` (opt-in) retries noisy, dim
    scans (e.g. infants) at higher surface thresholds when no face is found. Verification is skipped for ``region="ears"``
    (the face legitimately remains detectable).

    Raises RuntimeError if no view yields a validated face."""
    def resolve(d):
        if d == "mediapipe":
            return None
        if d == "dlib":
            return _dlib_detector(predictor_path, detector_upsample)
        if callable(d):
            return d
        raise ValueError(f"unknown detector {d!r} (mediapipe|dlib|callable)")

    def run(d, guard):
        return vol.deface_auto(
            detector=resolve(d),
            deface=dict(mode=mode, region=region, **deface_kwargs),
            yaws=[float(y) for y in yaws],
            pitch=float(pitch),
            size=int(size),
            threshold_frac=float(threshold_frac),
            threshold=threshold,
            verify=verify,
            max_verify_rounds=int(max_verify_rounds),
            verify_pitches=[float(p) for p in verify_pitches],
            verify_effort=verify_effort,
            escalate_to_block=escalate_to_block,
            brain_check=brain_check,
            brain_guard=guard,
            brain_guard_mm=float(brain_guard_mm),
            partial_face=partial_face,
            auto_threshold=auto_threshold,
            return_mask=return_mask,
        )

    if fallback_detector is not None:
        resolve(fallback_detector)  # reject a bad value before any work
    used, fallback_used = detector, False
    try:
        report = run(detector, brain_guard)
    except RuntimeError:
        # A refusal leaves the volume untouched, so the fallback starts
        # from the original data.
        if fallback_detector is None:
            raise
        used, fallback_used = fallback_detector, True
        report = run(fallback_detector, brain_guard or fallback_brain_guard)
    report["detector"] = used if isinstance(used, str) else "custom"
    report["fallback_used"] = fallback_used
    if used == "dlib":
        parts = _PREDICTOR_PARTS.get(os.path.abspath(_predictor_path(predictor_path)))
        report["landmark_model_points"] = parts
        report["mouth_landmark_inferred"] = parts == 5
    return report


def anonymization_mask(
    vol,
    mode: str = "blur",
    region: str = "face",
    mask_path: Optional[str] = None,
    yaws: Sequence[float] = (-20.0, 0.0, 20.0),
    pitch: float = 0.0,
    size: int = 512,
    detector="mediapipe",
    predictor_path: Optional[str] = None,
    brain_check: bool = False,
    brain_guard: bool = False,
    brain_guard_mm: float = 3.0,
    partial_face: bool = False,
    threshold_frac: float = 0.14,
    threshold: Optional[float] = None,
    detector_upsample: int = 1,
    **deface_kwargs,
) -> dict:
    """Escape hatch: the region :func:`deface_volume` would anonymize,
    without modifying anything.

    ``vol`` is a :class:`Volume` or a NIfTI path. Detection, validation, the
    safety refusals and (with ``brain_check``) the brain check run exactly
    as in :func:`deface_volume`, with the same arguments; then it stops. The
    report's ``mask`` is a uint8 array on the volume's grid (1 = voxel that
    would be rewritten); ``mask_path`` also saves it as NIfTI with the
    volume's affine (needs nibabel).

    Nothing is rewritten and nothing is verified: the mask says where
    RadVeil would anonymize, not that anonymizing there defeats
    re-identification. If you process the region yourself, check the
    result yourself (for example with :func:`detect_landmarks` on renders).

    Raises RuntimeError if no plausible face is found."""
    if isinstance(vol, (str, os.PathLike)):
        vol = Volume(os.fspath(vol))
    if detector == "mediapipe":
        det = None
    elif detector == "dlib":
        det = _dlib_detector(predictor_path, detector_upsample)
    elif callable(detector):
        det = detector
    else:
        raise ValueError(f"unknown detector {detector!r} (mediapipe|dlib|callable)")
    report = vol.anonymization_mask(
        detector=det,
        deface=dict(mode=mode, region=region, **deface_kwargs),
        yaws=[float(y) for y in yaws],
        pitch=float(pitch),
        size=int(size),
        threshold_frac=float(threshold_frac),
        threshold=threshold,
        brain_check=brain_check,
        brain_guard=brain_guard,
        brain_guard_mm=float(brain_guard_mm),
        partial_face=partial_face,
    )
    report["detector"] = detector if isinstance(detector, str) else "custom"
    if mask_path is not None:
        import nibabel as nib

        nib.save(nib.Nifti1Image(report["mask"].astype(np.uint8), vol.affine), mask_path)
        report["mask_path"] = mask_path
    return report


def deface_auto(
    input_path: str,
    output_path: str,
    mode: str = "blur",
    mask_path: Optional[str] = None,
    qa_dir: Optional[str] = None,
    gzip_level: int = 1,
    **kwargs,
) -> dict:
    """File-to-file pipeline: load -> :func:`deface_volume` -> save.

    Accepts every :func:`deface_volume` keyword argument. Additionally:
    ``mask_path`` writes the binary anonymization mask as NIfTI (needs
    nibabel), ``qa_dir`` writes before/after renders with landmark
    overlays, ``gzip_level`` controls output compression (1 = fast
    default). ``brain_check=True`` adds a ``brain`` block to the report
    with the number of brain voxels the anonymization modified."""
    vol = Volume(input_path)
    report = deface_volume(
        vol, mode=mode, return_mask=mask_path is not None, **kwargs
    )
    views = report["views"]
    # keep the report JSON-friendly: per-view angles and scores only
    report["views"] = [
        {k: v[k] for k in ("yaw_deg", "pitch_deg", "score")} for v in views
    ]
    report.pop("landmarks_3d", None)

    if mask_path is not None:
        mask = report.pop("mask")
        try:
            import nibabel as nib

            ref = nib.load(input_path)
            nib.save(
                nib.Nifti1Image(mask.astype(np.uint8), ref.affine, ref.header), mask_path
            )
            report["mask_path"] = mask_path
        except ImportError:
            report["mask_path"] = None

    vol.save(output_path, gzip_level=gzip_level)
    report["output"] = output_path

    if qa_dir is not None:
        os.makedirs(qa_dir, exist_ok=True)
        _save_qa(views, Volume(input_path), vol, qa_dir, kwargs.get("size", 512))
    return report


def deskull(
    input_path: str,
    output_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    gzip_level: int = 1,
    **kwargs,
) -> dict:
    """File-to-file brain extraction (skull stripping).

    Writes the stripped volume to ``output_path`` and/or the uint8 brain
    mask to ``mask_path`` (needs nibabel). Keyword arguments go to
    :meth:`Volume.deskull` (``method`` = ``"synthstrip"`` (default) or
    ``"classical"``, ``modality``, ``fill``; classical: ``frac``,
    ``robust``, ``face``, ``detector``; synthstrip: ``low_memory``). Returns the report (without the mask array).
    """
    if output_path is None and mask_path is None:
        raise ValueError("nothing to write: pass output_path and/or mask_path")
    vol = Volume(input_path)
    report = vol.deskull(strip=output_path is not None, **kwargs)
    mask = report.pop("mask")
    if output_path is not None:
        vol.save(output_path, gzip_level=gzip_level)
        report["output"] = output_path
    if mask_path is not None:
        import nibabel as nib

        ref = nib.load(input_path)
        hdr = ref.header.copy()
        hdr.set_data_dtype(np.uint8)
        hdr["scl_slope"], hdr["scl_inter"] = 1.0, 0.0
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), ref.affine, hdr), mask_path)
        report["mask_path"] = mask_path
    return report


def _save_qa(views, vol_before, vol_after, qa_dir: str, size: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    for view in views:
        yaw, pitch = view["yaw_deg"], view["pitch_deg"]
        tag = f"yaw{yaw:+.0f}_pitch{pitch:+.0f}"
        img = Image.fromarray(vol_before.render(yaw_deg=yaw, pitch_deg=pitch, size=size)["rgb"])
        d = ImageDraw.Draw(img)
        for x, y in view["landmarks"]:
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 80))
        img.save(os.path.join(qa_dir, f"before_{tag}.png"))
        after = vol_after.render(yaw_deg=yaw, pitch_deg=pitch, size=size)
        Image.fromarray(after["rgb"]).save(os.path.join(qa_dir, f"after_{tag}.png"))


# ---------------------------------------------------------------------------
# Batch processing


def _batch_init(rayon_threads: int) -> None:
    # Called in each worker process before any Rust code runs; divides the
    # machine between workers so subject-level and voxel-level parallelism
    # do not oversubscribe each other.
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)


def _batch_worker(task):
    inp, out, mode, kwargs = task
    try:
        rep = deface_auto(inp, out, mode=mode, **kwargs)
        rep["input"] = inp
        return rep
    except Exception as e:  # a refusal is a result, not a crash
        return {"input": inp, "output": out, "error": str(e)}


def deface_batch(
    inputs,
    output_dir: Optional[str] = None,
    outputs: Optional[Sequence[str]] = None,
    mode: str = "blur",
    workers: Optional[int] = None,
    executor: str = "threads",
    **kwargs,
) -> list:
    """Anonymize many volumes concurrently.

    ``inputs`` is a list of paths or a glob pattern. Outputs are either
    given explicitly (``outputs``) or derived as
    ``output_dir/<name>_defaced.nii.gz``. All ``deface_auto`` keyword
    arguments (mode parameters, verification, thresholds...) pass through
    unchanged.

    ``executor="threads"`` (default) runs ``workers`` volumes at a time in
    this process: the Rust core releases the GIL, so threads scale like
    processes (measured: 11 scans/s with 14 threads on a 14-core laptop vs
    10 with 14 processes), share one copy of the detector models, and need
    no ``__main__`` guard. ``executor="processes"`` isolates each volume in
    a spawned worker process (call it under ``if __name__ == "__main__":``).
    ``workers`` defaults to ``min(cpu_count, 32, len(inputs))``; each
    concurrent volume needs roughly 4-5x its voxel data in memory
    (~70 MB for a 256x256x150 T1).

    Returns one report dict per input, in order; failed/refused inputs
    get ``{"input", "output", "error"}`` instead of raising."""
    import concurrent.futures as cf
    import glob as _glob
    import multiprocessing as mp

    if isinstance(inputs, str):
        inputs = sorted(_glob.glob(os.path.expanduser(inputs)))
    inputs = [str(p) for p in inputs]
    if outputs is None:
        if output_dir is None:
            raise ValueError("provide output_dir or outputs")
        os.makedirs(output_dir, exist_ok=True)

        def _out(p):
            name = os.path.basename(p)
            for suf in (".nii.gz", ".nii"):
                if name.endswith(suf):
                    name = name[: -len(suf)]
                    break
            return os.path.join(output_dir, f"{name}_defaced.nii.gz")

        outputs = [_out(p) for p in inputs]
    if len(outputs) != len(inputs):
        raise ValueError("inputs and outputs length mismatch")

    ncpu = os.cpu_count() or 1
    workers = max(1, workers or min(ncpu, 32, len(inputs)))
    tasks = [(i, o, mode, kwargs) for i, o in zip(inputs, outputs)]
    if workers == 1 or len(tasks) <= 1:
        return [_batch_worker(t) for t in tasks]
    if executor == "threads":
        # One process: Rust's thread pool is shared by all jobs, so no
        # per-worker sizing is needed.
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_batch_worker, tasks))
    if executor != "processes":
        raise ValueError(f"executor must be 'threads' or 'processes', not {executor!r}")
    rayon_threads = max(1, ncpu // workers)
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_batch_init, initargs=(rayon_threads,)
    ) as ex:
        return list(ex.map(_batch_worker, tasks))
