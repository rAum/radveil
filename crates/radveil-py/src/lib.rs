// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for radveil-core.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use radveil_core::pipeline::Detection;
use radveil_core::{
    anonymization_mask as core_anonymization_mask, brain_mask, deface as core_deface,
    deface_auto as core_deface_auto, deskull as core_deskull, face_frame_from_landmarks,
    locate_face, render_view, AutoParams, AutoReport, DefaceMode, DefaceParams, DefaceReport,
    DeskullParams, DetectorParams, FaceDetector, FaceFrame, MediaPipeDetector, Method, Modality,
    Region, RenderParams, Volume as CoreVolume,
};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// A loaded NIfTI volume.
#[pyclass]
struct Volume {
    inner: CoreVolume,
    source: String,
}

#[pymethods]
impl Volume {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let inner = CoreVolume::load(&path).map_err(err)?;
        Ok(Volume {
            inner,
            source: path,
        })
    }

    /// Build a Volume from an in-memory float32 array (nx, ny, nz) and a
    /// 4x4 voxel->world (RAS+, mm) affine. Use this for DICOM series or raw
    /// data: read with pydicom/SimpleITK/nibabel, pass the array + affine.
    #[staticmethod]
    fn from_array(
        data: numpy::PyReadonlyArray3<f32>,
        affine: numpy::PyReadonlyArray2<f64>,
    ) -> PyResult<Self> {
        let arr = data.as_array().to_owned();
        let a = affine.as_array();
        if a.shape() != [4, 4] {
            return Err(PyValueError::new_err("affine must be 4x4"));
        }
        let mut m = [[0.0f64; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                m[r][c] = a[[r, c]];
            }
        }
        Ok(Volume {
            inner: CoreVolume::from_array(arr, m),
            source: "<array>".to_string(),
        })
    }

    /// Parse a NIfTI volume from bytes (.nii or .nii.gz content), e.g. an
    /// HTTP upload or object-store blob -- no temporary file needed.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        let inner = py.detach(|| CoreVolume::from_bytes(data)).map_err(err)?;
        Ok(Volume {
            inner,
            source: "<bytes>".to_string(),
        })
    }

    /// Encode as NIfTI bytes: gzipped at `gzip_level` (default 1), or plain
    /// .nii with `gzip_level=None`. Identical to what `save` writes.
    #[pyo3(signature = (gzip_level=Some(1)))]
    fn to_bytes<'py>(
        &self,
        py: Python<'py>,
        gzip_level: Option<u32>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = py.detach(|| self.inner.to_bytes(gzip_level)).map_err(err)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let [a, b, c] = self.inner.shape();
        (a, b, c)
    }

    #[getter]
    fn spacing(&self) -> (f64, f64, f64) {
        let s = self.inner.spacing;
        (s[0], s[1], s[2])
    }

    /// Orientation codes of the voxel axes, e.g. "RAS", "LPS".
    #[getter]
    fn orientation(&self) -> String {
        self.inner.orientation()
    }

    /// Voxel -> world (RAS+, mm) affine as a 4x4 float64 array.
    #[getter]
    fn affine<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut flat = Vec::with_capacity(16);
        for r in 0..4 {
            flat.extend_from_slice(&self.inner.affine[r]);
        }
        let arr = PyArray1::from_vec(py, flat);
        arr.reshape([4, 4]).unwrap()
    }

    /// Voxel data (scaled intensities) as float32 (nx, ny, nz). Copy.
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        PyArray3::from_array(py, &self.inner.data)
    }

    /// Intensity percentile (approximate, subsampled).
    fn percentile(&self, pct: f64) -> f32 {
        self.inner.percentile(pct)
    }

    /// Render one view. Returns dict with:
    ///   rgb  uint8   (h, w, 3)
    ///   pos  float32 (h, w, 3) world position per pixel (NaN = miss)
    ///   hit  bool    (h, w)
    ///   yaw_deg / pitch_deg / threshold / pixel_mm
    #[pyo3(signature = (yaw_deg=0.0, pitch_deg=0.0, size=512, threshold_frac=0.14,
                        threshold=None, step_mm=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn render<'py>(
        &self,
        py: Python<'py>,
        yaw_deg: f64,
        pitch_deg: f64,
        size: usize,
        threshold_frac: f32,
        threshold: Option<f32>,
        step_mm: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let params = RenderParams {
            size,
            threshold_frac,
            threshold_abs: threshold,
            step_mm,
        };
        let view = py.detach(|| render_view(&self.inner, yaw_deg, pitch_deg, &params));
        let d = PyDict::new(py);
        let rgb = PyArray1::from_vec(py, view.rgb.clone())
            .reshape([size, size, 3])
            .unwrap();
        let pos = PyArray1::from_vec(py, view.pos.clone())
            .reshape([size, size, 3])
            .unwrap();
        let hit = PyArray1::from_vec(py, view.hit.to_vec())
            .reshape([size, size])
            .unwrap();
        let boundary = PyArray1::from_vec(py, view.boundary.to_vec())
            .reshape([size, size])
            .unwrap();
        d.set_item("rgb", rgb)?;
        d.set_item("pos", pos)?;
        d.set_item("hit", hit)?;
        d.set_item("boundary", boundary)?;
        d.set_item("yaw_deg", yaw_deg)?;
        d.set_item("pitch_deg", pitch_deg)?;
        d.set_item("threshold", view.threshold)?;
        d.set_item("pixel_mm", view.camera.pixel_mm)?;
        Ok(d)
    }

    /// Anonymize using 3D world-space (RAS+, mm) 68-point landmarks.
    /// `points`: sequence of 68 items, each (x, y, z) or None.
    /// Returns a report dict; the volume is modified in place.
    /// `ear_mode` (default None = same as `mode`) sets a separate mode for
    /// the ear regions, e.g. `mode="adaptive", ear_mode="cut"`.
    /// `region` is "face" (default), "ears" or "face+ears". Ears are
    /// located by surface evidence: a protrusion scan in a calibrated
    /// window detects each pinna, and the cut follows the measured head
    /// surface (see the report's per-ROI `detected` / `protrusion_mm`
    /// fields; ear bounds are in the head-based ear frame).
    /// All geometric parameters default to *auto*: scaled from the detected
    /// inter-eye distance. Pass explicit values (mm) to override.
    /// `eye_surface_anchor` (default True) anchors the face ROI's depth
    /// bounds to the measured eye-skin surface, correcting the few-mm
    /// posterior landmark bias of thick-slice scans (report key
    /// `eye_bias_mm`); pass False for raw landmark-plane anchoring.
    #[pyo3(signature = (points, mode="blur", region="face", ear_mode=None, eye_surface_anchor=true,
                        eye_bias_mm=None, blur_sigma_mm=None,
                        adaptive_factor=None, depth_behind_mm=None, depth_front_mm=None,
                        width_factor=None, above_eyes_mm=None, below_chin_mm=None,
                        ear_inner_mm=None, ear_outer_mm=None, ear_guard_mm=None,
                        ear_down_mm=None, ear_back_mm=None,
                        ear_height_mm=None, ear_depth_mm=None,
                        return_mask=false, skin_depth_mm=None, spare_air=false,
                        level_face=false))]
    #[allow(clippy::too_many_arguments)]
    fn deface<'py>(
        &mut self,
        py: Python<'py>,
        points: Vec<Option<(f64, f64, f64)>>,
        mode: &str,
        region: &str,
        ear_mode: Option<&str>,
        eye_surface_anchor: bool,
        eye_bias_mm: Option<f64>,
        blur_sigma_mm: Option<f64>,
        adaptive_factor: Option<f64>,
        depth_behind_mm: Option<f64>,
        depth_front_mm: Option<f64>,
        width_factor: Option<f64>,
        above_eyes_mm: Option<f64>,
        below_chin_mm: Option<f64>,
        ear_inner_mm: Option<f64>,
        ear_outer_mm: Option<f64>,
        ear_guard_mm: Option<f64>,
        ear_down_mm: Option<f64>,
        ear_back_mm: Option<f64>,
        ear_height_mm: Option<f64>,
        ear_depth_mm: Option<f64>,
        return_mask: bool,
        skin_depth_mm: Option<f64>,
        spare_air: bool,
        level_face: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let lm: Vec<Option<[f64; 3]>> = points
            .into_iter()
            .map(|p| p.map(|(x, y, z)| [x, y, z]))
            .collect();
        let frame = face_frame_from_landmarks(&lm).map_err(PyValueError::new_err)?;
        let params = DefaceParams {
            mode: mode.parse::<DefaceMode>().map_err(PyValueError::new_err)?,
            region: region.parse::<Region>().map_err(PyValueError::new_err)?,
            ear_mode: ear_mode
                .map(|m| m.parse::<DefaceMode>().map_err(PyValueError::new_err))
                .transpose()?,
            eye_surface_anchor: Some(eye_surface_anchor),
            eye_bias_mm,
            blur_sigma_mm,
            adaptive_factor,
            depth_behind_mm,
            skin_depth_mm,
            depth_front_mm,
            width_factor,
            above_eyes_mm,
            below_chin_mm,
            ear_inner_mm,
            ear_outer_mm,
            ear_guard_mm,
            ear_down_mm,
            ear_back_mm,
            ear_height_mm,
            ear_depth_mm,
            spare_air,
            level_face,
            surface_threshold: None,
        };
        let (report, mask) = py.detach(|| core_deface(&mut self.inner, &frame, &params));
        let d = PyDict::new(py);
        fill_deface_report(&d, &report, &frame)?;
        if return_mask {
            d.set_item("mask", PyArray3::from_array(py, &mask))?;
        }
        Ok(d)
    }

    /// Full automatic pipeline on this volume (modified in place):
    /// render -> detect -> validate -> lift -> anonymize -> verify.
    ///
    /// `detector` is None for the built-in MediaPipe detector, or a Python
    /// callable ``f(rgb, try_rotations)`` returning None, an (68, 2) array
    /// of 68-point pixel landmarks, or ``(array, score)``.
    /// `deface` is a dict of :meth:`deface` keyword arguments (mode,
    /// region, geometry overrides).
    ///
    /// ``brain_check=True`` also extracts the brain from the input
    /// (SynthStrip) and reports, under ``brain``, how many brain voxels the
    /// anonymization modified (0 = brain untouched) and how close it came.
    /// ``brain_guard=True`` (opt-in, ~3 s) never rewrites the extracted
    /// brain or anything within ``brain_guard_mm`` of it.
    #[pyo3(signature = (detector=None, deface=None, yaws=vec![-20.0, 0.0, 20.0], pitch=0.0,
                        size=512, threshold_frac=0.14, threshold=None, verify=true,
                        max_verify_rounds=3, escalate_to_block=true, return_mask=false,
                        brain_check=false, verify_pitches=vec![0.0, -15.0],
                        brain_guard=false, brain_guard_mm=3.0, verify_effort="balanced",
                        partial_face=false, auto_threshold=false))]
    #[allow(clippy::too_many_arguments)]
    fn deface_auto<'py>(
        &mut self,
        py: Python<'py>,
        detector: Option<Py<PyAny>>,
        deface: Option<&Bound<'py, PyDict>>,
        yaws: Vec<f64>,
        pitch: f64,
        size: usize,
        threshold_frac: f32,
        threshold: Option<f32>,
        verify: bool,
        max_verify_rounds: usize,
        escalate_to_block: bool,
        return_mask: bool,
        brain_check: bool,
        verify_pitches: Vec<f64>,
        brain_guard: bool,
        brain_guard_mm: f64,
        verify_effort: &str,
        partial_face: bool,
        auto_threshold: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let params = AutoParams {
            deface: deface_params(deface)?,
            yaws,
            pitch,
            size,
            threshold_frac,
            threshold,
            verify,
            verify_effort: verify_effort.parse().map_err(PyValueError::new_err)?,
            verify_pitches,
            max_verify_rounds,
            escalate_to_block,
            return_mask,
            brain_check,
            brain_guard,
            brain_guard_mm,
            partial_face,
            auto_threshold,
            // A custom/dlib detector is checked by the built-in one too.
            verify_with_builtin: detector.is_some(),
        };
        let inner = &mut self.inner;
        let rep = py
            .detach(|| match detector {
                None => core_deface_auto(inner, &mut MediaPipeDetector::default(), &params),
                Some(f) => core_deface_auto(inner, &mut PyDetector(f), &params),
            })
            .map_err(err)?;
        auto_report_dict(py, &rep)
    }

    /// Escape hatch: the region :meth:`deface_auto` would anonymize, as a
    /// uint8 ``mask`` (1 = voxel it would rewrite), **without modifying the
    /// volume**. Detection, validation, the safety refusals and (with
    /// ``brain_check``) the brain check run as in :meth:`deface_auto`;
    /// nothing is rewritten and nothing is verified, so the mask says where
    /// RadVeil would anonymize, not that anonymizing there defeats
    /// re-identification. If you process the region yourself, verifying the
    /// result is up to you. Arguments as in :meth:`deface_auto`; returns its
    /// report dict (without the verification keys) plus ``mask``.
    #[pyo3(signature = (detector=None, deface=None, yaws=vec![-20.0, 0.0, 20.0], pitch=0.0,
                        size=512, threshold_frac=0.14, threshold=None, brain_check=false,
                        brain_guard=false, brain_guard_mm=3.0, partial_face=false))]
    #[allow(clippy::too_many_arguments)]
    fn anonymization_mask<'py>(
        &self,
        py: Python<'py>,
        detector: Option<Py<PyAny>>,
        deface: Option<&Bound<'py, PyDict>>,
        yaws: Vec<f64>,
        pitch: f64,
        size: usize,
        threshold_frac: f32,
        threshold: Option<f32>,
        brain_check: bool,
        brain_guard: bool,
        brain_guard_mm: f64,
        partial_face: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let params = AutoParams {
            deface: deface_params(deface)?,
            yaws,
            pitch,
            size,
            threshold_frac,
            threshold,
            brain_check,
            brain_guard,
            brain_guard_mm,
            partial_face,
            ..Default::default()
        };
        let inner = &self.inner;
        let rep = py
            .detach(|| match detector {
                None => core_anonymization_mask(inner, &mut MediaPipeDetector::default(), &params),
                Some(f) => core_anonymization_mask(inner, &mut PyDetector(f), &params),
            })
            .map_err(err)?;
        auto_report_dict(py, &rep)
    }

    /// Brain extraction (skull stripping). ``method`` is ``"synthstrip"``
    /// (default; the SynthStrip network in pure Rust, any contrast/CT) or
    /// ``"classical"`` (model-free). Returns a dict with ``mask`` (uint8 array, 1 = brain) and
    /// the report. With ``strip`` (default) the volume is modified in
    /// place: non-brain voxels are set to ``fill`` (default 0 for MRI, air
    /// for CT).
    ///
    /// ``low_memory=True`` trades ≤3% speed for ~0.4 GB less peak memory
    /// in the synthstrip method (identical output).
    ///
    /// ``face=True`` (classical method) first locates the face (``detector`` as in
    /// :meth:`deface_auto`; None = built-in) and uses it as an anatomical
    /// prior for MRI; if no face is found the plain method runs and
    /// ``face_prior`` is False in the report.
    #[pyo3(signature = (method="synthstrip", modality="auto", frac=0.5, fill=None, robust=true,
                        face=false, detector=None, strip=true, low_memory=false))]
    #[allow(clippy::too_many_arguments)]
    fn deskull<'py>(
        &mut self,
        py: Python<'py>,
        method: &str,
        modality: &str,
        frac: f64,
        fill: Option<f32>,
        robust: bool,
        face: bool,
        detector: Option<Py<PyAny>>,
        strip: bool,
        low_memory: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let method = Method::parse(method)
            .ok_or_else(|| PyValueError::new_err(format!("unknown method {method:?}")))?;
        let modality = Modality::parse(modality)
            .ok_or_else(|| PyValueError::new_err(format!("unknown modality {modality:?}")))?;
        let inner = &mut self.inner;
        let (mask, rep, frame, face_error) = py
            .detach(|| {
                let (frame, face_error) = if face {
                    let located = match detector {
                        None => locate_face(
                            inner,
                            &mut MediaPipeDetector::default(),
                            &AutoParams::default(),
                        ),
                        Some(f) => locate_face(inner, &mut PyDetector(f), &AutoParams::default()),
                    };
                    match located {
                        Ok(l) => (Some(l.frame), None),
                        Err(e) => (None, Some(e)),
                    }
                } else {
                    (None, None)
                };
                let params = DeskullParams {
                    method,
                    synthstrip: radveil_core::SynthStripParams {
                        low_memory,
                        ..Default::default()
                    },
                    modality,
                    frac,
                    fill,
                    robust,
                    face: frame.clone(),
                    ..Default::default()
                };
                let res = if strip {
                    core_deskull(inner, &params)
                } else {
                    brain_mask(inner, &params)
                };
                res.map(|(mask, rep)| (mask, rep, frame, face_error))
            })
            .map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("mask", PyArray3::from_array(py, &mask))?;
        d.set_item("method", &rep.method)?;
        d.set_item(
            "modality",
            match rep.modality {
                Modality::Ct => "ct",
                _ => "mri",
            },
        )?;
        d.set_item("brain_volume_ml", rep.brain_volume_ml)?;
        d.set_item("brain_voxels", rep.brain_voxels)?;
        d.set_item("centroid_world", rep.centroid_world.to_vec())?;
        d.set_item("fill", rep.fill)?;
        d.set_item("passes", rep.passes)?;
        d.set_item("face_prior", rep.face_prior)?;
        if let Some(f) = frame {
            d.set_item("eye_dist_mm", f.eye_dist)?;
        }
        if let Some(e) = face_error {
            d.set_item("face_error", e)?;
        }
        Ok(d)
    }

    /// Save the (possibly modified) volume, preserving the original header,
    /// datatype and value scaling. Path may end in .nii or .nii.gz.
    /// gzip_level 1 (default) is ~5x faster than 6 for ~3% larger files.
    #[pyo3(signature = (path, gzip_level=1))]
    fn save(&self, py: Python<'_>, path: String, gzip_level: u32) -> PyResult<()> {
        py.detach(|| self.inner.save_with_compression(&path, gzip_level))
            .map_err(err)
    }

    fn __repr__(&self) -> String {
        let [a, b, c] = self.inner.shape();
        format!(
            "Volume('{}', shape=({a}, {b}, {c}), orientation={}, dtype={:?})",
            self.source,
            self.inner.orientation(),
            self.inner.datatype
        )
    }
}

/// Anonymization parameters from a dict of :meth:`Volume.deface` keywords.
fn deface_params(kw: Option<&Bound<'_, PyDict>>) -> PyResult<DefaceParams> {
    let mut p = DefaceParams::default();
    let Some(kw) = kw else {
        return Ok(p);
    };
    for (k, v) in kw.iter() {
        let key: String = k.extract()?;
        let mm = || -> PyResult<Option<f64>> { v.extract() };
        match key.as_str() {
            "mode" => {
                p.mode = v
                    .extract::<String>()?
                    .parse()
                    .map_err(PyValueError::new_err)?
            }
            "region" => {
                p.region = v
                    .extract::<String>()?
                    .parse()
                    .map_err(PyValueError::new_err)?
            }
            "ear_mode" => {
                p.ear_mode = v
                    .extract::<Option<String>>()?
                    .map(|m| m.parse().map_err(PyValueError::new_err))
                    .transpose()?
            }
            "eye_surface_anchor" => p.eye_surface_anchor = Some(v.extract()?),
            "spare_air" => p.spare_air = v.extract()?,
            "level_face" => p.level_face = v.extract()?,
            "eye_bias_mm" => p.eye_bias_mm = mm()?,
            "blur_sigma_mm" => p.blur_sigma_mm = mm()?,
            "adaptive_factor" => p.adaptive_factor = mm()?,
            "depth_behind_mm" => p.depth_behind_mm = mm()?,
            "skin_depth_mm" => p.skin_depth_mm = mm()?,
            "depth_front_mm" => p.depth_front_mm = mm()?,
            "width_factor" => p.width_factor = mm()?,
            "above_eyes_mm" => p.above_eyes_mm = mm()?,
            "below_chin_mm" => p.below_chin_mm = mm()?,
            "ear_inner_mm" => p.ear_inner_mm = mm()?,
            "ear_outer_mm" => p.ear_outer_mm = mm()?,
            "ear_guard_mm" => p.ear_guard_mm = mm()?,
            "ear_down_mm" => p.ear_down_mm = mm()?,
            "ear_back_mm" => p.ear_back_mm = mm()?,
            "ear_height_mm" => p.ear_height_mm = mm()?,
            "ear_depth_mm" => p.ear_depth_mm = mm()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown anonymization parameter '{other}'"
                )))
            }
        }
    }
    Ok(p)
}

/// A Python callable used as the pipeline's face detector.
struct PyDetector(Py<PyAny>);

impl FaceDetector for PyDetector {
    fn detect(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
        try_rotations: bool,
    ) -> Result<Option<Detection>, String> {
        Python::attach(|py| -> PyResult<Option<Detection>> {
            let img = PyArray1::from_slice(py, rgb).reshape([h, w, 3])?;
            let out = self.0.call1(py, (img, try_rotations))?;
            let out = out.bind(py);
            if out.is_none() {
                return Ok(None);
            }
            let (arr, score) = match out.extract::<(Bound<'_, PyAny>, f64)>() {
                Ok(pair) => pair,
                Err(_) => (out.clone(), 1.0),
            };
            let arr: PyReadonlyArray2<f64> = arr.extract()?;
            let a = arr.as_array();
            if a.ncols() != 2 {
                return Err(PyValueError::new_err(
                    "detector must return an (N, 2) array",
                ));
            }
            Ok(Some(Detection {
                landmarks: a.rows().into_iter().map(|r| [r[0], r[1]]).collect(),
                score,
            }))
        })
        .map_err(|e| e.to_string())
    }
}

/// The report dict of `deface_auto` / `anonymization_mask`.
fn auto_report_dict<'py>(py: Python<'py>, rep: &AutoReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    fill_deface_report(&d, &rep.deface, &rep.frame)?;
    d.set_item("views_detected", rep.views_detected)?;
    d.set_item("orientation", &rep.orientation)?;
    d.set_item("orientation_sweep", rep.orientation_sweep)?;
    let pts: Vec<Option<(f64, f64, f64)>> = rep
        .landmarks_3d
        .iter()
        .map(|p| p.map(|p| (p[0], p[1], p[2])))
        .collect();
    d.set_item("landmarks_3d", pts)?;
    let views = PyList::empty(py);
    for v in &rep.views {
        let vd = PyDict::new(py);
        vd.set_item("yaw_deg", v.yaw_deg)?;
        vd.set_item("pitch_deg", v.pitch_deg)?;
        vd.set_item("score", v.score)?;
        let flat: Vec<f64> = v.landmarks.iter().flatten().copied().collect();
        let n = v.landmarks.len();
        vd.set_item("landmarks", PyArray1::from_vec(py, flat).reshape([n, 2])?)?;
        views.append(vd)?;
    }
    d.set_item("views", views)?;
    if let Some(r) = rep.verify_rounds {
        d.set_item("verify_rounds", r)?;
        d.set_item("residual_detections", rep.residual_detections)?;
        d.set_item("escalated_to_block", rep.escalated_to_block)?;
        d.set_item("final_blur_sigma_mm", rep.final_blur_sigma_mm)?;
    }
    if let Some(mask) = &rep.mask {
        d.set_item("mask", PyArray3::from_array(py, mask))?;
    }
    if let Some(b) = &rep.brain {
        let bd = PyDict::new(py);
        bd.set_item("voxels_modified", b.voxels_modified)?;
        bd.set_item("brain_volume_ml", b.brain_volume_ml)?;
        bd.set_item("margin_mm", b.margin_mm)?;
        d.set_item("brain", bd)?;
    }
    Ok(d)
}

fn fill_deface_report(
    d: &Bound<'_, PyDict>,
    report: &DefaceReport,
    frame: &FaceFrame,
) -> PyResult<()> {
    let py = d.py();
    let region = match report.region {
        Region::Face => "face",
        Region::Ears => "ears",
        Region::FaceAndEars => "face+ears",
    };
    d.set_item("mode", format!("{:?}", report.mode).to_lowercase())?;
    d.set_item("ear_mode", format!("{:?}", report.ear_mode).to_lowercase())?;
    d.set_item("region", region)?;
    d.set_item("voxels_changed", report.voxels_changed)?;
    d.set_item("voxels_total", report.voxels_total)?;
    d.set_item("eye_dist_mm", report.eye_dist_mm)?;
    d.set_item("eye_bias_mm", report.eye_bias_mm)?;
    let rois = PyList::empty(py);
    for r in &report.rois {
        let rd = PyDict::new(py);
        rd.set_item("label", &r.label)?;
        rd.set_item("u_mm", r.u_mm.to_vec())?;
        rd.set_item("v_mm", r.v_mm.to_vec())?;
        rd.set_item("w_mm", r.w_mm.to_vec())?;
        rd.set_item("voxels", r.voxels)?;
        if let Some(det) = r.detected {
            rd.set_item("detected", det)?;
            rd.set_item("protrusion_mm", r.protrusion_mm)?;
        }
        rois.append(rd)?;
    }
    d.set_item("rois", rois)?;
    // Legacy flat keys: the face ROI bounds, when the face is included.
    if let Some(f) = report.rois.iter().find(|r| r.label == "face") {
        d.set_item("roi_u_mm", f.u_mm.to_vec())?;
        d.set_item("roi_v_mm", f.v_mm.to_vec())?;
        d.set_item("roi_w_mm", f.w_mm.to_vec())?;
    }
    d.set_item("blur_sigma_mm", report.resolved.blur_sigma_mm)?;
    d.set_item("depth_behind_mm", report.resolved.depth_behind_mm)?;
    d.set_item("skin_depth_mm", report.resolved.skin_depth_mm)?;
    if let Some(a) = report.face_seen_through_air {
        d.set_item("face_seen_through_air", a)?;
    }
    if let Some(n) = report.protected_voxels_spared {
        d.set_item("protected_voxels_spared", n)?;
    }
    if let Some(n) = report.air_voxels_spared {
        d.set_item("air_voxels_spared", n)?;
    }
    d.set_item(
        "nose_tip",
        (frame.nose_tip[0], frame.nose_tip[1], frame.nose_tip[2]),
    )?;
    Ok(())
}

/// Built-in MediaPipe face detector on an RGB8 image (h, w, 3). Returns
/// ``(landmarks, score)`` -- (68, 2) float64 pixel coordinates in the 68-point
/// order plus the detection confidence -- or None when no face is found.
#[pyfunction]
#[pyo3(signature = (rgb, try_rotations=false, min_score=0.5))]
fn detect_face<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<u8>,
    try_rotations: bool,
    min_score: f64,
) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, f64)>> {
    let a = rgb.as_array();
    let (h, w, c) = a.dim();
    if c != 3 {
        return Err(PyValueError::new_err("rgb must be (h, w, 3) uint8"));
    }
    let buf: Vec<u8> = a.iter().copied().collect();
    let det = MediaPipeDetector::new(DetectorParams {
        min_score,
        ..Default::default()
    });
    let mesh = py
        .detach(|| det.detect(&buf, w, h, try_rotations))
        .map_err(err)?;
    let Some(m) = mesh else {
        return Ok(None);
    };
    let flat: Vec<f64> = m.landmarks68().into_iter().flatten().collect();
    Ok(Some((
        PyArray1::from_vec(py, flat).reshape([68, 2])?,
        m.detection.score,
    )))
}

#[pymodule]
fn _radveil(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Volume>()?;
    m.add_function(wrap_pyfunction!(detect_face, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
