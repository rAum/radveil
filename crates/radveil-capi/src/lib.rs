// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! C ABI for RadVeil.
//!
//! Conventions:
//! - Functions returning `i32` use 0 = success, negative = error; the
//!   message is available via `radveil_last_error()` (thread-local).
//! - `RadveilVolume` is opaque; free with `radveil_volume_free`.
//! - Arrays are C-order `(nx, ny, nz)` with `k` fastest.
//! - `radveil_deface_auto` runs the whole pipeline with the built-in
//!   MediaPipe face detector. Hosts with their own detector (or manual
//!   annotation) instead supply 68-point landmarks as 3D
//!   world-space (RAS+, mm) points for `radveil_deface`, lifting 2D
//!   detections through the position maps returned by `radveil_render`.

use ndarray::Array3;
use radveil_core::{
    anonymization_mask, brain_mask, deface, deface_auto, deskull, face_frame_from_landmarks,
    locate_face, render_view, AutoParams, DefaceMode, DefaceParams, DeskullParams,
    MediaPipeDetector, Method, Modality, Region, RenderParams, VerifyEffort, Volume,
};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_error(msg: impl std::fmt::Display) -> i32 {
    let c = CString::new(msg.to_string()).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
    -1
}

/// Message for the last failed call on this thread. Valid until the next
/// failing call on the same thread.
#[no_mangle]
pub extern "C" fn radveil_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Library version as a static string.
#[no_mangle]
pub extern "C" fn radveil_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

pub struct RadveilVolume(Volume);

/// Load a NIfTI volume (.nii / .nii.gz). NULL on failure.
///
/// # Safety
/// `path` must be a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn radveil_volume_load(path: *const c_char) -> *mut RadveilVolume {
    let path = match CStr::from_ptr(path).to_str() {
        Ok(p) => p,
        Err(e) => {
            set_error(e);
            return std::ptr::null_mut();
        }
    };
    match Volume::load(path) {
        Ok(v) => Box::into_raw(Box::new(RadveilVolume(v))),
        Err(e) => {
            set_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Build a volume from a C-order float32 array and a row-major 4x4
/// voxel-to-world (RAS+, mm) affine. Data is copied. NULL on failure.
///
/// # Safety
/// `data` must point to `nx*ny*nz` floats, `affine` to 16 doubles.
#[no_mangle]
pub unsafe extern "C" fn radveil_volume_from_array(
    nx: usize,
    ny: usize,
    nz: usize,
    data: *const f32,
    affine: *const f64,
) -> *mut RadveilVolume {
    if data.is_null() || affine.is_null() || nx * ny * nz == 0 {
        set_error("null or empty input");
        return std::ptr::null_mut();
    }
    let slice = std::slice::from_raw_parts(data, nx * ny * nz);
    let arr = match Array3::from_shape_vec((nx, ny, nz), slice.to_vec()) {
        Ok(a) => a,
        Err(e) => {
            set_error(e);
            return std::ptr::null_mut();
        }
    };
    let a = std::slice::from_raw_parts(affine, 16);
    let mut m = [[0.0f64; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            m[r][c] = a[r * 4 + c];
        }
    }
    Box::into_raw(Box::new(RadveilVolume(Volume::from_array(arr, m))))
}

/// # Safety
/// `vol` must be a pointer returned by a radveil constructor (or NULL).
#[no_mangle]
pub unsafe extern "C" fn radveil_volume_free(vol: *mut RadveilVolume) {
    if !vol.is_null() {
        drop(Box::from_raw(vol));
    }
}

/// Writes the shape into `dims[3]`.
///
/// # Safety
/// Valid `vol`; `dims` points to 3 u64.
#[no_mangle]
pub unsafe extern "C" fn radveil_volume_shape(vol: *const RadveilVolume, dims: *mut u64) {
    let v = &(*vol).0;
    for (i, &d) in v.shape().iter().enumerate() {
        *dims.add(i) = d as u64;
    }
}

/// Copies the voxel-axis orientation codes (e.g. "LPS") into `buf`
/// (needs >= 4 bytes, NUL-terminated).
///
/// # Safety
/// Valid `vol`; `buf` points to at least `len` bytes, `len >= 4`.
#[no_mangle]
pub unsafe extern "C" fn radveil_volume_orientation(
    vol: *const RadveilVolume,
    buf: *mut c_char,
    len: usize,
) {
    let s = (*vol).0.orientation();
    let bytes = s.as_bytes();
    let n = bytes.len().min(len.saturating_sub(1));
    std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, n);
    *buf.add(n) = 0;
}

/// Save preserving datatype/scaling/header. gzip_level 0-9 (1 = fast
/// default). Returns 0 on success.
///
/// # Safety
/// Valid `vol`; `path` NUL-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn radveil_volume_save(
    vol: *const RadveilVolume,
    path: *const c_char,
    gzip_level: u32,
) -> i32 {
    let path = match CStr::from_ptr(path).to_str() {
        Ok(p) => p,
        Err(e) => return set_error(e),
    };
    match (*vol).0.save_with_compression(path, gzip_level) {
        Ok(()) => 0,
        Err(e) => set_error(e),
    }
}

/// Render one surface view.
///
/// Outputs (caller-allocated): `rgb` = size*size*3 bytes, `pos` =
/// size*size*3 floats (world mm; NaN where no hit), `flags` = size*size
/// bytes (0 = miss, 1 = hit, 2 = hit on volume boundary). Any output
/// pointer may be NULL to skip it. `threshold_abs` = NaN selects the
/// automatic threshold. Returns 0 on success.
///
/// # Safety
/// Valid `vol`; output pointers NULL or sized as documented.
#[no_mangle]
pub unsafe extern "C" fn radveil_render(
    vol: *const RadveilVolume,
    yaw_deg: f64,
    pitch_deg: f64,
    size: u32,
    threshold_frac: f32,
    threshold_abs: f32,
    rgb: *mut u8,
    pos: *mut f32,
    flags: *mut u8,
) -> i32 {
    if size == 0 || size > 8192 {
        return set_error("size out of range");
    }
    let params = RenderParams {
        size: size as usize,
        threshold_frac,
        threshold_abs: if threshold_abs.is_nan() {
            None
        } else {
            Some(threshold_abs)
        },
        step_mm: 0.0,
    };
    let view = render_view(&(*vol).0, yaw_deg, pitch_deg, &params);
    let n = (size * size) as usize;
    if !rgb.is_null() {
        std::ptr::copy_nonoverlapping(view.rgb.as_ptr(), rgb, n * 3);
    }
    if !pos.is_null() {
        std::ptr::copy_nonoverlapping(view.pos.as_ptr(), pos, n * 3);
    }
    if !flags.is_null() {
        for i in 0..n {
            *flags.add(i) = if view.hit[i] {
                if view.boundary[i] {
                    2
                } else {
                    1
                }
            } else {
                0
            };
        }
    }
    0
}

/// Geometric parameters; any negative/NaN field means "auto" (scaled
/// from the detected inter-eye distance).
#[repr(C)]
pub struct RadveilDefaceParams {
    pub width_factor: f64,
    pub above_eyes_mm: f64,
    pub below_chin_mm: f64,
    pub depth_behind_mm: f64,
    pub depth_front_mm: f64,
    pub blur_sigma_mm: f64,
    pub adaptive_factor: f64,
    /// Depth below the measured skin that the face ROI reaches, mm.
    pub skin_depth_mm: f64,
    /// Opt-in bits: [`RADVEIL_SPARE_AIR`], [`RADVEIL_LEVEL_FACE`],
    /// [`RADVEIL_PARTIAL_FACE`], [`RADVEIL_AUTO_THRESHOLD`].
    pub options: u32,
}

/// `RadveilDefaceParams.options`: leave background air untouched.
pub const RADVEIL_SPARE_AIR: u32 = 1;
/// `RadveilDefaceParams.options`: measure the face skin along the level
/// head axes (the lower lip and chin are removed too).
pub const RADVEIL_LEVEL_FACE: u32 = 2;
/// `RadveilDefaceParams.options` (radveil_deface_auto and
/// radveil_anonymization_mask only): also handle faces cut by the field of
/// view (an axial slab ending at the nose); implies `RADVEIL_LEVEL_FACE`.
pub const RADVEIL_PARTIAL_FACE: u32 = 4;
/// `RadveilDefaceParams.options` (radveil_deface_auto and
/// radveil_anonymization_mask only): noisy, dim scans (e.g. infants) are
/// retried at higher surface thresholds when no face is found.
pub const RADVEIL_AUTO_THRESHOLD: u32 = 8;

/// Ear-ROI geometry for `radveil_deface_region`; any negative/NaN field
/// means "auto" (scaled from the detected inter-eye distance). Ears are
/// located by surface evidence (protrusion detection inside a calibrated
/// window; the cut wall follows the measured scalp minus `guard_mm`, never
/// medial of the skull or `inner_mm`). down/back center the fallback box
/// (used when no pinna is detected) below the eye line / behind the eye
/// plane in the head-based ear frame; height/depth are its full v/w
/// extents.
#[repr(C)]
pub struct RadveilEarParams {
    pub inner_mm: f64, /* absolute medial floor */
    pub outer_mm: f64,
    pub guard_mm: f64, /* cut depth below the detected surface */
    pub down_mm: f64,
    pub back_mm: f64,
    pub height_mm: f64,
    pub depth_mm: f64,
}

#[repr(C)]
pub struct RadveilReport {
    pub voxels_changed: u64,
    pub voxels_total: u64,
    pub eye_dist_mm: f64,
    pub blur_sigma_mm: f64,
}

fn opt(v: f64) -> Option<f64> {
    if v.is_finite() && v >= 0.0 {
        Some(v)
    } else {
        None
    }
}

/// Anonymize in place using 68-point world-space (RAS+, mm)
/// landmarks: `points` = 68*3 doubles; a NaN in any coordinate marks the
/// landmark as missing. `mode` is "blur", "cut", "block" or "adaptive".
/// `params` may be NULL (all auto). `mask_out` (nx*ny*nz bytes, C-order)
/// may be NULL. Returns 0 on success.
///
/// # Safety
/// Valid `vol`; `points` = 68*3 doubles; `mode` NUL-terminated;
/// `mask_out` NULL or nx*ny*nz bytes; `report` NULL or valid.
#[no_mangle]
pub unsafe extern "C" fn radveil_deface(
    vol: *mut RadveilVolume,
    points: *const f64,
    mode: *const c_char,
    params: *const RadveilDefaceParams,
    mask_out: *mut u8,
    report: *mut RadveilReport,
) -> i32 {
    deface_impl(
        vol,
        points,
        mode,
        std::ptr::null(),
        Region::Face,
        params,
        std::ptr::null(),
        mask_out,
        report,
    )
}

/// Like `radveil_deface`, with a region selector: "face", "ears" or
/// "face+ears". Each pinna is detected from surface evidence relative to
/// the eyes (no ear landmarks exist in the 68-point set); `ear_params` may
/// be NULL (all auto). The report's `voxels_changed` covers all regions.
///
/// # Safety
/// Same as `radveil_deface`; `region` NUL-terminated; `ear_params` NULL
/// or valid.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn radveil_deface_region(
    vol: *mut RadveilVolume,
    points: *const f64,
    mode: *const c_char,
    region: *const c_char,
    params: *const RadveilDefaceParams,
    ear_params: *const RadveilEarParams,
    mask_out: *mut u8,
    report: *mut RadveilReport,
) -> i32 {
    let region = match CStr::from_ptr(region).to_str() {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    let region: Region = match region.parse() {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    deface_impl(
        vol,
        points,
        mode,
        std::ptr::null(),
        region,
        params,
        ear_params,
        mask_out,
        report,
    )
}

/// Like `radveil_deface_region`, with a separate mode for the ear regions:
/// `ear_mode` is "blur", "cut", "block", "adaptive", or NULL for the same
/// mode as `mode` (e.g. `mode="adaptive"`, `ear_mode="cut"`). One pass;
/// `mask_out` covers all regions.
///
/// # Safety
/// Same as `radveil_deface_region`; `ear_mode` NULL or NUL-terminated.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn radveil_deface_regions(
    vol: *mut RadveilVolume,
    points: *const f64,
    mode: *const c_char,
    ear_mode: *const c_char,
    region: *const c_char,
    params: *const RadveilDefaceParams,
    ear_params: *const RadveilEarParams,
    mask_out: *mut u8,
    report: *mut RadveilReport,
) -> i32 {
    let region: Region = match CStr::from_ptr(region)
        .to_str()
        .map_err(|e| e.to_string())
        .and_then(|r| r.parse())
    {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    deface_impl(
        vol, points, mode, ear_mode, region, params, ear_params, mask_out, report,
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn deface_impl(
    vol: *mut RadveilVolume,
    points: *const f64,
    mode: *const c_char,
    ear_mode: *const c_char,
    region: Region,
    params: *const RadveilDefaceParams,
    ear_params: *const RadveilEarParams,
    mask_out: *mut u8,
    report: *mut RadveilReport,
) -> i32 {
    let mode = match CStr::from_ptr(mode).to_str() {
        Ok(m) => m,
        Err(e) => return set_error(e),
    };
    let mode: DefaceMode = match mode.parse() {
        Ok(m) => m,
        Err(e) => return set_error(e),
    };
    let ear_mode: Option<DefaceMode> = if ear_mode.is_null() {
        None
    } else {
        match CStr::from_ptr(ear_mode)
            .to_str()
            .map_err(|e| e.to_string())
            .and_then(|m| m.parse())
        {
            Ok(m) => Some(m),
            Err(e) => return set_error(e),
        }
    };
    let pts = std::slice::from_raw_parts(points, 68 * 3);
    let lm: Vec<Option<[f64; 3]>> = (0..68)
        .map(|i| {
            let p = [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]];
            if p.iter().all(|c| c.is_finite()) {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    let frame = match face_frame_from_landmarks(&lm) {
        Ok(f) => f,
        Err(e) => return set_error(e),
    };
    let mut dp = build_params(mode, region, params, ear_params);
    dp.ear_mode = ear_mode;
    let (rep, mask) = deface(&mut (*vol).0, &frame, &dp);
    if !mask_out.is_null() {
        copy_mask(&mask, mask_out);
    }
    if !report.is_null() {
        *report = RadveilReport {
            voxels_changed: rep.voxels_changed as u64,
            voxels_total: rep.voxels_total as u64,
            eye_dist_mm: rep.eye_dist_mm,
            blur_sigma_mm: rep.resolved.blur_sigma_mm,
        };
    }
    0
}

unsafe fn build_params(
    mode: DefaceMode,
    region: Region,
    params: *const RadveilDefaceParams,
    ear_params: *const RadveilEarParams,
) -> DefaceParams {
    let mut dp = DefaceParams {
        mode,
        region,
        ..Default::default()
    };
    if !params.is_null() {
        let p = &*params;
        dp.width_factor = opt(p.width_factor);
        dp.above_eyes_mm = opt(p.above_eyes_mm);
        dp.below_chin_mm = opt(p.below_chin_mm);
        dp.depth_behind_mm = opt(p.depth_behind_mm);
        dp.depth_front_mm = opt(p.depth_front_mm);
        dp.blur_sigma_mm = opt(p.blur_sigma_mm);
        dp.adaptive_factor = opt(p.adaptive_factor);
        dp.skin_depth_mm = opt(p.skin_depth_mm);
        dp.spare_air = p.options & RADVEIL_SPARE_AIR != 0;
        dp.level_face = p.options & RADVEIL_LEVEL_FACE != 0;
    }
    if !ear_params.is_null() {
        let e = &*ear_params;
        dp.ear_inner_mm = opt(e.inner_mm);
        dp.ear_outer_mm = opt(e.outer_mm);
        dp.ear_guard_mm = opt(e.guard_mm);
        dp.ear_down_mm = opt(e.down_mm);
        dp.ear_back_mm = opt(e.back_mm);
        dp.ear_height_mm = opt(e.height_mm);
        dp.ear_depth_mm = opt(e.depth_mm);
    }
    dp
}

unsafe fn copy_mask(mask: &Array3<u8>, out: *mut u8) {
    let flat: Vec<u8> = if let Some(s) = mask.as_slice() {
        s.to_vec()
    } else {
        mask.iter().copied().collect()
    };
    std::ptr::copy_nonoverlapping(flat.as_ptr(), out, flat.len());
}

/// Result of `radveil_deface_auto`.
#[repr(C)]
pub struct RadveilAutoReport {
    pub base: RadveilReport,
    /// Rendered views the face was validated in.
    pub views_detected: u32,
    /// 1 when the stored orientation was wrong and the face was found by
    /// the 6-direction orientation sweep.
    pub orientation_sweep: u8,
    /// 1 when verification escalated to an opaque block fill.
    pub escalated_to_block: u8,
    /// Verification escalation rounds; -1 when verification did not run.
    pub verify_rounds: i32,
    /// Detections at the anonymized face after the last round; -1 when
    /// verification did not run.
    pub residual_detections: i32,
    /// Brain voxels the anonymization modified (0 = brain untouched);
    /// -1 when the brain check did not run.
    pub brain_voxels_modified: i64,
    /// Distance from the brain to the closest modified voxel, mm; NaN when
    /// the check did not run or the brain was touched.
    pub brain_margin_mm: f64,
    /// Region voxels the brain guard kept unmodified; -1 when it is off.
    pub brain_guard_spared: i64,
}

/// `brain` flag for `radveil_deface_auto` / `radveil_anonymization_mask`:
/// report how many brain voxels were modified.
pub const RADVEIL_BRAIN_CHECK: u8 = 1;
/// `brain` flag: never modify the brain or anything within 3 mm of it.
pub const RADVEIL_BRAIN_GUARD: u8 = 2;

/// `verify` argument of `radveil_deface_auto`: off.
pub const RADVEIL_VERIFY_OFF: u8 = 0;
/// On, default effort (balanced: detection views + a 3x3 yaw/pitch grid).
pub const RADVEIL_VERIFY_ON: u8 = 1;
/// On, detection views only (fastest).
pub const RADVEIL_VERIFY_FAST: u8 = 2;
/// On, detection views + a 5x5 yaw/pitch grid (slowest).
pub const RADVEIL_VERIFY_THOROUGH: u8 = 3;

/// Detect the face with the built-in MediaPipe detector and anonymize it
/// in place: render -> detect -> validate -> lift -> deface -> verify
/// (re-render and re-detect, escalating the blur and finally an opaque
/// fill until no face is found; `verify` = 0 disables). `brain` is a
/// bit set: `RADVEIL_BRAIN_CHECK` (1) extracts the brain (SynthStrip) from
/// the input and reports how many brain voxels the anonymization modified;
/// `RADVEIL_BRAIN_GUARD` (2) never rewrites the brain or anything within
/// 3 mm of it. Fails (returns
/// -1, nothing modified) when no validated face is found, e.g. on an
/// already-defaced volume. `mode`: "blur", "cut", "block", "adaptive";
/// `region`: "face", "ears", "face+ears". `params`, `ear_params`,
/// `mask_out` and `report` may be NULL.
///
/// # Safety
/// Valid `vol`; `mode`/`region` NUL-terminated; `mask_out` NULL or
/// nx*ny*nz bytes; the other pointers NULL or valid.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn radveil_deface_auto(
    vol: *mut RadveilVolume,
    mode: *const c_char,
    region: *const c_char,
    params: *const RadveilDefaceParams,
    ear_params: *const RadveilEarParams,
    verify: u8,
    brain: u8,
    mask_out: *mut u8,
    report: *mut RadveilAutoReport,
) -> i32 {
    let mode: DefaceMode = match CStr::from_ptr(mode).to_str().map(str::parse) {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => return set_error(e),
        Err(e) => return set_error(e),
    };
    let region: Region = match CStr::from_ptr(region).to_str().map(str::parse) {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return set_error(e),
        Err(e) => return set_error(e),
    };
    let options = if params.is_null() {
        0
    } else {
        (*params).options
    };
    let params = AutoParams {
        deface: build_params(mode, region, params, ear_params),
        partial_face: options & RADVEIL_PARTIAL_FACE != 0,
        auto_threshold: options & RADVEIL_AUTO_THRESHOLD != 0,
        verify: verify != RADVEIL_VERIFY_OFF,
        verify_effort: match verify {
            RADVEIL_VERIFY_FAST => VerifyEffort::Fast,
            RADVEIL_VERIFY_THOROUGH => VerifyEffort::Thorough,
            _ => VerifyEffort::default(),
        },
        brain_check: brain & RADVEIL_BRAIN_CHECK != 0,
        brain_guard: brain & RADVEIL_BRAIN_GUARD != 0,
        return_mask: !mask_out.is_null(),
        // The primary detector is the built-in one already.
        verify_with_builtin: false,
        ..Default::default()
    };
    let rep = match deface_auto(&mut (*vol).0, &mut MediaPipeDetector::default(), &params) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    if let Some(mask) = &rep.mask {
        copy_mask(mask, mask_out);
    }
    if !report.is_null() {
        *report = RadveilAutoReport {
            base: RadveilReport {
                voxels_changed: rep.deface.voxels_changed as u64,
                voxels_total: rep.deface.voxels_total as u64,
                eye_dist_mm: rep.deface.eye_dist_mm,
                blur_sigma_mm: rep.deface.resolved.blur_sigma_mm,
            },
            views_detected: rep.views_detected as u32,
            orientation_sweep: rep.orientation_sweep as u8,
            escalated_to_block: rep.escalated_to_block.unwrap_or(false) as u8,
            verify_rounds: rep.verify_rounds.map_or(-1, |r| r as i32),
            residual_detections: rep.residual_detections.map_or(-1, |r| r as i32),
            brain_voxels_modified: rep.brain.as_ref().map_or(-1, |b| b.voxels_modified as i64),
            brain_margin_mm: rep
                .brain
                .as_ref()
                .and_then(|b| b.margin_mm)
                .unwrap_or(f64::NAN),
            brain_guard_spared: rep.deface.protected_voxels_spared.map_or(-1, |n| n as i64),
        };
    }
    0
}

/// Fill `params` with "auto" for every field (scaled from the detected
/// inter-eye distance). Parameter structs treat negative/NaN fields as auto,
/// so a zero-initialised struct would instead request zero-sized margins;
/// call this first and override only what you need.
///
/// # Safety
/// `params` must point to a writable `RadveilDefaceParams`.
#[no_mangle]
pub unsafe extern "C" fn radveil_deface_params_init(params: *mut RadveilDefaceParams) {
    if !params.is_null() {
        *params = RadveilDefaceParams {
            width_factor: f64::NAN,
            above_eyes_mm: f64::NAN,
            below_chin_mm: f64::NAN,
            depth_behind_mm: f64::NAN,
            depth_front_mm: f64::NAN,
            blur_sigma_mm: f64::NAN,
            adaptive_factor: f64::NAN,
            skin_depth_mm: f64::NAN,
            options: 0,
        };
    }
}

/// Fill `params` with "auto" for every field; see
/// `radveil_deface_params_init`.
///
/// # Safety
/// `params` must point to a writable `RadveilEarParams`.
#[no_mangle]
pub unsafe extern "C" fn radveil_ear_params_init(params: *mut RadveilEarParams) {
    if !params.is_null() {
        *params = RadveilEarParams {
            inner_mm: f64::NAN,
            outer_mm: f64::NAN,
            guard_mm: f64::NAN,
            down_mm: f64::NAN,
            back_mm: f64::NAN,
            height_mm: f64::NAN,
            depth_mm: f64::NAN,
        };
    }
}

/// Escape hatch: the region `radveil_deface_auto` would anonymize, written
/// to `mask_out` (nx*ny*nz bytes, 1 = voxel it would rewrite), **without
/// modifying the volume**. Detection, validation, the safety refusals and
/// the `brain` flags run as in `radveil_deface_auto`;
/// nothing is rewritten and nothing is verified. The report's
/// `voxels_changed` counts the voxels that would change; its verification
/// fields are -1 / 0.
///
/// # Safety
/// `vol` must be a live volume; `mode` and `region` NUL-terminated strings;
/// `mask_out` must hold nx*ny*nz bytes; the other pointers may be NULL.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn radveil_anonymization_mask(
    vol: *const RadveilVolume,
    mode: *const c_char,
    region: *const c_char,
    params: *const RadveilDefaceParams,
    ear_params: *const RadveilEarParams,
    brain: u8,
    mask_out: *mut u8,
    report: *mut RadveilAutoReport,
) -> i32 {
    if mask_out.is_null() {
        return set_error("mask_out must not be NULL");
    }
    let mode: DefaceMode = match CStr::from_ptr(mode).to_str().map(str::parse) {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => return set_error(e),
        Err(e) => return set_error(e),
    };
    let region: Region = match CStr::from_ptr(region).to_str().map(str::parse) {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return set_error(e),
        Err(e) => return set_error(e),
    };
    let options = if params.is_null() {
        0
    } else {
        (*params).options
    };
    let params = AutoParams {
        deface: build_params(mode, region, params, ear_params),
        partial_face: options & RADVEIL_PARTIAL_FACE != 0,
        auto_threshold: options & RADVEIL_AUTO_THRESHOLD != 0,
        brain_check: brain & RADVEIL_BRAIN_CHECK != 0,
        brain_guard: brain & RADVEIL_BRAIN_GUARD != 0,
        verify_with_builtin: false,
        ..Default::default()
    };
    let rep = match anonymization_mask(&(*vol).0, &mut MediaPipeDetector::default(), &params) {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    if let Some(mask) = &rep.mask {
        copy_mask(mask, mask_out);
    }
    if !report.is_null() {
        *report = RadveilAutoReport {
            base: RadveilReport {
                voxels_changed: rep.deface.voxels_changed as u64,
                voxels_total: rep.deface.voxels_total as u64,
                eye_dist_mm: rep.deface.eye_dist_mm,
                blur_sigma_mm: rep.deface.resolved.blur_sigma_mm,
            },
            views_detected: rep.views_detected as u32,
            orientation_sweep: rep.orientation_sweep as u8,
            escalated_to_block: 0,
            verify_rounds: -1,
            residual_detections: -1,
            brain_voxels_modified: rep.brain.as_ref().map_or(-1, |b| b.voxels_modified as i64),
            brain_margin_mm: rep
                .brain
                .as_ref()
                .and_then(|b| b.margin_mm)
                .unwrap_or(f64::NAN),
            brain_guard_spared: rep.deface.protected_voxels_spared.map_or(-1, |n| n as i64),
        };
    }
    0
}

/// Result of `radveil_deskull`.
#[repr(C)]
pub struct RadveilDeskullReport {
    /// Brain volume, millilitres.
    pub brain_volume_ml: f64,
    pub brain_voxels: u64,
    /// Mask centroid, world RAS+ mm.
    pub centroid: [f64; 3],
    /// 0 = MRI, 1 = CT (after resolving "auto").
    pub modality: u8,
    /// 1 when the face prior was used.
    pub face_prior: u8,
    /// Value written outside the brain (NaN when `strip` = 0).
    pub fill: f32,
}

/// Brain extraction (skull stripping). `method`: "synthstrip" (the
/// SynthStrip network, any contrast/CT) or "classical" (model-free);
/// `modality`: "auto", "mri" or "ct". `frac` (MRI) is the fractional
/// intensity threshold, 0.5 by default (smaller = larger brain); pass NaN
/// for the default. `use_face` = 1 locates the face with the built-in
/// detector and uses it as an anatomical prior (MRI; silently skipped when
/// no face is found -- check `report->face_prior`). With `strip` = 1 the
/// volume is modified in place (non-brain voxels = `fill`, NaN = default:
/// 0 for MRI, air for CT). `mask_out` (nx*ny*nz bytes, 1 = brain) and
/// `report` may be NULL. `low_memory` = 1 trades ≤3% speed for ~0.4 GB
/// less peak memory in the synthstrip method (identical output).
///
/// # Safety
/// Valid `vol`; `method` and `modality` NUL-terminated; `mask_out` NULL or nx*ny*nz
/// bytes; `report` NULL or valid.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn radveil_deskull(
    vol: *mut RadveilVolume,
    method: *const c_char,
    modality: *const c_char,
    frac: f64,
    use_face: u8,
    strip: u8,
    low_memory: u8,
    fill: f32,
    mask_out: *mut u8,
    report: *mut RadveilDeskullReport,
) -> i32 {
    let modality = match CStr::from_ptr(modality).to_str().map(Modality::parse) {
        Ok(Some(m)) => m,
        Ok(None) => return set_error("modality must be \"auto\", \"mri\" or \"ct\""),
        Err(e) => return set_error(e),
    };
    let method = match CStr::from_ptr(method).to_str().map(Method::parse) {
        Ok(Some(m)) => m,
        Ok(None) => return set_error("method must be \"synthstrip\" or \"classical\""),
        Err(e) => return set_error(e),
    };
    let v = &mut (*vol).0;
    let face = if use_face != 0 {
        locate_face(v, &mut MediaPipeDetector::default(), &AutoParams::default())
            .ok()
            .map(|f| f.frame)
    } else {
        None
    };
    let params = DeskullParams {
        method,
        synthstrip: radveil_core::SynthStripParams {
            low_memory: low_memory != 0,
            ..Default::default()
        },
        modality,
        frac: if frac.is_finite() { frac } else { 0.5 },
        fill: fill.is_finite().then_some(fill),
        face,
        ..Default::default()
    };
    let res = if strip != 0 {
        deskull(v, &params)
    } else {
        brain_mask(v, &params)
    };
    let (mask, rep) = match res {
        Ok(r) => r,
        Err(e) => return set_error(e),
    };
    if !mask_out.is_null() {
        copy_mask(&mask, mask_out);
    }
    if !report.is_null() {
        *report = RadveilDeskullReport {
            brain_volume_ml: rep.brain_volume_ml,
            brain_voxels: rep.brain_voxels as u64,
            centroid: rep.centroid_world,
            modality: u8::from(rep.modality == Modality::Ct),
            face_prior: u8::from(rep.face_prior),
            fill: rep.fill.unwrap_or(f32::NAN),
        };
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_roundtrip() {
        // phantom via from_array -> render -> deface -> report
        let n = 64usize;
        let mut data = vec![0.0f32; n * n * n];
        let c = n as f64 / 2.0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let (x, y, z) = (i as f64 - c, j as f64 - c, k as f64 - c);
                    let head = (x / 22.0).powi(2) + (y / 25.0).powi(2) + (z / 27.0).powi(2);
                    let nose =
                        (x / 5.0).powi(2) + ((y + 28.0) / 8.0).powi(2) + ((z + 4.0) / 6.0).powi(2);
                    if head <= 1.0 || nose <= 1.0 {
                        data[(i * n + j) * n + k] = 800.0;
                    }
                }
            }
        }
        let aff = [
            2.0, 0.0, 0.0, -64.0, 0.0, 2.0, 0.0, -64.0, 0.0, 0.0, 2.0, -64.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let vol = unsafe { radveil_volume_from_array(n, n, n, data.as_ptr(), aff.as_ptr()) };
        assert!(!vol.is_null());
        let mut dims = [0u64; 3];
        unsafe { radveil_volume_shape(vol, dims.as_mut_ptr()) };
        assert_eq!(dims, [64, 64, 64]);

        let size = 128u32;
        let mut pos = vec![f32::NAN; (size * size * 3) as usize];
        let mut flags = vec![0u8; (size * size) as usize];
        let rc = unsafe {
            radveil_render(
                vol,
                0.0,
                0.0,
                size,
                0.14,
                f32::NAN,
                std::ptr::null_mut(),
                pos.as_mut_ptr(),
                flags.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0);
        assert!(flags.iter().filter(|&&f| f > 0).count() > 500);

        // hand-placed landmarks like the core phantom tests
        let mut pts = vec![f64::NAN; 68 * 3];
        let set = |pts: &mut Vec<f64>, i: usize, p: [f64; 3]| {
            pts[i * 3] = p[0];
            pts[i * 3 + 1] = p[1];
            pts[i * 3 + 2] = p[2];
        };
        for i in 36..42 {
            set(&mut pts, i, [-16.0, -44.0, 8.0]);
        }
        for i in 42..48 {
            set(&mut pts, i, [16.0, -44.0, 8.0]);
        }
        set(&mut pts, 30, [0.0, -70.0, -8.0]);
        for i in 48..68 {
            set(&mut pts, i, [0.0, -46.0, -26.0]);
        }
        let mode = CString::new("cut").unwrap();
        let mut rep = RadveilReport {
            voxels_changed: 0,
            voxels_total: 0,
            eye_dist_mm: 0.0,
            blur_sigma_mm: 0.0,
        };
        let rc = unsafe {
            radveil_deface(
                vol,
                pts.as_ptr(),
                mode.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut rep,
            )
        };
        assert_eq!(rc, 0, "err: {:?}", unsafe {
            CStr::from_ptr(radveil_last_error())
        });
        assert!(rep.voxels_changed > 500);
        assert!((rep.eye_dist_mm - 32.0).abs() < 1.0);

        // Region variant: ABI smoke test only — this small phantom has no
        // ears, so the fallback boxes may fall outside the head entirely;
        // geometry is covered by the core tests.
        let region = CString::new("ears").unwrap();
        let rc = unsafe {
            radveil_deface_region(
                vol,
                pts.as_ptr(),
                mode.as_ptr(),
                region.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut rep,
            )
        };
        assert_eq!(rc, 0, "err: {:?}", unsafe {
            CStr::from_ptr(radveil_last_error())
        });
        assert_eq!(rep.voxels_total, 64 * 64 * 64);

        let bad = CString::new("nose").unwrap();
        let rc = unsafe {
            radveil_deface_region(
                vol,
                pts.as_ptr(),
                mode.as_ptr(),
                bad.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert!(rc < 0);

        // Separate ear mode in one pass; NULL ear_mode = same as mode.
        let face_ears = CString::new("face+ears").unwrap();
        let blur = CString::new("blur").unwrap();
        for ear_mode in [mode.as_ptr(), std::ptr::null()] {
            let rc = unsafe {
                radveil_deface_regions(
                    vol,
                    pts.as_ptr(),
                    blur.as_ptr(),
                    ear_mode,
                    face_ears.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    &mut rep,
                )
            };
            assert_eq!(rc, 0, "err: {:?}", unsafe {
                CStr::from_ptr(radveil_last_error())
            });
        }
        let rc = unsafe {
            radveil_deface_regions(
                vol,
                pts.as_ptr(),
                blur.as_ptr(),
                bad.as_ptr(),
                face_ears.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert!(rc < 0, "invalid ear_mode must be rejected");
        unsafe { radveil_volume_free(vol) };
    }
}
