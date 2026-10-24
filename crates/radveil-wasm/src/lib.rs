// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! WebAssembly bindings: the full RadVeil pipeline (render, built-in
//! MediaPipe detection, validation, anonymization, verification) on a
//! NIfTI file held in browser memory. Nothing leaves the page.

use radveil_core::pipeline::{deface_auto, AutoParams};
use radveil_core::render::{render_view, RenderParams};
use radveil_core::{MediaPipeDetector, Volume};
use std::io::Write;
use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Parse the embedded detector models ahead of the first scan.
#[wasm_bindgen(js_name = warmUp)]
pub fn warm_up() -> Result<(), JsError> {
    MediaPipeDetector::warm_up().map_err(js_err)
}

/// A loaded scan plus the file bytes it came from (the output reuses the
/// original header and extensions byte-for-byte).
#[wasm_bindgen]
pub struct Scan {
    vol: Volume,
    original: Vec<u8>,
}

#[wasm_bindgen]
impl Scan {
    /// Load a `.nii` or `.nii.gz` file.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: Vec<u8>) -> Result<Scan, JsError> {
        let vol = Volume::from_bytes(&bytes).map_err(js_err)?;
        Ok(Scan {
            vol,
            original: bytes,
        })
    }

    /// Shape, voxel spacing (mm) and orientation code, as JSON.
    pub fn info(&self) -> String {
        serde_json::json!({
            "shape": self.vol.shape(),
            "spacing": self.vol.spacing,
            "orientation": self.vol.orientation(),
        })
        .to_string()
    }

    /// Surface render as RGBA8 (`size` x `size`, ready for `ImageData`).
    pub fn render(&self, yaw_deg: f64, pitch_deg: f64, size: usize) -> Vec<u8> {
        let params = RenderParams {
            size,
            ..RenderParams::default()
        };
        let view = render_view(&self.vol, yaw_deg, pitch_deg, &params);
        let mut rgba = Vec::with_capacity(view.rgb.len() / 3 * 4);
        for px in view.rgb.as_chunks::<3>().0 {
            rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        rgba
    }

    /// Run the automatic pipeline in place. `mode`: blur | cut | block |
    /// adaptive; `region`: face | ears | face+ears; `spare_air` and
    /// `level_face`, and `partial_face` are the opt-ins of the same names. Returns the report
    /// as JSON; fails (volume untouched) when no face is validated.
    pub fn anonymize(
        &mut self,
        mode: &str,
        region: &str,
        spare_air: bool,
        level_face: bool,
        partial_face: bool,
    ) -> Result<String, JsError> {
        let mut params = AutoParams::default();
        params.deface.mode = mode.parse().map_err(js_err)?;
        params.deface.region = region.parse().map_err(js_err)?;
        params.deface.spare_air = spare_air;
        params.deface.level_face = level_face;
        params.partial_face = partial_face;
        // The primary detector is the built-in one already.
        params.verify_with_builtin = false;
        let mut det = MediaPipeDetector::default();
        let report = deface_auto(&mut self.vol, &mut det, &params).map_err(js_err)?;
        serde_json::to_string(&report).map_err(js_err)
    }

    /// The current voxels as a NIfTI file with the original header;
    /// gzipped (level 1) when `gzip`.
    pub fn encode(&self, gzip: bool) -> Result<Vec<u8>, JsError> {
        let nii = self.vol.encode_like(&self.original).map_err(js_err)?;
        if !gzip {
            return Ok(nii);
        }
        let mut enc = flate2::write::GzEncoder::new(
            Vec::with_capacity(nii.len() / 3),
            flate2::Compression::new(1),
        );
        enc.write_all(&nii).map_err(js_err)?;
        enc.finish().map_err(js_err)
    }
}
