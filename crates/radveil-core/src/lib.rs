// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! radveil-core: CPU-only NIfTI face anonymization.
//!
//! Pipeline:
//! 1. Load a NIfTI volume, resolve its voxel->world (RAS+) affine from
//!    sform/qform, and report orientation codes (RAS/LPS/...).
//! 2. Software volumetric renderer (orthographic ray caster) producing
//!    natural-looking face renders plus a per-pixel 3D world-position map.
//! 3. 2D face landmarks are detected on one or more rendered views (built-in
//!    MediaPipe detector, feature `detector`, or any [`FaceDetector`]) and
//!    unprojected through the position map into 3D world space.
//! 4. A face-aligned region of interest is built from the 3D landmarks and
//!    anonymized in the volume (blur / cut / block), leaving every voxel
//!    outside the region bit-exact.

pub mod affine;
#[cfg(any(feature = "synthstrip", feature = "rvbrain"))]
pub mod conform;
pub mod deface;
pub mod deskull;
#[cfg(feature = "detector")]
pub mod detect;
pub(crate) mod kernels;
pub mod landmarks;
#[cfg(feature = "detector")]
pub mod nn;
pub mod pipeline;
pub mod render;
#[cfg(feature = "rvbrain")]
pub mod rvbrain;
#[cfg(feature = "synthstrip")]
pub mod synthstrip;
#[cfg(any(feature = "synthstrip", feature = "rvbrain"))]
pub(crate) mod unet;
pub mod volume;

#[cfg(any(feature = "synthstrip", feature = "rvbrain"))]
pub use conform::{conform, normalise, ConformSpec, Conformed};
pub use deface::{
    apply, deface, plan, plan_mask, DefaceMode, DefaceParams, DefacePlan, DefaceReport, Region,
    RoiReport,
};
pub use deskull::{
    brain_mask, brain_mask_with, default_fill, deskull, strip, BrainExtractor, ClassicalExtractor,
    DeskullParams, DeskullReport, Method, Modality,
};
#[cfg(feature = "detector")]
pub use detect::{DetectorParams, MediaPipeDetector};
pub use landmarks::{
    combine_views, face_frame_from_landmarks, unproject, FaceFrame, ViewLandmarks,
};
pub use pipeline::{
    anonymization_mask, deface_auto, locate_face, AutoParams, AutoReport, Detection, FaceDetector,
    LocatedFace, VerifyEffort,
};
pub use render::{render_view, RenderParams, RenderView};
#[cfg(feature = "rvbrain")]
pub use rvbrain::{Arch, RvBrain, RvBrainParams};
#[cfg(feature = "synthstrip")]
pub use synthstrip::{SynthStrip, SynthStripParams};
pub use volume::{Volume, VolumeError};

/// Golden test data from `tests/data/`, read at run time so the published
/// crate can leave it out (it would push the `.crate` past the crates.io
/// size limit). `None` (skip) only when the whole directory is absent, as
/// in a crate downloaded from crates.io; a missing file in a checkout
/// fails the test.
#[cfg(all(
    test,
    any(feature = "detector", feature = "synthstrip", feature = "rvbrain")
))]
pub(crate) fn golden_data(name: &str) -> Option<Vec<u8>> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    if !dir.is_dir() {
        eprintln!("skipping: {} not present (packaged crate)", dir.display());
        return None;
    }
    let path = dir.join(name);
    Some(std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}
