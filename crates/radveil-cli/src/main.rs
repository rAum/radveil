// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! radveil CLI: NIfTI face rendering + anonymization.
//!
//! `radveil auto` runs the whole pipeline with the built-in MediaPipe face
//! detector (no Python, no model download). `radveil deface` consumes
//! landmarks from an external detector as JSON — either 2D per-view
//! (produced together with `radveil render` output) or already-lifted 3D.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use radveil_core::{
    brain_mask, combine_views, conform, deface, deface_auto, deskull, face_frame_from_landmarks,
    locate_face, render_view, unproject, AutoParams, ConformSpec, DefaceMode, DefaceParams,
    DeskullParams, MediaPipeDetector, Method, Modality, RenderParams, RvBrainParams,
    SynthStripParams, Volume,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "radveil",
    version,
    about = "CPU-only NIfTI face renderer, anonymizer & brain extractor"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print volume geometry: shape, spacing, orientation (RAS/LPS/...), affine.
    Info { input: PathBuf },
    /// Render face views (PNG + per-view camera metadata JSON).
    Render {
        input: PathBuf,
        /// Output directory for view_*.png and views.json.
        #[arg(short, long, default_value = "renders")]
        out_dir: PathBuf,
        /// Yaw angles in degrees (0 = frontal, + = patient-left side of face).
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "-20,0,20",
            allow_hyphen_values = true
        )]
        yaw: Vec<f64>,
        /// Pitch angle in degrees.
        #[arg(long, default_value_t = 0.0)]
        pitch: f64,
        #[arg(long, default_value_t = 512)]
        size: usize,
        /// Surface threshold as fraction of the p99 intensity.
        #[arg(long, default_value_t = 0.14)]
        threshold_frac: f32,
        /// Absolute surface threshold (overrides --threshold-frac).
        #[arg(long)]
        threshold: Option<f32>,
    },
    /// Detect the face and anonymize it in one step (built-in detector).
    ///
    /// Renders the head, finds the face with the embedded MediaPipe
    /// models, anonymizes it, then re-renders and re-detects to verify —
    /// escalating the blur (and finally an opaque fill) until no face is
    /// found. Refuses volumes without a plausible face.
    Auto {
        input: PathBuf,
        output: PathBuf,
        /// blur | cut | block | adaptive
        #[arg(short, long, default_value = "blur")]
        mode: String,
        /// Anatomy to anonymize: face | ears | face+ears.
        #[arg(short, long, default_value = "face")]
        region: String,
        /// Mode for the ear regions (blur | cut | block | adaptive);
        /// default: same as --mode. E.g. `--mode adaptive --ear-mode cut`.
        #[arg(long)]
        ear_mode: Option<String>,
        /// Also write the binary anonymization mask here (.nii.gz).
        #[arg(long)]
        mask_out: Option<PathBuf>,
        /// Write the JSON report here ("-" = stdout).
        #[arg(long)]
        report: Option<PathBuf>,
        /// Gaussian sigma in mm for blur/adaptive modes (default: auto).
        #[arg(long)]
        blur_sigma: Option<f64>,
        /// Hull floor: how far behind the eye plane the face ROI may reach,
        /// mm (default: auto). The skin-following wall is never deeper.
        #[arg(long)]
        depth_behind: Option<f64>,
        /// Disable anchoring the face depth to the measured eye-skin surface.
        #[arg(long)]
        no_eye_anchor: bool,
        /// Detection view yaw angles in degrees.
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "-20,0,20",
            allow_hyphen_values = true
        )]
        yaw: Vec<f64>,
        #[arg(long, default_value_t = 512)]
        size: usize,
        #[arg(long, default_value_t = 0.14)]
        threshold_frac: f32,
        /// Absolute surface threshold (overrides --threshold-frac).
        #[arg(long)]
        threshold: Option<f32>,
        /// Skip the re-detection verification loop.
        #[arg(long)]
        no_verify: bool,
        /// Verification effort (speed vs quality): fast (detection views
        /// only), balanced (+ a 3x3 yaw/pitch grid), thorough (+ 5x5).
        #[arg(long, default_value = "balanced")]
        verify_effort: String,
        /// Safety check: extract the brain (deskull) from the input and
        /// report how many brain voxels the anonymization modified.
        /// Exits with status 3 if the brain was touched.
        #[arg(long)]
        brain_check: bool,
        /// Brain guard: extract the brain (deskull) from the input and never
        /// rewrite it, nor anything within --brain-guard-mm of it (~3 s).
        #[arg(long)]
        brain_guard: bool,
        #[arg(long, default_value_t = 3.0)]
        brain_guard_mm: f64,
        /// Faces cut by the field of view (an axial slab ending at the nose):
        /// also detect from above/below, place the frame without a mouth,
        /// and measure the skin along the level head axes (opt-in).
        #[arg(long)]
        partial_face: bool,
        /// Noisy, dim scans (e.g. infants): if no face is found, retry at
        /// higher surface thresholds (opt-in).
        #[arg(long)]
        auto_threshold: bool,
        /// Face cut depth below the measured skin, mm (default: auto, 0.07 x
        /// inter-eye distance).
        #[arg(long)]
        skin_depth: Option<f64>,
        /// Leave background air untouched: rewrite a region voxel only if
        /// it is tissue before or after (off by default).
        #[arg(long)]
        spare_air: bool,
        /// Measure the face skin along the level head axes, so the lips and
        /// chin are not hidden behind each other (off by default).
        #[arg(long)]
        level_face: bool,
        /// gzip level for .nii.gz output (1 = fast).
        #[arg(long, default_value_t = 1)]
        gzip_level: u32,
    },
    /// Write the resampled grid a brain-extraction network is fed: the
    /// volume reoriented to LIA, resampled to an isotropic spacing, cropped
    /// to the head and padded to the network's block size, and intensity
    /// normalised. Useful for debugging, and for building training data with
    /// exactly the preprocessing inference uses (see training/README.md on the
    /// `training` branch).
    Conform {
        input: PathBuf,
        output: PathBuf,
        /// Isotropic voxel size (mm). SynthStrip uses 1.0.
        #[arg(long, default_value_t = 1.0)]
        voxel_mm: f64,
        /// Pad each axis to a multiple of this (2^levels for the network).
        #[arg(long, default_value_t = 64)]
        multiple: usize,
        /// Per-axis clamp, in voxels.
        #[arg(long, default_value_t = 192)]
        min: usize,
        #[arg(long, default_value_t = 320)]
        max: usize,
        /// Take the grid from a .rvb model instead of the options above.
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        gzip_level: u32,
    },
    /// Extract the brain (skull stripping): write the stripped volume
    /// and/or the binary brain mask.
    Deskull {
        input: PathBuf,
        /// Stripped volume (non-brain voxels set to background).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Binary brain mask (uint8, same grid as the input).
        #[arg(long)]
        mask_out: Option<PathBuf>,
        /// synthstrip (learned, any contrast/CT) | rvbrain (RadVeil's own
        /// smaller network, needs --model) | classical (no model).
        #[arg(long, default_value = "synthstrip")]
        method: String,
        /// Model file for --method rvbrain (.rvb). Falls back to
        /// $RADVEIL_RVBRAIN_MODEL, then to a model compiled into this build.
        #[arg(long, env = "RADVEIL_RVBRAIN_MODEL")]
        model: Option<PathBuf>,
        /// rvbrain: resample the predicted distance with trilinear instead
        /// of cubic interpolation.
        #[arg(long)]
        no_cubic: bool,
        /// auto | mri | ct
        #[arg(long, default_value = "auto")]
        modality: String,
        /// Classical MRI fractional intensity threshold (smaller = larger brain).
        #[arg(short, long, default_value_t = 0.5)]
        frac: f64,
        /// Background value outside the brain (default: 0 MRI, air CT).
        #[arg(long, allow_hyphen_values = true)]
        fill: Option<f32>,
        /// Disable re-centring passes on the extracted brain (MRI).
        #[arg(long)]
        no_robust: bool,
        /// Trade speed for peak memory in the synthstrip method
        /// (~0.4 GB less, ≤3% slower; identical output).
        #[arg(long)]
        low_memory: bool,
        /// Classical method: detect the face (built-in detector) and use it
        /// as an anatomical prior (brain-centred start, no surface below the
        /// measured brain floor). Falls back to the plain method when no
        /// face is found.
        #[arg(long)]
        face: bool,
        /// Write the JSON report here ("-" = stdout).
        #[arg(long)]
        report: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        gzip_level: u32,
    },
    /// Anonymize the face given landmarks (2D per-view or 3D world-space JSON).
    Deface {
        input: PathBuf,
        output: PathBuf,
        /// blur | cut | block | adaptive
        #[arg(short, long, default_value = "blur")]
        mode: String,
        /// Anatomy to anonymize: face | ears | face+ears. Each pinna is
        /// detected from surface evidence relative to the detected eyes (no
        /// ear landmarks needed).
        #[arg(short, long, default_value = "face")]
        region: String,
        /// Mode for the ear regions (blur | cut | block | adaptive);
        /// default: same as --mode. E.g. `--mode adaptive --ear-mode cut`.
        #[arg(long)]
        ear_mode: Option<String>,
        /// JSON with 2D landmarks per rendered view (e.g. from Python
        /// `radveil.detect_landmarks` or your own detector).
        #[arg(long)]
        landmarks: Option<PathBuf>,
        /// JSON with 3D world-space landmarks: {"points": [[x,y,z]|null, ...68]}.
        #[arg(long)]
        landmarks3d: Option<PathBuf>,
        /// Also write the binary anonymization mask here (.nii.gz).
        #[arg(long)]
        mask_out: Option<PathBuf>,
        /// Gaussian sigma in mm for blur/adaptive modes (default: auto,
        /// scaled from the detected inter-eye distance).
        #[arg(long)]
        blur_sigma: Option<f64>,
        /// Hull floor: how far behind the eye plane the face ROI may reach,
        /// mm (default: auto from inter-eye distance). The skin-following
        /// wall is never deeper.
        #[arg(long)]
        depth_behind: Option<f64>,
        /// How deep below the measured skin the face ROI reaches, mm
        /// (default: auto, ~4.5 mm; nose and lips are removed entirely).
        #[arg(long)]
        skin_depth: Option<f64>,
        /// Leave background air untouched: rewrite a region voxel only if
        /// it is tissue before or after (off by default).
        #[arg(long)]
        spare_air: bool,
        /// Measure the face skin along the level head axes, so the lips and
        /// chin are not hidden behind each other (off by default).
        #[arg(long)]
        level_face: bool,
        /// Disable anchoring the face depth bounds to the measured
        /// eye-skin surface (the anchor corrects the posterior landmark
        /// bias of thick-slice scans; without it the raw landmark plane
        /// is used).
        #[arg(long)]
        no_eye_anchor: bool,
        /// Render size used when lifting 2D landmarks.
        #[arg(long, default_value_t = 512)]
        size: usize,
        #[arg(long, default_value_t = 0.14)]
        threshold_frac: f32,
        /// Write before/after QA renders next to the output.
        #[arg(long)]
        qa: bool,
    },
}

#[derive(Serialize, Deserialize)]
struct ViewMeta {
    file: String,
    yaw_deg: f64,
    pitch_deg: f64,
    size: usize,
    threshold: f32,
}

#[derive(Serialize, Deserialize)]
struct ViewsFile {
    views: Vec<ViewMeta>,
}

#[derive(Deserialize)]
struct Landmarks2dFile {
    views: Vec<radveil_core::ViewLandmarks>,
}

#[derive(Deserialize)]
struct Landmarks3dFile {
    points: Vec<Option<[f64; 3]>>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info { input } => {
            let v = Volume::load(&input).with_context(|| format!("loading {:?}", input))?;
            let [nx, ny, nz] = v.shape();
            println!("file:        {}", input.display());
            println!("shape:       {} x {} x {}", nx, ny, nz);
            println!(
                "spacing:     {:.3} x {:.3} x {:.3} mm",
                v.spacing[0], v.spacing[1], v.spacing[2]
            );
            println!(
                "orientation: {} (voxel axes -> world RAS+)",
                v.orientation()
            );
            println!("datatype:    {:?}", v.datatype);
            println!("scaling:     slope={} inter={}", v.scl_slope, v.scl_inter);
            println!("affine (voxel -> world RAS+, mm):");
            for r in 0..3 {
                println!(
                    "  [{:9.4} {:9.4} {:9.4} {:10.3}]",
                    v.affine[r][0], v.affine[r][1], v.affine[r][2], v.affine[r][3]
                );
            }
            let (lo, hi) = v.world_bounds();
            println!(
                "world bbox:  [{:.1},{:.1},{:.1}] .. [{:.1},{:.1},{:.1}] mm",
                lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
            );
        }
        Cmd::Conform {
            input,
            output,
            voxel_mm,
            multiple,
            min,
            max,
            model,
            gzip_level,
        } => {
            let v = Volume::load(&input).with_context(|| format!("loading {:?}", input))?;
            let spec = match &model {
                Some(path) => {
                    let m = radveil_core::rvbrain::Model::load(path)
                        .map_err(anyhow::Error::msg)
                        .with_context(|| format!("loading {:?}", path))?;
                    ConformSpec {
                        voxel_mm: m.arch.voxel_mm,
                        multiple: m.arch.grid_multiple,
                        min: m.arch.grid_min,
                        max: m.arch.grid_max,
                    }
                }
                None => ConformSpec {
                    voxel_mm,
                    multiple,
                    min,
                    max,
                },
            };
            let c = conform(&v, &spec);
            let [d, h, w] = c.shape;
            let arr = ndarray::Array3::from_shape_vec((d, h, w), c.data)
                .context("conformed grid shape")?;
            let out = Volume::from_array(arr, c.affine);
            out.save_with_compression(&output, gzip_level)
                .with_context(|| format!("writing {:?}", output))?;
            eprintln!("conformed to {d}x{h}x{w} at {:.3} mm", spec.voxel_mm);
        }
        Cmd::Deskull {
            input,
            output,
            mask_out,
            method,
            model,
            no_cubic,
            modality,
            frac,
            fill,
            no_robust,
            face,
            low_memory,
            report,
            gzip_level,
        } => {
            if output.is_none() && mask_out.is_none() {
                bail!("nothing to write: pass --output and/or --mask-out");
            }
            let mut v = Volume::load(&input).with_context(|| format!("loading {:?}", input))?;
            let located = if face {
                match locate_face(
                    &v,
                    &mut MediaPipeDetector::default(),
                    &AutoParams::default(),
                ) {
                    Ok(f) => Some(f),
                    Err(e) => {
                        eprintln!("no face prior: {e}");
                        None
                    }
                }
            } else {
                None
            };
            let params = DeskullParams {
                method: Method::parse(&method).with_context(|| {
                    format!("unknown method {method:?} (synthstrip | rvbrain | classical)")
                })?,
                model,
                face: located.as_ref().map(|f| f.frame.clone()),
                modality: Modality::parse(&modality)
                    .with_context(|| format!("unknown modality {modality:?}"))?,
                frac,
                fill,
                robust: !no_robust,
                synthstrip: SynthStripParams {
                    low_memory,
                    ..Default::default()
                },
                rvbrain: RvBrainParams {
                    cubic: !no_cubic,
                    ..Default::default()
                },
                ..Default::default()
            };
            let t0 = std::time::Instant::now();
            let (mask, rep) = if output.is_some() {
                deskull(&mut v, &params)
            } else {
                brain_mask(&v, &params)
            }
            .map_err(anyhow::Error::msg)?;
            eprintln!(
                "{:?} brain: {:.0} ml in {:.2}s",
                rep.modality,
                rep.brain_volume_ml,
                t0.elapsed().as_secs_f64()
            );
            if let Some(out) = output {
                v.save_with_compression(&out, gzip_level)?;
                eprintln!("stripped volume written to {}", out.display());
            }
            if let Some(mp) = mask_out {
                write_mask(&v, &mask, &mp)?;
            }
            if let Some(rp) = report {
                let mut js = serde_json::to_value(&rep)?;
                if let Some(f) = &located {
                    js["face_frame"] = serde_json::to_value(&f.frame)?;
                }
                let js = serde_json::to_string_pretty(&js)?;
                if rp.as_os_str() == "-" {
                    println!("{js}");
                } else {
                    std::fs::write(rp, js)?;
                }
            }
        }
        Cmd::Render {
            input,
            out_dir,
            yaw,
            pitch,
            size,
            threshold_frac,
            threshold,
        } => {
            let v = Volume::load(&input)?;
            eprintln!("orientation: {}", v.orientation());
            std::fs::create_dir_all(&out_dir)?;
            let params = RenderParams {
                size,
                threshold_frac,
                threshold_abs: threshold,
                ..Default::default()
            };
            let mut metas = Vec::new();
            for (i, &y) in yaw.iter().enumerate() {
                let view = render_view(&v, y, pitch, &params);
                let name = format!("view_{i:02}_yaw{y:+.0}.png");
                view.save_png(out_dir.join(&name))?;
                eprintln!("wrote {} (threshold {:.1})", name, view.threshold);
                metas.push(ViewMeta {
                    file: name,
                    yaw_deg: y,
                    pitch_deg: pitch,
                    size,
                    threshold: view.threshold,
                });
            }
            std::fs::write(
                out_dir.join("views.json"),
                serde_json::to_string_pretty(&ViewsFile { views: metas })?,
            )?;
        }
        Cmd::Auto {
            input,
            output,
            mode,
            region,
            ear_mode,
            mask_out,
            report,
            blur_sigma,
            depth_behind,
            no_eye_anchor,
            yaw,
            size,
            threshold_frac,
            threshold,
            no_verify,
            verify_effort,
            brain_check,
            brain_guard,
            brain_guard_mm,
            partial_face,
            auto_threshold,
            skin_depth,
            spare_air,
            level_face,
            gzip_level,
        } => {
            let mut v = Volume::load(&input).with_context(|| format!("loading {:?}", input))?;
            let params = AutoParams {
                deface: DefaceParams {
                    mode: mode.parse::<DefaceMode>().map_err(anyhow::Error::msg)?,
                    region: region
                        .parse::<radveil_core::Region>()
                        .map_err(anyhow::Error::msg)?,
                    ear_mode: ear_mode
                        .map(|m| m.parse::<DefaceMode>().map_err(anyhow::Error::msg))
                        .transpose()?,
                    blur_sigma_mm: blur_sigma,
                    depth_behind_mm: depth_behind,
                    skin_depth_mm: skin_depth,
                    eye_surface_anchor: Some(!no_eye_anchor),
                    spare_air,
                    level_face,
                    ..Default::default()
                },
                yaws: yaw,
                size,
                threshold_frac,
                threshold,
                verify: !no_verify,
                verify_effort: verify_effort
                    .parse::<radveil_core::VerifyEffort>()
                    .map_err(anyhow::Error::msg)?,
                brain_check,
                brain_guard,
                brain_guard_mm,
                partial_face,
                auto_threshold,
                return_mask: mask_out.is_some(),
                // The primary detector is the built-in one already.
                verify_with_builtin: false,
                ..Default::default()
            };
            let mut det = MediaPipeDetector::default();
            let rep = deface_auto(&mut v, &mut det, &params)
                .map_err(|e| anyhow::anyhow!("{}: {e}", input.display()))?;
            eprintln!(
                "orientation {}{}, face in {} view(s), eye_dist {:.1} mm",
                rep.orientation,
                if rep.orientation_sweep {
                    " (sweep)"
                } else {
                    ""
                },
                rep.views_detected,
                rep.deface.eye_dist_mm
            );
            eprintln!(
                "anonymized {} voxels ({:.2}% of volume), mode {:?}, region {:?}",
                rep.deface.voxels_changed,
                100.0 * rep.deface.voxels_changed as f64 / rep.deface.voxels_total as f64,
                rep.deface.mode,
                rep.deface.region,
            );
            if let (Some(r), Some(res)) = (rep.verify_rounds, rep.residual_detections) {
                eprintln!(
                    "verify: {r} escalation round(s), {res} residual detection(s){}",
                    if rep.escalated_to_block == Some(true) {
                        ", escalated to block"
                    } else {
                        ""
                    }
                );
            }
            if let Some(spared) = rep.deface.protected_voxels_spared {
                eprintln!(
                    "brain guard: on ({brain_guard_mm} mm), {spared} region voxel(s) kept untouched"
                );
            }
            let brain_touched = match &rep.brain {
                Some(b) if b.voxels_modified > 0 => {
                    eprintln!(
                        "BRAIN CHECK FAILED: {} brain voxel(s) modified ({:.0} ml brain)",
                        b.voxels_modified, b.brain_volume_ml
                    );
                    true
                }
                Some(b) => {
                    eprintln!(
                        "brain check: brain untouched ({:.0} ml), closest modified voxel {} from the brain",
                        b.brain_volume_ml,
                        b.margin_mm
                            .map_or("--".to_string(), |d| format!("{d:.1} mm"))
                    );
                    false
                }
                None => false,
            };
            v.save_with_compression(&output, gzip_level)?;
            println!("{}", output.display());
            if let (Some(mp), Some(mask)) = (mask_out, rep.mask.as_ref()) {
                write_mask(&v, mask, &mp)?;
            }
            if let Some(rp) = report {
                let json = serde_json::to_string_pretty(&rep)?;
                if rp.as_os_str() == "-" {
                    println!("{json}");
                } else {
                    std::fs::write(&rp, json)?;
                }
            }
            if brain_touched {
                // The output is written either way; the status code lets a
                // pipeline stop on it.
                std::process::exit(3);
            }
        }
        Cmd::Deface {
            input,
            output,
            mode,
            region,
            ear_mode,
            landmarks,
            landmarks3d,
            mask_out,
            blur_sigma,
            depth_behind,
            skin_depth,
            spare_air,
            level_face,
            no_eye_anchor,
            size,
            threshold_frac,
            qa,
        } => {
            let mut v = Volume::load(&input)?;
            eprintln!("orientation: {}", v.orientation());

            let lm3d: Vec<Option<[f64; 3]>> = if let Some(p) = landmarks3d {
                let f: Landmarks3dFile = serde_json::from_str(&std::fs::read_to_string(p)?)?;
                f.points
            } else if let Some(p) = landmarks {
                let f: Landmarks2dFile = serde_json::from_str(&std::fs::read_to_string(p)?)?;
                if f.views.is_empty() {
                    bail!("landmark file contains no views");
                }
                let params = RenderParams {
                    size,
                    threshold_frac,
                    ..Default::default()
                };
                let mut per_view = Vec::new();
                for vl in &f.views {
                    let view = render_view(&v, vl.yaw_deg, vl.pitch_deg, &params);
                    per_view.push(unproject(&view, &vl.points));
                }
                combine_views(&per_view)
            } else {
                bail!("provide --landmarks (2D per-view) or --landmarks3d");
            };

            let frame = face_frame_from_landmarks(&lm3d).map_err(anyhow::Error::msg)?;
            eprintln!(
                "face frame: eye_dist {:.1} mm, nose tip at [{:.1}, {:.1}, {:.1}]",
                frame.eye_dist, frame.nose_tip[0], frame.nose_tip[1], frame.nose_tip[2]
            );

            let params = DefaceParams {
                mode: mode.parse::<DefaceMode>().map_err(anyhow::Error::msg)?,
                region: region
                    .parse::<radveil_core::Region>()
                    .map_err(anyhow::Error::msg)?,
                ear_mode: ear_mode
                    .map(|m| m.parse::<DefaceMode>().map_err(anyhow::Error::msg))
                    .transpose()?,
                blur_sigma_mm: blur_sigma,
                depth_behind_mm: depth_behind,
                skin_depth_mm: skin_depth,
                eye_surface_anchor: Some(!no_eye_anchor),
                spare_air,
                level_face,
                ..Default::default()
            };
            let (report, mask) = deface(&mut v, &frame, &params);
            eprintln!(
                "anonymized {} voxels ({:.2}% of volume), mode {:?} (ears {:?}), region {:?}, eye bias {:.1} mm",
                report.voxels_changed,
                100.0 * report.voxels_changed as f64 / report.voxels_total as f64,
                report.mode,
                report.ear_mode,
                report.region,
                report.eye_bias_mm
            );
            for r in &report.rois {
                eprintln!("  {:>9}: {} voxels", r.label, r.voxels);
            }
            v.save(&output)?;
            println!("{}", output.display());

            if let Some(mp) = mask_out {
                write_mask(&v, &mask, &mp)?;
            }

            if qa {
                let stem = output
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("out")
                    .trim_end_matches(".gz")
                    .trim_end_matches(".nii")
                    .to_string();
                let dir = output.parent().unwrap_or(std::path::Path::new("."));
                let params = RenderParams::default();
                let after = render_view(&v, 0.0, 0.0, &params);
                after.save_png(dir.join(format!("{stem}_qa_after.png")))?;
                let orig = Volume::load(&input)?;
                let before = render_view(&orig, 0.0, 0.0, &params);
                before.save_png(dir.join(format!("{stem}_qa_before.png")))?;
                eprintln!("QA renders written next to output");
            }
        }
    }
    Ok(())
}

fn write_mask(v: &Volume, mask: &ndarray::Array3<u8>, path: &std::path::Path) -> Result<()> {
    let mut mv = Volume {
        accel: Default::default(),
        data: mask.mapv(|m| m as f32),
        affine: v.affine,
        world_to_voxel: v.world_to_voxel,
        spacing: v.spacing,
        header: v.header.clone(),
        datatype: nifti::NiftiType::Uint8,
        scl_slope: 1.0,
        scl_inter: 0.0,
    };
    mv.header.scl_slope = 1.0;
    mv.header.scl_inter = 0.0;
    mv.save(path)?;
    eprintln!("mask written to {}", path.display());
    Ok(())
}
