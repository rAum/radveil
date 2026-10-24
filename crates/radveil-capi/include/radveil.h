/* SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
 * SPDX-License-Identifier: Apache-2.0 */

/* RadVeil C API — face anonymization for volumetric head imaging.
 *
 * Conventions:
 *  - i32-returning functions: 0 = success, negative = error;
 *    radveil_last_error() gives the message (thread-local).
 *  - Voxel arrays are C-order (nx, ny, nz), k fastest.
 *  - World coordinates are NIfTI RAS+ millimeters.
 *  - radveil_deface_auto detects the face itself (built-in MediaPipe
 *    detector). Hosts with their own detector supply 68-point
 *    3D landmarks to radveil_deface instead (lift 2D detections through
 *    radveil_render's position map, or use any other source).
 *
 * License: Apache-2.0
 */
#ifndef RADVEIL_H
#define RADVEIL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RadveilVolume RadveilVolume;

/* Any negative/NaN field = auto (scaled from detected inter-eye distance). */
typedef struct RadveilDefaceParams {
    double width_factor;
    double above_eyes_mm;
    double below_chin_mm;
    double depth_behind_mm;
    double depth_front_mm;
    double blur_sigma_mm;
    double adaptive_factor;
    double skin_depth_mm;  /* depth below the measured skin; ~4.5 mm auto */
    uint32_t options;      /* RADVEIL_SPARE_AIR | _LEVEL_FACE | _PARTIAL_FACE | _AUTO_THRESHOLD */
} RadveilDefaceParams;

/* RadveilDefaceParams.options (opt-in, default off):
 *   RADVEIL_SPARE_AIR   leave background air untouched: write only tissue,
 *                       the skin edge and a smooth shell around the head
 *   RADVEIL_LEVEL_FACE  measure the face skin along the level head axes, so
 *                       the lower lip and chin are removed too */
#define RADVEIL_SPARE_AIR 1u
#define RADVEIL_LEVEL_FACE 2u
/*   RADVEIL_PARTIAL_FACE (radveil_deface_auto / radveil_anonymization_mask)
 *                       also handle faces cut by the field of view (an axial
 *                       slab ending at the nose); implies RADVEIL_LEVEL_FACE */
#define RADVEIL_PARTIAL_FACE 4u
/*   RADVEIL_AUTO_THRESHOLD (radveil_deface_auto / radveil_anonymization_mask)
 *                       noisy, dim scans (e.g. infants): retry at higher
 *                       surface thresholds when no face is found */
#define RADVEIL_AUTO_THRESHOLD 8u

/* Ear-ROI geometry for radveil_deface_region; any negative/NaN field = auto
 * (scaled from detected inter-eye distance). Ears are located by surface
 * evidence: a protrusion scan inside a calibrated window detects each
 * pinna, and the medial cut wall follows the measured scalp minus guard_mm,
 * never medial of the skull or inner_mm. down/back center the fallback box
 * (used when no pinna is detected) below the eye line / behind the eye
 * plane in the head-based ear frame (eye axis + scanner superior);
 * height/depth are its full v/w extents. */
typedef struct RadveilEarParams {
    double inner_mm; /* absolute medial floor */
    double outer_mm;
    double guard_mm; /* cut depth below the detected head surface */
    double down_mm;
    double back_mm;
    double height_mm;
    double depth_mm;
} RadveilEarParams;

typedef struct RadveilReport {
    uint64_t voxels_changed;
    uint64_t voxels_total;
    double eye_dist_mm;
    double blur_sigma_mm; /* resolved (auto-scaled) value */
} RadveilReport;

typedef struct RadveilAutoReport {
    RadveilReport base;
    uint32_t views_detected;     /* rendered views the face was validated in */
    uint8_t orientation_sweep;   /* 1 = found via the 6-direction sweep */
    uint8_t escalated_to_block;  /* 1 = verification ended in an opaque fill */
    int32_t verify_rounds;       /* escalation rounds; -1 = not verified */
    int32_t residual_detections; /* at the face after the last round; -1 = n/a */
    int64_t brain_voxels_modified; /* 0 = brain untouched; -1 = check not run */
    double brain_margin_mm;        /* brain-to-modified-voxel distance; NaN if n/a */
    int64_t brain_guard_spared;    /* region voxels kept by the brain guard; -1 if off */
} RadveilAutoReport;

/* Set every field of a params struct to "auto". Do this before overriding
 * individual fields: a zero-initialised struct means 0 mm, not auto. */
void radveil_deface_params_init(RadveilDefaceParams *params);
void radveil_ear_params_init(RadveilEarParams *params);

const char *radveil_version(void);
const char *radveil_last_error(void);

/* Constructors / destructor */
RadveilVolume *radveil_volume_load(const char *path); /* .nii / .nii.gz */
RadveilVolume *radveil_volume_from_array(size_t nx, size_t ny, size_t nz,
                                         const float *data /* nx*ny*nz */,
                                         const double *affine /* 4x4 row-major,
                                                                 voxel->RAS mm */);
void radveil_volume_free(RadveilVolume *vol);

/* Introspection */
void radveil_volume_shape(const RadveilVolume *vol, uint64_t dims[3]);
void radveil_volume_orientation(const RadveilVolume *vol, char *buf, size_t len);

/* Save (.nii / .nii.gz); preserves datatype, scaling and header; voxels
 * outside the deface mask are bit-exact. gzip_level 0-9 (1 = fast default). */
int32_t radveil_volume_save(const RadveilVolume *vol, const char *path,
                            uint32_t gzip_level);

/* Render one surface view (orthographic; yaw 0 faces anterior, orbiting the
 * superior axis; pitch tilts).  Caller-allocated outputs, any may be NULL:
 *   rgb   size*size*3 bytes
 *   pos   size*size*3 floats — world position of the surface hit (NaN = miss)
 *   flags size*size bytes — 0 miss, 1 hit, 2 hit on a volume boundary
 * threshold_abs = NaN selects the automatic threshold (percentile-based for
 * MRI, -400 HU for CT-like data). */
int32_t radveil_render(const RadveilVolume *vol, double yaw_deg, double pitch_deg,
                       uint32_t size, float threshold_frac, float threshold_abs,
                       uint8_t *rgb, float *pos, uint8_t *flags);

/* Anonymize in place. points = 68*3 doubles (standard 68-point landmark order), world RAS
 * mm; NaN coordinates mark a landmark as missing. mode: "blur", "cut",
 * "block", "adaptive". params may be NULL (all auto). mask_out may be NULL
 * or nx*ny*nz bytes (1 = modified voxel). report may be NULL.
 * The face ROI's depth bounds are automatically anchored to the measured
 * eye-skin surface (corrects the posterior landmark bias of thick-slice
 * scans; a no-op on well-resolved data). */
int32_t radveil_deface(RadveilVolume *vol, const double *points, const char *mode,
                       const RadveilDefaceParams *params, uint8_t *mask_out,
                       RadveilReport *report);

/* Like radveil_deface, with a region selector: "face", "ears" or
 * "face+ears". Ears are biometric identifiers; the 68-point set has no ear
 * landmarks, so each pinna is detected from surface evidence relative to
 * the eyes. ear_params may be NULL (all auto). The report's
 * voxels_changed covers all regions. */
int32_t radveil_deface_region(RadveilVolume *vol, const double *points,
                              const char *mode, const char *region,
                              const RadveilDefaceParams *params,
                              const RadveilEarParams *ear_params,
                              uint8_t *mask_out, RadveilReport *report);

/* One-call pipeline with the built-in MediaPipe face detector: render ->
 * detect -> validate -> lift -> anonymize -> verify (re-render + re-detect,
 * escalating blur and finally an opaque fill until no face is found).
 * verify: RADVEIL_VERIFY_OFF, _ON (default effort: the detection views plus
 * a 3x3 yaw/pitch grid), _FAST (detection views only) or _THOROUGH (a 5x5
 * grid; slower, rarely finds more). brain: RADVEIL_BRAIN_CHECK and/or RADVEIL_BRAIN_GUARD
 * (see below), or 0.
 * Returns -1 and leaves the volume untouched when no
 * validated face is found (e.g. an already-defaced volume). mode: "blur",
 * "cut", "block", "adaptive"; region: "face", "ears", "face+ears".
 * params, ear_params, mask_out (nx*ny*nz bytes) and report may be NULL. */
/* Flags for the `brain` argument of radveil_deface_auto and
 * radveil_anonymization_mask (combine with |; 0 = neither). Both extract the
 * brain from the input with SynthStrip (~3 s, ~3.2 GB peak).
 *   RADVEIL_BRAIN_CHECK  report brain voxels modified (brain_voxels_modified)
 *   RADVEIL_BRAIN_GUARD  never modify the brain or anything within 3 mm of it */
#define RADVEIL_BRAIN_CHECK 1
#define RADVEIL_BRAIN_GUARD 2

#define RADVEIL_VERIFY_OFF 0
#define RADVEIL_VERIFY_ON 1
#define RADVEIL_VERIFY_FAST 2
#define RADVEIL_VERIFY_THOROUGH 3

int32_t radveil_deface_auto(RadveilVolume *vol, const char *mode, const char *region,
                            const RadveilDefaceParams *params,
                            const RadveilEarParams *ear_params, uint8_t verify,
                            uint8_t brain, uint8_t *mask_out,
                            RadveilAutoReport *report);

/* Escape hatch: the region radveil_deface_auto would anonymize, written to
 * mask_out (nx*ny*nz bytes, required; 1 = voxel it would rewrite), WITHOUT
 * modifying the volume. Detection, validation, the safety refusals and
 * the brain flags run as in radveil_deface_auto; nothing
 * is rewritten and nothing is verified, so the mask says where RadVeil
 * would anonymize, not that doing so defeats re-identification. Returns -1
 * when no plausible face is found. params, ear_params, report may be NULL;
 * report->base.voxels_changed counts the voxels that would change and the
 * verification fields are -1. */
int32_t radveil_anonymization_mask(const RadveilVolume *vol, const char *mode,
                                   const char *region, const RadveilDefaceParams *params,
                                   const RadveilEarParams *ear_params, uint8_t brain,
                                   uint8_t *mask_out, RadveilAutoReport *report);

typedef struct {
    double brain_volume_ml;
    uint64_t brain_voxels;
    double centroid[3];   /* mask centroid, world RAS+ mm */
    uint8_t modality;     /* 0 = MRI, 1 = CT (after resolving "auto") */
    uint8_t face_prior;   /* 1 when the face prior was used */
    float fill;           /* value written outside the brain; NaN if strip = 0 */
} RadveilDeskullReport;

/* Brain extraction (skull stripping). method: "synthstrip" (the SynthStrip
 * network, any contrast/CT) or "classical" (model-free). modality: "auto",
 * "mri", "ct". frac: MRI fractional intensity threshold
 * (classical; NaN = default 0.5). use_face = 1 (classical) locates the
 * face with the built-in detector and uses it as an anatomical prior (MRI;
 * skipped when no face is found, see report->face_prior). strip = 1
 * modifies the volume in place, non-brain voxels = fill (NaN = default: 0
 * for MRI, air for CT). low_memory = 1 trades ≤3% speed for ~0.4 GB less
 * peak memory (synthstrip; identical output). mask_out (nx*ny*nz bytes,
 * 1 = brain) and report may be NULL. */
int32_t radveil_deskull(RadveilVolume *vol, const char *method, const char *modality, double frac,
                        uint8_t use_face, uint8_t strip, uint8_t low_memory, float fill,
                        uint8_t *mask_out, RadveilDeskullReport *report);

/* Like radveil_deface_region, with a separate mode for the ear regions in the
 * same pass: ear_mode "blur", "cut", "block", "adaptive", or NULL for the same
 * mode as `mode` (e.g. mode "adaptive" + ear_mode "cut"). mask_out covers all
 * regions. */
int32_t radveil_deface_regions(RadveilVolume *vol, const double *points,
                               const char *mode, const char *ear_mode,
                               const char *region,
                               const RadveilDefaceParams *params,
                               const RadveilEarParams *ear_params,
                               uint8_t *mask_out, RadveilReport *report);

#ifdef __cplusplus
}
#endif

#endif /* RADVEIL_H */
