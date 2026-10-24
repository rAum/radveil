// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Brain extraction on layered synthetic heads (no patient data needed).

use ndarray::Array3;
use radveil_core::{brain_mask, deskull, DeskullParams, Method, Modality, Volume};

fn classical() -> DeskullParams {
    DeskullParams {
        method: Method::Classical,
        ..Default::default()
    }
}

const N: usize = 128;
const SP: f64 = 1.5;

/// Tissue layers of a synthetic head, world mm (RAS, origin at the brain
/// centre): brain ellipsoid, CSF rim, skull, scalp, plus a neck column
/// that leaves the field of view and a foramen in the skull base.
#[derive(PartialEq)]
enum Tissue {
    Air,
    Scalp,
    Bone,
    Csf,
    Brain,
    Neck,
}

fn tissue(x: f64, y: f64, z: f64) -> Tissue {
    let e = |s: f64| (x / (62.0 * s)).powi(2) + (y / 78.0 / s).powi(2) + (z / (55.0 * s)).powi(2);
    let r = e(1.0);
    let foramen = x.hypot(y + 5.0) < 14.0 && z < 0.0;
    if r <= 1.0 {
        Tissue::Brain
    } else if e(1.05) <= 1.0 {
        Tissue::Csf
    } else if e(1.14) <= 1.0 {
        if foramen {
            Tissue::Neck
        } else {
            Tissue::Bone
        }
    } else if e(1.24) <= 1.0 {
        Tissue::Scalp
    } else if z < -40.0 && x.hypot(y + 5.0) < 45.0 {
        Tissue::Neck
    } else {
        Tissue::Air
    }
}

fn phantom(value: impl Fn(&Tissue, f64, f64, f64) -> f32) -> (Volume, Array3<u8>) {
    let c = N as f64 / 2.0;
    // Brain centre sits above the volume centre so the neck is cut by the
    // bottom of the field of view, as in real scans.
    let off = 25.0;
    let mut data = Array3::<f32>::zeros((N, N, N));
    let mut truth = Array3::<u8>::zeros((N, N, N));
    for ((i, j, k), v) in data.indexed_iter_mut() {
        let (x, y, z) = (
            (i as f64 - c) * SP,
            (j as f64 - c) * SP,
            (k as f64 - c) * SP - off,
        );
        let t = tissue(x, y, z);
        *v = value(&t, x, y, z);
        truth[[i, j, k]] = u8::from(t == Tissue::Brain || t == Tissue::Csf);
    }
    let affine = [
        [SP, 0.0, 0.0, -c * SP],
        [0.0, SP, 0.0, -c * SP],
        [0.0, 0.0, SP, -c * SP - off],
        [0.0, 0.0, 0.0, 1.0],
    ];
    (Volume::from_array(data, affine), truth)
}

fn dice(a: &Array3<u8>, b: &Array3<u8>) -> f64 {
    let (mut inter, mut sa, mut sb) = (0usize, 0usize, 0usize);
    for (&x, &y) in a.iter().zip(b.iter()) {
        sa += (x != 0) as usize;
        sb += (y != 0) as usize;
        inter += (x != 0 && y != 0) as usize;
    }
    2.0 * inter as f64 / (sa + sb).max(1) as f64
}

#[test]
fn mri_t1_phantom() {
    // T1-like contrast: brain bright with a WM/GM gradient, CSF and bone
    // dark, scalp fat brightest.
    let (vol, truth) = phantom(|t, x, y, z| match t {
        Tissue::Air => 5.0,
        Tissue::Scalp => 900.0,
        Tissue::Bone => 40.0,
        Tissue::Csf => 120.0,
        Tissue::Brain => {
            let r = (x / 62.0).powi(2) + (y / 78.0).powi(2) + (z / 55.0).powi(2);
            (700.0 - 250.0 * r) as f32
        }
        Tissue::Neck => 450.0,
    });
    let (mask, rep) = brain_mask(&vol, &classical()).unwrap();
    assert_eq!(rep.modality, Modality::Mri);
    let d = dice(&mask, &truth);
    assert!(
        d > 0.9,
        "MRI phantom dice {d:.3}, {:.0} ml",
        rep.brain_volume_ml
    );
}

#[test]
fn ct_phantom_strips_to_air() {
    let (mut vol, truth) = phantom(|t, _, _, _| match t {
        Tissue::Air => -1000.0,
        Tissue::Scalp => 30.0,
        Tissue::Bone => 900.0,
        Tissue::Csf => 8.0,
        Tissue::Brain => 35.0,
        Tissue::Neck => 45.0,
    });
    let (mask, rep) = deskull(&mut vol, &classical()).unwrap();
    assert_eq!(rep.modality, Modality::Ct);
    let d = dice(&mask, &truth);
    assert!(d > 0.9, "CT phantom dice {d:.3}");
    // Everything outside the mask is air now; inside is untouched.
    for ((idx, &m), &v) in mask.indexed_iter().zip(vol.data.iter()) {
        if m == 0 {
            assert_eq!(v, -1000.0, "at {idx:?}");
        }
    }
    assert_eq!(rep.fill, Some(-1000.0));
}
