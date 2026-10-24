// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Small 4x4 affine / 3-vector helpers (f64, row-major).

pub type Mat4 = [[f64; 4]; 4];
pub type Vec3 = [f64; 3];

pub fn apply(m: &Mat4, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3],
    ]
}

/// Apply only the linear (rotation/scale) part.
pub fn apply_linear(m: &Mat4, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Invert an affine transform (linear 3x3 + translation).
/// Matrix product `a * b` (apply `b` first).
pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = (0..4).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

pub fn invert(m: &Mat4) -> Mat4 {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let id = 1.0 / det;
    let inv = [
        [
            (e * i - f * h) * id,
            (c * h - b * i) * id,
            (b * f - c * e) * id,
        ],
        [
            (f * g - d * i) * id,
            (a * i - c * g) * id,
            (c * d - a * f) * id,
        ],
        [
            (d * h - e * g) * id,
            (b * g - a * h) * id,
            (a * e - b * d) * id,
        ],
    ];
    let t = [m[0][3], m[1][3], m[2][3]];
    let nt = [
        -(inv[0][0] * t[0] + inv[0][1] * t[1] + inv[0][2] * t[2]),
        -(inv[1][0] * t[0] + inv[1][1] * t[1] + inv[1][2] * t[2]),
        -(inv[2][0] * t[0] + inv[2][1] * t[1] + inv[2][2] * t[2]),
    ];
    [
        [inv[0][0], inv[0][1], inv[0][2], nt[0]],
        [inv[1][0], inv[1][1], inv[1][2], nt[1]],
        [inv[2][0], inv[2][1], inv[2][2], nt[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

pub fn normalize(a: Vec3) -> Vec3 {
    let n = norm(a);
    if n == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / n)
    }
}

/// Orientation codes for the voxel axes of an affine, nibabel-style.
/// Each voxel axis is labeled by the world direction it mostly points to:
/// R/L, A/P, S/I (world frame is NIfTI RAS+).
#[allow(clippy::needless_range_loop)]
pub fn axcodes(affine: &Mat4) -> [char; 3] {
    let mut codes = ['?'; 3];
    // Track which world axis each voxel axis claims, greedily by magnitude.
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for vax in 0..3 {
        for wax in 0..3 {
            pairs.push((vax, wax, affine[wax][vax]));
        }
    }
    pairs.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
    let mut vused = [false; 3];
    let mut wused = [false; 3];
    for (vax, wax, val) in pairs {
        if vused[vax] || wused[wax] {
            continue;
        }
        vused[vax] = true;
        wused[wax] = true;
        let pos = ['R', 'A', 'S'];
        let neg = ['L', 'P', 'I'];
        codes[vax] = if val >= 0.0 { pos[wax] } else { neg[wax] };
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_roundtrip() {
        let m: Mat4 = [
            [0.0, -1.2, 0.0, 10.0],
            [1.1, 0.0, 0.0, -5.0],
            [0.0, 0.1, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv = invert(&m);
        let p = [3.0, -7.0, 2.5];
        let q = apply(&inv, apply(&m, p));
        for k in 0..3 {
            assert!((p[k] - q[k]).abs() < 1e-9);
        }
    }

    #[test]
    fn axcodes_identity_is_ras() {
        let m: Mat4 = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(axcodes(&m), ['R', 'A', 'S']);
    }

    #[test]
    fn axcodes_lps() {
        let m: Mat4 = [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(axcodes(&m), ['L', 'P', 'S']);
    }
}
