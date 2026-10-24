// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Time the embedded face networks on the built-in engine:
//! `cargo run --release --example nn_bench -p radveil-core`
//! (`PROFILE=1` adds a per-op breakdown).
use radveil_core::nn::Net;
use std::time::Instant;

fn main() {
    println!("kernels: {}", radveil_core::nn::simd_level());
    let models: [(&str, &[u8]); 2] = [
        (
            "blazeface",
            include_bytes!("../models/blazeface_short_range.rvnn"),
        ),
        (
            "face_mesh_v2",
            include_bytes!("../models/face_mesh_v2.rvnn"),
        ),
    ];
    for (name, bytes) in models {
        let t = Instant::now();
        let net = Net::from_bytes(bytes).unwrap();
        let load = t.elapsed().as_secs_f64() * 1e3;
        let [h, w, c] = net.input_shape();
        let x: Vec<f32> = (0..h * w * c)
            .map(|i| ((i * 7919) % 255) as f32 / 255.0)
            .collect();
        let mut ws = net.workspace();
        net.run(&mut ws, &x);
        let n = 200;
        let t = Instant::now();
        for _ in 0..n {
            std::hint::black_box(net.run(&mut ws, &x));
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
        println!(
            "{name}: {ms:.2} ms per run (load {load:.1} ms, workspace {:.1} MB per thread)",
            net.workspace_bytes() as f64 / 1e6
        );
        if std::env::var_os("PROFILE").is_some() {
            let mut acc: std::collections::BTreeMap<String, f64> = Default::default();
            for _ in 0..50 {
                for (label, ms) in net.profile(&mut ws, &x) {
                    let kind = label.split(' ').next().unwrap().to_string();
                    *acc.entry(kind).or_default() += ms / 50.0;
                }
            }
            let mut v: Vec<_> = acc.into_iter().collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (k, ms) in v {
                println!("    {ms:6.3} ms  {k}");
            }
        }
    }
}
