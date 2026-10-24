// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Render experiment harness: speed, render parity and pipeline outcome
//! per config. Usage: render_eval <list-file> <config>...
//! A config is `label:ENV=VAL,ENV=VAL` (e.g. `base:` or `x:MY_KNOB=1`); the first
//! config is the reference the others are compared against.
use radveil_core::*;
use std::time::Instant;

struct Out {
    ms_view: f64,
    ms_auto: f64,
    view: RenderView,
    auto: Result<AutoReport, String>,
}

fn run(path: &str, env: &[(String, String)]) -> Out {
    for (k, v) in env {
        std::env::set_var(k, v);
    }
    let v = Volume::load(path).unwrap();
    let p = RenderParams::default();
    let _ = render_view(&v, 0.0, 0.0, &p); // warm caches / GPU upload
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        for y in [-20.0, 0.0, 20.0] {
            std::hint::black_box(render_view(&v, y, 0.0, &p));
        }
    }
    let ms_view = t.elapsed().as_secs_f64() * 1e3 / (3 * reps) as f64;
    let view = render_view(&v, 0.0, 0.0, &p);
    let mut vol = Volume::load(path).unwrap();
    let mut det = MediaPipeDetector::default();
    let t = Instant::now();
    let auto = deface_auto(&mut vol, &mut det, &AutoParams::default());
    let ms_auto = t.elapsed().as_secs_f64() * 1e3;
    for (k, _) in env {
        std::env::remove_var(k);
    }
    Out {
        ms_view,
        ms_auto,
        view,
        auto,
    }
}

fn parity(a: &RenderView, b: &RenderView) -> (usize, usize, f64, f64) {
    let n = a.hit.len();
    let hit_mis = (0..n).filter(|&i| a.hit[i] != b.hit[i]).count();
    let rgb_diff = a
        .rgb
        .iter()
        .zip(&b.rgb)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count()
        / 3;
    let mut ds: Vec<f64> = (0..n)
        .filter(|&i| a.hit[i] && b.hit[i])
        .map(|i| {
            (0..3)
                .map(|c| (a.pos[i * 3 + c] - b.pos[i * 3 + c]) as f64)
                .map(|d| d * d)
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    ds.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let p99 = ds.get(ds.len() * 99 / 100).copied().unwrap_or(0.0);
    let max = ds.last().copied().unwrap_or(0.0);
    (hit_mis, rgb_diff, p99, max)
}

fn lm_dist(a: &AutoReport, b: &AutoReport) -> f64 {
    let d: Vec<f64> = a
        .landmarks_3d
        .iter()
        .zip(&b.landmarks_3d)
        .filter_map(|(x, y)| Some(affine::norm(affine::sub((*x)?, (*y)?))))
        .collect();
    if d.is_empty() {
        f64::NAN
    } else {
        d.iter().sum::<f64>() / d.len() as f64
    }
}

fn summary(r: &Result<AutoReport, String>) -> String {
    match r {
        Ok(r) => format!(
            "face v={} rounds={:?} blk={}",
            r.views_detected,
            r.verify_rounds.unwrap_or(0),
            r.escalated_to_block.unwrap_or(false) as u8
        ),
        Err(e) if e.contains("no plausible") => "no-face".into(),
        Err(_) => "refused".into(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let list = std::fs::read_to_string(&args[1]).unwrap();
    let configs: Vec<(String, Vec<(String, String)>)> = args[2..]
        .iter()
        .map(|c| {
            let (l, e) = c.split_once(':').unwrap_or((c, ""));
            let env = e
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|kv| {
                    let (k, v) = kv.split_once('=').unwrap();
                    (k.into(), v.into())
                })
                .collect();
            (l.to_string(), env)
        })
        .collect();
    MediaPipeDetector::warm_up().unwrap();
    println!(
        "{:<28} {:<10} {:>8} {:>9} {:>7} {:>6} {:>7} {:>7} {:>8}  outcome",
        "volume", "config", "ms/view", "ms/auto", "hitMis", "rgb>8", "p99mm", "maxmm", "lm-mm"
    );
    for path in list
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let name: String = std::path::Path::new(path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .chars()
            .take(28)
            .collect();
        let outs: Vec<Out> = configs.iter().map(|(_, e)| run(path, e)).collect();
        for ((label, _), o) in configs.iter().zip(&outs) {
            let (hm, rd, p99, mx) = parity(&outs[0].view, &o.view);
            let lm = match (&outs[0].auto, &o.auto) {
                (Ok(a), Ok(b)) => lm_dist(a, b),
                _ => f64::NAN,
            };
            println!(
                "{:<28} {:<10} {:>8.2} {:>9.0} {:>7} {:>6} {:>7.3} {:>7.3} {:>8.3}  {}",
                name,
                label,
                o.ms_view,
                o.ms_auto,
                hm,
                rd,
                p99,
                mx,
                lm,
                summary(&o.auto)
            );
        }
    }
}
