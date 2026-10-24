# Performance notes

## Why the renderer is CPU-only

During development RadVeil had an optional wgpu compute renderer (cargo
feature `gpu`, Metal / Vulkan / DX12). It was removed before the first
release after the CPU ray caster
was tuned, because the CPU path finished the full pipeline faster on every
volume we tested, even against a fully optimized GPU path.

### What was compared

- **CPU**: the current renderer. Voxel-space ray march, macro-grid
  empty-space skipping, memoized intensity percentiles, interior fast path
  for trilinear sampling.
- **GPU**: the wgpu renderer *after* it got the same optimizations (the
  same voxel-space march and macro-grid skipping in the shader, voxels
  uploaded in memory order and indexed by stride, reused output/readback
  buffers, parallel unpack). The original GPU path was slower
  still: about 4.3 ms per view, with 37–59 ms per volume upload. This
  optimized version was the strongest case for keeping the GPU.

Hardware: Apple M4 Pro (14 CPU cores, 20 GPU cores), 512 × 512 renders,
release build, built-in MediaPipe detector, default `deface_auto`
parameters (3 yaws, verify on).


### Per render (min of 20 runs)

| Volume | Shape | CPU | GPU | GPU re-upload after deface |
|---|---|---:|---:|---:|
| IXI012 T1 | 256×256×150 | 3.37 ms | 1.49 ms | 19.2 ms |
| OASIS-1 0001 | 256×256×128 | 4.62 ms | 1.53 ms | 16.4 ms |

A single GPU render is 2–3× faster. But each anonymization pass changes
voxels, and the GPU copy of the volume then has to be uploaded again. One
re-upload costs as much as 4–6 CPU renders.

### End-to-end pipeline (`deface_auto`, all 23 volumes)

| | Median per view | Total pipeline time |
|---|---:|---:|
| CPU | 3.67 ms | 8.99 s |
| GPU | 3.00 ms | 9.62 s |

**The GPU build was slower on 23 of 23 volumes (median +8%).** A subject
needs only a handful of renders (3 detection views, validation views,
3 per verify round), about 20–40 ms of CPU work in a 170–1300 ms pipeline
dominated by face detection and anonymization. The GPU cannot win back its
first upload plus one re-upload per verify round in that budget.

Both builds gave identical pipeline outcomes: the same views accepted, the
same verify rounds, the same OASIS refusals. The optimized GPU path
matched the CPU closely (0–5 pixels with a different hit/miss result,
landmarks within 0.02–1.2 mm). So the removal is about speed and upkeep,
not correctness.

## Other rejected options

- **Coarser ray step** (0.6–1.0 voxel instead of 0.4): only ~10–20% faster
  per render. It shifted landmarks by 0.5–2 mm, changed verify-round
  counts on some IXI subjects, and broke up the thin shells left on
  defaced OASIS volumes (99th-percentile surface error up to 14 mm).
- **Direction-aware step**: frontal rays run anterior-posterior, which is
  the fine in-plane axis of axial scans, so it barely changes the sample
  count where it matters.
