# Deployment: batch jobs and on-the-fly services

RadVeil runs the same pipeline in both settings. The Rust core releases
Python's GIL and keeps per-thread scratch memory, so **one process with
many threads** is the fastest setup in both cases: no worker start-up, one
shared copy of the models, and every core busy.

Measured on a 14-core Apple M-series laptop with real T1/FLAIR volumes
(256×256×150–160). Scale roughly with core count on x86/ARM servers.

| Setting | Setup | Result |
|---|---|---|
| one request, warm process | bytes in → `deface_volume` → bytes out | **~0.3 s** (0.2–0.45 s) |
| service, 14 concurrent requests | threads in one process | **16.6 requests/s** |
| batch, `deface_batch` defaults | 14 threads, files in/out | **11.3 scans/s** |
| batch, `executor="processes"` | 14 worker processes | 10.1 scans/s |
| batch, one worker | sequential | 2.6 scans/s |
| CLI, one file per process | `radveil auto in out` | ~0.5 s per file |

Where one request's time goes (T1, 256×256×160): load .nii.gz 55–85 ms,
~6 surface renders of ~37 ms each (the dominant cost; each render already
uses all cores), face detection ~8 ms per view, anonymization 10–100 ms,
.nii.gz encoding 55–80 ms (plain .nii: ~12 ms). Model loading happens once
per process and takes ~4 ms.

Memory: ~500 MB baseline for a Python process plus ~70 MB per volume
being processed concurrently (roughly 4–5× the voxel data). Face
detection needs ~5 MB of scratch memory per thread.

## Batch (lab datasets, training-data preparation)

```python
import radveil

reports = radveil.deface_batch("/data/cohort/*.nii.gz", output_dir="/data/defaced",
                               mode="adaptive")
refused = [r for r in reports if "error" in r]   # review these manually
```

- Defaults: threads, `workers = min(cpu_count, 32, n_files)`. Lower
  `workers` if memory is tight; the throughput knee is at about half the
  cores (7 workers reached 90% of the 14-worker rate).
- A refusal (no validated face, e.g. already defaced or non-head data) is
  returned as `{"input", "output", "error"}`; the batch continues.
- `executor="processes"` isolates each volume in its own process, which is
  useful when inputs are untrusted or unusually large. Call it under
  `if __name__ == "__main__":`.
- For very large cohorts, shard the file list across machines (one
  `deface_batch` per machine); there is no shared state.
- Without Python: the `radveil` CLI processes one volume per call, e.g.
  `ls *.nii.gz | xargs -P 8 -I{} radveil auto {} out/{}`. Each call pays
  process start-up and file I/O (~0.5 s per file in total).

## Service (production, on the fly)

```python
vol = radveil.Volume.from_bytes(upload)          # .nii or .nii.gz bytes
report = radveil.deface_volume(vol, mode="blur")  # RuntimeError -> refuse (HTTP 422)
return vol.to_bytes()                             # .nii.gz bytes, nothing on disk
```

[`examples/service_http.py`](../examples/service_http.py) is a complete
standard-library HTTP service built on this: `POST /deface` with the
volume as the body, `GET /health`, a bounded number of concurrent volumes
(`--workers`), HTTP 422 for refusals and 400 for unreadable input. Put it
(or the same four lines in your framework of choice) behind a reverse
proxy for TLS, authentication and upload limits.

- Keep the process long-lived; it holds no per-request state.
- Size `--workers` for memory (~70 MB per concurrent 256×256×150 volume),
  not just cores: extra concurrency beyond the core count adds latency,
  not throughput.
- Latency is dominated by rendering. `mode`, `region` and `verify`
  change it little; `verify=False` saves the ~3 re-renders but also skips
  the guarantee that the detector no longer finds the face. Keep it on
  for production.
- C/C++ hosts: `radveil_deface_auto` in the C API runs the same pipeline
  in-process (see `crates/radveil-capi/include/radveil.h`).

## CPU support

The face-detection kernels pick NEON (ARM64) or AVX2+FMA (x86-64) at run
time and fall back to portable, auto-vectorized kernels on older CPUs
(e.g. x86 servers without AVX2). Results are identical on every path.
`RADVEIL_SIMD=off` forces the portable kernels (about 20% slower
detection). `radveil_core::nn::simd_level()` reports the active set.
