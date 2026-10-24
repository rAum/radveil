# radveil-wasm

WebAssembly build of the full RadVeil pipeline. The whole `deface_auto` pipeline runs client side: rendering, the embedded MediaPipe detector, validation, anonymization and verification.

It makes possible to make website where the scan never leaves the page and user computer. Output is byte-identical to `radveil auto` with the same settings.

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129   # must match Cargo.lock
crates/radveil-wasm/build.sh                        # writes site/demo/pkg/
python3 -m http.server -d site                      # open /demo/
```

The Pages workflow runs the same build on deploy, so `site/demo/pkg/` is not
committed.

Notes:
- Single-threaded. rayon falls back to the calling thread on
  `wasm32-unknown-unknown`, and the pipeline still runs in about 0.6 s on a
  256x256x160 T1 (Apple M-series, Chrome).
- Built with `+simd128`, supported by every current browser.
- Size: about 17 MB raw and 6.6 MB gzipped. About 3.2 MB of that is the two
  detector models; most of the rest is tract.
- Only the `detector` feature of radveil-core is enabled. Brain extraction
  (SynthStrip weights) is left out to keep the download small.
