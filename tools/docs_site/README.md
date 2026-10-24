# Documentation site generator

`site/docs.html` is generated: edit `template/*.html` (concatenated in name
order) and the example programs in `examples/`, then run

```bash
python tools/docs_site/gen.py
```

## Testing the examples

Every example on the page runs as written. Before publishing a change, run
them against a real head volume named `t1.nii.gz` (any public T1, e.g. an IXI
subject or 3D Slicer's "MRHead" sample):

```bash
cargo build --release -p radveil-cli -p radveil-capi
pip install maturin nibabel && maturin develop --release
export PATH="$PWD/target/release:$PATH"
cd <scratch dir with t1.nii.gz, a cohort/ folder and points.json>

python tools/docs_site/examples/py/quickstart.py          # ... each .py
bash -e tools/docs_site/examples/sh/quickstart.sh         # ... each .sh
cargo run --release --manifest-path tools/docs_site/examples/rs/Cargo.toml --bin quickstart
cc -I crates/radveil-capi/include tools/docs_site/examples/c/quickstart.c \
   -L target/release -lradveil -lm -o quickstart && ./quickstart
```

`points.json` (for `steps.sh`) and `points.txt` (for `landmarks.c`) hold 68
world-space landmarks; take them from `radveil.anonymization_mask(...)["landmarks_3d"]`.
