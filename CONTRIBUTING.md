# Contributing

Maintainer does not promise that PRs will be reviewed and merged, and all potential contributions should follow the below rules to be considered.

This is open source project, and everyone can make their own private version.

## Development setup

```bash
git lfs install                           # once per machine (scan data is in Git LFS)
git clone https://github.com/rAum/radveil && cd radveil
cargo test --workspace --release          # Rust tests (synthetic phantoms only)
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy nibabel pytest
maturin develop --release
pytest tests/
```

The face detector is built in (no extra setup). Optional: `pip install
dlib` + `python -c "import radveil; radveil.fetch_predictor()"` to test
the `detector="dlib"` path. `RADVEIL_SIMD=off` tests the portable
(non-intrinsics) kernels. Renderer changes: compare speed, render
parity and pipeline outcomes with `cargo run --release --example
render_eval` (see [docs/performance.md](docs/performance.md)). The
embedded models (`.rvnn`) are regenerated with
`tools/convert_mediapipe_models.py` (see `crates/radveil-core/models/README.md`).

## Ground rules

- `cargo fmt --all` and `cargo clippy --workspace --all-targets
  --all-features -- -D warnings` must pass; CI enforces both.
- Never commit imaging data or rendered faces.
  Tests use synthetic phantoms only.
- Changes to the anonymization geometry or verification logic need an
  accompanying test that would fail without them.
- Report security privately, through GitHub's "Report a
  vulnerability" form on the repository's Security tab, rather than in a
  public issue. Never attach identifiable images or scans to any report.

By contributing you agree that your contributions are licensed under
Apache-2.0 (section 5 of the license). You keep the copyright to your
contributions. New source files start with the same two SPDX lines as the
existing ones:

```
SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
SPDX-License-Identifier: Apache-2.0
```

## Dependencies and licenses

Adding or updating a Rust dependency changes what the binaries bundle:
run `python tools/third_party_licenses.py` and commit the regenerated
`THIRD_PARTY_LICENSES.txt` (CI checks it). Dependencies must be
permissively licensed (see the allow-list in that script); copyleft
licenses are not accepted.

