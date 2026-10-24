#!/bin/sh
# Build the in-browser demo bundle into site/demo/pkg.
# Needs: rustup target add wasm32-unknown-unknown
#        cargo install wasm-bindgen-cli --version <wasm-bindgen version in Cargo.lock>
set -eu
cd "$(dirname "$0")/../.."
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build -p radveil-wasm --target wasm32-unknown-unknown --release
wasm-bindgen --target web --no-typescript --remove-name-section --remove-producers-section \
  --out-dir site/demo/pkg target/wasm32-unknown-unknown/release/radveil_wasm.wasm
ls -l site/demo/pkg
