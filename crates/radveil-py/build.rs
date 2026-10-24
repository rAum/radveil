// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Let the extension module link on macOS outside maturin (e.g.
//! `cargo build --all-features`): libpython symbols resolve at import time.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
