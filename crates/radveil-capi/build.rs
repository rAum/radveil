// SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
// SPDX-License-Identifier: Apache-2.0

//! Give the shared library a relocatable identity, so a program linked
//! against a copy of it does not record the path it happened to be built
//! at: `@rpath/libradveil.dylib` on macOS, soname `libradveil.so` on Linux.

fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libradveil.dylib")
        }
        "linux" | "android" | "freebsd" => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libradveil.so")
        }
        _ => {}
    }
}
