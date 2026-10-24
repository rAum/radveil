# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate THIRD_PARTY_LICENSES.txt: license texts of every Rust crate
compiled into RadVeil's shipped artifacts (Python extension, CLI, C
library), plus the embedded model weights.

    python tools/third_party_licenses.py            # rewrite the file
    python tools/third_party_licenses.py --check    # CI: fail if stale

Crates are collected from `cargo metadata` (normal + build-time-linked
dependencies of the workspace members, default features). License texts come
from the files each crate publishes (LICENSE*, LICENCE*, COPYING*,
NOTICE*); identical texts are printed once with the list of crates that
share them, so every copyright line is preserved without repeating the
Apache-2.0 text 150 times.
"""

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "THIRD_PARTY_LICENSES.txt")
LICENSE_PREFIXES = ("license", "licence", "copying", "notice", "unlicense")
# Accepted SPDX identifiers (any crate whose expression allows one of these
# is fine for commercial, closed or open redistribution).
PERMISSIVE = {
    "MIT", "Apache-2.0", "Apache-2.0 WITH LLVM-exception", "BSD-2-Clause",
    "BSD-3-Clause", "ISC", "Zlib", "Unlicense", "0BSD", "CC0-1.0",
    "Unicode-3.0", "Unicode-DFS-2016", "BSL-1.0", "BlueOak-1.0.0",
}

MODELS = """\
MediaPipe BlazeFace (short range) and Face Mesh v2 model weights
  embedded as crates/radveil-core/models/*.rvnn
  Copyright Google LLC. Licensed under the Apache License, Version 2.0
  (model cards: "LICENSED UNDER Apache License, Version 2.0").
  Modified by RadVeil: exported from TFLite to RadVeil's .rvnn format
  (tools/export_rvnn.py); float16 weights and computation are
  unchanged. Source: https://storage.googleapis.com/mediapipe-models/
  face_landmarker/face_landmarker/float16/latest/face_landmarker.task
  The Apache License 2.0 text is reproduced below (crates using it).
"""
RUST_STD = """\
Rust standard library
The binaries statically link the Rust standard library and runtime
(core, alloc, std and their dependencies), Copyright The Rust Project
Developers, licensed under MIT OR Apache-2.0. Both texts are reproduced
below; per-component notices: https://github.com/rust-lang/rust/blob/master/COPYRIGHT
"""
# SynthStrip ships in every default build (feature `synthstrip`); its MIT
# notice, provenance and modification note are reproduced verbatim.
SYNTHSTRIP = os.path.join(ROOT, "crates", "radveil-core", "models", "LICENSE-synthstrip")


def spdx_ids(expr):
    """Very small SPDX expression splitter: OR/AND/parentheses and the
    legacy '/' separator."""
    e = expr.replace("(", " ").replace(")", " ").replace("/", " OR ")
    ids, buf = [], []
    for tok in e.split():
        if tok in ("OR", "AND"):
            if buf:
                ids.append(" ".join(buf))
            buf = []
        else:
            buf.append(tok)
    if buf:
        ids.append(" ".join(buf))
    return ids


def acceptable(expr):
    ids = spdx_ids(expr)
    if " AND " in expr and "OR" not in expr:
        return all(i in PERMISSIVE for i in ids)
    return any(i in PERMISSIVE for i in ids)


def crates(all_features):
    cmd = ["cargo", "metadata", "--format-version", "1", "--locked"]
    if all_features:
        cmd.append("--all-features")
    m = json.loads(subprocess.check_output(cmd, cwd=ROOT))
    pkgs = {p["id"]: p for p in m["packages"]}
    nodes = {n["id"]: n for n in m["resolve"]["nodes"]}
    seen, stack = set(), list(m["workspace_members"])
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        for d in nodes[i]["deps"]:
            kinds = {k["kind"] for k in d["dep_kinds"]}
            if kinds & {None, "build"}:
                stack.append(d["pkg"])
    out = [pkgs[i] for i in seen if not pkgs[i]["name"].startswith("radveil")]
    return sorted(out, key=lambda p: (p["name"], p["version"]))


def license_texts(pkg):
    d = os.path.dirname(pkg["manifest_path"])
    files = []
    for f in sorted(os.listdir(d)):
        if f.lower().startswith(LICENSE_PREFIXES) and os.path.isfile(os.path.join(d, f)):
            files.append(f)
    if pkg.get("license_file"):
        lf = pkg["license_file"]
        if lf not in files and os.path.isfile(os.path.join(d, lf)):
            files.append(lf)
    return [(f, open(os.path.join(d, f), encoding="utf-8", errors="replace").read().strip())
            for f in files]


def render(all_features):
    pkgs = crates(all_features)
    bad = [(p["name"], p.get("license")) for p in pkgs if not acceptable(p.get("license") or "")]
    by_text = collections.OrderedDict()
    missing = []
    for p in pkgs:
        texts = license_texts(p)
        if not texts:
            missing.append(p)
        for fname, text in texts:
            h = hashlib.sha256(text.encode()).hexdigest()
            by_text.setdefault(h, (text, []))[1].append(f"{p['name']} {p['version']} ({fname})")
    lines = [
        "THIRD-PARTY SOFTWARE NOTICES AND LICENSES",
        "=========================================",
        "",
        "RadVeil is licensed under Apache-2.0 (see LICENSE and NOTICE). Its",
        "binaries include the third-party components listed below.",
        "Generated by tools/third_party_licenses.py -- do not edit by hand.",
        "",
        "Embedded model weights",
        "----------------------",
        MODELS,
        open(SYNTHSTRIP, encoding="utf-8").read().strip(),
        "",
        RUST_STD,
        f"Rust crates ({len(pkgs)})",
        "-" * len(f"Rust crates ({len(pkgs)})"),
    ]
    for p in pkgs:
        repo = p.get("repository") or ""
        lines.append(f"{p['name']} {p['version']}  [{p.get('license')}]  {repo}".rstrip())
    if missing:
        lines += ["", "Crates without a bundled license file (license per the SPDX",
                  "expression above; standard texts reproduced in this file):"]
        lines += [f"  {p['name']} {p['version']} [{p.get('license')}]" for p in missing]
    lines += ["", "License texts", "-------------"]
    for text, users in by_text.values():
        lines += ["", "=" * 78, "Used by: " + ", ".join(users), "=" * 78, "", text]
    return "\n".join(lines) + "\n", bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--all-features", action="store_true")
    a = ap.parse_args()
    text, bad = render(a.all_features)
    if bad:
        print("crates with licenses outside the allow-list:", bad, file=sys.stderr)
        sys.exit(1)
    if a.check:
        cur = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if cur != text:
            print("THIRD_PARTY_LICENSES.txt is stale; run tools/third_party_licenses.py",
                  file=sys.stderr)
            sys.exit(1)
        print("THIRD_PARTY_LICENSES.txt up to date")
        return
    open(OUT, "w", encoding="utf-8").write(text)
    print(f"wrote {OUT} ({len(text) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
