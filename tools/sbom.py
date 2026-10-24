# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate CycloneDX 1.6 SBOMs for RadVeil's shipped artifacts.

    python tools/sbom.py                        # all artifacts -> sbom/
    python tools/sbom.py --artifact cli         # one artifact
    python tools/sbom.py --all-features         # every optional feature (same set today)
    python tools/sbom.py --target x86_64-unknown-linux-gnu   # one platform only
    python tools/sbom.py --validate             # also schema-check the output
                                                # (pip install "cyclonedx-python-lib[json-validation]")

One SBOM per artifact, because each links a different dependency set:

    python  radveil-<ver>.cdx.json       Python wheel (radveil-py extension + python/radveil)
    cli     radveil-cli-<ver>.cdx.json   `radveil` command-line binary
    capi    radveil-capi-<ver>.cdx.json  libradveil C library

Each lists every Rust crate compiled into the artifact (normal and
build-time-linked dependencies, as tools/third_party_licenses.py does) with
its version, purl, SPDX license expression and the crates.io SHA-256 from
Cargo.lock; the Rust standard library it statically links; the embedded
MediaPipe model weights (SHA-256 of the exact bytes compiled in, plus their
upstream provenance); runtime-external Python dependencies; and the full
dependency graph. Without --target the crate set is the union over all
platforms (e.g. both windows-sys and libc appear).

Output is reproducible: the timestamp is SOURCE_DATE_EPOCH or the HEAD
commit time, and the serial number is derived from artifact, version,
commit and options.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tomllib
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = "https://github.com/rAum/radveil"
SPEC = "1.6"

ARTIFACTS = {
    # key: (root crate, component type, name, purl type, description)
    "python": ("radveil-py", "library", "radveil", "pypi",
               "RadVeil Python package (wheel): radveil._radveil extension module "
               "(Rust) plus the pure-Python radveil package"),
    "cli": ("radveil-cli", "application", "radveil-cli", "cargo",
            "RadVeil command-line tool (`radveil` binary)"),
    "capi": ("radveil-capi", "library", "radveil-capi", "cargo",
             "RadVeil C library (libradveil) and header radveil.h"),
}

GOOGLE = {"name": "Google LLC", "url": ["https://ai.google.dev/edge/mediapipe"]}
TASK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task")
TASK_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODELS = [
    # (file, name, upstream tflite, task, input, output)
    ("blazeface_short_range.rvnn", "MediaPipe BlazeFace (short range)",
     "face_detector.tflite (sha256 b4578f35...0152f)", "object-detection",
     "1x128x128x3 RGB in [-1, 1]",
     "896 SSD anchors x (box + 6 keypoints), scores"),
    ("face_mesh_v2.rvnn", "MediaPipe Face Mesh v2",
     "face_landmarks_detector.tflite (sha256 c7d54204...d43d5)", "keypoint-detection",
     "1x256x256x3 RGB in [0, 1]",
     "478 x (x, y, z) landmarks in crop pixels, face-presence logit"),
]


def run(cmd):
    return subprocess.check_output(cmd, cwd=ROOT, text=True)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def spdx_expr(lic):
    """Normalize Cargo's legacy '/' separator to SPDX 'OR'."""
    if not lic:
        return None
    return " OR ".join(p.strip() for p in lic.split("/")) if "/" in lic else lic


def load_metadata(all_features, target):
    cmd = ["cargo", "metadata", "--format-version", "1", "--locked"]
    if all_features:
        cmd.append("--all-features")
    if target:
        cmd += ["--filter-platform", target]
    return json.loads(run(cmd))


def lock_checksums():
    with open(os.path.join(ROOT, "Cargo.lock"), "rb") as f:
        lock = tomllib.load(f)
    return {(p["name"], p["version"]): p["checksum"]
            for p in lock.get("package", []) if "checksum" in p}


def linked_graph(meta, root_name):
    """Crates reachable from root_name through normal/build edges, and those
    edges (dev-dependencies are never linked into the artifact)."""
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = next(i for i in meta["workspace_members"] if pkgs[i]["name"] == root_name)
    edges, stack = {}, [root]
    while stack:
        i = stack.pop()
        if i in edges:
            continue
        deps = []
        for d in nodes[i]["deps"]:
            kinds = {k["kind"] for k in d["dep_kinds"]}
            if kinds & {None, "build"}:
                deps.append(d["pkg"])
        edges[i] = sorted(set(deps))
        stack.extend(deps)
    return root, pkgs, edges


def crate_ref(p):
    return f"pkg:cargo/{p['name']}@{p['version']}"


def crate_component(p, checksums):
    c = {
        "type": "library",
        "bom-ref": crate_ref(p),
        "name": p["name"],
        "version": p["version"],
        "purl": crate_ref(p),
        "scope": "required",
    }
    if p.get("description"):
        c["description"] = " ".join(p["description"].split())
    if p.get("authors"):
        c["authors"] = [{"name": a} for a in p["authors"]]
    expr = spdx_expr(p.get("license"))
    if expr:
        c["licenses"] = [{"expression": expr}]
    sha = checksums.get((p["name"], p["version"]))
    if sha:
        c["hashes"] = [{"alg": "SHA-256", "content": sha}]
    refs = []
    if p.get("repository"):
        refs.append({"type": "vcs", "url": p["repository"]})
    if p.get("homepage"):
        refs.append({"type": "website", "url": p["homepage"]})
    if p["source"] and p["source"].startswith("registry+"):
        refs.append({"type": "distribution",
                     "url": f"https://crates.io/crates/{p['name']}/{p['version']}"})
    if refs:
        c["externalReferences"] = refs
    return c


SYNTHSTRIP = {
    "file": "synthstrip.1.rvss",
    "source": "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt",
    "source_sha256": "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33",
}


def synthstrip_component():
    path = os.path.join(ROOT, "crates", "radveil-core", "models", SYNTHSTRIP["file"])
    authors = "A. Hoopes, J. S. Mora, A. V. Dalca, B. Fischl, M. Hoffmann"
    return {
        "type": "machine-learning-model",
        "bom-ref": f"model:{SYNTHSTRIP['file']}",
        "supplier": {"name": "The SynthStrip authors", "url": ["https://synthstrip.io"]},
        "authors": [{"name": a.strip()} for a in authors.split(",")],
        "name": "SynthStrip",
        "version": f"synthstrip.1, sha256:{SYNTHSTRIP['source_sha256'][:12]}",
        "description": "Brain-extraction U-Net weights embedded in radveil-core "
                       f"({SYNTHSTRIP['file']}, compiled in with include_bytes!).",
        "scope": "required",
        "hashes": [{"alg": "SHA-256", "content": sha256_file(path)}],
        "licenses": [{"license": {"id": "MIT"}}],
        "copyright": f"Copyright (c) 2022 The SynthStrip authors ({authors})",
        "pedigree": {
            "ancestors": [{
                "type": "machine-learning-model",
                "name": "SynthStrip PyTorch checkpoint (synthstrip.1.pt)",
                "version": "1",
                "hashes": [{"alg": "SHA-256", "content": SYNTHSTRIP["source_sha256"]}],
                "licenses": [{"license": {"id": "MIT"}}],
                "externalReferences": [{"type": "distribution", "url": SYNTHSTRIP["source"]}],
            }],
            "notes": "Converted by tools/export_synthstrip.py to the .rvss layout with "
                     "weights rounded to float16 (biases float32); the network is "
                     "re-implemented in radveil-core src/synthstrip.rs. Offered by the "
                     "authors under MIT or CC BY 4.0; used under MIT. See "
                     "crates/radveil-core/models/LICENSE-synthstrip.",
        },
        "externalReferences": [
            {"type": "documentation", "url": "https://doi.org/10.1016/j.neuroimage.2022.119474"},
        ],
        "modelCard": {
            "modelParameters": {
                "task": "image-segmentation",
                "inputs": [{"format": "conformed 3D MRI/CT volume"}],
                "outputs": [{"format": "signed distance to the brain boundary"}],
            },
        },
    }


def model_components():
    out = []
    for fname, name, tflite, task, inp, outp in MODELS:
        path = os.path.join(ROOT, "crates", "radveil-core", "models", fname)
        out.append({
            "type": "machine-learning-model",
            "bom-ref": f"model:{fname}",
            "supplier": GOOGLE,
            "name": name,
            "version": f"face_landmarker float16, sha256:{TASK_SHA256[:12]}",
            "description": f"Model weights embedded in radveil-core ({fname}, "
                           "compiled in with include_bytes!).",
            "scope": "required",
            "hashes": [{"alg": "SHA-256", "content": sha256_file(path)}],
            "licenses": [{"license": {"id": "Apache-2.0"}}],
            "copyright": "Copyright Google LLC",
            "pedigree": {
                "ancestors": [{
                    "type": "machine-learning-model",
                    "supplier": GOOGLE,
                    "name": "MediaPipe Face Landmarker bundle (face_landmarker.task)",
                    "version": "float16/latest",
                    "hashes": [{"alg": "SHA-256", "content": TASK_SHA256}],
                    "licenses": [{"license": {"id": "Apache-2.0"}}],
                    "externalReferences": [{"type": "distribution", "url": TASK_URL}],
                }],
                "notes": f"Extracted {tflite} from the bundle and exported by "
                         "tools/export_rvnn.py to RadVeil's .rvnn op list (padding "
                         "resolved, ReLU/PReLU fused, float16 weights unchanged), "
                         "verified against TensorFlow Lite. Runs on RadVeil's own "
                         "engine (radveil-core src/nn.rs). See "
                         "crates/radveil-core/models/README.md.",
            },
            "externalReferences": [
                {"type": "documentation",
                 "url": "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"},
                {"type": "documentation",
                 "url": f"{REPO}/blob/main/crates/radveil-core/models/README.md"},
            ],
            "modelCard": {
                "modelParameters": {
                    "task": task,
                    "inputs": [{"format": inp}],
                    "outputs": [{"format": outp}],
                },
                "considerations": {
                    "useCases": ["Locating faces in rendered head scans so RadVeil can "
                                 "remove them; faces are never stored, compared or "
                                 "identified."],
                    "technicalLimitations": [
                        "Upstream model cards list human-life-critical decisions and "
                        "facial recognition/identification as out of scope."],
                },
            },
        })
    return out


def rust_std_component():
    v = run(["rustc", "--version"]).split()[1]
    return {
        "type": "library",
        "bom-ref": f"pkg:generic/rust-std@{v}",
        "supplier": {"name": "The Rust Project", "url": ["https://www.rust-lang.org"]},
        "name": "rust-std",
        "version": v,
        "description": "Rust standard library (core, alloc, std), statically linked by rustc",
        "scope": "required",
        "purl": f"pkg:generic/rust-std@{v}",
        "licenses": [{"expression": "MIT OR Apache-2.0"}],
        "externalReferences": [{"type": "vcs", "url": "https://github.com/rust-lang/rust"}],
    }


def python_components():
    """Runtime dependencies of the wheel that are installed separately, not
    bundled (declared in pyproject.toml)."""
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        proj = tomllib.load(f)["project"]
    out = []
    reqs = [(r, "required") for r in proj.get("dependencies", [])]
    for extra, rs in proj.get("optional-dependencies", {}).items():
        reqs += [(r, "optional") for r in rs]
    for req, scope in reqs:
        name = req.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
        spec = req[len(name):].strip()
        out.append({
            "type": "library",
            "bom-ref": f"pkg:pypi/{name}",
            "name": name,
            "scope": scope,
            "description": "Runtime dependency declared in pyproject.toml, installed "
                           "separately by pip (not bundled in the wheel).",
            # CycloneDX 1.7 has isExternal/versionRange for this; 1.6 does not.
            "properties": [{"name": "radveil:bundled", "value": "false"},
                           {"name": "radveil:version-constraint", "value": spec or "*"}],
            "externalReferences": [{"type": "distribution",
                                    "url": f"https://pypi.org/project/{name}/"}],
        })
    return out


def build(artifact, meta, checksums, commit, epoch, all_features, target):
    root_crate, ctype, name, ptype, desc = ARTIFACTS[artifact]
    root_id, pkgs, edges = linked_graph(meta, root_crate)
    version = pkgs[root_id]["version"]
    root_ref = f"pkg:{ptype}/{name}@{version}"

    components, deps = [], []
    ref = {i: crate_ref(pkgs[i]) for i in edges}
    ref[root_id] = root_ref if artifact != "python" else crate_ref(pkgs[root_id])
    for i in sorted(edges, key=lambda i: (pkgs[i]["name"], pkgs[i]["version"])):
        p = pkgs[i]
        if i == root_id and artifact != "python":
            continue
        c = crate_component(p, checksums)
        if p["source"] is None:  # workspace crate: part of this artifact, built from source
            c.pop("scope")
            c["supplier"] = {"name": "RadVeil authors", "url": [REPO]}
            c["externalReferences"] = [{"type": "vcs", "url": REPO}]
        components.append(c)
        deps.append({"ref": ref[i], "dependsOn": [ref[d] for d in edges[i]]})

    std = rust_std_component()
    components.append(std)
    deps.append({"ref": std["bom-ref"], "dependsOn": []})

    core = next((i for i in edges if pkgs[i]["name"] == "radveil-core"), None)
    core_features = set()
    if core:
        node = next(n for n in meta["resolve"]["nodes"] if n["id"] == core)
        core_features = set(node.get("features", []))
    models = []
    if "detector" in core_features:
        models += model_components()
    if "synthstrip" in core_features:
        models.append(synthstrip_component())
    if models:
        components += models
        for d in deps:
            if d["ref"] == ref[core]:
                d["dependsOn"] += [m["bom-ref"] for m in models]
        deps += [{"ref": m["bom-ref"], "dependsOn": []} for m in models]

    root_deps = [ref[d] for d in edges[root_id]] + [std["bom-ref"]]
    if artifact == "python":
        pycomps = python_components()
        components += pycomps
        deps += [{"ref": c["bom-ref"], "dependsOn": []} for c in pycomps]
        root_deps = [ref[root_id], std["bom-ref"]] + [
            c["bom-ref"] for c in pycomps if c["scope"] == "required"]
    deps.insert(0, {"ref": root_ref, "dependsOn": sorted(root_deps)})

    props = [{"name": "radveil:git-commit", "value": commit},
             {"name": "radveil:cargo-features",
              "value": "all" if all_features else "default"},
             {"name": "radveil:target", "value": target or "all platforms (union)"}]
    ts = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    serial = uuid.uuid5(uuid.NAMESPACE_URL,
                        f"{REPO}#{artifact}@{version}@{commit}@{all_features}@{target}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC,
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": ts,
            "lifecycles": [{"phase": "build"}],
            "tools": {"components": [{
                "type": "application", "name": "tools/sbom.py",
                "supplier": {"name": "RadVeil authors", "url": [REPO]},
                "externalReferences": [{"type": "vcs", "url": REPO}]}]},
            "authors": [{"name": "RadVeil authors"}],
            "supplier": {"name": "RadVeil authors", "url": [REPO]},
            "component": {
                "type": ctype,
                "bom-ref": root_ref,
                "supplier": {"name": "RadVeil authors", "url": [REPO]},
                "name": name,
                "version": version,
                "description": desc,
                "licenses": [{"license": {"id": "Apache-2.0"}}],
                "purl": root_ref,
                "externalReferences": [
                    {"type": "vcs", "url": REPO},
                    {"type": "issue-tracker", "url": f"{REPO}/issues"},
                    {"type": "security-contact", "url": f"{REPO}/security"},
                ],
            },
            "properties": props,
        },
        "components": components,
        "dependencies": deps,
    }


def validate(path):
    try:
        from cyclonedx.schema import SchemaVersion
        from cyclonedx.validation.json import JsonStrictValidator
    except ImportError:
        sys.exit('--validate needs: pip install "cyclonedx-python-lib[json-validation]"')
    with open(path, encoding="utf-8") as f:
        err = JsonStrictValidator(SchemaVersion.V1_6).validate_str(f.read())
    if err:
        sys.exit(f"{path}: invalid CycloneDX {SPEC}: {str(err).splitlines()[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", choices=sorted(ARTIFACTS), action="append",
                    help="artifact(s) to describe (default: all)")
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--target", help="Rust target triple (default: union of all platforms)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "sbom"))
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    meta = load_metadata(a.all_features, a.target)
    checksums = lock_checksums()
    commit = run(["git", "rev-parse", "HEAD"]).strip()
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH")
                or run(["git", "log", "-1", "--format=%ct"]).strip())
    os.makedirs(a.out_dir, exist_ok=True)
    for art in a.artifact or sorted(ARTIFACTS):
        bom = build(art, meta, checksums, commit, epoch, a.all_features, a.target)
        name, version = bom["metadata"]["component"]["name"], bom["metadata"]["component"]["version"]
        path = os.path.join(a.out_dir, f"{name}-{version}.cdx.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bom, f, indent=2, ensure_ascii=False)
            f.write("\n")
        if a.validate:
            validate(path)
        print(f"wrote {path} ({len(bom['components'])} components"
              f"{', valid' if a.validate else ''})")


if __name__ == "__main__":
    main()
