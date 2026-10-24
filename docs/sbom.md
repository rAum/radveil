# SBOM & vulnerability handling

*Engineering documentation for integrators, not regulatory or legal advice.*

RadVeil publishes a machine-readable **Software Bill of Materials** for
every release, so manufacturers and hospitals that build RadVeil into
their own systems can list it as a third-party component (SOUP / OTS
software in IEC 62304 terms) and track its vulnerabilities. This covers
the SBOM expectations of FDA premarket cybersecurity submissions (FD&C Act
§524B) and the component inventory of IEC 81001-5-1 / EU MDR Annex I §17.

RadVeil itself is **not a medical device** ([legal](legal.md#not-a-medical-device)).
An SBOM is one input to *your* risk management file (ISO 14971, IEC 62304
SOUP analysis); it does not replace it.

**This project is also best effort, it is not actively maintaned or funded.**

## Where to get it

Each GitHub release has one [CycloneDX 1.6](https://cyclonedx.org/) JSON
file per artifact, because each artifact links a different set of crates:

| File | Artifact |
|---|---|
| `radveil-<ver>.cdx.json` | Python wheel (`pip install radveil`) |
| `radveil-cli-<ver>.cdx.json` | `radveil` command-line binary |
| `radveil-capi-<ver>.cdx.json` | `libradveil` C library + `radveil.h` |

To generate them for any checkout (default features, same output as the
release):

```bash
python tools/sbom.py
```

Options: `--all-features` includes every optional cargo feature (today
the same crate set as the default build);
`--target <triple>` limits the crate set to one platform (the default is
the union over all platforms); `--validate` checks the output against the
CycloneDX schema (`pip install "cyclonedx-python-lib[json-validation]"`).
Output is reproducible: the timestamp is the commit time (or
`SOURCE_DATE_EPOCH`), and the serial number is derived from the artifact,
version, commit and options.

## What it contains

| Element | Source |
|---|---|
| Every Rust crate compiled or build-time linked into the artifact: name, version, purl, SPDX license expression, authors, repository | `cargo metadata --locked` (the same crate set as `THIRD_PARTY_LICENSES.txt`) |
| SHA-256 of each crate as downloaded from crates.io | `Cargo.lock` |
| Rust standard library (statically linked) with the compiler version | `rustc --version` |
| **Embedded model weights** (MediaPipe BlazeFace short range, Face Mesh v2) as `machine-learning-model` components: supplier Google LLC, Apache-2.0, SHA-256 of the exact bytes compiled in, upstream bundle URL and hash, the TFLite → .rvnn export, inputs/outputs and out-of-scope uses; the SynthStrip brain-extraction weights (MIT) with their PyTorch source hash | `crates/radveil-core/models/` |
| Python runtime dependencies installed separately by pip (`numpy`; `dlib` optional), marked `radveil:bundled=false` | `pyproject.toml` |
| Complete dependency graph (`dependencies`) | resolver output |
| Build metadata: git commit, cargo features, target | properties on `metadata` |

This covers the NTIA minimum elements (supplier, component name, version,
unique identifier, dependency relationship, author of SBOM data,
timestamp).

Not included: the C runtime and system libraries of the host (`libc`,
`libm`, the Python interpreter), which RadVeil links dynamically and does
not ship.

### Support level and end of support

FDA asks for each component's support level and end-of-support date. For
RadVeil's dependencies:

- **Rust crates**: community-maintained open source. RadVeil tracks the
  RustSec advisory database (below) and updates or replaces crates that
  are vulnerable or flagged unmaintained. No component has a vendor
  end-of-support date.
- **Model weights**: static files. They cannot receive security patches.
  Their risk is behavioural (missed faces), and the pipeline's validation
  and verification steps control it
  ([README](../README.md)).
- **RadVeil**: the latest release receives fixes. There is no LTS
  branch yet.

## Vulnerability handling

- **Monitoring**: `cargo audit` checks `Cargo.lock` against the
  [RustSec advisory database](https://rustsec.org/) on every dependency
  change and **weekly** on `main` (`.github/workflows/audit.yml`). A new
  advisory against a shipped version therefore shows up even if the code
  has not changed.
- **Triage**: an advisory is fixed by upgrading. When RadVeil cannot
  upgrade yet and is demonstrably not affected (for example, the
  vulnerable function is never called), the advisory is recorded in
  `.cargo/audit.toml` with a written justification. That file is the
  project's record of "not affected" decisions (the information a VEX
  statement carries).
- **Reporting**: see [SECURITY.md](../SECURITY.md).

Current exceptions: none. (RUSTSEC-2026-0176 and RUSTSEC-2026-0177 in
`pyo3` 0.25 were fixed by the upgrade to `pyo3`/`numpy` 0.29 before 0.1.0.)

## Checking an SBOM against current advisories

The SBOMs use standard purls, so existing tools can scan them, for
example:

```bash
osv-scanner scan --sbom radveil-0.1.0.cdx.json
```
