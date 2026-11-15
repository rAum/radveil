# Changelog

## Unreleased

### Added
- `deep` (CLI `--deep`, C `RADVEIL_DEEP`), opt-in: a depth-adaptive face
  region. It also rewrites the tissue below the skin, down to
  `deep_depth_mm` (0.40 x inter-eye distance), and stops `deep_guard_mm`
  (5 mm) in front of the extracted brain. After the skin-deep default,
  renders at 2x and 4x the skin threshold still matched the subject; with
  `deep` they no longer do on 120 IXI subjects. Implies a brain extraction.
- `reshape` (CLI `--reshape`, C `RADVEIL_RESHAPE`), experimental, with
  `deep`: moves the face tissue onto a generic face surface before the mode
  rewrites it.
- Reports give the brain margin used (`brain_guard_mm`) and whether the
  region was `deep`.

## 0.1.1 - 2026-10-25

Patch release: dependency updates and release-build fixes. No API or
output changes.

### Added
- Zenodo DOI for citing the software.

### Changed
- Dependencies updated: png 0.18, pyo3 0.29.3, serde_json, thiserror, clap.
- Release builds (wheels, CLI, C library) use fat LTO and stripped symbols.
- `cargo install radveil-cli` now builds with the same optimized profile
  (LTO, one codegen unit) as the prebuilt binaries.

### Fixed
- Release pipeline: Linux Python 3.9 tests run on Ubuntu 24.04, and wheel
  tests install their dependencies before the wheel.

## 0.1.0 - 2026-10-24

### Added
- Initial release of the algorithm.
- Initial release of all language wrappers.

### Changed
- None.

### Fixed
- None.