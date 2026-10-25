# Changelog

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