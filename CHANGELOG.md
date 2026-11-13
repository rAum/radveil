# Changelog

## 0.1.2 - 2026-11-13

Patch release: the face frame no longer trusts a mouth that lifted
implausibly, which fixes partially-kept noses and spurious refusals on
T2/PD scans. No API changes; outputs change only for scans whose mouth
landmarks had mis-lifted.

### Fixed
- Face frame: on T2/PD scans the lips often sit below the render surface
  threshold, so the lifted mouth landed far behind the face and pitched
  the frame — leaving the nose only shaved skin-deep instead of cut to
  the surrounding skin level, or failing the seen-through-air check and
  refusing the volume. The frame now derives a reference vertical from
  the nose axis (calibrated on 54 IXI runs), keeps the mouth only when
  it agrees with that axis, and otherwise re-places it on the axis at
  the calibrated distance (reported as `mouth_reconstructed`), widening
  the protrusion zone to absorb the remaining uncertainty.

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