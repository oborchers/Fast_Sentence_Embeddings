# Changelog

## 0.2.0 (unreleased)

### Changed

- Require fse>=2.1.0,<3 and NumPy directly. This version is prepared, not released.
- Validate configuration within the package instead of importing private fse code.
- Raise TypeError for invalid types and ValueError for invalid values. Invalid workers always raise ValueError, matching fse.

### Added

- Export only the adapter classes through `__all__` and include `py.typed`.
- Accept text PathLike model references and NumPy integer worker counts.

### Fixed

- Validate inherited LlamaIndex options before framework conversion or model loading. Reject invalid callback managers, caches and rate limiters with TypeError. Reject coerced batch sizes and async worker counts, and reject out-of-range values with ValueError.

- Reject invalid workers during construction, including before an empty batch.
- Keep path objects local and serialize local model paths as absolute paths.

## 0.1.0 (2026-11-02)

- Publish the standalone fse integration.
- Keep the package in the fse repository with isolated CI and a separately approved release.
- Require fse 2.0.0 or newer for publication.
- Add a real POTION retrieval example for end-to-end validation.
