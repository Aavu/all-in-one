# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-08-25

Fork maintenance release. No change to model weights, architecture, or outputs --
verified identical to 1.1.0 on the pretrained checkpoint, down to beat times,
downbeat times, BPM and segment boundaries.

### Changed

- Neighborhood attention no longer requires natten. `allin1/models/dinat.py` now
  imports from the new `allin1/models/neighborhood_attention.py`, which uses
  natten's unfused ops when an installed natten still has them (0.16-0.17.x) and
  otherwise falls back to an equivalent pure-PyTorch implementation. natten 0.20
  deleted the unfused RPB ops this model needs, and none of its fused backends
  accept an additive bias, so no natten >= 0.20 can serve them. Select explicitly
  with `ALLIN1_NA_BACKEND=auto|natten|torch`.
- Lifted the effective `torch <= 2.6` ceiling that natten 0.17.5 imposed. Tested
  on torch 2.13.0. torch is still not declared as a dependency (demucs requires
  `torch>=2.1`, and installing your preferred build first keeps that choice
  yours), exactly as upstream.
- Blackwell / `sm_120` (RTX 5090) works with a stock cu128+ torch wheel. No CUDA
  toolkit, no natten source build, no patched `setup.py`.
- `requires-python` raised to `>=3.9`; added 3.12 and 3.13 classifiers.

### Added

- `natten` extra for anyone who wants natten's own kernels: `pip install
  "allin1[natten] @ git+https://github.com/Aavu/all-in-one"`.
- `tests/test_neighborhood_attention.py`, which pins the torch implementation
  against a naive transcription of natten 0.17.5's reference CPU kernels (no
  natten install needed) and against natten itself where available.

### Removed

- The `natten; platform_system == 'Darwin'` dependency.

## [1.1.0] - 2023-10-10

### Added

- Training code and instructions.

[unreleased]: https://github.com/Aavu/all-in-one/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/Aavu/all-in-one/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.3...v1.1.0
