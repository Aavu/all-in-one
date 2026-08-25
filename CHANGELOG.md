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

### Fixed

- `analyze()` no longer hangs when called from a script without an
  `if __name__ == '__main__':` guard. `spectrogram.py` created a `Pool()`
  unconditionally; on macOS and Windows, where multiprocessing defaults to the
  `spawn` start method, every child re-imported the caller's `__main__`,
  re-entered `analyze()`, tried to spawn again and died with "An attempt has
  been made to start a new process before the current process has finished its
  bootstrapping phase" -- leaving the parent blocked on a pool of dead workers.
  A pool is now only created when there is more than one item to process, which
  also removes the pointless per-CPU worker spin-up for the single-file case.
  Same change in `visualize()` and `sonify()`. Batch callers still get the pool,
  and for them the standard Python rule applies: guard the entry point, or pass
  `multiprocess=False`.
- `visualize()` and `sonify()` worked against `demucs.separate.load_track`,
  which demucs removed in 4.1, so both raised `AttributeError` on any current
  demucs. They now use `demucs.audio.AudioFile`, the public API `load_track`
  wrapped.

### Known issues

- `visualize=True` with more than one track fails on macOS: matplotlib Figures
  returned from pool workers cannot be unpickled under the interactive MacOSX
  backend ("Cannot create a GUI FigureManager outside the main thread"). The
  plot files are still written correctly; only the returned Figure objects fail.
  Pass `multiprocess=False`, or select a non-interactive backend
  (`matplotlib.use('Agg')`) before calling. Pre-existing upstream behaviour.

### Removed

- The `natten; platform_system == 'Darwin'` dependency.

## [1.1.0] - 2023-10-10

### Added

- Training code and instructions.

[unreleased]: https://github.com/Aavu/all-in-one/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/Aavu/all-in-one/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.3...v1.1.0
