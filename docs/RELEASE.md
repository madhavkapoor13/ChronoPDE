# Negative-result recovery release

## Scope

This release closes the original ChronoPDE study without weakening its Week 6
gate. The target audit passed, loss–metric alignment substantially improved both
continuous-time backbones, and the boundary-aware DCT model had lower error on
all 16 matched diagnostic samples. Neither model reached the registered `0.01`
velocity-nRMSE threshold, so production retraining, sparse-time experiments,
and OOD evaluation remain out of scope.

## Reproduce the result

From a fresh Python 3.11 environment:

```bash
python -m pip install -e ".[dev]"
python scripts/analyze_week6_failure.py
pytest -q
ruff check .
mypy chronopde
```

The analysis is CPU-only and works without the dataset, checkpoints, network,
or Kaggle. Original archive checksums and source commits are recorded in
`reports/diagnostics/week6/evidence_manifest.json`; pass `--archive-root` to
verify local copies byte-for-byte.

## Release checklist

- [x] Seven source archives identified by SHA-256 and Git commit.
- [x] Only lightweight JSON, CSV, configuration, and plot evidence committed.
- [x] Deterministic paired analysis with fixed bootstrap seed and sample IDs.
- [x] Historical and conventional even-sample median definitions disclosed.
- [x] Negative-result report rendered and visually inspected on all seven pages.
- [x] Offline Streamlit explorer reads only committed final evidence.
- [x] Model card states intended use, evidence status, and unsupported claims.
- [x] Full test, Ruff, and mypy suites pass in a clean Python 3.11 environment.

## Resume-ready project description

> Built a reproducible PyTorch research pipeline for parameterized 2D
> reaction-diffusion dynamics, including deterministic data generation,
> continuous-time FFT/DCT neural operators, spline-derived velocity targets,
> RK4 rollout, checkpoint-safe GPU diagnostics, and frozen evidence analysis.

> Diagnosed a predeclared negative result instead of weakening its gate:
> corrected an unrepresentative batch, aligned the loss with per-sample relative
> error, improved DCT velocity nRMSE by 94%, and showed DCT lower error on all 16
> matched samples while transparently reporting that neither model passed.

## Release boundary

Tag this state as `v0.1.0-negative-result`. Any future work on whitening,
channel-balanced objectives, residual vector fields, multiple seeds, sparse
time, or OOD evaluation must begin as a separately registered study.
