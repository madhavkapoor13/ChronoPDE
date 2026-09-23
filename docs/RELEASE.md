# Controlled study and portfolio release

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

The scientific stopping decision remains frozen at `v0.1.0-negative-result`.
The public-facing documentation and visualization pass is tagged separately as
`v0.1.1-portfolio`. Any future work on whitening, channel-balanced objectives,
residual vector fields, multiple seeds, sparse time, or OOD evaluation must
begin as a separately registered study.

## Additive errata and future-study boundary

ChronoPDE V2 Phase 1 preserves this release while adding a machine-readable
[claim ledger](../reports/chronopde_v2/phase1/claim_ledger.yaml). It records the
Week 5 four-trajectory gate as unverified pending its exact source artifact,
labels the unequal-history ID results exploratory, and prevents the exposed
legacy ID set from serving as a fresh V2 confirmatory test. These annotations do
not rewrite the original results or release tags.

Phase 2 is documented separately in
[`reports/chronopde_v2/phase2/README.md`](../reports/chronopde_v2/phase2/README.md).
It validates the numerical ground truth and a fresh data protocol but performs
no model training; therefore it does not revise this release's model conclusions.

The additive [Phase 3 data report](../reports/chronopde_v2/phase3/README.md)
records the new exact-RHS development dataset. Confirmatory trajectories remain
sealed, and the V1 model conclusions remain unchanged.

The [Phase 4 protocol](../reports/chronopde_v2/phase4/README.md) adds a bounded,
development-only feasibility trainer for the matched V2 models. Its seed-0 run
produced a split result—DCT passed and FFT failed—which triggers the separate
[Phase 4B integration audit](../reports/chronopde_v2/phase4b/README.md). Neither
development result revises the frozen V1 conclusions or authorizes a V2
superiority claim.

Phase 4B classified the repeated FFT divergence as learned-vector-field
instability rather than an RK4-resolution artifact and authorized the
[Phase 5 multi-seed development study](../reports/chronopde_v2/phase5/README.md).
Phase 5 uses fresh seeds and keeps all confirmatory states sealed. Even a Phase
5 pass only freezes a candidate for later confirmation; it does not revise the
public V1 scientific claim boundary.
