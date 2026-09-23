# ChronoPDE V2 confirmatory portfolio release (`v0.2.0`)

## Result

ChronoPDE V2 completed a one-shot comparison of five frozen FFT-DCT checkpoint
pairs on 256 sealed reaction-diffusion trajectories. The strict predeclared gate
passed:

- DCT had lower rollout relative L2 and exact-velocity nRMSE in `5/5` seeds.
- Median paired rollout improvement was `21.14%`; the fixed 20,000-resample
  hierarchical-bootstrap 95% interval was `16.35%-27.72%`.
- DCT had zero divergence across 1,280 seed-trajectory rollouts; FFT diverged
  on 255.
- DCT beat persistence and had no worse divergence in every seed.
- Five directional wins give the predeclared one-sided sign-test `p=0.03125`.
- Boundary-strip and first-interior normal-derivative errors were lower in all
  five seeds.

The result supports a narrower basis-aligned inductive-bias claim for this PDE,
`64 x 64` grid, central parameter range, and frozen protocol. It does not prove
boundary enforcement, physical wall-flux correctness, sparse-time performance,
OOD generalization, grid transfer, or superiority on other PDEs.

## Frozen inputs

- Development data: 512 training and 128 validation trajectories, exact
  discrete-RHS labels, training-only normalization.
- Confirmatory data: 256 identities frozen in Phase 2 and generated once after
  the ten Phase 5 checkpoints were frozen.
- Models: matched width-29, four-block, 1,973,657-parameter FFT `12 x 12` and
  DCT `24 x 24` vector fields with equal active real spectral DOF.
- Selection: fixed validation velocity nRMSE only.
- Evaluation: RK4 with eight steps per stored interval.

Important hashes:

- Phase 5 archive: `cf3eede191de4f9e1dd43b36f1d9f1000d314d3a0e01bc90f71a80d4ebee3c54`
- Phase 6 protocol: `7a869166876c6bd2a6143bd334a71128f692b7dc77cc9554d69e510b09c26464`
- Confirmatory HDF5: `0a432d24d34605274a4e4bfa06d97831602062b016b44f4d5b81f91cc14e64e3`
- Phase 6 results ZIP: `151e0fb628735b21f7d63a159abfa9695253f2a210147c29555ced06d587cfc1`

The full local/private inventory is recorded in the release artifact manifest.
Large datasets, checkpoints and archives remain outside Git.

## Reproduce and inspect

From a clean Python 3.11 environment:

```bash
python -m pip install -e ".[dev,demo,report]"
python scripts/chronopde_v2.py phase1 --format json
python scripts/chronopde_v2.py phase2 --format json
pytest -q
ruff check .
mypy chronopde
```

Use `python scripts/build_v2_confirmatory_report.py` to rebuild the report and
`streamlit run demo/app.py` to inspect the committed evidence offline. The full
private-artifact workflow is in [`REPRODUCTION.md`](REPRODUCTION.md).

## Historical V1 release

The original Week 1-6 study remains a valid controlled negative result. Its
spline-target models did not pass the unchanged velocity gate, and its unequal-
history ID comparison is exploratory. V2 began as a separate registered study
with exact-RHS targets, a capacity-matched comparison, fresh splits, five seeds,
and a sealed holdout.

The following remain unchanged:

- `v0.1.0-negative-result`
- `v0.1.1-portfolio`
- `reports/final/`
- `output/pdf/chronopde_negative_result_report.pdf`

## Release checklist

- [x] Every public number derives from committed Phase 6 evidence.
- [x] Architecture figure matches the frozen V2 models.
- [x] V1 history remains explicit and unmodified.
- [x] Large artifacts are excluded from Git and tracked by hashes.
- [x] Confirmatory report and offline explorer consume committed evidence only.
- [x] Phase 1, Phase 2, pytest, Ruff, mypy, and Git whitespace checks pass.
- [x] Resume remains a private one-page artifact outside the repository.
