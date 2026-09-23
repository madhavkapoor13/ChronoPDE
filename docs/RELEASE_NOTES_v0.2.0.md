# ChronoPDE V2 - sealed confirmatory release

`v0.2.0` completes the separately registered ChronoPDE V2 study. It preserves
the original V1 negative result while adding exact discrete-RHS supervision,
capacity-matched FFT/DCT operators, five training seeds, and a one-shot sealed
confirmatory evaluation.

## Headline result

On 256 sealed in-distribution reaction-diffusion trajectories, the matched DCT
operator had lower rollout relative L2 and exact-velocity nRMSE than FFT in all
five frozen seeds. Median paired rollout improvement was 21.14% with a fixed
hierarchical-bootstrap 95% interval of 16.35%-27.72%. DCT had zero divergence
across 1,280 rollouts; FFT diverged on 255.

The authorized claim is restricted to this PDE, `64 x 64` grid, central
parameter range, exact-RHS objective, frozen checkpoints, and RK4 evaluation
protocol. Sparse-time, OOD, grid-transfer and other-PDE claims remain untested.

## Release assets

- Evidence-first README and model card.
- Five-page confirmatory technical report.
- Offline Streamlit evidence explorer with separate V2 and historical V1 views.
- V2 method and two-tier reproduction guides.
- Lightweight artifact inventory with frozen SHA-256 hashes.
- Corrected matched-architecture and five-seed result figures.

## Frozen hashes

- Phase 5 checkpoint archive:
  `cf3eede191de4f9e1dd43b36f1d9f1000d314d3a0e01bc90f71a80d4ebee3c54`
- Phase 6 protocol:
  `7a869166876c6bd2a6143bd334a71128f692b7dc77cc9554d69e510b09c26464`
- Confirmatory HDF5:
  `0a432d24d34605274a4e4bfa06d97831602062b016b44f4d5b81f91cc14e64e3`
- Phase 6 results ZIP:
  `151e0fb628735b21f7d63a159abfa9695253f2a210147c29555ced06d587cfc1`

The large artifacts remain in private Kaggle datasets and are not part of the
source release.

## Historical continuity

Tags `v0.1.0-negative-result` and `v0.1.1-portfolio` remain unchanged. The V1
report, evidence, and stopped Week 6 gate are retained as historical records.
