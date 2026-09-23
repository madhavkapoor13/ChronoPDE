# ChronoPDE V2 Phase 5 — multi-seed development study

Phase 5 is a fresh five-seed development comparison of the matched FFT and DCT
vector fields. It follows the Phase 4B finding that the DCT checkpoint was
stable under RK4 refinement while the seed-0 FFT checkpoint had persistent
learned-vector-field instability.

## Frozen protocol

- Seeds: `0, 1, 2, 3, 4` for both backbones.
- Budget: 10,000 optimizer steps per run.
- Objective: mean per-sample full-field relative exact-RHS error.
- Selection: minimum fixed-validation velocity nRMSE, with validation relative
  loss as the tie-breaker.
- Rollout: 16 fixed development-validation trajectories, RK4 with 8 steps per
  stored interval.
- Models: matched 1,973,657-parameter FFT 12x12 and DCT 24x24 fields.

The Phase 4 checkpoints are not reused. All ten runs start from their declared
seed. Training and model selection use development data only.

## Decision gate

Phase 6 confirmatory-data generation is authorized only when all ten runs
complete and all of the following hold:

1. DCT has zero rollout divergence for all five seeds.
2. Median DCT validation velocity nRMSE is at most 0.15.
3. DCT beats persistence on rollout error for every seed.
4. DCT beats FFT on rollout error and velocity nRMSE for at least four seeds.
5. DCT divergence is no worse than FFT for every seed.

Passing freezes the DCT candidate and the 8-step RK4 evaluation protocol for
Phase 6. It does **not** authorize a DCT-superiority claim; that requires the
sealed confirmatory evaluation.

## Kaggle

Use `notebooks/chronopde_v2_phase5_kaggle.ipynb` with the private Phase 3
development dataset and `GPU T4 x2`. Each seed's FFT and DCT runs execute in
parallel on separate GPUs. The notebook produces
`chronopde_v2_phase5_outputs.zip` on success and can create a resumable partial
ZIP after an interruption.
