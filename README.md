# ChronoPDE

ChronoPDE investigates whether a cosine spectral neural operator matched to
homogeneous Neumann boundary conditions improves continuous-time PDE modeling
over a parameter-matched FFT control. The project covers deterministic PDE data
generation, learned velocity fields, RK4 rollout, and reproducible failure
analysis in PyTorch.

> **Headline result:** in a one-shot evaluation of five frozen checkpoint pairs
> on 256 sealed trajectories, the matched DCT operator had lower rollout and
> exact-velocity error in **5/5 seeds**, with **21.14% median paired rollout
> improvement** (hierarchical-bootstrap 95% interval **16.35%–27.72%**) and
> zero divergence across all **1,280** DCT rollouts.

## Motivation

Fourier neural operators are naturally periodic, while this reaction-diffusion
system uses homogeneous Neumann, or no-flux, boundaries. ChronoPDE tests a
specific architectural hypothesis: does replacing the Fourier basis with a
real cosine basis improve a continuous-time neural operator when everything
else is held as constant as possible?

The original study used predefined gates and stopped when its Week 6 diagnostic
failed. ChronoPDE V2 was registered as a separate study: it replaced spline
targets with exact discrete PDE velocities, matched spectral degrees of
freedom and parameter counts, used five fresh training seeds, and reserved 256
new trajectories for one-shot confirmation. Sparse-time and OOD claims remain
outside the completed confirmatory scope.

## Method and architecture

Both continuous-time models receive a normalized two-channel state, physical
time, and three PDE parameters. A FiLM network conditions four spectral blocks.
The model predicts the instantaneous velocity, and fixed-step RK4 integrates it
to future states. The controlled difference is the spatial spectral basis.

![ChronoPDE architecture](figures/architecture_overview.svg)

| Component | FFT control | ChronoPDE DCT |
| --- | --- | --- |
| Spectral basis | Complex Fourier modes | Real orthonormal cosine modes |
| Boundary assumption | Periodic | Homogeneous Neumann / no-flux |
| Retained modes | 12 x 12 | 24 x 24 |
| Active real spectral DOF/block | 484,416 | 484,416 |
| Parameters | 1,973,657 | 1,973,657 |
| Conditioning | Time and `[Du, Dv, k]` through FiLM | Same |
| Rollout | Fixed-step RK4 | Same |

The repository also includes residual U-Net and autoregressive FFT-FNO
baselines, quintic spline velocity targets, and shared evaluation metrics.

## V2 dataset

- 512 training, 128 validation and 256 sealed confirmatory trajectories of a
  two-species parameterized reaction-diffusion system.
- State layout: `[trajectory, time, channel, 64, 64]`.
- 101 stored states over physical time `[0, 50]`.
- Exact discrete simulator RHS supervision in physical-time units.
- Normalization fitted only on the 512 training trajectories.
- Confirmatory identities were frozen before generation and evaluated once.

The raw HDF5 dataset is intentionally excluded from Git. Its frozen SHA-256 is
recorded in the evidence and figure provenance.

## Sealed confirmatory result

All ten frozen Phase 5 checkpoints completed evaluation on every confirmatory
trajectory. DCT beat FFT on rollout relative L2 and exact-velocity nRMSE in all
five seeds, beat persistence in all five seeds, and had zero divergence. FFT
diverged on 255 of 1,280 seed–trajectory rollouts; finite divergent predictions
remained included in its rollout-error calculation. The directional result has
the predeclared one-sided exact sign-test value `p=0.03125`.

| Seed | FFT rollout L2 | DCT rollout L2 | Paired improvement |
| ---: | ---: | ---: | ---: |
| 0 | 1.3300 | 0.9403 | 29.30% |
| 1 | 0.9927 | 0.8259 | 16.80% |
| 2 | 1.1966 | 0.9437 | 21.14% |
| 3 | 1.1212 | 0.9336 | 16.73% |
| 4 | 1.1364 | 0.8941 | 21.32% |

Boundary-strip and first-interior normal-derivative errors were also lower for
DCT in all five seeds. This supports lower boundary-region error, not boundary
enforcement or physical wall-flux correctness. See the frozen
[Phase 6 decision report](reports/chronopde_v2/phase6/decision_report.json).

## Original-study diagnostic

The decisive diagnostic used seed 0, the same 16 samples from four trajectories,
batch size 16, no spline perturbation, constant learning rate `3e-4`, no weight
decay, and 5,000 optimizer steps. The aligned objective directly minimized
per-sample full-field relative velocity error.

| Model | Original objective nRMSE | Aligned objective nRMSE | 0.01 gate |
| --- | ---: | ---: | :---: |
| CT-FFT | 0.1205 | 0.01895 | Fail |
| ChronoPDE DCT | 0.2480 | **0.01421** | Fail |

![Original objective-alignment diagnostic](reports/final/objective_alignment.png)

Both models exceeded the required 1,000x optimization-loss reduction. DCT had
lower velocity nRMSE for every matched identity; the paired mean DCT-minus-FFT
difference was `-0.00503`, with a deterministic 10,000-resample bootstrap 95%
interval of `[-0.00668, -0.00354]`. This is a fixed diagnostic result, not a
multi-seed generalization claim.

## What went wrong?

The original objective combined global MSE with relative error in only the
lowest 12 x 12 spectral modes. High-energy samples therefore had greater
influence, while the gate weighted each sample through its own full-field target
energy. Aligning the objective with the gate improved both models dramatically,
showing that loss-metric mismatch was real.

It did not fully explain the failure. The CPU target audit found finite targets
and a median spline-to-PDE velocity nRMSE of `0.00151`, far below the repair
threshold. After the complete aligned budget, both backbones still remained
above `0.01`. The declared route was therefore to document a model or
conditioning limitation and stop the study.

## Exploratory held-out rollout

The following visualization replays existing frozen ID checkpoints on
`id-0042`, selected deterministically as the trajectory nearest the median
combined error rank. It is qualitative context only: CT-FFT trained for 150
epochs while DCT stopped after 85, so it is not a confirmatory comparison.

![Exploratory held-out rollout](figures/qualitative_rollout.png)

The companion [figure provenance](figures/qualitative_rollout.json) records the
dataset, checkpoint hashes, source commits, selection rule, time indices, and
per-channel errors.

## Reproduce the main result

The headline result can be recomputed from committed lightweight evidence on a
CPU without the dataset, checkpoints, GPU, Kaggle, or Internet:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/reproduce_diagnostic.py --model both
```

Expected core output:

```text
CT-FFT
  Best step: 4100
  Median velocity nRMSE: 0.01895
  Gate: 0.01000
  Result: FAIL

ChronoPDE DCT
  Best step: 5000
  Median velocity nRMSE: 0.01421
  Gate: 0.01000
  Result: FAIL

Matched samples: DCT lower error on 16 / 16
```

Use `--format json` for machine-readable output and `--archive-root PATH` to
verify all seven original Kaggle archives byte-for-byte. Full dataset,
training, evaluation, figure-generation, and Kaggle instructions are in the
[reproduction guide](docs/REPRODUCTION.md).

## Repository structure

```text
ChronoPDE/
├── chronopde/         # data, models, numerics, training, evaluation, diagnostics
├── configs/           # frozen experiment and dataset configuration
├── scripts/           # generation, training, evaluation, and reproduction CLIs
├── tests/             # numerical, model, CLI, and evidence contracts
├── figures/           # public architecture and qualitative figures
├── reports/           # lightweight scientific evidence and generated analysis
├── notebooks/         # canonical Kaggle workflow plus archived provenance
├── demo/              # offline Streamlit evidence explorer
├── docs/              # method, roadmap, reproduction, and portfolio notes
└── output/pdf/        # controlled study and failure-analysis report
```

## Limitations and claim boundary

- Confirmation covers one reaction-diffusion PDE, a `64×64` grid, the central
  parameter range, five fixed seeds and one frozen training protocol.
- Sparse-time, hidden-time, parameter-OOD and initial-condition-OOD performance
  remain untested in V2.
- Lower boundary-region errors do not establish exact boundary enforcement or
  physical wall-flux correctness.
- The original one-seed diagnostic and unequal-history rollout figure remain
  exploratory V1 evidence and are not the basis of the V2 claim.
- Checkpoints are research artifacts, not validated scientific simulators.

See the [model card](MODEL_CARD.md),
[experimental report](output/pdf/chronopde_negative_result_report.pdf),
[release record](docs/RELEASE.md), and [portfolio summary](docs/PORTFOLIO.md).

The additive [ChronoPDE V2 Phase 1 evidence audit](reports/chronopde_v2/phase1/README.md)
records claim-level errata and provenance without changing the frozen study.

The [ChronoPDE V2 Phase 2 numerical audit](reports/chronopde_v2/phase2/README.md)
validates the discrete Neumann operator, freezes exact-RHS supervision and fresh
split identities, and records a successful CPU pilot without making a new model
quality claim.

The [ChronoPDE V2 Phase 3 data report](reports/chronopde_v2/phase3/README.md)
records a validated 512-trajectory training and 128-trajectory validation
dataset with exact discrete-RHS labels. At Phase 3, the 256 confirmatory
identities remained sealed and ungenerated; Phase 6 later generated them once.

The [Phase 4 feasibility protocol](reports/chronopde_v2/phase4/README.md)
implements matched FFT-12 and DCT-24 training on that development dataset,
including deterministic resume, validation-only selection, and verified
recovery packages. The completed seed-0 development run produced a split gate:
DCT passed with velocity nRMSE `0.0846` and zero divergence, while FFT failed
with velocity nRMSE `0.3791` and `0.625` divergence at its selected step. This
does not establish general DCT superiority.

The [Phase 4B integration audit](reports/chronopde_v2/phase4b/README.md) found
that DCT was stable through RK4 refinement and that FFT's `0.625` divergence
fraction persisted at every tested resolution. It classifies the seed-0 FFT
failure as learned-vector-field instability and freezes 8 RK4 steps per stored
interval for subsequent evaluation.

The [Phase 5 protocol](reports/chronopde_v2/phase5/README.md) implements the
authorized fresh five-seed development study. It retrains both matched models
for seeds 0–4, selects checkpoints only on fixed validation velocity error, and
applies a predeclared paired stability and performance gate. Confirmatory data
remain sealed, so Phase 5 cannot itself support a superiority claim.

The [Phase 6 report](reports/chronopde_v2/phase6/README.md) records the completed
one-shot sealed confirmation and its predeclared five-of-five paired gate. The
passing result authorizes only the narrow claim stated above for this PDE,
grid, central parameter range and frozen training protocol.

## License and attribution

ChronoPDE is released under the [MIT License](LICENSE). The implementation is
independent; PDEBench reference fixtures are used only for numerical comparison.
