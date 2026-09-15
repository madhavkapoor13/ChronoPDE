# ChronoPDE Roadmap

## Week 1 — Foundation

- [x] Freeze equations, shapes, splits, metrics, and claim policy.
- [x] Create validated YAML configuration.
- [x] Create deterministic experiment naming and environment capture.
- [x] Create data/model/integrator interfaces.
- [x] Create dry-run CLIs and CPU-only tests.
- [x] Configure CI and repository hygiene.

## Week 2 — Simulator and pilot

- [x] Implement cell-centred Neumann Laplacian.
- [x] Implement reaction-diffusion right-hand side and adaptive integration.
- [x] Implement DCT-filtered initial-condition generator.
- [x] Compare small deterministic runs with PDEBench.
- [x] Execute the frozen 24-trajectory stability pilot.
- [x] Publish pilot plots and confirm that no range adjustment is required.

## Week 3 — Dataset and numerical primitives

- [x] Generate deterministic split manifests and temporal masks.
- [x] Generate and validate the HDF5 dataset.
- [x] Implement DCT, splines, and fixed-step integrators with reference tests.

## Week 4 — Autoregressive baselines

- [x] Implement lazy HDF5 pair/rollout datasets and shared normalization.
- [x] Implement parameter-matched residual U-Net and FFT-FNO baselines.
- [x] Implement training, checkpoint resume, rollout, and ID evaluation infrastructure.
- [x] Add the reproducible Colab GPU workflow.
- [ ] Pass both four-trajectory GPU overfit gates.
- [x] Complete seed-0 full training and recover frozen ID summaries from Drive.
- [x] Record that both original full-training gates were not reached; retain the
  ID reports as exploratory evidence rather than silently redefining the gate.

## Week 5 — Continuous-time FFT baseline

- [x] Implement deterministic local quintic velocity sampling.
- [x] Implement the FiLM-conditioned continuous-time FFT vector field.
- [x] Implement the shared velocity/spectral objective and RK4 training path.
- [x] Implement frozen ID evaluation and the Colab workflow.
- [x] Pass the four-trajectory GPU overfit gate.
- [x] Beat persistence on validation and publish the frozen ID comparison.

## Week 6 — Boundary-aware ChronoPDE

- [x] Implement a real DCT spectral convolution aligned with Neumann boundaries.
- [x] Add a parameter-matched FiLM-conditioned continuous-time DCT vector field.
- [x] Make `chronopde` operational in training, resume, and frozen ID evaluation.
- [x] Add checkpoint model-identity validation and CPU contract tests.
- [x] Add an unattended, resumable Kaggle workflow.
- [x] Close the four-trajectory gate as a valid negative result after the full
  target, mechanics, balanced-batch, and loss-alignment diagnostic sequence.
- [x] Preserve and checksum the failed resampled smoke result and the 85-epoch
  exploratory run.
- [x] Implement the CPU target audit, true fixed-batch memorization controls,
  matched fixed-trajectory comparison, and conditional DCT variants.
- [x] Run the isolated diagnostic suite and follow its predeclared route.
- [x] Apply the stop rule: Week 7 and OOD evaluation are canceled for this
  release because the corrected gate did not pass.
- [x] Run the shared optimizer/loss mechanics sweep and record the matched-control result.
- [x] Preserve the original gate statistic and report the conventional median
  separately; neither definition changes the failed decision.
- [x] Preserve the exploratory ID comparison, where both continuous-time models
  beat persistence, without promoting it to confirmatory evidence.

## Recovery release

- [x] Freeze archive checksums and import lightweight evidence only.
- [x] Add deterministic paired failure analysis and plots.
- [x] Recover the Week 4 seed-0 summaries from Google Drive.
- [x] Publish the negative-result report, offline demo, model card, and clean-environment release.

## Portfolio release

- [x] Make the README the evidence-first project landing page.
- [x] Add a CPU-only command that reproduces the frozen headline result.
- [x] Publish an architecture overview and a checksum-verified qualitative rollout.
- [x] Consolidate runnable notebooks and archive historical Kaggle workflows.
- [x] Add a detailed reproduction guide and concise resume-ready project summary.
- [x] Preserve `v0.1.0-negative-result` and publish the additive `v0.1.1-portfolio` release.

Original Weeks 7-10 are intentionally not executed in this release. The project
continues through a narrower reproducibility and communication track rather than
manufacturing a positive result after a predeclared stop rule fired.
