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
- [ ] Complete seed-0 full training and publish frozen ID curves.

## Week 5 — Continuous-time FFT baseline

- [x] Implement deterministic local quintic velocity sampling.
- [x] Implement the FiLM-conditioned continuous-time FFT vector field.
- [x] Implement the shared velocity/spectral objective and RK4 training path.
- [x] Implement frozen ID evaluation and the Colab workflow.
- [ ] Pass the four-trajectory GPU overfit gate.
- [ ] Beat persistence on validation and publish the frozen ID comparison.

Later weeks follow the twelve-week project plan, subject to the documented gates.
