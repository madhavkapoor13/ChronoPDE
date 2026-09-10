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

Later weeks follow the twelve-week project plan, subject to the documented gates.
