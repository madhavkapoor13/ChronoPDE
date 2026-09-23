# ChronoPDE V2 Phase 4B — integration and instability audit

Phase 4B is an evaluation-only control audit triggered by the split Phase 4
result: the DCT model passed its feasibility gate and the matched FFT model did
not. It loads the frozen `best.pt` checkpoints and repeats the exact same 16
development-validation rollouts with RK4 using 1, 2, 4, and 8 substeps per
stored interval.

The audit records per-trajectory rollout, final-time, persistence, correlation,
stability, boundary-strip, interior, and first-interior normal-derivative
errors. It also measures prediction convergence between 4 and 8 RK4 substeps.
No optimizer is created, no checkpoint is written, and confirmatory identities
remain sealed.

The Phase 4 source archive, both checkpoints, run manifests, configuration,
dataset, and normalization are verified before evaluation. This audit cannot
authorize a superiority claim or confirmatory evaluation. It only determines
whether Phase 5 multi-seed development experiments are methodologically
justified and freezes the future integration resolution when they are.

## Recorded result

The completed audit found the DCT model stable on all 16 trajectories at every
tested resolution, with 4-vs-8-step p95 relative disagreement of approximately
`2.79e-5`. The same 10 of 16 FFT trajectories diverged at every resolution.
The result therefore records persistent learned-vector-field instability for
the seed-0 FFT checkpoint, freezes 8 RK4 steps per stored interval, and permits
the development-only Phase 5 multi-seed study. It does not authorize a
superiority claim.
