# ChronoPDE V2 Phase 2 — numerical ground truth and fresh protocol

Phase 2 passed. It performed no model training and produced no model-quality
result. It validated the numerical premise, froze an exact-discrete-RHS target,
created new non-overlapping development and sealed confirmatory identities, and
generated a 24-trajectory CPU pilot.

## Decision

`phase2_complete_phase3_data_generation_allowed`

- The sparse finite-volume, reflected-stencil, and analytical DCT Neumann
  operators agree within the predeclared tolerances.
- All 24 pilot trajectories completed and remained below an absolute state
  magnitude of 10; the observed maximum was approximately `1.8365`.
- Across six tight-tolerance repeats, the 95th-percentile state relative L2 was
  approximately `4.47e-7` and the maximum final-state error was approximately
  `1.11e-8`.
- The selected FFT `12×12`, width-29 and DCT `24×24`, width-29 operators each
  have `1,973,657` real parameters and `484,416` active real spectral degrees
  of freedom per block. Their maximum physical cutoffs differ by approximately
  `4.35%`.

The DCT is described only as a **Neumann-aligned inductive bias**. The complete
network is not described as enforcing physical wall flux: pointwise paths,
nonlinearities, FiLM, lifting, and projection are not explicit physical-wall
constraints.

## Evidence

- `decision_report.json` is the complete machine-readable decision.
- `numerical_audit.json` records the independent operator and RHS checks.
- `comparator.json` records model-size, spectral-DOF, and physical-cutoff matching.
- `pilot_manifest.csv` identifies the 24 pilot trajectories.
- `future_manifest.csv` freezes 512 training, 128 validation, and 256 sealed
  confirmatory identities without generating their trajectories.
- `pilot_summary.json` records convergence and logical dataset hashes. Pilot
  normalization is explicitly diagnostic-only.

The approximately 153 MB pilot HDF5 and recovery ZIP remain under the ignored
`artifacts/chronopde_v2/runs/` tree. No HDF5, checkpoint, or ZIP is committed.
