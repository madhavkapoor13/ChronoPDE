# ChronoPDE V2 Phase 3 — development dataset

Phase 3 passed and authorizes bounded feasibility training in Phase 4. It
generated the complete development dataset while leaving all confirmatory
trajectories sealed and ungenerated.

## Result

- Training trajectories: `512`
- Validation trajectories: `128`
- Confirmatory trajectories generated: `0`
- Successful development trajectories: `640/640`
- Maximum absolute state: approximately `2.8198`, below the frozen limit of `10`
- Exact-RHS spot checks: `16`
- Maximum RHS spot-check relative L2: approximately `2.70e-8`
- Dataset logical SHA-256:
  `4eb0482bdadb5fe836076131ecb01588409bdb7c3aa0914727a69107deaeded6`
- Training-only normalization SHA-256:
  `c9b334d077ef1075dc3d7e0c0da230954a6b4abfae4b0ce02ecfa69505581c1d`

No model or GPU training occurred. Pilot normalization was not reused. Legacy
data was not used. The confirmatory split remains represented only by the
identity manifest frozen in Phase 2.

## Artifacts

The ignored Phase 3 artifact directory contains a resumable per-trajectory
cache, the approximately 4.0 GB development HDF5, a runtime summary, and a
lightweight recovery ZIP. The directory occupies approximately 7.6 GB in total.

Committed evidence contains only the protocol snapshot, development manifest,
per-trajectory QA table, dataset summary, and decision report. Phase 4 may use
training data for fitting and validation data for selection; it may not generate
or inspect confirmatory states.
