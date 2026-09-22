# ChronoPDE V2 Phase 4 — matched feasibility training

Phase 4 is implemented but has not been scientifically executed in this
repository snapshot. It trains the matched Phase 2 FFT-12 and DCT-24 models on
the Phase 3 exact-RHS development dataset using seed 0. Training uses only the
512 training trajectories; checkpoint selection and the feasibility gate use
only the 128 validation trajectories.

The confirmatory identities remain sealed and their states must not exist in
the input HDF5. This phase cannot authorize a DCT-superiority claim. If both
models pass the declared feasibility gate, Phase 5 may perform a multi-seed
development-only comparison before a single frozen confirmatory evaluation.

The frozen command is:

```bash
python scripts/chronopde_v2.py phase4 \
  --data-path /path/to/chronopde_v2_development.h5 \
  --model both --device cuda --format json
```

Use `--check-only` to validate the protocol and comparator without training.
Use `--resume` only when the same run manifest remains in the `running` state
and its `last.pt` checkpoint is present.
