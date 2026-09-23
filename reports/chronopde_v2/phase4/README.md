# ChronoPDE V2 Phase 4 — matched feasibility training

Phase 4 trained the matched Phase 2 FFT-12 and DCT-24 models on the Phase 3
exact-RHS development dataset using seed 0. Training used only the 512 training
trajectories; checkpoint selection and the feasibility gate used only the 128
validation trajectories. The source archive and headline result are frozen in
`result_summary.json`.

The DCT model passed; the FFT control failed because of velocity error and
rollout divergence. The confirmatory identities remain sealed. This phase does
not authorize a DCT-superiority claim and routes to the Phase 4B integration
audit before any multi-seed development comparison.

The frozen command is:

```bash
python scripts/chronopde_v2.py phase4 \
  --data-path /path/to/chronopde_v2_development.h5 \
  --model both --device cuda --format json
```

Use `--check-only` to validate the protocol and comparator without training.
Use `--resume` only when the same run manifest remains in the `running` state
and its `last.pt` checkpoint is present.
