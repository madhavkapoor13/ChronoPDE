# Week 3 Dataset Validation

- Status: **PASS**
- Successful trajectories: **720/720**
- Maximum state magnitude: **4.535845**
- Median runtime: **1.387 seconds**
- Median function evaluations: **4,457**
- Manifest SHA-256: `8906e1bce6a3c1879323623fb61e9bfa4368fca3dfb5eda04f1b4587542deef1`
- Configuration SHA-256: `a761704ff22b1f479f781c605137bf02803cda4ad9bd31e1d8a0edb7ba5689e8`

The parameter ranges frozen after Week 2 were unchanged. The manifest balances
the parameter-OOD split across eight low/high band combinations and provides
unique initial-condition seeds for every trajectory. Both sparse temporal masks
always retain the first and final stored times.

The ignored `data/chronopde.h5` file is 2.2 GB. Its five split groups contain
float32 states in `[N,T,2,H,W]` order, float64 times and parameters, temporal
masks, and simulator diagnostics. Normalization statistics use the training
split only. Ten deterministic source trajectories were compared byte-for-byte
with their HDF5 states during validation.

## Versioned outputs

- `manifest.csv` and `manifest.csv.sha256`
- `diagnostics.csv` and `summary.json`
- `parameter_coverage.png`
- `observation_masks.png` and `mask_retention.png`
- `diagnostic_distributions.png` and `state_distributions.png`
- `state_panels.png`

Raw trajectory checkpoints remain in the ignored `artifacts/dataset/week3/`
directory so generation can be resumed or audited.
