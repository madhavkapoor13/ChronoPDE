# ChronoPDE V2 Phase 6 — sealed confirmation

Phase 6 is the one-shot confirmatory stage. It first generates the 256 identities
sealed in Phase 2, then evaluates all five frozen FFT–DCT checkpoint pairs. No
training, checkpoint selection, development-data access, OOD evaluation or
changed-setting repeat is permitted.

The strict claim gate requires five of five DCT wins on rollout and exact
velocity, at least 10% median paired rollout improvement, zero DCT divergence,
five persistence wins and no worse divergence in every seed. A passing result
supports only a claim scoped to this PDE, grid, parameter range and protocol.
Boundary wording additionally requires five of five wins on boundary-strip and
first-interior normal-derivative error; the model is never described as
enforcing physical wall flux.

Use `chronopde_v2_phase6_generate_kaggle.ipynb` on CPU, save the HDF5, its
`.sha256` sidecar and the generation-evidence ZIP together as one private
dataset, then use `chronopde_v2_phase6_evaluate_kaggle.ipynb` with T4x2. Large
datasets and checkpoints remain outside Git.
