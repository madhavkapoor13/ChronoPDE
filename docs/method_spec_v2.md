# ChronoPDE V2 method specification

Status: completed, frozen confirmatory study. The authoritative machine-readable
contracts live under `configs/chronopde_v2/`; the lightweight decisions live
under `reports/chronopde_v2/`.

## Research question

For the frozen two-species reaction-diffusion PDE, does a Neumann-aligned cosine
operator learn the exact discrete velocity and produce more accurate stable
rollouts than a physically bandwidth-, spectral-DOF-, parameter-, optimizer-
and data-matched Fourier operator?

The hypothesis concerns a basis-aligned inductive bias. The complete DCT neural
network does not enforce a boundary condition because lifting, pointwise paths,
FiLM, nonlinearities and projection can change boundary behaviour.

## Numerical target and data

- Cell-centred `64 x 64` finite-volume grid with homogeneous Neumann boundaries.
- Physical interval `[0, 50]` with 101 stored states.
- Supervised target: exact discrete simulator RHS per unit physical time.
- 512 training and 128 validation trajectories generated in Phase 3.
- 256 confirmatory identities frozen in Phase 2 and generated once in Phase 6.
- State and parameter normalization fitted only on the 512 training trajectories.
- Legacy V1 data and exposed ID results are excluded from V2 confirmation.

The numerical audit independently checked the sparse operator, reflected-cell
stencil, DCT diagonalization, RHS implementations, time units, float32 storage,
and solver convergence. Phase 2 evidence contains the frozen tolerances and
results.

## Matched models

Both vector fields use width 29, four spectral blocks, identical FiLM
conditioning, pointwise paths, lifting, projection, initialization and RK4
rollout. Each has 1,973,657 real trainable parameters and 484,416 active real
spectral degrees of freedom per block.

| Model | Basis | Retained modes | Spatial assumption |
| --- | --- | ---: | --- |
| FFT control | complex Fourier | `12 x 12` | periodic |
| ChronoPDE | real orthonormal DCT | `24 x 24` | Neumann-aligned |

The maximum-wavenumber mismatch is 4.35%, within the frozen 10% tolerance.

## Training and selection

Five fresh seeds (`0` through `4`) were trained for each backbone. Both used the
same exact-RHS objective, optimizer, budget and development data. Checkpoints
were selected only by fixed validation velocity nRMSE, with validation relative
loss as tie-breaker. The confirmatory states did not exist during selection.

## Confirmatory evaluation

All ten frozen checkpoints were evaluated once on all 256 sealed trajectories
using RK4 with eight steps per stored interval. The primary metric was rollout
relative L2 over physical time. Secondary metrics included exact-velocity
nRMSE, final-time error, divergence, correlation horizon, persistence, and
boundary-strip/interior attribution.

The predeclared gate required five of five DCT wins on rollout and velocity,
zero DCT divergence, at least 10% median paired improvement, DCT beating
persistence in every seed, and divergence no worse than FFT in every seed.

## Result and claim boundary

The gate passed. DCT won rollout and exact-velocity error in all five seeds,
achieved 21.14% median paired rollout improvement (20,000-resample hierarchical
bootstrap 95% interval 16.35%-27.72%), and had zero divergence across 1,280
seed-trajectory rollouts. The one-sided exact sign-test value is `p=0.03125`.

The authorized claim is restricted to this PDE, grid, central parameter range,
and frozen protocol. Boundary-region errors were lower, but the result does not
establish boundary enforcement, physical wall-flux correctness, sparse-time
performance, OOD generalization, grid transfer, or superiority on other PDEs.
