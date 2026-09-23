# ChronoPDE portfolio summary

**ChronoPDE - Continuous-Time Neural Operators for Reaction-Diffusion Dynamics**

PyTorch | Scientific ML | Neural Operators | DCT/FFT | RK4

- Developed matched 1.97M-parameter continuous-time FFT and Neumann-aligned DCT
  neural operators for parameter-conditioned 2D reaction-diffusion dynamics,
  learning exact discrete PDE velocities and integrating them with RK4.
- Ran a preregistered five-seed comparison on 256 sealed trajectories: DCT won
  rollout and exact-velocity error in `5/5` seeds, achieved `21.1%` median paired
  rollout improvement (95% bootstrap interval `16.4%–27.7%`), and had zero
  divergence across 1,280 confirmatory rollouts.
- Built reproducible research infrastructure spanning deterministic data
  generation, exact-RHS and solver audits, atomic checkpoint recovery,
  checksum-pinned Kaggle workflows, sealed holdouts, paired hierarchical
  bootstrap analysis, and predeclared decision gates.
