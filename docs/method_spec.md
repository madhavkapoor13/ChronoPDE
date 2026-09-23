# ChronoPDE V1 Method Specification (historical)

This file records the original Week 1-6 protocol and is retained unchanged in
substance for provenance. It does not describe the matched V2 confirmatory
study. See [`method_spec_v2.md`](method_spec_v2.md) for the current method.

Status: frozen Week 1 contract. Changes require a dated decision record before
final experiments begin.

## 1. Scientific question

ChronoPDE tests two conditional hypotheses under matched trajectory, parameter,
and optimization budgets:

1. Continuous-time velocity learning is more robust than direct autoregressive
   state prediction when trajectory observations are sparse or irregular.
2. A discrete cosine spectral backbone aligned with homogeneous Neumann boundary
   conditions reduces boundary-gradient and spectral errors relative to an FFT
   backbone.

The work is a controlled scientific-ML portfolio project, not a claim of a new
state-of-the-art algorithm. A reproducible negative result is successful if the
failure regime and error source are analysed.

## 2. Governing problem

The state is `x(t) = [u(t,x,y), v(t,x,y)]` and evolves according to

```text
du/dt = Du * Laplacian(u) + u - u^3 - k - v
dv/dt = Dv * Laplacian(v) + u - v
```

Both fields satisfy zero normal derivative on all four boundaries. The spatial
domain is `[-1,1] × [-1,1]`, discretised on a cell-centred `64×64` grid. The
reference interval is `[0,50]`, with 101 stored states. Reference integration
uses SciPy `solve_ivp` with DOP853, `rtol=1e-7`, and `atol=1e-9`.

The discretisation follows the PDEBench 2D diffusion-reaction convention: a
sparse cell-centred finite-volume Laplacian whose boundary diagonals encode zero
normal flux. The implementation will be independently written and checked
against PDEBench on small deterministic examples.

The committed reference fixture uses PDEBench commit
`4ff3e3a4aa1561721b5571fa3a048a0a463e0568`, an 8×8 grid, five stored times on
`[0,1]`, seed 0, and SciPy's default RK45 tolerances. ChronoPDE matches the
fixture within `1e-6` relative L2 error.

## 3. Data protocol

| Split | Count | Parameters | Initial conditions |
|---|---:|---|---|
| Train | 320 | central ranges | smooth, modes 1–8 |
| Validation | 60 | unseen central values | unseen seeds |
| ID test | 100 | central ranges | unseen seeds |
| Parameter OOD | 120 | disjoint lower/upper bands | train-like spectrum |
| IC OOD | 120 | central ranges | modes 9–16, 1.5× amplitude |

Exact coefficient ranges live in `configs/project.yaml`. A 24-trajectory pilot
must cover centre and boundary values. If a solver fails or more than 10% of
trajectories exceed absolute state magnitude 10, the offending OOD endpoint is
moved 20% toward the training boundary and the pilot is repeated. Ranges freeze
before any model tuning.

### Week 2 pilot outcome

The frozen pilot completed successfully on 10 September 2026. All 24
trajectories integrated successfully, with a maximum observed state magnitude
of 4.237602 against the divergence threshold of 10. Median runtime was 1.356
seconds and median function evaluations were 4,463. The original training and
OOD coefficient ranges therefore remain frozen without adjustment. Reproducible
diagnostics and figures are stored in `reports/pilot/week2/`.

Observation regimes retain 100%, 50%, or 25% of stored times. Both endpoints
are always present. Interior masks are deterministic functions of trajectory ID,
regime, and mask seed. Splitting occurs before interval construction. Normalizers
are fitted only on the training split.

The HDF5 state layout is `[N,T,2,H,W]`, float32. Each trajectory stores its time
array, `[Du,Dv,k]`, IC seed, trajectory ID, split, masks, and simulator metadata.

### Week 3 dataset outcome

The frozen 720-row manifest has SHA-256
`8906e1bce6a3c1879323623fb61e9bfa4368fca3dfb5eda04f1b4587542deef1`.
All 720 trajectories completed successfully on 10 September 2026 and were
validated in the HDF5 container. The maximum state magnitude was 4.535845,
median runtime was 1.387 seconds, and median function evaluations were 4,457.
The dataset configuration SHA-256 is
`a761704ff22b1f479f781c605137bf02803cda4ad9bd31e1d8a0edb7ba5689e8`.
Normalizers use every stored training state, independent of observation mask.
Raw trajectories and the HDF5 file remain ignored; manifests, diagnostics, and
QA figures are versioned in `reports/dataset/week3/`.

## 4. Learning targets and models

The continuous-time path uses a CFO-style quintic polynomial between adjacent
retained knots. Knot derivatives come from finite differences; the analytic
path derivative is the supervised target. The primary model predicts
`v_theta(x(t), t, p) ≈ dx/dt`. A smooth endpoint-vanishing perturbation begins at
`gamma=1e-5`. Linear paths are diagnostic only.

ChronoPDE uses a real orthonormal 2D DCT-II/DCT-III pair, four residual spectral
blocks, 12×12 retained modes, pointwise residual paths, GELU, and FiLM
conditioning on normalized `[t,Du,Dv,k]`. RK4 is the default inference solver;
Euler and Heun are controlled cost/accuracy comparisons.

Baselines are:

- `unet_ar`: convolutional residual next-state predictor.
- `fno_ar`: FFT spectral residual next-state predictor.
- `fno_ct`: continuous-time FFT velocity model with the ChronoPDE training path.
- `chronopde`: continuous-time DCT velocity model aligned with Neumann boundaries.

Autoregressive models receive `delta_t` and physical parameters. Trainable
parameter counts in headline comparisons must be within 10%. Shared data,
normalization, optimizer family, early-stopping budget, and evaluation
trajectories are mandatory.

The Week 4 baselines predict normalized residuals rather than absolute next
states. Both receive identical broadcast maps for normalized `delta_t`, `Du`,
`Dv`, and `k`, plus cell-centred coordinates. The U-Net uses reflection-padded
convolutions with channel widths 32/64/128/256. The FFT-FNO uses width 29, four
blocks, and 12x12 retained modes. Complex parameters count as two real degrees
of freedom, giving 1,929,890 U-Net parameters and 1,943,263 FNO parameters, a
0.69% difference. Week 4 model selection uses validation rollout nRMSE; OOD
splits remain unopened.

The Week 5 `fno_ct` baseline uses width 29, four FFT blocks, and 12x12 modes.
It lifts normalized states and applies independent FiLM scale/bias values in
each block from a two-layer conditioner over normalized time and parameters.
It predicts normalized-state velocity per physical-time unit. Its 1,973,657
real trainable parameters are within 2.3% of both autoregressive baselines.
Training exposes one deterministic quintic-path sample per retained interval
and epoch; the full regime therefore contains 32,000 velocity examples per
epoch. The common auxiliary loss compares the lowest 12x12 orthonormal DCT
coefficients, independently of the model's spatial backbone.

The Week 6 `chronopde` model changes only the spatial spectral basis. Each of
its four blocks applies an orthonormal DCT-II, learns a real channel-mixing
tensor over the lowest 12x12 cosine modes, zero-fills omitted modes, and applies
the inverse DCT-III. Width 57 yields 1,951,125 real trainable parameters. The
training samples, loss, FiLM conditioner, integration method, and validation
gate remain identical to the Week 5 continuous-time control.

## 5. Training and evaluation

Continuous-time loss is channel-normalized velocity MSE plus spectral loss with
weight 0.05. Default optimization is AdamW with learning rate `3e-4`, weight
decay `1e-4`, five warm-up epochs, cosine decay, gradient clipping at 1.0, and
float32 precision until numerical stability is established.

Each model must overfit four trajectories before a full run. Development uses
seed 0. Final three-seed work begins only after configurations freeze.

Required per-trajectory metrics are relative L2, nRMSE, final-time error,
error-versus-time, 0.9 correlation horizon, low/high DCT-band error,
first-interior boundary gradient, divergence rate, OOD/ID error ratio, latency,
peak memory, parameter count, and function evaluations. Confidence intervals
use 10,000 paired bootstrap samples.

Published figures must include median and failure cases, not only the best
trajectory. All tables and plots are regenerated from archived raw metrics.

## 6. Claim and failure policy

OOD ranges and test seeds are not changed after inspection. Model selection uses
validation data only. If DCT improves boundary behaviour but not global nRMSE,
that narrower finding is reported. If continuous time loses at short horizons,
the crossover horizon is reported. If the main hypothesis fails, the report
becomes a regime map and error-attribution study; the metric or split is not
changed to manufacture a win.

## 7. Attribution boundary

- CFO supplies the continuous-time flow-matching motivation and spline-path
  precedent. ChronoPDE reimplements the method in PyTorch and adds the
  parameter-conditioned DCT/FFT controlled comparison.
- PDEBench supplies the reaction-diffusion equation and reference discretisation
  convention. ChronoPDE independently implements the simulator and creates its
  own frozen parameter and OOD protocol.
- External source code copied verbatim, if ever required, must be isolated,
  licensed, and identified. The default is independent implementation.

Primary references:

1. Hou, Huang, and Perdikaris. “CFO: Learning Continuous-Time PDE Dynamics via
   Flow-Matched Neural Operators.” ICLR 2026.
2. Takamoto et al. “PDEBench: An Extensive Benchmark for Scientific Machine
   Learning.” NeurIPS Datasets and Benchmarks, 2022.
3. Li et al. “Fourier Neural Operator for Parametric Partial Differential
   Equations.” ICLR 2021.
