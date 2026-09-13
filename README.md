# ChronoPDE

ChronoPDE is a from-scratch PyTorch project for studying boundary-aware,
continuous-time neural operators on coupled two-dimensional reaction-diffusion
dynamics. The intended model learns a parameter-conditioned velocity field from
irregularly sampled trajectories and integrates it at arbitrary query times.

The current repository state includes the completed Week 3 dataset/numerical
primitives and the CPU-validated Week 4-5 learning stack: autoregressive U-Net
and FFT-FNO baselines plus a FiLM-conditioned continuous-time FFT operator.
GPU training results remain measured release gates and are not represented as
completed locally.

## Research question

Under matched data, parameter count, and optimization budgets:

1. Does continuous-time velocity learning improve forecasts as temporal
   observations become sparse or irregular?
2. Does a cosine spectral backbone aligned with homogeneous Neumann boundaries
   improve boundary-gradient or spectral errors relative to an FFT backbone?

The project does not assume that ChronoPDE will win every metric. Negative and
conditional results will be retained and analysed.

## Quick start

ChronoPDE targets Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
ruff check .
mypy chronopde
```

Run the non-mutating Week 1 interface checks:

```bash
python scripts/generate_data.py --config configs/data.yaml --dry-run
python scripts/train.py --config configs/project.yaml --model chronopde --regime irreg25 --seed 0 --dry-run
python scripts/evaluate.py --config configs/project.yaml --experiment id_rollout --checkpoint placeholder.pt --dry-run
```

Run or resume the frozen 24-trajectory simulator pilot:

```bash
python scripts/generate_data.py --config configs/project.yaml --pilot
```

Raw pilot trajectories are written to `artifacts/pilot/week2/` and remain
ignored by Git. The publishable summary, diagnostic table, and plots are written
to `reports/pilot/week2/`. The completed pilot passed 24/24 trajectories without
changing any parameter range.

Freeze the full dataset manifest, then generate or resume all 720 trajectories:

```bash
python scripts/generate_data.py --config configs/project.yaml --manifest-only
python scripts/generate_data.py --config configs/project.yaml --full
```

The manifest and its SHA-256 file are committed under
`reports/dataset/week3/`. Raw per-trajectory checkpoints live under the ignored
`artifacts/dataset/week3/` directory. The validated dataset is written atomically
to the ignored `data/chronopde.h5`; rerunning `--full` validates and reuses all
successful checkpoints.

The HDF5 state layout is `[N,T,2,H,W]`. Each split contains parameters, seeds,
trajectory IDs, diagnostics, and the `full`, `irregular_50`, and `irregular_25`
masks. Channel and parameter normalizers are fitted from the training split only.

The reusable numerical APIs are available from `chronopde.numerics`: orthonormal
`dct2`/`idct2`, irregular quintic spline construction and evaluation, conditional
path sampling, and differentiable fixed-step integration.

## Week 4 autoregressive baselines

ChronoPDE includes parameter-matched residual U-Net and FFT-FNO baselines. Both
consume normalized states, the time increment, physical parameters, and spatial
coordinates, then predict a normalized state residual.

Run the mandatory four-trajectory checks on a CUDA machine before full training:

```bash
python scripts/train.py --config configs/project.yaml --model unet_ar --regime full --seed 0 --smoke-overfit --device cuda
python scripts/train.py --config configs/project.yaml --model fno_ar --regime full --seed 0 --smoke-overfit --device cuda
```

Run or resume full seed-0 training:

```bash
python scripts/train.py --config configs/project.yaml --model unet_ar --regime full --seed 0 --device cuda --resume
python scripts/train.py --config configs/project.yaml --model fno_ar --regime full --seed 0 --device cuda --resume
```

Use `--data-path` when the HDF5 file is stored outside the repository, as in the
provided `notebooks/week4_baselines_colab.ipynb`. Checkpoints and logs are saved
under the ignored `artifacts/runs/` directory.

After training, generate the frozen ID report with:

```bash
python scripts/evaluate.py --config configs/project.yaml --model unet_ar --experiment id_rollout --checkpoint artifacts/runs/unet_ar-full-train-s0/best.pt
```

## Week 5 continuous-time FFT baseline

`fno_ct` learns a normalized state velocity from deterministic samples on the
quintic observation paths. Time and normalized PDE parameters condition every
FFT block through FiLM. Inference integrates the learned velocity over physical
time with the differentiable fixed-step RK4 solver.

Run the mandatory four-trajectory gate before a full run:

```bash
python scripts/train.py --config configs/project.yaml --model fno_ct --regime full --seed 0 --smoke-overfit --device cuda
```

Train, resume, and evaluate with:

```bash
python scripts/train.py --config configs/project.yaml --model fno_ct --regime full --seed 0 --device cuda --resume
python scripts/evaluate.py --config configs/project.yaml --model fno_ct --experiment id_rollout --checkpoint artifacts/runs/fno_ct-full-train-s0/best.pt --device cuda
```

Use `--learning-rate 1e-4` only for the predefined validation recovery run and
`--steps-per-interval 4` only for the integration-resolution diagnostic. The
Colab notebook verifies the exact dataset hash before launching any GPU job.

The completed seed-0 baseline ran for 150 epochs and achieved median ID final
nRMSE 0.4219 with no divergent trajectories. Its archived results are the
locked comparison point for Week 6.

## Week 6 boundary-aware ChronoPDE

`chronopde` replaces periodic FFT modes with a real orthonormal DCT basis that
matches the simulator's Neumann boundary condition. It retains the same
continuous-time velocity target, FiLM conditioning, RK4 rollout, optimizer,
and 12x12 spectral budget as `fno_ct`. Width 57 gives 1,951,125 trainable
parameters, within 10% of all three controlled baselines.

Run the mandatory four-trajectory gate first:

```bash
python scripts/train.py --config configs/project.yaml --model chronopde --regime full --seed 0 --smoke-overfit --device cuda
```

Only after that summary reports `passed: true`, launch the full run and frozen
ID evaluation:

```bash
python scripts/train.py --config configs/project.yaml --model chronopde --regime full --seed 0 --device cuda --resume
python scripts/evaluate.py --config configs/project.yaml --model chronopde --experiment id_rollout --checkpoint artifacts/runs/chronopde-full-train-s0/best.pt --device cuda
```

For unattended Kaggle execution, use
`notebooks/week6_chronopde_kaggle.ipynb`. It checkpoints after every epoch,
stops before Kaggle's wall-time limit, and packages resumable state.

The demo becomes active after a trained checkpoint exists:

```bash
python -m pip install -e ".[demo]"
streamlit run demo/app.py
```

## Repository contract

- `chronopde/`: reusable configuration, interfaces, experiment planning, and reproducibility code.
- `configs/`: versioned scientific and runtime decisions.
- `scripts/`: stable command-line interfaces.
- `tests/`: fast CPU-only contract tests.
- `docs/`: mathematical specification, development log, and roadmap.

Generated datasets and experiment artifacts are excluded from version control.
Every future run will save its resolved configuration, environment, Git commit,
metrics, checkpoints, and figures under a deterministic experiment identifier.

## Attribution

ChronoPDE is independently implemented in PyTorch. Its continuous-time training
design is inspired by *CFO: Learning Continuous-Time PDE Dynamics via
Flow-Matched Neural Operators*. The reaction-diffusion equations and reference
simulation conventions are based on PDEBench. See `docs/method_spec.md` for the
precise boundary between referenced ideas and original implementation.
