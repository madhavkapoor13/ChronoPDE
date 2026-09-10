# ChronoPDE

ChronoPDE is a from-scratch PyTorch project for studying boundary-aware,
continuous-time neural operators on coupled two-dimensional reaction-diffusion
dynamics. The intended model learns a parameter-conditioned velocity field from
irregularly sampled trajectories and integrates it at arbitrary query times.

The current repository state includes the completed **Week 3 dataset and
numerical primitives**: a validated 720-trajectory HDF5 dataset pipeline,
deterministic split manifests and temporal masks, differentiable DCT transforms,
CFO-style quintic paths, and fixed-step Euler, Heun, and RK4 integration. Neural
models remain scheduled for later phases.

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
