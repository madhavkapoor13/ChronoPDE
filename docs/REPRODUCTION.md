# ChronoPDE reproduction guide

## Environment

ChronoPDE targets Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,demo,report]"
pytest -q
ruff check .
mypy chronopde
```

## Frozen headline result

Recompute the Week 6 decision from committed lightweight evidence:

```bash
python scripts/reproduce_diagnostic.py --model both
python scripts/reproduce_diagnostic.py --model both --format json
```

Optionally verify the original local archives:

```bash
python scripts/reproduce_diagnostic.py \
  --model both \
  --archive-root /path/to/archive/directory
```

This command does not train a model. It validates the frozen protocol, selected
steps, 16 paired identities, loss reduction, gate statistic, bootstrap result,
and evidence hashes.

## Dataset generation

Run a 24-trajectory simulator pilot, freeze the manifest, and generate or resume
the full 720-trajectory dataset:

```bash
python scripts/generate_data.py --config configs/project.yaml --pilot
python scripts/generate_data.py --config configs/project.yaml --manifest-only
python scripts/generate_data.py --config configs/project.yaml --full
```

Raw checkpoints and `data/chronopde.h5` remain ignored by Git. Re-running full
generation validates and reuses successful per-trajectory checkpoints.

## Training and frozen ID evaluation

Autoregressive baselines:

```bash
python scripts/train.py --config configs/project.yaml --model unet_ar --regime full --seed 0 --device cuda --resume
python scripts/train.py --config configs/project.yaml --model fno_ar --regime full --seed 0 --device cuda --resume
```

Continuous-time controls:

```bash
python scripts/train.py --config configs/project.yaml --model fno_ct --regime full --seed 0 --device cuda --resume
python scripts/train.py --config configs/project.yaml --model chronopde --regime full --seed 0 --device cuda --resume
```

Example ID evaluation:

```bash
python scripts/evaluate.py \
  --config configs/project.yaml \
  --model chronopde \
  --experiment id_rollout \
  --checkpoint artifacts/runs/chronopde-full-train-s0/best.pt \
  --device cuda
```

These commands remain available for provenance. The published release does not
authorize new training as a continuation of the stopped Week 6 experiment.

## Week 6 diagnostic sequence

The completed sequence was:

```bash
python scripts/diagnose_continuous.py --config configs/project.yaml --data-path data/chronopde.h5 --device cuda
python scripts/diagnose_mechanics.py --config configs/project.yaml --data-path data/chronopde.h5 --device cuda
python scripts/audit_velocity.py --config configs/project.yaml --data-path data/chronopde.h5 --mechanics-root artifacts/diagnostics/week6/mechanics
python scripts/diagnose_balanced_batch.py --config configs/project.yaml --data-path data/chronopde.h5 --device cuda
python scripts/diagnose_loss_alignment.py --config configs/project.yaml --data-path data/chronopde.h5 --device cuda
```

The final command selected
`stop_and_document_model_or_conditioning_limitation`. Earlier commands are not
rerun by the CPU-only headline reproducer.

## Qualitative figure

Generate the frozen exploratory figure from the exact dataset and extracted
checkpoints:

```bash
python scripts/build_qualitative_comparison.py \
  --data-path /path/to/chronopde.h5 \
  --fft-checkpoint /path/to/fno_ct-best.pt \
  --dct-checkpoint /path/to/chronopde-best.pt \
  --device auto
```

The command verifies all three SHA-256 hashes before inference and writes the
PNG plus a JSON provenance record under `figures/`.

## Offline evidence explorer and report

```bash
streamlit run demo/app.py
python scripts/analyze_week6_failure.py
python scripts/build_negative_result_report.py
```

The explorer and frozen analysis require no model checkpoint or dataset. The
report builder consumes only committed final evidence and figures.
