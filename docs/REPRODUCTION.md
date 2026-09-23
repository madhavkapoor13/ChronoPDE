# ChronoPDE V2 reproduction guide

ChronoPDE targets Python 3.11. The release provides a lightweight evidence
verification path and a separate private-artifact path for full regeneration.

## Environment and lightweight verification

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,demo,report]"

python scripts/chronopde_v2.py phase1 --format json
python scripts/chronopde_v2.py phase2 --format json
pytest -q
ruff check .
mypy chronopde
git diff --check
```

This path uses committed lightweight evidence. It reproduces the V1 diagnostic,
validates the V2 numerical and provenance contracts, and checks that the public
release numbers agree with the frozen Phase 6 decision. It does not train a
model or require CUDA.

The confirmatory release inputs are pinned by SHA-256 in
`reports/chronopde_v2/release/artifact_manifest.json`. The final decision and
paired seed values are under `reports/chronopde_v2/phase6/`.

## Full private-artifact workflow

Large data and checkpoints are deliberately excluded from Git. Recover the
private Phase 3 development HDF5 and frozen Phase 5 ZIP from the corresponding
private Kaggle datasets before running these notebooks:

1. `notebooks/chronopde_v2_phase5_kaggle.ipynb` - five matched training seeds.
2. `notebooks/chronopde_v2_phase6_generate_kaggle.ipynb` - generate the 256
   identities sealed in Phase 2.
3. `notebooks/chronopde_v2_phase6_evaluate_kaggle.ipynb` - evaluate all ten
   frozen checkpoints once and collect the decision.

Equivalent Phase 6 commands are:

```bash
python scripts/chronopde_v2.py phase6 generate \
  --phase5-archive chronopde_v2_phase5_outputs.zip

python scripts/chronopde_v2.py phase6 evaluate \
  --phase5-archive chronopde_v2_phase5_outputs.zip \
  --data-path chronopde_v2_confirmatory.h5 \
  --model dct --seed 0 --device cuda

python scripts/chronopde_v2.py phase6 collect \
  --phase5-archive chronopde_v2_phase5_outputs.zip \
  --data-path chronopde_v2_confirmatory.h5
```

Repeat `evaluate` for both models and seeds 0-4 before `collect`. A corrupt or
incomplete execution permits only an identical rerun; the confirmatory data may
not be used for model selection or retuning.

## Public report and offline explorer

```bash
python scripts/build_v2_confirmatory_report.py
streamlit run demo/app.py
```

Both consume committed evidence only. The historical V1 diagnostic remains
reproducible with `python scripts/reproduce_diagnostic.py --model both`; its
complete original workflow is preserved in the V1 method specification and
the `v0.1.1-portfolio` tag.
