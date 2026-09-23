# Notebook index

## Canonical workflow

The active V2 workflows are:

| Notebook | Purpose |
| --- | --- |
| `chronopde_v2_phase4_kaggle.ipynb` | Completed matched feasibility training |
| `chronopde_v2_phase4b_kaggle.ipynb` | Evaluation-only RK4 integration audit |
| `chronopde_v2_phase5_kaggle.ipynb` | Fresh five-seed matched development study |
| `chronopde_v2_phase6_generate_kaggle.ipynb` | Generate sealed confirmatory trajectories |
| `chronopde_v2_phase6_evaluate_kaggle.ipynb` | One-shot frozen confirmatory evaluation |

Phase 6A generated the sealed confirmatory dataset on CPU, and Phase 6B
completed the one-shot five-seed evaluation on T4x2. The strict gate passed.
These notebooks are now frozen reproduction workflows; rerunning with altered
settings is not part of the completed study.

## Frozen V1 workflow

`week6_loss_alignment_kaggle.ipynb` is the final reproducible Kaggle workflow.
It verifies the dataset, runs the matched CT-FFT and ChronoPDE DCT
loss-alignment diagnostic, displays progress, and packages partial or completed
outputs safely.

The final run selected
`stop_and_document_model_or_conditioning_limitation`. The notebook is retained
for provenance and should not be interpreted as authorization to reopen the
stopped experiment.

## Archived workflows

The `archive/` directory preserves earlier Colab and Kaggle stages:

| Notebook | Historical purpose |
| --- | --- |
| `week4_baselines_colab.ipynb` | Autoregressive U-Net and FFT-FNO training |
| `week6_chronopde_kaggle.ipynb` | Exploratory full DCT training and resume |
| `week6_chronopde_smoke_kaggle.ipynb` | Original resampled smoke gate |
| `week6_gate_diagnostics_kaggle.ipynb` | Target audit and initial fixed controls |
| `week6_mechanics_diagnostics_kaggle.ipynb` | Optimizer/loss mechanics sweep |
| `week6_velocity_audit_kaggle.ipynb` | CPU checkpoint and metric audit |
| `week6_balanced_batch_kaggle.ipynb` | Corrected four-trajectory batch |

These files document the decision trail. They are not current entry points.
