# Notebooks

`week4_baselines_colab.ipynb` covers the original autoregressive workflows.
`week6_chronopde_smoke_kaggle.ipynb` runs only the corrected strict smoke gate
and packages its diagnostic without launching full training.
`week6_chronopde_kaggle.ipynb` runs the boundary-aware model in Kaggle with a
smoke gate, epoch-level checkpointing, a safe wall-time cutoff, resumable ZIP
packaging, and conditional frozen ID evaluation. Notebook outputs are cleared;
reusable logic belongs in the `chronopde` package rather than notebook-only cells.

`week6_gate_diagnostics_kaggle.ipynb` is the current Week 6 entry point. It runs
the CPU target audit and the controlled fixed-sample DCT/CT-FFT diagnostics,
then always creates `chronopde_week6_gate_diagnostics.zip`. It does not read or
write production checkpoints and it never evaluates OOD data.

`week6_mechanics_diagnostics_kaggle.ipynb` is the short follow-up when the first
suite routes to `debug_model_loss_optimizer`. It tests only the fixed 16-example
batch and packages a best-step-aware optimizer/loss diagnosis.

`week6_velocity_audit_kaggle.ipynb` is the CPU-only follow-up for completed
mechanics checkpoints. It replays `best.pt` and `last.pt`, compares median and
pooled metrics without changing the gate, records exact sample/checkpoint
provenance, and always packages its report. No GPU or training is used.

`week6_balanced_batch_kaggle.ipynb` runs the next bounded GPU decision: four
fixed intervals from each of the four training trajectories for both DCT and
CT-FFT. It packages checkpoints and partial results on failure and does not
launch the four-trajectory or OOD phases automatically.

`week6_loss_alignment_kaggle.ipynb` is the follow-up after the balanced global
objective fails. It trains both backbones on the same 16 samples with the fixed
full-field relative objective, preserves the original gate, and always packages
the resulting checkpoints and metrics without launching later phases.
