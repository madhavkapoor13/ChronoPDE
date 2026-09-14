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
