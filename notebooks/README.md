# Notebooks

`week4_baselines_colab.ipynb` covers the original autoregressive workflows.
`week6_chronopde_kaggle.ipynb` runs the boundary-aware model in Kaggle with a
smoke gate, epoch-level checkpointing, a safe wall-time cutoff, resumable ZIP
packaging, and conditional frozen ID evaluation. Notebook outputs are cleared;
reusable logic belongs in the `chronopde` package rather than notebook-only cells.
