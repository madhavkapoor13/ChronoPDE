# Week 4 baseline status

The autoregressive U-Net and FFT-FNO implementation, datasets, training loop,
checkpoint resumption, evaluation pipeline, and Colab workflow are complete.

The following empirical gates remain pending because the development Mac has no
available CUDA or MPS device:

- four-trajectory overfit for `unet_ar`;
- four-trajectory overfit for `fno_ar`;
- full seed-0 training for both models;
- frozen ID-rollout comparison and figures.

Run `notebooks/week4_baselines_colab.ipynb` with the exact Week 3 dataset. Do not
create the `week4-ar-baselines` tag until both smoke checks and full runs pass.
