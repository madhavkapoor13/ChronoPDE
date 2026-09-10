# Week 5 continuous-time FFT status

The deterministic velocity dataset, FiLM-conditioned CT-FFT model, common
velocity/spectral loss, RK4 validation, checkpoint resumption, frozen ID
evaluation, and Colab commands are implemented and CPU-tested.

The following release gates require Colab CUDA execution and remain pending:

- the four-trajectory CT-FFT overfit gate;
- seed-0 full training;
- validation rollout nRMSE below persistence with zero divergence;
- the frozen 100-trajectory ID report.

Do not create `week5-ct-fft` until those gates pass. Parameter and initial-
condition OOD splits remain unopened.
