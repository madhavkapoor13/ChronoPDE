# ChronoPDE portfolio summary

**ChronoPDE - Continuous-Time Neural Operators for Reaction-Diffusion Dynamics**

PyTorch | Scientific ML | Neural Operators | DCT/FFT | RK4

- Developed approximately 1.95M-parameter continuous-time spectral neural
  operators for parameter-conditioned 2D reaction-diffusion dynamics,
  integrating learned velocity fields with RK4 across a deterministic
  720-trajectory dataset.
- Designed a Neumann boundary-aware DCT operator and matched FFT control;
  metric-aligned optimization reduced DCT velocity nRMSE from `0.2480` to
  `0.0142`, with DCT lower-error on all 16 matched diagnostic samples while
  transparently reporting that neither model passed the predefined gate.
- Built reproducible evaluation infrastructure including PDE/spline target
  audits, spectral diagnostics, autoregressive U-Net/FNO baselines, checkpoint-
  safe Kaggle workflows, paired bootstrap analysis, and predefined experimental
  stopping criteria.
