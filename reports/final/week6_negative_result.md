# Week 6 decision: valid negative result

## Decision

The fixed 5,000-step loss-alignment experiment completed, but neither matched
continuous-time model reached the unchanged `0.01` velocity-nRMSE gate. The
frozen route is `stop_and_document_model_or_conditioning_limitation`. Production retraining, four-trajectory
comparison, sparse-time experiments, and OOD evaluation are not authorized.

## Gate result

| Model | Best eligible step | Loss reduction | Historical gate nRMSE | Result |
| --- | ---: | ---: | ---: | --- |
| CT-FFT | 4100 | 188684x | 0.01895 | Fail |
| ChronoPDE DCT | 5000 | 7980x | 0.01421 | Fail |

The historical gate used `torch.median`, which returns the lower middle value
for an even sample count. Conventional medians are
`0.01950` for CT-FFT and
`0.01514` for DCT. Both
definitions preserve the failed decision.

## What was learned

- The 400-sample target audit passed; median spline/PDE nRMSE was
  `0.00151`.
- Full-field relative loss reduced the balanced error by
  `84.3%`
  for CT-FFT and `94.3%`
  for DCT.
- DCT had lower error on all `16` paired samples. The paired
  mean DCT-minus-FFT difference was `-0.00503`
  with a deterministic bootstrap 95% interval of
  `[-0.00668, -0.00354]`. This is a diagnostic result, not a
  generalization claim.
- The exploratory 100-trajectory ID runs were stable and beat persistence, but
  used unequal training histories. Their final nRMSE values were
  `0.4219` for CT-FFT and
  `0.4668` for DCT and are retained
  only as context.

## Claim boundary

The project may claim a reproducible objective-metric alignment finding and a
matched diagnostic DCT advantage. It may not claim that ChronoPDE passed Week 6,
is generally superior to FFT, or has validated sparse-time or OOD performance.
