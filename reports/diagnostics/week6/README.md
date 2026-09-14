# Week 6 gate diagnosis

The two existing GPU result archives are preserved outside Git. Their checksums
and extracted scientific summaries are recorded in `observed_runs.json` so the
evidence remains identifiable without committing model binaries.

The 85-epoch DCT run is exploratory pending the fixed-sample gate diagnosis. The
10,000-step resampled smoke run is a valid failed diagnostic, not a production
checkpoint. Neither result authorizes OOD evaluation.

The smaller failed-smoke archive is stored as one Drive ZIP. Because the Drive
connector timed out twice on the 230 MB exploratory ZIP, that exact file is
stored losslessly as four numbered parts beside a reconstruction command and
both part-level and original-file SHA-256 checksums.

Run `scripts/diagnose_continuous.py` to write the new isolated evidence under
`artifacts/diagnostics/week6/`. The generated suite chooses one of the declared
routes: repair targets, debug mechanics, repair the metric, proceed to Week 7,
or retrain a selected DCT variant.

The first controlled suite completed in 378 seconds. Its target audit passed,
but both backbones missed the `0.01` single-batch velocity nRMSE threshold even
after exceeding the required 1,000x loss reduction. The selected route is a
small shared optimizer/loss sweep; the four-trajectory comparison remains
blocked until both models pass one identical protocol.
