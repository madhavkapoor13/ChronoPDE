# ChronoPDE V2 Phase 6 — sealed confirmation

Phase 6 completed the one-shot confirmatory stage. It generated the 256
identities sealed in Phase 2 and evaluated all five frozen FFT–DCT checkpoint
pairs without training, checkpoint selection, development-data access, OOD
evaluation or changed-setting repeats.

## Result

The strict confirmatory gate passed:

- DCT had lower rollout relative L2 and exact-velocity nRMSE in `5/5` seeds.
- Median paired rollout improvement was `21.14%`.
- The fixed 20,000-resample hierarchical-bootstrap 95% interval was
  `[16.35%, 27.72%]`.
- DCT had zero divergence across all `1,280` seed–trajectory rollouts and beat
  persistence in every seed.
- The predeclared one-sided exact sign-test result was `p=0.03125`.
- Boundary-strip and first-interior normal-derivative errors were lower in all
  five seeds.

The authorized claim remains scoped to this reaction-diffusion PDE, `64×64`
grid, central parameter range and frozen training protocol. The result does not
establish boundary enforcement, physical wall-flux correctness, sparse-time
performance or OOD generalization.

## Evidence

- Confirmatory dataset SHA-256:
  `0a432d24d34605274a4e4bfa06d97831602062b016b44f4d5b81f91cc14e64e3`
- Phase 6 results ZIP SHA-256:
  `151e0fb628735b21f7d63a159abfa9695253f2a210147c29555ced06d587cfc1`
- Protocol SHA-256:
  `7a869166876c6bd2a6143bd334a71128f692b7dc77cc9554d69e510b09c26464`
- [Decision report](decision_report.json)
- [Paired seed results](paired_seed_results.csv)
- [Protocol snapshot](protocol_snapshot.json)

The CPU generation and T4x2 evaluation notebooks remain the canonical
reproduction workflows. Large datasets, checkpoints and result archives remain
outside Git; only the lightweight frozen evidence is committed here.
