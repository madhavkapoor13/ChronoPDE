# ChronoPDE research artifact model card

## Intended use

ChronoPDE and CT-FFT are research implementations for studying continuous-time
neural operators on a controlled two-species reaction-diffusion dataset. The
checkpoints support reproducibility and diagnostic analysis; they are not
validated simulators for scientific, operational, or safety-critical use.

## Training and evaluation status

- V2 compares 1,973,657-parameter FFT-12 and DCT-24 models with matched active
  real spectral degrees of freedom, shared conditioning and shared training.
- Five fresh seeds per backbone were selected only on development validation
  velocity error before the confirmatory set was generated.
- All ten frozen checkpoints were evaluated once on 256 sealed trajectories.
- DCT had lower rollout and exact-velocity error in all five seeds, zero
  divergence across all 1,280 rollouts, and a 21.14% median paired rollout
  improvement (95% hierarchical-bootstrap interval 16.35%–27.72%).
- Sparse-time, hidden-time, parameter-OOD and initial-condition-OOD claims have
  not been evaluated in V2.

## Confirmatory result

The strict Phase 6 gate passed in full. DCT beat FFT and persistence in every
seed, had no worse divergence in every seed, and met the predeclared 10%
practical-effect threshold. Five of five directional wins correspond to the
predeclared one-sided exact sign-test result `p=0.03125`. Boundary-strip and
first-interior normal-derivative errors were also lower in all five seeds.

The permitted claim is limited to this reaction-diffusion PDE, `64×64` grid,
central parameter range and frozen protocol. Boundary-region improvement is not
evidence of exact boundary enforcement or physical wall-flux correctness.

## Preserved original diagnostic

On 16 fixed samples covering four trajectories and four time intervals, the
full-field relative objective produced historical gate nRMSE values of `0.01421`
for DCT and `0.01895` for FFT. DCT was lower on all 16 paired identities, but
both exceeded the required `0.01` threshold. This is a diagnostic result rather
than evidence of generalization.

## Qualitative context

The public held-out rollout figure uses frozen checkpoints on `id-0042`, chosen
by a deterministic median-rank rule. Its JSON sidecar records the dataset and
checkpoint hashes, source commits, time indices, and per-channel errors. Because
the FFT and DCT checkpoints have unequal training histories, the figure is
explicitly exploratory and is not used to change the diagnostic decision.

## Limitations

- The V2 confirmation covers five fixed seeds and one PDE family, grid,
  parameter range, objective, optimizer and integration protocol.
- OOD, sparse-time and hidden-time behavior remain unknown.
- The historical V1 checkpoints have unequal training histories and remain
  exploratory; they are separate from the matched V2 confirmation.
- The confirmatory comparison establishes relative performance, not suitability
  as a scientific or operational simulator.
- Checkpoints must not be used outside the frozen dataset ranges without new
  validation.

## Future work

Any continuation must be registered as a new study with new holdout identities.
Candidate directions include sparse-time observations, parameter and
initial-condition OOD evaluation, grid transfer and broader PDE families. The
original failed V1 gate and completed V2 confirmation both remain in the record.
