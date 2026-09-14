# ChronoPDE research artifact model card

## Intended use

ChronoPDE and CT-FFT are research implementations for studying continuous-time
neural operators on a controlled two-species reaction-diffusion dataset. The
checkpoints support reproducibility and diagnostic analysis; they are not
validated simulators for scientific, operational, or safety-critical use.

## Training and evaluation status

- ChronoPDE DCT: approximately 1.95 million parameters; exploratory seed-0 run
  stopped after 85 epochs by validation early stopping.
- CT-FFT: approximately 1.97 million parameters; exploratory seed-0 run ran for
  150 epochs.
- Both exploratory ID evaluations were stable and beat persistence.
- Neither model passed the predeclared Week 6 fixed-sample velocity-nRMSE gate.
- Sparse-time, hidden-time, parameter-OOD, and initial-condition-OOD claims were
  not evaluated after the stop rule fired.

## Valid diagnostic result

On 16 fixed samples covering four trajectories and four time intervals, the
full-field relative objective produced historical gate nRMSE values of `0.01421`
for DCT and `0.01895` for FFT. DCT was lower on all 16 paired identities, but
both exceeded the required `0.01` threshold. This is a diagnostic result rather
than evidence of generalization.

## Limitations

- The aligned diagnostic has one development seed and only 16 fixed samples.
- The full ID runs have unequal training histories and are exploratory.
- The DCT implementation was slower in its recorded session, but cross-session
  latency is not a controlled comparison.
- The learned vector fields remain sensitive to loss scaling, target energy,
  channel balance, and late physical time.
- Checkpoints must not be used outside the frozen dataset ranges without new
  validation.

## Future work

Any continuation must be registered as a new study. Candidate hypotheses are
target whitening, channel-balanced relative loss, residual vector-field
parameterization, and replicated aligned training. The original failed gate
must remain in the record.
