# Final recovery evidence

Run `python scripts/analyze_week6_failure.py` from the repository root to
regenerate every JSON, CSV, and PNG in this directory from the committed frozen
evidence under `reports/diagnostics/week6/evidence/`.

The authoritative decision is `stop_and_document_model_or_conditioning_limitation`.
The results are a valid negative finding: objective alignment helped both
continuous-time models substantially, and DCT was lower-error on all 16 paired
diagnostic samples, but neither model passed the unchanged `0.01` gate.

For a concise terminal reproduction, run
`python scripts/reproduce_diagnostic.py --model both`. The architecture and
exploratory held-out rollout are published under `figures/`; the latter remains
outside the confirmatory decision because its checkpoints have unequal training
histories.
