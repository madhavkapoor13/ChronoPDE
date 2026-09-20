# ChronoPDE V2 Phase 1 — evidence preservation

Phase 1 is an additive provenance and reliability release. It does not train a
model, generate a dataset, alter a numerical method, or create a new scientific
result. The original Week 1–6 artifacts and stopping decision remain frozen.

The machine-readable claim ledger distinguishes verified diagnostics,
exploratory context, unsupported interpretations, contradicted wording, and
experiments that were not run. In particular, the Week 5 four-trajectory gate
is treated as unverified because the committed status documents conflict and no
distinct passing smoke-gate artifact is present.

Run the audit with:

```bash
python scripts/chronopde_v2.py phase1 --format json
```

Pass `--archive-root PATH` to recursively discover renamed ZIP files by content
hash. Missing external archives cause exit code 2; checksum/provenance conflicts
cause exit code 3; claim or study-contract errors cause exit code 4.

Future V2 work must use `artifacts/chronopde_v2/runs/` and must not treat the
legacy ID evaluation as a fresh confirmatory test.
