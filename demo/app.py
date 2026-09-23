"""Offline Streamlit explorer for frozen ChronoPDE V2 and historical V1 evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronopde.v2.publication import load_release_evidence  # noqa: E402

V1_REPORT = ROOT / "reports/final"
PHASE6_REPORT = ROOT / "reports/chronopde_v2/phase6"
FIGURES = ROOT / "figures"


def load_v1_summary() -> dict[str, Any]:
    """Load the historical V1 summary without treating it as current evidence."""

    value = json.loads((V1_REPORT / "final_summary.json").read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


st.set_page_config(page_title="ChronoPDE V2 evidence", page_icon="⏱️", layout="wide")
st.title("ChronoPDE V2")
st.caption("Matched continuous-time neural operators for reaction-diffusion dynamics")

try:
    evidence = load_release_evidence(ROOT)
except (OSError, ValueError, KeyError) as error:
    st.error(f"Frozen Phase 6 evidence could not be validated: {error}")
    st.stop()

st.success(
    "The sealed confirmatory gate passed: DCT had lower rollout and exact-velocity "
    "error in all five frozen seeds."
)

left, middle, right, last = st.columns(4)
left.metric("Median rollout improvement", f"{100 * evidence.median_relative_improvement:.2f}%")
middle.metric(
    "Hierarchical-bootstrap 95% CI",
    f"{100 * evidence.bootstrap_ci_low:.2f}%-{100 * evidence.bootstrap_ci_high:.2f}%",
)
right.metric("DCT directional wins", "5 / 5")
last.metric("DCT divergent rollouts", f"{evidence.dct_divergent_rollouts} / 1,280")

overview, seeds_tab, architecture, provenance, historical = st.tabs(
    ("Confirmed result", "Five frozen seeds", "Matched architecture", "Provenance", "Historical V1")
)

with overview:
    st.image(str(FIGURES / "v2_confirmatory_summary.png"), width="stretch")
    st.subheader("Authorized claim")
    st.write(evidence.permitted_claim)
    st.subheader("Boundary-region wording")
    st.write(evidence.permitted_boundary_claim)
    st.info(
        "Scope is limited to this PDE, 64x64 grid, central parameter range and frozen "
        "training protocol. Sparse-time, OOD and grid-transfer behavior remain unknown."
    )

with seeds_tab:
    rows = pd.DataFrame(
        [
            {
                "seed": row.seed,
                "FFT rollout L2": row.fft_rollout_relative_l2,
                "DCT rollout L2": row.dct_rollout_relative_l2,
                "DCT improvement (%)": 100 * row.relative_improvement,
                "FFT velocity nRMSE": row.fft_velocity_nrmse,
                "DCT velocity nRMSE": row.dct_velocity_nrmse,
                "FFT divergence (%)": 100 * row.fft_divergence_fraction,
                "DCT divergence (%)": 100 * row.dct_divergence_fraction,
            }
            for row in evidence.seed_results
        ]
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.subheader("Rollout relative L2")
    st.bar_chart(rows.set_index("seed")[["FFT rollout L2", "DCT rollout L2"]])
    st.caption(
        f"Five of five directional wins give the predeclared one-sided exact sign-test "
        f"p={evidence.sign_test_p:.5f}."
    )

with architecture:
    st.image(str(FIGURES / "architecture_overview.svg"), width="stretch")
    st.markdown(
        "Both fields have **1,973,657 parameters** and **484,416 active real spectral "
        "degrees of freedom per block**. FFT retains 12x12 complex modes; DCT retains "
        "24x24 real cosine modes. Conditioning, width, block layout, pointwise paths, "
        "initialization and RK4 rollout are shared."
    )
    st.warning(
        "The DCT basis is Neumann-aligned, but the complete neural network does not "
        "enforce the boundary condition."
    )

with provenance:
    st.code(evidence.decision)
    st.json(
        {
            "confirmatory_hdf5_sha256": evidence.dataset_sha256,
            "phase6_results_zip_sha256": evidence.results_sha256,
            "protocol_sha256": evidence.protocol_sha256,
            "confirmatory_trajectories": 256,
            "checkpoint_pairs": 5,
            "rk4_steps_per_interval": 8,
        }
    )
    st.write(
        "The explorer reads committed JSON, CSV and figures only. It performs no network "
        "requests, checkpoint loading, model selection or inference."
    )

with historical:
    v1 = load_v1_summary()
    models = cast(dict[str, dict[str, Any]], v1["models"])
    st.warning(
        "V1 remains a valid controlled negative result. Its spline-target models did not "
        "pass the unchanged 0.01 velocity gate; the unequal-history ID comparison is exploratory."
    )
    a, b, c = st.columns(3)
    a.metric("V1 DCT gate nRMSE", f"{models['chronopde']['historical_gate_median_nrmse']:.5f}")
    b.metric("V1 FFT gate nRMSE", f"{models['fno_ct']['historical_gate_median_nrmse']:.5f}")
    c.metric("Matched-sample DCT wins", f"{v1['dct_wins']} / 16")
    st.image(str(V1_REPORT / "objective_alignment.png"), width="stretch")
    st.caption(
        "V2 is a separately registered study with exact-RHS targets, matched capacity, "
        "fresh identities and a sealed confirmatory set; it does not rewrite V1."
    )
