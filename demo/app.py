"""Offline Streamlit explorer for the frozen ChronoPDE Week 6 evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/final"


def load_json(name: str) -> dict[str, Any]:
    value = json.loads((REPORT / name).read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


st.set_page_config(page_title="ChronoPDE evidence", page_icon="⏱️", layout="wide")
st.title("ChronoPDE")
st.caption("A controlled study of boundary-aware continuous-time neural operators")

summary_path = REPORT / "final_summary.json"
if not summary_path.is_file():
    st.error("Final evidence is missing. Run `python scripts/analyze_week6_failure.py`.")
    st.stop()

summary = load_json("final_summary.json")
models = cast(dict[str, dict[str, Any]], summary["models"])
st.warning(
    "Valid negative result: neither continuous-time model passed the unchanged "
    "0.01 Week 6 gate. Sparse-time and OOD claims were not evaluated."
)

left, middle, right, last = st.columns(4)
left.metric("DCT gate nRMSE", f"{models['chronopde']['historical_gate_median_nrmse']:.5f}")
middle.metric("FFT gate nRMSE", f"{models['fno_ct']['historical_gate_median_nrmse']:.5f}")
right.metric("Paired DCT wins", f"{summary['dct_wins']} / 16")
last.metric("Target-audit samples", str(summary["target_audit"]["samples"]))

overview, samples, context, provenance = st.tabs(
    ("Outcome", "Matched samples", "Exploratory ID context", "Provenance")
)

with overview:
    st.subheader("Objective alignment")
    st.image(str(REPORT / "objective_alignment.png"), use_container_width=True)
    st.subheader("Training dynamics")
    st.image(str(REPORT / "loss_alignment_convergence.png"), use_container_width=True)
    st.markdown(
        "The aligned objective improved both models, but the registered gate remained "
        "failed. DCT's within-diagnostic advantage is not presented as a generalization claim."
    )

with samples:
    frame = pd.read_csv(REPORT / "paired_sample_comparison.csv")
    trajectory = st.selectbox(
        "Trajectory", ["all", *sorted(frame["trajectory_id"].unique().tolist())]
    )
    shown = frame if trajectory == "all" else frame[frame["trajectory_id"] == trajectory]
    st.image(str(REPORT / "paired_model_comparison.png"), use_container_width=True)
    st.image(str(REPORT / "conditioning_analysis.png"), use_container_width=True)
    st.dataframe(shown, hide_index=True, use_container_width=True)

with context:
    st.info(
        "These 100-trajectory runs are context only: DCT stopped after 85 epochs while "
        "CT-FFT ran 150 epochs, and the DCT gate did not pass."
    )
    id_results = cast(dict[str, Any], summary["exploratory_id"])
    rows = []
    for model in ("unet_ar", "fno_ar", "fno_ct", "chronopde"):
        result = cast(dict[str, Any], id_results[model])
        rows.append(
            {
                "model": model,
                "median final nRMSE": result["median_final_nrmse"],
                "median relative L2": result["median_relative_l2"],
                "persistence final nRMSE": result["persistence_final_nrmse"],
                "divergence fraction": result["divergence_fraction"],
                "parameters": result["parameter_count"],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with provenance:
    st.code(summary["decision"])
    st.write(
        "Every headline result is regenerated from committed lightweight evidence. "
        "Original archive hashes and Git commits are recorded in the evidence manifest."
    )
    st.json(
        {
            "gate": summary["gate"],
            "bootstrap": summary["bootstrap"],
            "target_audit": summary["target_audit"],
        }
    )
