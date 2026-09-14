"""Build the seven-page ChronoPDE negative-result technical report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/final"
OUTPUT = ROOT / "output/pdf/chronopde_negative_result_report.pdf"

NAVY = colors.HexColor("#0f172a")
BLUE = colors.HexColor("#2563eb")
PURPLE = colors.HexColor("#7c3aed")
SLATE = colors.HexColor("#475569")
PALE = colors.HexColor("#f1f5f9")
RED = colors.HexColor("#b91c1c")


def load_summary() -> dict[str, Any]:
    value = json.loads((REPORT_DIR / "final_summary.json").read_text(encoding="utf-8"))
    return cast(dict[str, Any], value)


def footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)
    canvas.drawString(18 * mm, 9 * mm, "ChronoPDE | controlled negative result")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def figure(name: str, width: float = 168 * mm) -> Image:
    image = Image(str(REPORT_DIR / name))
    image._restrictSize(width, 112 * mm)
    return image


def build() -> Path:
    summary = load_summary()
    models = cast(dict[str, dict[str, Any]], summary["models"])
    bootstrap = cast(dict[str, Any], summary["bootstrap"])
    interval = cast(list[float], bootstrap["confidence_interval_95"])
    exploratory = cast(dict[str, Any], summary["exploratory_id"])

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=33,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=10 * mm,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=13,
        leading=19,
        textColor=SLATE,
        alignment=TA_CENTER,
    )
    heading = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=NAVY,
        spaceAfter=5 * mm,
    )
    subheading = ParagraphStyle(
        "SubheadingCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=BLUE,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        textColor=NAVY,
        spaceAfter=3 * mm,
    )
    small = ParagraphStyle(
        "SmallCustom",
        parent=body,
        fontSize=8,
        leading=11,
        textColor=SLATE,
    )
    callout = ParagraphStyle(
        "Callout",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=16,
        textColor=RED,
        backColor=colors.HexColor("#fef2f2"),
        borderColor=colors.HexColor("#fecaca"),
        borderWidth=0.7,
        borderPadding=8,
        spaceBefore=4 * mm,
        spaceAfter=5 * mm,
    )

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="ChronoPDE: A Controlled Negative Result",
        author="Madhav Kapoor",
    )
    story: list[Any] = []

    story.extend(
        [
            Spacer(1, 32 * mm),
            Paragraph("ChronoPDE", title),
            Paragraph(
                "A controlled study of boundary-aware continuous-time neural operators", subtitle
            ),
            Spacer(1, 14 * mm),
            Paragraph(
                "A reproducible PyTorch investigation of FFT and DCT neural operators for "
                "parameterized two-species reaction-diffusion dynamics.",
                subtitle,
            ),
            Spacer(1, 18 * mm),
            Paragraph(
                "Outcome: the main Week 6 gate did not pass. Objective alignment improved both "
                "models sharply and DCT was lower-error on all 16 matched diagnostic samples, "
                "but neither reached the predeclared 0.01 velocity-nRMSE threshold.",
                callout,
            ),
            Spacer(1, 20 * mm),
            Paragraph("Madhav Kapoor | September 2026", subtitle),
            Paragraph("Negative-result recovery release", small),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            Paragraph("1. Problem and contribution", heading),
            Paragraph(
                "ChronoPDE asks whether a cosine spectral backbone aligned with homogeneous "
                "Neumann boundaries improves a continuous-time learned velocity field relative "
                "to a parameter-matched Fourier control. Both vector fields are conditioned on "
                "physical parameters and time, and both are integrated with RK4.",
                body,
            ),
            Paragraph("Implemented system", subheading),
            Paragraph(
                "The repository contains an independently implemented reaction-diffusion "
                "simulator, a deterministic 720-trajectory 64 x 64 dataset, temporal masks, "
                "orthonormal DCT transforms, quintic spline targets, fixed-step integrators, "
                "autoregressive U-Net and FFT-FNO baselines, and approximately 1.95M-parameter "
                "continuous-time FFT and DCT models.",
                body,
            ),
            Paragraph("Original hypotheses", subheading),
            Spacer(1, 2 * mm),
            Paragraph(
                "H1: continuous-time velocity learning improves behavior with sparse or irregular "
                "observations. H2: a DCT backbone improves boundary and spectral behavior under "
                "Neumann conditions. The execution plan required a four-trajectory gate before "
                "sparse-time or OOD testing.",
                body,
            ),
            Paragraph("Recovery contribution", subheading),
            Paragraph(
                "After the gate failed, the project followed its stop rule. It audited targets, "
                "tested matched optimizer and loss controls, corrected an unrepresentative batch, "
                "and aligned the objective with the gate metric. The resulting evidence separates "
                "dataset validity, objective mismatch, architecture behavior, and unsupported "
                "claims.",
                body,
            ),
            Table(
                [
                    ["Component", "Frozen choice"],
                    ["Grid and state", "64 x 64; two channels; no-flow boundary"],
                    ["Models", "CT-FFT 1,973,657; DCT 1,951,125 parameters"],
                    ["Diagnostic batch", "4 trajectories x intervals 0, 33, 66, 99"],
                    ["Optimization", "AdamW; lr 3e-4; zero decay; 5,000 steps"],
                    ["Gate", ">=1,000x loss reduction and median nRMSE <=0.01"],
                ],
                colWidths=(42 * mm, 118 * mm),
                style=[
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, -1), PALE),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("LEADING", (0, 0), (-1, -1), 12),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ],
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            Paragraph("2. Protocol and claim discipline", heading),
            Paragraph(
                "The diagnostic protocol was fixed before inspection: seed 0; the same 16 sample "
                "identities for both backbones; gamma 0; batch size 16; constant learning rate "
                "3e-4; no weight decay; evaluation every 100 steps; and a maximum of 5,000 steps.",
                body,
            ),
            Paragraph("Metric contract", subheading),
            Paragraph(
                "The aligned objective is the mean across samples of squared full-field error "
                "divided by target energy. The gate uses per-sample full-field velocity nRMSE. "
                "Historical code used torch.median, which returns the lower middle observation for "
                "16 values. This report preserves that decision and also states the conventional "
                "interpolated median; neither definition changes the outcome.",
                body,
            ),
            Paragraph("Evidence chain", subheading),
            Paragraph(
                "Seven archives are identified by SHA-256 and source commit. Only lightweight "
                "JSON, CSV, configuration, and image evidence is committed. Checkpoints, datasets, "
                "nested repositories, and ZIP files remain outside version control.",
                body,
            ),
            Paragraph("Permitted claims", subheading),
            Paragraph(
                "The target audit passed; alignment improved both models; and DCT was lower-error "
                "on every matched sample in this diagnostic. The analysis does not claim that DCT "
                "generalizes better, that ChronoPDE passed Week 6, or that sparse-time and OOD "
                "performance was validated.",
                body,
            ),
            Paragraph("Stop rule", subheading),
            Paragraph(
                "Because both models missed the registered threshold after the complete budget, "
                "the selected route is stop_and_document_model_or_conditioning_limitation. No "
                "10,000-step comparison, architecture variant, production retraining, or OOD run "
                "is treated as authorized by this experiment.",
                callout,
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            Paragraph("3. Diagnostic sequence", heading),
            Paragraph(
                "The initial resampled smoke run reduced loss without reaching the desired "
                "velocity or rollout behavior. A CPU target audit then tested whether the spline "
                "derivative, physical time scaling, spectral truncation, or normalization contract "
                "was corrupt.",
                body,
            ),
            Paragraph("Target audit", subheading),
            Paragraph(
                f"All {summary['target_audit']['samples']} fixed samples were finite. Median "
                "spline-to-PDE velocity nRMSE was "
                f"{summary['target_audit']['median_spline_pde_nrmse']:.5f}, far below the 0.15 "
                "repair threshold. Twelve DCT modes retained about 79.2% of target energy, "
                "while 20 modes retained about 97.4%.",
                body,
            ),
            Paragraph("Matched mechanics and representative sampling", subheading),
            Paragraph(
                "Shared learning-rate and loss controls left FFT and DCT closely matched and below "
                "the required absolute accuracy. The first batch was then found to contain only "
                "the "
                "first trajectory. A balanced follow-up selected intervals 0, 33, 66, and 99 from "
                "each of four trajectories. Both backbones again failed despite large global-loss "
                "reductions: best velocity nRMSE was 0.1205 for FFT and 0.2480 for DCT.",
                body,
            ),
            Paragraph("Why align the objective?", subheading),
            Paragraph(
                "Global MSE gives high-energy examples greater influence, while the gate weights "
                "each "
                "sample through its own target energy. The production auxiliary spectral term also "
                "measures only the lowest 12 x 12 modes. The final diagnostic changed only its "
                "objective to full-field relative error, leaving production defaults untouched.",
                body,
            ),
            figure("objective_alignment.png", 145 * mm),
            Paragraph(
                "Figure 1. Objective alignment improved both models sharply, but both bars remain "
                "above the unchanged 0.01 gate.",
                small,
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            Paragraph("4. Loss-alignment result", heading),
            Table(
                [
                    ["Model", "Best step", "Loss reduction", "Gate nRMSE", "Decision"],
                    [
                        "CT-FFT",
                        str(models["fno_ct"]["best_eligible_step"]),
                        f"{models['fno_ct']['loss_reduction']:.0f}x",
                        f"{models['fno_ct']['historical_gate_median_nrmse']:.5f}",
                        "Fail",
                    ],
                    [
                        "ChronoPDE DCT",
                        str(models["chronopde"]["best_eligible_step"]),
                        f"{models['chronopde']['loss_reduction']:.0f}x",
                        f"{models['chronopde']['historical_gate_median_nrmse']:.5f}",
                        "Fail",
                    ],
                ],
                colWidths=(43 * mm, 26 * mm, 34 * mm, 30 * mm, 25 * mm),
                style=[
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, -1), PALE),
                    ("TEXTCOLOR", (-1, 1), (-1, -1), RED),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (-1, 1), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ],
            ),
            Spacer(1, 5 * mm),
            figure("loss_alignment_convergence.png"),
            Paragraph(
                "Figure 2. Both models exceed the loss-reduction requirement, but their selected "
                "velocity errors remain above the gate. FFT's best eligible step is 4100; DCT's "
                "is 5000.",
                small,
            ),
            Paragraph(
                "The conventional medians are "
                f"{models['fno_ct']['conventional_sample_median_nrmse']:.5f} for FFT and "
                f"{models['chronopde']['conventional_sample_median_nrmse']:.5f} for DCT. Reporting "
                "that distinction improves metric transparency but does not convert either result "
                "into a pass.",
                body,
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            Paragraph("5. Matched sample analysis", heading),
            figure("paired_model_comparison.png", 137 * mm),
            Paragraph(
                "Figure 3. Every marker lies below the equality line: DCT has lower velocity nRMSE "
                "for all 16 matched trajectory-interval identities at the selected checkpoints.",
                small,
            ),
            Paragraph(
                f"The paired mean DCT-minus-FFT difference is "
                f"{bootstrap['paired_mean_difference_dct_minus_fft']:.5f}. A deterministic "
                f"10,000-resample paired bootstrap gives a 95% interval of "
                f"[{interval[0]:.5f}, {interval[1]:.5f}]. The interval describes this selected "
                "diagnostic batch; it is not a population-level or multi-seed generalization "
                "result.",
                body,
            ),
            Paragraph("Channel behavior", subheading),
            Paragraph(
                f"At the selected DCT step, the historical channel medians were "
                f"{models['chronopde']['velocity_nrmse_u']:.5f} for u and "
                f"{models['chronopde']['velocity_nrmse_v']:.5f} for v. FFT recorded "
                f"{models['fno_ct']['velocity_nrmse_u']:.5f} and "
                f"{models['fno_ct']['velocity_nrmse_v']:.5f}. Late intervals remained among the "
                "hardest examples, motivating future work on conditioning and target balancing.",
                body,
            ),
        ]
    )
    story.append(PageBreak())

    story.extend(
        [
            Paragraph("6. Context, limitations, and conclusion", heading),
            figure("conditioning_analysis.png", 137 * mm),
            Paragraph(
                "Figure 4. Diagnostic error varies with physical time and target energy.", small
            ),
            Paragraph("Exploratory ID context", subheading),
            Paragraph(
                f"The 100-trajectory exploratory evaluations were stable and beat persistence. "
                f"CT-FFT recorded final nRMSE {exploratory['fno_ct']['median_final_nrmse']:.4f}; "
                f"DCT recorded {exploratory['chronopde']['median_final_nrmse']:.4f}. U-Net AR and "
                f"FFT-FNO AR recorded {exploratory['unet_ar']['median_final_nrmse']:.4f} and "
                f"{exploratory['fno_ar']['median_final_nrmse']:.4f}. These values are context "
                "only: "
                "the continuous-time runs used unequal histories, and the original AR training "
                "summaries also recorded unmet gates.",
                body,
            ),
            Paragraph("Conclusion", subheading),
            Paragraph(
                "The project does not establish the original headline claim. It does establish a "
                "reproducible failure-analysis result: the target construction is numerically "
                "consistent, global objectives can hide large per-sample relative errors, and the "
                "boundary-aware DCT model responds favorably to an aligned loss without clearing a "
                "strict memorization threshold. This is a scientifically meaningful outcome "
                "because "
                "the stopping rule, failed threshold, and unsupported experiments remain visible.",
                body,
            ),
            Paragraph("References", subheading),
            Paragraph(
                "[1] Hou, Huang, and Perdikaris. CFO: Learning Continuous-Time PDE Dynamics via "
                "Flow-Matched Neural Operators. ICLR 2026.<br/>"
                "[2] Takamoto et al. PDEBench: An Extensive Benchmark for Scientific Machine "
                "Learning. NeurIPS Datasets and Benchmarks, 2022.<br/>"
                "[3] Li et al. Fourier Neural Operator for Parametric Partial Differential "
                "Equations. ICLR 2021.",
                small,
            ),
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
