"""Build the ChronoPDE V2 confirmatory figure and seven-page release report."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronopde.v2.publication import ReleaseEvidence, load_release_evidence  # noqa: E402

FIGURE = ROOT / "figures/v2_confirmatory_summary.png"
OUTPUT = ROOT / "output/pdf/chronopde_v2_confirmatory_report.pdf"


def build_figure(evidence: ReleaseEvidence) -> Path:
    """Render the compact public confirmatory summary."""

    rows = evidence.seed_results
    seeds = np.asarray([row.seed for row in rows])
    fft_rollout = np.asarray([row.fft_rollout_relative_l2 for row in rows])
    dct_rollout = np.asarray([row.dct_rollout_relative_l2 for row in rows])
    improvement = 100 * np.asarray([row.relative_improvement for row in rows])
    fft_divergence = 100 * np.asarray([row.fft_divergence_fraction for row in rows])
    dct_divergence = 100 * np.asarray([row.dct_divergence_fraction for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    width = 0.34
    axes[0].bar(seeds - width / 2, fft_rollout, width, label="FFT", color="#2563eb")
    axes[0].bar(seeds + width / 2, dct_rollout, width, label="DCT", color="#7c3aed")
    axes[0].set(title="Rollout relative L2", xlabel="seed", ylabel="lower is better")
    axes[0].set_xticks(seeds)
    axes[0].legend(frameon=False)

    axes[1].bar(seeds, improvement, color="#0f766e")
    axes[1].axhline(10, color="#b45309", linestyle="--", linewidth=1.5, label="10% gate")
    axes[1].axhspan(
        100 * evidence.bootstrap_ci_low,
        100 * evidence.bootstrap_ci_high,
        color="#99f6e4",
        alpha=0.35,
        label="95% hierarchical CI",
    )
    axes[1].set(title="Paired DCT improvement", xlabel="seed", ylabel="percent")
    axes[1].set_xticks(seeds)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(seeds - width / 2, fft_divergence, width, label="FFT", color="#2563eb")
    axes[2].bar(seeds + width / 2, dct_divergence, width, label="DCT", color="#7c3aed")
    axes[2].set(title="Divergent rollouts", xlabel="seed", ylabel="percent")
    axes[2].set_xticks(seeds)
    axes[2].legend(frameon=False)

    figure.suptitle(
        "ChronoPDE V2 sealed confirmation: five frozen checkpoint pairs, 256 trajectories",
        fontsize=13,
        fontweight="bold",
    )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return FIGURE


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 9 * mm, "ChronoPDE V2 | sealed confirmatory study")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def _table(rows: list[list[str]], widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def build_report(evidence: ReleaseEvidence) -> Path:
    """Build the fixed seven-page V2 technical report."""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleV2",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceAfter=10 * mm,
    )
    heading = ParagraphStyle(
        "HeadingV2",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#4c1d95"),
        spaceAfter=5 * mm,
    )
    subheading = ParagraphStyle(
        "SubheadingV2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
    )
    body = ParagraphStyle(
        "BodyV2",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=3.5 * mm,
    )
    callout = ParagraphStyle(
        "CalloutV2",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#4c1d95"),
        alignment=TA_CENTER,
        borderColor=colors.HexColor("#c4b5fd"),
        borderWidth=1,
        borderPadding=10,
        backColor=colors.HexColor("#f5f3ff"),
        spaceAfter=5 * mm,
    )
    bullet = ParagraphStyle("BulletV2", parent=body, leftIndent=5 * mm, firstLineIndent=-3 * mm)

    story: list[Any] = []
    story.extend(
        [
            Spacer(1, 20 * mm),
            Paragraph("ChronoPDE V2", title),
            Paragraph(
                "A sealed five-seed comparison of matched cosine and Fourier neural operators",
                callout,
            ),
            Spacer(1, 8 * mm),
            Paragraph("Confirmatory outcome", heading),
            Paragraph(
                "The predeclared Phase 6 gate passed. Across five frozen checkpoint pairs "
                "and 256 sealed trajectories, the DCT operator had lower rollout and exact-"
                "velocity error in every seed.",
                body,
            ),
            _table(
                [
                    ["Measure", "Frozen result"],
                    ["Median paired rollout improvement", "21.14%"],
                    ["Hierarchical-bootstrap 95% interval", "16.35%-27.72%"],
                    ["Directional wins", "5/5 (one-sided p=0.03125)"],
                    ["DCT divergence", "0 / 1,280 rollouts"],
                    ["FFT divergence", "255 / 1,280 rollouts"],
                ],
                [92 * mm, 72 * mm],
            ),
            Spacer(1, 6 * mm),
            Paragraph(
                "Claim boundary: this result applies to the frozen reaction-diffusion PDE, "
                "64 x 64 grid, central parameter range and training protocol. It is not a "
                "claim of boundary enforcement, wall-flux correctness, OOD generalization "
                "or universal DCT superiority.",
                body,
            ),
            PageBreak(),
        ]
    )

    pages = [
        (
            "1. From a negative result to a fresh study",
            [
                "The original V1 study used spline-derived velocity targets and unequal-history "
                "continuous-time checkpoints. Its unchanged Week 6 gate did not pass. That "
                "negative result remains preserved under the v0.1 tags.",
                "V2 did not retune on the exposed V1 test set. It introduced exact discrete-RHS "
                "supervision, capacity matching, fresh development identities, five training "
                "seeds and a new confirmatory split whose states were generated only after all "
                "ten checkpoints and evaluation settings were frozen.",
                "This separation makes the positive V2 outcome additive rather than a rewrite of "
                "the historical result.",
            ],
        ),
        (
            "2. Numerical ground truth",
            [
                "The supervised target is the exact finite-volume simulator RHS per unit physical "
                "time. Phase 2 checked the cell-centred Neumann Laplacian for symmetry, negative "
                "semidefiniteness, constant nullspace, diffusion mass conservation and the "
                "discrete Green identity.",
                "Sparse-matrix, reflected-stencil and DCT-spectral operators agreed within frozen "
                "float64 tolerances. Float32 storage and a tight DOP853 convergence repeat also "
                "passed. Phase 3 generated 512 training and 128 validation trajectories; the "
                "normalizer used training data only.",
                "The DCT basis is aligned with the discrete Neumann operator. The full network is "
                "not boundary enforcing because pointwise paths, nonlinearities, FiLM, lifting "
                "and projection can alter boundary values.",
            ],
        ),
        (
            "3. Matched operator comparison",
            [
                "Both vector fields use width 29, four blocks, the same FiLM conditioner, "
                "pointwise paths, lifting, projection, initialization and RK4 rollout. Each has "
                "1,973,657 real trainable parameters and 484,416 active real spectral degrees "
                "of freedom per block.",
                "The FFT control retains 12 x 12 complex Fourier modes. ChronoPDE retains 24 x 24 "
                "real orthonormal cosine modes. The 4.35% physical-cutoff mismatch is within the "
                "predeclared 10% tolerance.",
                "Consequently, the comparison isolates the basis-aligned spatial inductive bias "
                "far more tightly than the V1 models.",
            ],
        ),
        (
            "4. Development and sealed protocol",
            [
                "Seeds 0-4 were trained for each backbone with identical exact-RHS objective, "
                "optimizer, budget and development data. Checkpoints were selected only by fixed "
                "validation velocity nRMSE, with validation relative loss as tie-breaker.",
                "The 256 confirmatory identities were frozen in Phase 2 and remained ungenerated "
                "through model development. Phase 6 generated them once, verified 32 exact-RHS "
                "samples and 12 tight-solver repeats, then evaluated all frozen checkpoints once.",
                "Finite predictions above magnitude 10 remained in rollout errors and were also "
                "marked divergent. Nonfinite predictions were unavailable and treated as infinite "
                "for the architecture decision.",
            ],
        ),
    ]
    for index, (page_title, paragraphs) in enumerate(pages):
        story.append(Paragraph(page_title, heading))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, body))
        if page_title.startswith("3."):
            story.append(
                _table(
                    [
                        ["Component", "FFT control", "ChronoPDE DCT"],
                        ["Retained modes", "12 x 12", "24 x 24"],
                        ["Parameters", "1,973,657", "1,973,657"],
                        ["Real spectral DOF/block", "484,416", "484,416"],
                        ["Spatial assumption", "periodic", "Neumann-aligned"],
                    ],
                    [55 * mm, 52 * mm, 57 * mm],
                )
            )
        if index in (1, 3):
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 7 * mm))

    story.extend(
        [
            Paragraph("5. Confirmatory results", heading),
            Image(str(FIGURE), width=174 * mm, height=53 * mm),
            Spacer(1, 4 * mm),
            _table(
                [
                    [
                        "Seed",
                        "FFT rollout",
                        "DCT rollout",
                        "DCT improvement",
                        "FFT div.",
                        "DCT div.",
                    ],
                    *[
                        [
                            str(row.seed),
                            f"{row.fft_rollout_relative_l2:.4f}",
                            f"{row.dct_rollout_relative_l2:.4f}",
                            f"{100 * row.relative_improvement:.2f}%",
                            f"{100 * row.fft_divergence_fraction:.2f}%",
                            f"{100 * row.dct_divergence_fraction:.2f}%",
                        ]
                        for row in evidence.seed_results
                    ],
                ],
                [17 * mm, 28 * mm, 28 * mm, 34 * mm, 27 * mm, 27 * mm],
            ),
            Spacer(1, 4 * mm),
            Paragraph(
                "Every seed cleared the directional rollout and velocity requirements. The median "
                "paired improvement exceeded the 10% practical-effect gate, and the bootstrap "
                "interval remained wholly positive.",
                body,
            ),
            PageBreak(),
            Paragraph("6. Stability, boundary attribution and provenance", heading),
            Paragraph(
                "DCT produced zero divergent rollouts. FFT divergence was concentrated in seed 0 "
                "but appeared in four of five seeds overall. Phase 4B had already shown that the "
                "seed-0 FFT instability persisted under RK4 refinement, supporting a learned-"
                "vector-field explanation rather than an integration-resolution artifact.",
                body,
            ),
            Paragraph(
                "DCT also had lower boundary-strip and first-interior normal-derivative error in "
                "all five seeds. This authorizes the phrase lower boundary-region error only; the "
                "first-interior metric is not physical wall flux.",
                body,
            ),
            Paragraph("Frozen provenance", subheading),
            *[
                Paragraph(f"- {line}", bullet)
                for line in (
                    f"Confirmatory HDF5 SHA-256: {evidence.dataset_sha256}",
                    f"Phase 6 results ZIP SHA-256: {evidence.results_sha256}",
                    f"Protocol SHA-256: {evidence.protocol_sha256}",
                    "Decision evidence: reports/chronopde_v2/phase6/decision_report.json",
                    "Paired seed evidence: reports/chronopde_v2/phase6/paired_seed_results.csv",
                )
            ],
            Paragraph("Limitations", subheading),
            Paragraph(
                "The study covers one PDE family, grid, parameter range, objective, optimizer, "
                "five fixed seeds and one rollout protocol. Sparse time, OOD, grid transfer and "
                "other PDEs require new registered studies and new holdouts. The Phase 6 set must "
                "not be reused for retuning.",
                body,
            ),
        ]
    )

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=16 * mm,
        title="ChronoPDE V2 Confirmatory Report",
        author="Madhav Kapoor",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT


def main() -> None:
    evidence = load_release_evidence(ROOT)
    build_figure(evidence)
    output = build_report(evidence)
    print(output)


if __name__ == "__main__":
    main()
