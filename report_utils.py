"""Diagnostic report generation (PDF and CSV) for VibeGuard Cloud.

Generates downloadable, self-contained PDF and CSV reports from a completed
signal diagnosis without altering model weights, scores, or preprocessing,
and without exposing any private cloud identifiers or credentials.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from data_utils import CLASS_DISPLAY_NAMES, CLASS_NAMES, NOMINAL_SAMPLING_HZ, WINDOW_SIZE


def build_diagnostic_csv_bytes(diag: dict[str, Any]) -> bytes:
    """Create a window-by-window + summary CSV export from a diagnosis state dict."""
    pred = diag["pred_result"]
    sig_meta = diag["sig_meta"]
    seg_meta = diag["seg_meta"]
    source_name = str(diag["source_name"])
    probs = np.asarray(pred["window_probabilities"], dtype=np.float64)
    win_preds = np.asarray(pred["window_predictions"], dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for idx in range(probs.shape[0]):
        cls_idx = int(win_preds[idx])
        cls_key = CLASS_NAMES[cls_idx]
        rows.append(
            {
                "source_file": source_name,
                "window_index": idx,
                "window_size_samples": WINDOW_SIZE,
                "channel_key": sig_meta.get("channel_key", "csv_column_0"),
                "nominal_sampling_hz": sig_meta.get("sampling_hz_nominal", NOMINAL_SAMPLING_HZ),
                "window_predicted_key": cls_key,
                "window_predicted_condition": CLASS_DISPLAY_NAMES[cls_key],
                "window_top_uncalibrated_score": round(float(probs[idx, cls_idx]), 6),
                "score_normal": round(float(probs[idx, 0]), 6),
                "score_inner": round(float(probs[idx, 1]), 6),
                "score_ball": round(float(probs[idx, 2]), 6),
                "score_outer": round(float(probs[idx, 3]), 6),
                "recording_predicted_condition": pred["predicted_display"],
                "recording_top_uncalibrated_score": round(float(pred["top_uncalibrated_score"]), 6),
                "recording_segment_agreement": round(float(pred["segment_agreement"]), 6),
                "manual_review_required": bool(pred["manual_review_required"]),
                "total_windows_extracted": int(seg_meta["num_windows"]),
            }
        )
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def _render_report_chart_png(
    raw_signal: np.ndarray,
    first_window_norm: np.ndarray,
    mean_scores: dict[str, float],
    sampling_hz: int = NOMINAL_SAMPLING_HZ,
) -> bytes:
    """Render a compact, high-contrast 4-panel Matplotlib summary figure for PDF embedding."""
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.6), dpi=150, facecolor="#030407")
    ax_raw, ax_norm, ax_fft, ax_bar = axes.flatten()
    for ax in (ax_raw, ax_norm, ax_fft, ax_bar):
        ax.set_facecolor("#0B1220")
        ax.tick_params(colors="#9DAAC2", labelsize=7)
        ax.grid(color="#243248", linestyle=":", linewidth=0.6, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#243248")

    n_plot = min(len(raw_signal), 4096)
    t_sec = np.arange(n_plot) / float(sampling_hz)
    ax_raw.plot(t_sec, raw_signal[:n_plot], color="#1558C9", linewidth=0.7)
    ax_raw.set_title("01 / Raw Vibration Waveform", color="#F6F8FF", fontsize=8.5, fontweight="bold")
    ax_raw.set_xlabel("Time (s)", color="#9DAAC2", fontsize=7.5)
    ax_raw.set_ylabel("Acceleration (g)", color="#9DAAC2", fontsize=7.5)

    ax_norm.plot(first_window_norm, color="#D4FF24", linewidth=0.65)
    ax_norm.set_title("02 / Normalized Window #0 (Z-Score)", color="#F6F8FF", fontsize=8.5, fontweight="bold")
    ax_norm.set_xlabel("Sample Index", color="#9DAAC2", fontsize=7.5)
    ax_norm.set_ylabel("Z-Score", color="#9DAAC2", fontsize=7.5)

    freqs_khz = np.fft.rfftfreq(len(first_window_norm), d=1.0 / sampling_hz) / 1000.0
    mag = np.abs(np.fft.rfft(first_window_norm)) / len(first_window_norm)
    ax_fft.plot(freqs_khz, mag, color="#83B6FF", linewidth=0.85)
    ax_fft.fill_between(freqs_khz, mag, color="#1558C9", alpha=0.22)
    ax_fft.set_title("03 / Single-Sided FFT Spectrum", color="#F6F8FF", fontsize=8.5, fontweight="bold")
    ax_fft.set_xlabel("Frequency (kHz)", color="#9DAAC2", fontsize=7.5)
    ax_fft.set_ylabel("FFT Magnitude", color="#9DAAC2", fontsize=7.5)

    vals = [float(mean_scores[k]) for k in CLASS_NAMES]
    best_idx = int(np.argmax(vals))
    bar_colors = ["#D4FF24" if i == best_idx else "#1558C9" for i in range(len(CLASS_NAMES))]
    short_labels = [CLASS_DISPLAY_NAMES[k].split(" (")[0] for k in CLASS_NAMES]
    y_pos = np.arange(len(CLASS_NAMES))
    ax_bar.barh(y_pos, vals, color=bar_colors, height=0.55)
    ax_bar.set_yticks(y_pos, short_labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, 1.08)
    ax_bar.set_title("04 / Uncalibrated Softmax Scores", color="#F6F8FF", fontsize=8.5, fontweight="bold")
    ax_bar.set_xlabel("Mean Softmax Score", color="#9DAAC2", fontsize=7.5)
    for i, v in enumerate(vals):
        ax_bar.text(min(v + 0.02, 0.98), i, f"{v:.4f}", va="center", color="#F6F8FF", fontsize=7.5, fontweight="bold")

    fig.tight_layout(pad=1.1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def build_diagnostic_pdf_bytes(diag: dict[str, Any]) -> bytes:
    """Generate a clean, downloadable PDF diagnostic report for the analyzed signal."""
    pred = diag["pred_result"]
    sig_meta = diag["sig_meta"]
    seg_meta = diag["seg_meta"]
    source_name = str(diag["source_name"])
    input_mode = str(diag.get("input_mode", "Official CWRU 3 HP Benchmark Recording"))
    checkpoint_sha = str(diag.get("checkpoint_sha", "3cd1ac3b823ea903"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=f"VibeGuard Cloud Diagnostic Report - {source_name}",
        author="VibeGuard Cloud Research Prototype",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "VGTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0B1220"),
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "VGSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "VGH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#1558C9"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "VGBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
    )

    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story: list[Any] = []

    story.append(Paragraph("VibeGuard Cloud — Motor Bearing Diagnostic Report", title_style))
    story.append(
        Paragraph(
            f"Generated: {ts_utc} | Model: 1D TinyFaultCNN (15,572 params, checkpoint SHA-256: {checkpoint_sha}...) | "
            f"Input Source: {input_mode}",
            sub_style,
        )
    )

    review_status = (
        "MANUAL REVIEW REQUIRED (Below Heuristic Threshold)"
        if pred["manual_review_required"]
        else "CONSISTENT CLASSIFICATION (Exceeds Heuristic Thresholds)"
    )

    story.append(Paragraph("1. Diagnostic Summary & Signal Provenance", h2_style))
    summary_data = [
        ["Analyzed Recording", source_name, "Detected Condition", pred["predicted_display"]],
        [
            "Top Model Score (Uncalibrated)",
            f"{pred['top_uncalibrated_score']:.4f}",
            "Segment Agreement",
            f"{pred['segment_agreement']:.1%} ({pred['per_class_window_counts'][pred['predicted_label']]}/{pred['num_windows']} windows)",
        ],
        [
            "Channel Identifier",
            str(sig_meta.get("channel_key", "N/A")),
            "Nominal Sampling Rate",
            f"{int(sig_meta.get('sampling_hz_nominal', NOMINAL_SAMPLING_HZ)):,} Hz",
        ],
        [
            "Raw Signal Length",
            f"{int(sig_meta.get('num_samples', 0)):,} samples",
            "Windowing & Normalization",
            f"{WINDOW_SIZE:,} samples/window, Z-score clamped [-10, 10]",
        ],
        ["Review Status", review_status, "Recorded Motor Speed", f"{sig_meta.get('recorded_rpm') or 'Not specified'} RPM"],
    ]
    summary_table = Table(summary_data, colWidths=[1.65 * inch, 1.95 * inch, 1.65 * inch, 2.15 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F1F5F9")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(summary_table)

    story.append(Paragraph("2. Four-Class Uncalibrated Model Score Breakdown", h2_style))
    score_rows = [["Class Key", "Bearing Condition", "Mean Uncalibrated Softmax Score", "Winning Windows", "Window Share"]]
    total_w = max(1, int(pred["num_windows"]))
    for cls_key in CLASS_NAMES:
        w_cnt = int(pred["per_class_window_counts"][cls_key])
        score_rows.append(
            [
                cls_key,
                CLASS_DISPLAY_NAMES[cls_key],
                f"{float(pred['mean_scores'][cls_key]):.6f}",
                str(w_cnt),
                f"{(w_cnt / total_w):.1%}",
            ]
        )
    score_table = Table(score_rows, colWidths=[0.95 * inch, 2.75 * inch, 1.65 * inch, 1.05 * inch, 1.0 * inch])
    score_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1220")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#F6F8FF")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(score_table)

    story.append(Paragraph("3. Vibration Waveform, Normalized Segment & Spectral Evidence", h2_style))
    chart_png = _render_report_chart_png(
        raw_signal=diag["signal"],
        first_window_norm=diag["windows"][0],
        mean_scores=pred["mean_scores"],
        sampling_hz=int(sig_meta.get("sampling_hz_nominal", NOMINAL_SAMPLING_HZ)),
    )
    img_buf = io.BytesIO(chart_png)
    story.append(RLImage(img_buf, width=7.2 * inch, height=4.04 * inch))
    story.append(Spacer(1, 4))

    story.append(Paragraph("4. Research Scope, Provenance & Methodological Limitations", h2_style))
    story.append(
        Paragraph(
            "<b>Research Prototype Disclaimer:</b> VibeGuard Cloud is an academic research prototype evaluated on the "
            "Case Western Reserve University (CWRU) Bearing Data Center benchmark (12 kHz drive-end accelerometer signals, "
            "0.007-inch electro-discharge machined single-point faults). Model outputs are <b>uncalibrated softmax scores</b> "
            "averaged across non-overlapping 2,048-sample windows and do not represent calibrated posterior probabilities of "
            "correctness. High accuracy on the locked 3 HP test split (4 recordings from the same laboratory test stand) "
            "demonstrates cross-load consistency from 0/1 HP training to 3 HP testing, not generalization across diverse "
            "industrial machinery or natural progressive wear. This report does not estimate Remaining Useful Life (RUL) "
            "and must not substitute for certified vibration engineering inspection.",
            body_style,
        )
    )

    doc.build(story)
    return buf.getvalue()
