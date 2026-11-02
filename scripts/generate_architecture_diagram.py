"""Generate the VibeGuard Cloud system architecture diagram (docs/architecture.png)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def generate_architecture_diagram(output_path: str | Path = "docs/architecture.png") -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=180)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Background title banner
    banner = patches.FancyBboxPatch(
        (0.3, 6.55),
        13.4,
        0.75,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor="#0f172a",
        edgecolor="#1e293b",
        linewidth=1.5,
    )
    ax.add_patch(banner)
    ax.text(
        7.0,
        6.98,
        "VibeGuard Cloud — End-to-End Load-Shift-Aware Bearing Fault Classification Architecture",
        color="white",
        fontsize=13.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        7.0,
        6.72,
        "1D Convolutional Neural Network (TinyFaultCNN) + Random Forest Baseline | AWS Free Plan (EC2 + EBS + Private S3 via IAM Role)",
        color="#93c5fd",
        fontsize=9.5,
        ha="center",
        va="center",
    )

    # AWS Cloud Boundary Box
    aws_box = patches.FancyBboxPatch(
        (4.15, 0.45),
        9.55,
        5.8,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#f8fafc",
        edgecolor="#f59e0b",
        linewidth=2.2,
        linestyle="--",
    )
    ax.add_patch(aws_box)
    ax.text(
        4.35,
        6.02,
        "AWS Cloud Boundary (Single Region: us-east-1 | Free Plan Budget Guard: USD 5/mo Alert)",
        color="#b45309",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="center",
    )

    def draw_card(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        lines: list[str],
        header_color: str,
        body_color: str = "#ffffff",
        border_color: str = "#cbd5e1",
    ) -> None:
        body = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=body_color,
            edgecolor=border_color,
            linewidth=1.5,
        )
        ax.add_patch(body)
        hdr_h = 0.42
        hdr = patches.FancyBboxPatch(
            (x, y + h - hdr_h),
            w,
            hdr_h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=header_color,
            edgecolor=header_color,
            linewidth=1.0,
        )
        ax.add_patch(hdr)
        ax.text(
            x + w / 2,
            y + h - hdr_h / 2,
            title,
            color="white",
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="center",
        )
        text_y = y + h - hdr_h - 0.22
        for line in lines:
            ax.text(
                x + 0.14,
                text_y,
                line,
                color="#1e293b",
                fontsize=8.2,
                ha="left",
                va="top",
            )
            text_y -= 0.27

    # Left Column: Data Source & Authorized User
    draw_card(
        0.35,
        3.55,
        3.4,
        2.65,
        "1. CWRU Bearing Data Center",
        [
            "• 16 Official MATLAB (.mat) Files",
            "• Sensor: Drive-End (*_DE_time)",
            "• Nominal Sampling: 12 kHz",
            "• 4 Classes: Normal, Inner (0.007\"),",
            "  Ball (0.007\"), Outer@6:00 (0.007\")",
            "• SHA-256 Verified Manifest",
            "• Disjoint Split by Motor Load:",
            "  Train: 0 & 1 HP | Val: 2 HP | Test: 3 HP",
        ],
        header_color="#1e40af",
        body_color="#eff6ff",
        border_color="#93c5fd",
    )

    draw_card(
        0.35,
        0.55,
        3.4,
        2.65,
        "5. Authorized Engineer / Examiner",
        [
            "• Local Browser: http://localhost:8501",
            "• Encrypted SSH Port-Forward Tunnel:",
            "  ssh -i key.pem -L 8501:127.0.0.1:8501",
            "• Security Group: Inbound 22 (My IP)",
            "• Public Port 8501 CLOSED",
            "• Source of Truth: GitHub Repository",
            "  shaziyasultanablr-byte/",
            "  Cloud-Computing-Final",
        ],
        header_color="#0f766e",
        body_color="#f0fdfa",
        border_color="#5eead4",
    )

    # Middle Column inside AWS: EC2 + EBS Compute & Pipeline
    draw_card(
        4.4,
        3.2,
        4.6,
        2.6,
        "2. Amazon EC2 (Ubuntu 24.04 x86_64) + gp3 EBS",
        [
            "• Preprocessing (data_utils.py):",
            "  - Non-overlapping 2,048-sample windows (max 200)",
            "  - Per-window z-score (mean=0, std=1), clamp [-10, 10]",
            "• Deep Learning Model (model.py / train.py):",
            "  - TinyFaultCNN: 3x (Conv1D + BN + ReLU + Pool)",
            "  - Kernel sizes: 15 -> 9 -> 5, Dropout 0.2, 4-class head",
            "• Classical Baseline: Random Forest (18 T/F features)",
            "• Selection on 2 HP Val Macro-F1; Locked 3 HP Test",
        ],
        header_color="#b45309",
        body_color="#fffbeb",
        border_color="#fcd34d",
    )

    draw_card(
        4.4,
        0.65,
        4.6,
        2.3,
        "4. Streamlit Web Dashboard (app.py @ 127.0.0.1:8501)",
        [
            "• Upload .mat (*_DE_time) or 1-column .csv (<= 25 MiB)",
            "• Or inspect held-out 3 HP test files (100/108/121/133)",
            "• Displays Predicted Class & 4 Uncalibrated Scores",
            "• Shows Window-Level Segment Agreement (%)",
            "• Manual-Review Flag (Score < 0.70 or Agree < 80%)",
            "• Waveform, Normalized Segment & 12 kHz FFT Plots",
        ],
        header_color="#4338ca",
        body_color="#eef2ff",
        border_color="#a5b4fc",
    )

    # Right Column inside AWS: IAM Role & Private S3
    draw_card(
        9.35,
        3.45,
        4.1,
        2.35,
        "3A. IAM Instance Profile (VibeGuardEC2Role)",
        [
            "• Zero Hard-Coded AWS Access Keys",
            "• Attached Directly to EC2 Instance",
            "• Least-Privilege Inline Policy:",
            "  - s3:ListBucket on bucket ARN",
            "  - s3:GetObject, s3:PutObject on",
            "    bucket ARN/*",
            "• Root MFA Enabled & Password Rotated",
        ],
        header_color="#be123c",
        body_color="#fff1f2",
        border_color="#fda4af",
    )

    draw_card(
        9.35,
        0.65,
        4.1,
        2.5,
        "3B. Private Amazon S3 Bucket (SSE-S3)",
        [
            "• Block All Public Access: ON",
            "• ACLs Disabled, AES-256 Encryption",
            "• s3://<bucket>/raw/{normal,inner,ball,outer}/",
            "  (16 validated CWRU .mat recordings)",
            "• s3://<bucket>/artifacts/",
            "  (model.pt, metrics.json, run_manifest.json,",
            "   test_predictions.csv, plots)",
        ],
        header_color="#15803d",
        body_color="#f0fdf4",
        border_color="#86efac",
    )

    # Connectors
    arrow_kw = dict(arrowstyle="->,head_width=0.25,head_length=0.3", lw=2.0, color="#334155")
    ax.annotate("", xy=(4.38, 4.65), xytext=(3.77, 4.65), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.7, 2.98), xytext=(6.7, 3.18), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.33, 4.65), xytext=(9.02, 4.65), arrowprops=arrow_kw)
    ax.annotate("", xy=(11.4, 3.18), xytext=(11.4, 3.43), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.33, 1.85), xytext=(9.02, 1.85), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.77, 1.80), xytext=(4.38, 1.80), arrowprops=dict(arrowstyle="<->,head_width=0.25,head_length=0.3", lw=2.0, color="#0f766e"))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = generate_architecture_diagram()
    print(f"Generated architecture diagram: {p}")
