"""Streamlit dashboard for VibeGuard Cloud: Load-Shift-Aware Bearing Fault Classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cloud_sync import probe_s3_status
from data_utils import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    DataValidationError,
    MAX_UPLOAD_BYTES,
    MAX_WINDOWS_PER_FILE,
    NOMINAL_SAMPLING_HZ,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    compute_sha256,
    extract_de_signal_from_mat,
    extract_signal_from_csv,
    segment_and_normalize_signal,
)
from model import (
    DEFAULT_AGREEMENT_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    load_model_checkpoint,
    predict_windows,
)

MODEL_PATH = Path("outputs/model.pt")
METRICS_PATH = Path("outputs/metrics.json")
FILE_METRICS_PATH = Path("outputs/file_level_metrics.json")
RUN_MANIFEST_PATH = Path("outputs/run_manifest.json")

HELD_OUT_DEMO_FILES: dict[str, dict[str, Any]] = {
    "None (Use File Uploader)": {"path": None},
    "Held-Out 3 HP Demo — 100.mat (Normal Baseline, 3 HP)": {
        "path": Path("data/raw/normal/100.mat"),
        "true_label": "normal",
        "file_id": 100,
        "load_hp": 3,
    },
    "Held-Out 3 HP Demo — 108.mat (Inner Race 0.007 in, 3 HP)": {
        "path": Path("data/raw/inner/108.mat"),
        "true_label": "inner",
        "file_id": 108,
        "load_hp": 3,
    },
    "Held-Out 3 HP Demo — 121.mat (Ball Fault 0.007 in, 3 HP)": {
        "path": Path("data/raw/ball/121.mat"),
        "true_label": "ball",
        "file_id": 121,
        "load_hp": 3,
    },
    "Held-Out 3 HP Demo — 133.mat (Outer Race @ 6:00 0.007 in, 3 HP)": {
        "path": Path("data/raw/outer/133.mat"),
        "true_label": "outer",
        "file_id": 133,
        "load_hp": 3,
    },
}


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Safely read a JSON file if it exists."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_uploaded_recording(
    filename: str,
    raw_bytes: bytes,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and extract 1D vibration signal from an uploaded .mat or .csv file."""
    lower_name = filename.lower().strip()
    if lower_name.endswith(".mat"):
        return extract_de_signal_from_mat(raw_bytes, max_bytes=MAX_UPLOAD_BYTES)
    if lower_name.endswith(".csv"):
        return extract_signal_from_csv(raw_bytes, max_bytes=MAX_UPLOAD_BYTES)
    raise DataValidationError(
        f"Unsupported file format for '{filename}'. Please upload a CWRU '.mat' file or a single-column '.csv' file."
    )


def render_waveform_and_spectrum(
    raw_signal: np.ndarray,
    first_window_norm: np.ndarray,
    sampling_hz: int = NOMINAL_SAMPLING_HZ,
) -> plt.Figure:
    """Plot raw drive-end waveform, first normalized 2,048-sample window, and FFT amplitude spectrum."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), dpi=140)

    # Panel 1: Raw signal preview (first 4,096 samples)
    preview_len = min(len(raw_signal), 4096)
    t_axis = np.arange(preview_len) / float(sampling_hz)
    axes[0].plot(t_axis, raw_signal[:preview_len], color="#1d4ed8", linewidth=0.8)
    axes[0].set_title(f"Raw Signal Preview (First {preview_len} Samples)", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Time (s) @ 12 kHz Nominal")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: First normalized 2,048-sample window
    axes[1].plot(first_window_norm, color="#059669", linewidth=0.8)
    axes[1].set_title("Window #1 Normalized (Z-Score, Clamped [-10, 10])", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Window Sample Index (0..2047)")
    axes[1].set_ylabel("Normalized Amplitude")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Single-sided FFT magnitude spectrum of Window #1
    freqs = np.fft.rfftfreq(len(first_window_norm), d=1.0 / sampling_hz)
    mag = np.abs(np.fft.rfft(first_window_norm)) / len(first_window_norm)
    axes[2].plot(freqs / 1000.0, mag, color="#dc2626", linewidth=0.9)
    axes[2].set_title("Window #1 FFT Magnitude Spectrum", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Frequency (kHz) @ 12 kHz Nominal")
    axes[2].set_ylabel("Spectral Magnitude")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="VibeGuard Cloud — Bearing Fault Diagnosis",
        page_icon="⚙️",
        layout="wide",
    )

    st.title("⚙️ VibeGuard Cloud: Load-Shift-Aware Bearing Fault Classification")
    st.markdown(
        "**Cloud-Hosted Research Prototype (`1D TinyFaultCNN` + AWS EC2 & Private S3)**  \n"
        "Evaluates whether a 1D Convolutional Neural Network trained on **0 HP & 1 HP** drive-end vibration "
        "recordings generalizes to previously unseen **3 HP** operating loads across four bearing conditions: "
        "**Normal**, **Inner Race Fault**, **Ball Fault**, and **Outer Race Fault**."
    )

    st.info(
        "**Research Scope & Calibration Disclaimer:** This application is a university research prototype built on "
        "recorded laboratory accelerometer data from the **Case Western Reserve University (CWRU) Bearing Data Center** "
        "(nominal 12 kHz drive-end channel `*_DE_time`). Reported class scores are **uncalibrated softmax outputs**, "
        "not calibrated posterior probabilities. The system diagnoses predefined seeded faults and **does not** "
        "estimate Remaining Useful Life (RUL) or replace certified industrial maintenance inspection."
    )

    metrics_data = load_json_if_exists(METRICS_PATH)
    file_metrics_data = load_json_if_exists(FILE_METRICS_PATH)
    manifest_data = load_json_if_exists(RUN_MANIFEST_PATH)
    s3_status = probe_s3_status()

    with st.sidebar:
        st.header("🔍 Model & Cloud Status")
        if MODEL_PATH.exists():
            model_sha = compute_sha256(MODEL_PATH)[:16]
            st.success(f"Checkpoint Loaded (`outputs/model.pt`)\nSHA-256: `{model_sha}...`")
        else:
            st.error("Model checkpoint `outputs/model.pt` not found. Run `python train.py` first.")

        if metrics_data:
            st.markdown(f"- **Run ID:** `{metrics_data.get('run_id', 'N/A')}`")
            st.markdown(f"- **Selected Epoch (2 HP Val):** `{metrics_data.get('selected_epoch', 'N/A')}`")
            cnn_test = metrics_data.get("cnn_1d", {}).get("test_3hp", {})
            rf_test = metrics_data.get("random_forest_baseline", {}).get("test_3hp", {})
            if cnn_test:
                st.markdown(
                    f"- **3 HP Held-Out Window Acc (CNN):** `{cnn_test.get('accuracy', 0.0):.2%}` "
                    f"(Macro-F1: `{cnn_test.get('macro_f1', 0.0):.3f}`)"
                )
            if rf_test:
                st.markdown(
                    f"- **3 HP Held-Out Window Acc (RF):** `{rf_test.get('accuracy', 0.0):.2%}` "
                    f"(Macro-F1: `{rf_test.get('macro_f1', 0.0):.3f}`)"
                )

        st.divider()
        st.subheader("☁️ AWS S3 Integration")
        if s3_status["connected"]:
            st.success(s3_status["message"])
        else:
            st.caption(s3_status["message"])

        st.divider()
        st.subheader("🛡️ Manual-Review Thresholds")
        st.caption(
            f"- Minimum Top Uncalibrated Score: `{DEFAULT_CONFIDENCE_THRESHOLD:.2f}`\n"
            f"- Minimum Segment Agreement: `{DEFAULT_AGREEMENT_THRESHOLD:.0%}`\n"
            "- Pre-specified heuristic (never tuned on 3 HP test set)."
        )

    tab_infer, tab_eval, tab_arch = st.tabs(
        [
            "1. Signal Upload & Fault Inference",
            "2. Held-Out 3 HP Evaluation & Baseline Comparison",
            "3. Architecture, Provenance & Limitations",
        ]
    )

    with tab_infer:
        col_in1, col_in2 = st.columns([1.2, 1.0])
        with col_in1:
            st.subheader("Upload Vibration Recording")
            uploaded_file = st.file_uploader(
                "Upload a CWRU MATLAB (.mat with `*_DE_time`) or single-column numeric CSV (>= 2,048 samples, max 25 MiB):",
                type=["mat", "csv"],
            )
        with col_in2:
            st.subheader("Or Select a Held-Out 3 HP Sample")
            demo_choice = st.selectbox(
                "Inspect a locked 3 HP test recording (never used in training or model selection):",
                options=list(HELD_OUT_DEMO_FILES.keys()),
            )

        if not MODEL_PATH.exists():
            st.warning(
                "Model checkpoint `outputs/model.pt` is missing. Please run `python train.py --epochs 14 --batch-size 64` "
                "to train and save the verified model before running inference."
            )
        else:
            source_name: str | None = None
            raw_bytes: bytes | None = None
            demo_meta: dict[str, Any] | None = None

            if uploaded_file is not None:
                source_name = uploaded_file.name
                raw_bytes = uploaded_file.getvalue()
            elif demo_choice != "None (Use File Uploader)":
                demo_info = HELD_OUT_DEMO_FILES[demo_choice]
                demo_path: Path | None = demo_info.get("path")
                if demo_path is not None and demo_path.exists():
                    source_name = demo_path.name
                    raw_bytes = demo_path.read_bytes()
                    demo_meta = demo_info
                else:
                    st.warning(
                        f"Selected demo recording `{demo_path}` is not present locally in `data/raw/`. "
                        "Run `python download_data.py` or upload a `.mat`/`.csv` file directly."
                    )

            if source_name is not None and raw_bytes is not None:
                try:
                    signal, sig_meta = parse_uploaded_recording(source_name, raw_bytes)
                    windows, seg_meta = segment_and_normalize_signal(
                        signal,
                        window_size=WINDOW_SIZE,
                        stride=WINDOW_STRIDE,
                        max_windows=MAX_WINDOWS_PER_FILE,
                    )
                    model, ckpt_meta = load_model_checkpoint(MODEL_PATH, device="cpu")
                    pred_result = predict_windows(model, windows, device="cpu")

                    st.divider()
                    st.subheader(f"Diagnostic Result for `{source_name}`")
                    if demo_meta is not None:
                        st.caption(
                            f"Held-Out Test Recording ID: **{demo_meta['file_id']}** | "
                            f"Operating Load: **{demo_meta['load_hp']} HP** | "
                            f"Ground-Truth Class: **{CLASS_DISPLAY_NAMES[demo_meta['true_label']]}**"
                        )

                    if pred_result["manual_review_required"]:
                        reasons_text = " ".join(pred_result["reasons"])
                        st.warning(
                            f"⚠️ **MANUAL REVIEW REQUIRED:** {reasons_text} "
                            "Do not rely on automated classification without engineer inspection."
                        )
                    else:
                        st.success(
                            "✅ **CONSISTENT SIGNAL CLASSIFICATION:** Top uncalibrated score and window-level "
                            "agreement exceed pre-specified heuristic thresholds."
                        )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Predicted Bearing Condition", pred_result["predicted_display"])
                    m2.metric(
                        "Top Uncalibrated Score",
                        f"{pred_result['top_uncalibrated_score']:.4f}",
                    )
                    m3.metric(
                        "Segment Agreement",
                        f"{pred_result['segment_agreement']:.1%}",
                    )
                    m4.metric(
                        "Analyzed Windows",
                        f"{pred_result['num_windows']} × {WINDOW_SIZE}",
                    )

                    score_col, meta_col = st.columns([1.2, 1.0])
                    with score_col:
                        st.markdown("#### Four-Class Uncalibrated Model Scores")
                        scores_df = pd.DataFrame(
                            [
                                {
                                    "Class Key": cls_name,
                                    "Bearing Condition": CLASS_DISPLAY_NAMES[cls_name],
                                    "Mean Uncalibrated Softmax Score": pred_result["mean_scores"][cls_name],
                                    "Winning Windows": pred_result["per_class_window_counts"][cls_name],
                                }
                                for cls_name in CLASS_NAMES
                            ]
                        )
                        st.dataframe(scores_df, width="stretch", hide_index=True)
                        st.bar_chart(
                            scores_df.set_index("Bearing Condition")["Mean Uncalibrated Softmax Score"]
                        )

                    with meta_col:
                        st.markdown("#### Signal & Preprocessing Metadata")
                        st.json(
                            {
                                "source_file": source_name,
                                "channel_key": sig_meta["channel_key"],
                                "raw_samples": sig_meta["num_samples"],
                                "recorded_rpm": sig_meta["recorded_rpm"],
                                "nominal_sampling_hz": sig_meta["sampling_hz_nominal"],
                                "windows_extracted": seg_meta["num_windows"],
                                "sample_index_span": f"[0, {seg_meta['end_sample']})",
                                "normalization": "Per-window z-score, clamped [-10, 10]",
                                "calibration_notice": pred_result["calibration_note"],
                            }
                        )

                    st.markdown("#### Vibration Waveform & Spectral Inspection (12 kHz Nominal)")
                    fig = render_waveform_and_spectrum(signal, windows[0], sampling_hz=NOMINAL_SAMPLING_HZ)
                    st.pyplot(fig)
                    plt.close(fig)

                except DataValidationError as val_err:
                    st.error(f"**Input Validation Error:** {val_err}")
                except Exception as exc:
                    st.error(
                        f"**Unable to process uploaded file:** {type(exc).__name__}. "
                        "Please verify that the file is a valid CWRU `.mat` or single-column numeric `.csv` recording."
                    )

    with tab_eval:
        st.subheader("Locked Cross-Load Evaluation (Trained on 0 & 1 HP → Validated on 2 HP → Tested on 3 HP)")
        if metrics_data is None or file_metrics_data is None:
            st.info("No evaluation artifacts found yet. Run `python train.py` to generate `outputs/metrics.json`.")
        else:
            cnn_test = metrics_data["cnn_1d"]["test_3hp"]
            rf_test = metrics_data["random_forest_baseline"]["test_3hp"]
            cnn_val = metrics_data["cnn_1d"]["val_2hp"]
            rf_val = metrics_data["random_forest_baseline"]["val_2hp"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("1D CNN — 2 HP Val Macro-F1", f"{cnn_val['macro_f1']:.4f}")
            c2.metric("1D CNN — 3 HP Test Macro-F1", f"{cnn_test['macro_f1']:.4f}")
            c3.metric("Random Forest — 2 HP Val Macro-F1", f"{rf_val['macro_f1']:.4f}")
            c4.metric("Random Forest — 3 HP Test Macro-F1", f"{rf_test['macro_f1']:.4f}")

            st.markdown("#### Recording-Level Predictions on Held-Out 3 HP Test Split (4 Recordings)")
            rec_rows = file_metrics_data.get("cnn_1d", {}).get("test_3hp", {}).get("recordings", [])
            rf_rows = file_metrics_data.get("random_forest_baseline", {}).get("test_3hp", {}).get("recordings", [])
            rf_by_id = {r["file_id"]: r for r in rf_rows}
            if rec_rows:
                clean_rec_rows = []
                for r in rec_rows:
                    rf_r = rf_by_id.get(r["file_id"], {})
                    clean_rec_rows.append(
                        {
                            "File ID": r["file_id"],
                            "Recording": r["filename"],
                            "Load (HP)": r["load_hp"],
                            "True Class": r["true_label"],
                            "Windows": r["num_windows"],
                            "1D CNN Prediction": r["predicted_label"],
                            "1D CNN Top Score": round(float(r["top_uncalibrated_score"]), 6),
                            "1D CNN Margin": round(float(r["second_best_margin"]), 6),
                            "RF Prediction": rf_r.get("predicted_label", "N/A"),
                            "RF Top Score": round(float(rf_r.get("top_uncalibrated_score", 0.0)), 6),
                            "RF Margin": round(float(rf_r.get("second_best_margin", 0.0)), 6),
                            "Segment Agreement": f"{float(r['segment_agreement']):.1%}",
                            "Manual Review Flag": bool(r["manual_review_required"]),
                        }
                    )
                st.dataframe(pd.DataFrame(clean_rec_rows), width="stretch", hide_index=True)

            fig_col1, fig_col2 = st.columns(2)
            cm_img = Path("outputs/confusion_matrix.png")
            loss_img = Path("outputs/training_loss.png")
            if cm_img.exists():
                with fig_col1:
                    st.image(str(cm_img), caption="3 HP Test Confusion Matrices & CNN vs. RF F1 Comparison")
            if loss_img.exists():
                with fig_col2:
                    st.image(str(loss_img), caption="1D TinyFaultCNN Training & Validation Curves")

    with tab_arch:
        st.subheader("System Architecture, Dataset Provenance & Methodological Limitations")
        arch_img = Path("docs/architecture.png")
        if arch_img.exists():
            st.image(str(arch_img), caption="VibeGuard Cloud End-to-End Architecture")

        st.markdown(
            """
### Key Methodological Boundaries
1. **Load-Conditional Split:** Training uses `0 HP` and `1 HP` recordings (`8` files); validation uses `2 HP` (`4` files); held-out testing uses `3 HP` (`4` files). Zero windows or file IDs are shared across splits.
2. **Sample Size Limitation:** Each non-training operating load (`2 HP` validation and `3 HP` test) contains exactly **one recording per bearing class** (`200` windows per recording). High accuracy on this split demonstrates cross-load consistency on the same CWRU test rig and 0.007-inch seeded EDM faults, **not** generalization across different motors, bearing geometries, or natural progressive wear.
3. **Uncalibrated Confidence Scores:** Neural network softmax outputs are uncalibrated and can be overconfident under domain shift. The manual-review flag is a pre-specified decision-support heuristic.
4. **Cloud Security Model:** Hosted on an AWS Free Plan eligible Ubuntu x86_64 EC2 instance with an attached least-privilege IAM role (`VibeGuardEC2Role`) accessing a private Amazon S3 bucket (`Block Public Access = ON`). Primary demonstration uses an encrypted SSH tunnel (`ssh -L 8501:127.0.0.1:8501`).
            """
        )
        if manifest_data:
            with st.expander("Inspect Reproducibility Run Manifest (`outputs/run_manifest.json`)"):
                st.json(manifest_data)


if __name__ == "__main__":
    main()
