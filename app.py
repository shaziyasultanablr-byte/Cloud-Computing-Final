"""Streamlit dashboard for VibeGuard Cloud: Load-Shift-Aware Bearing Fault Classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui_design import (
    mount_theme, hero, scope_note, section, sidebar_status,
    metrics_row, result_hero, provenance_card, eval_limitation_note,
    score_plot, signal_plot, confusion_plot, training_plot, architecture,
)

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
from download_data import download_single_file
from model import (
    DEFAULT_AGREEMENT_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    TinyFaultCNN,
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
        "url": "https://engineering.case.edu/sites/default/files/100.mat",
        "sha256": "88a5990cb541320e91505a1d72139e1993500ffe6e292a451011667f4138ca78",
        "true_label": "normal",
        "file_id": 100,
        "load_hp": 3,
    },
    "Held-Out 3 HP Demo — 108.mat (Inner Race 0.007 in, 3 HP)": {
        "path": Path("data/raw/inner/108.mat"),
        "url": "https://engineering.case.edu/sites/default/files/108.mat",
        "sha256": "d415f0e65128bfa0b118c2051c988fb0dc6ce3e4c8aa45fdc5ad1d0967462e7d",
        "true_label": "inner",
        "file_id": 108,
        "load_hp": 3,
    },
    "Held-Out 3 HP Demo — 121.mat (Ball Fault 0.007 in, 3 HP)": {
        "path": Path("data/raw/ball/121.mat"),
        "url": "https://engineering.case.edu/sites/default/files/121.mat",
        "sha256": "52f686e984ba8e9b04a047a57416271cdb322cee57ee105dce58926a24b966cb",
        "true_label": "ball",
        "file_id": 121,
        "load_hp": 3,
    },
    "Held-Out 3 HP Demo — 133.mat (Outer Race @ 6:00 0.007 in, 3 HP)": {
        "path": Path("data/raw/outer/133.mat"),
        "url": "https://engineering.case.edu/sites/default/files/133.mat",
        "sha256": "53f076cb0d905cf46bffaee7736abce8224ffed777deed03c0c863d762a90c71",
        "true_label": "outer",
        "file_id": 133,
        "load_hp": 3,
    },
}


@st.cache_resource(show_spinner=False)
def get_cached_model(checkpoint_path_str: str) -> tuple[TinyFaultCNN, dict[str, Any]]:
    """Load and cache TinyFaultCNN in CPU eval mode across Streamlit sessions."""
    return load_model_checkpoint(Path(checkpoint_path_str), device="cpu")


@st.cache_data(show_spinner=False)
def fetch_official_cwru_demo_bytes(
    file_id: int,
    url: str,
    dest_path_str: str,
    expected_sha256: str,
) -> bytes:
    """Retrieve an official held-out 3 HP CWRU .mat recording from disk or CWRU Bearing Data Center."""
    dest_path = Path(dest_path_str)
    if dest_path.exists() and dest_path.stat().st_size > 2048:
        if compute_sha256(dest_path) == expected_sha256:
            return dest_path.read_bytes()

    ok, msg = download_single_file(url=url, dest_path=dest_path)
    if not ok or not dest_path.exists():
        raise DataValidationError(
            f"Unable to retrieve official CWRU recording {file_id}.mat from {url}: {msg}"
        )
    actual_sha = compute_sha256(dest_path)
    if actual_sha != expected_sha256:
        raise DataValidationError(
            f"SHA-256 mismatch for {file_id}.mat (expected {expected_sha256[:12]}..., got {actual_sha[:12]}...)."
        )
    return dest_path.read_bytes()


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
    """Dark-themed three-panel vibration analysis; data are unchanged."""
    return signal_plot(raw_signal, first_window_norm, sampling_hz)


def main() -> None:
    st.set_page_config(
        page_title="VibeGuard Cloud — Bearing Fault Diagnosis",
        page_icon="⚙️",
        layout="wide",
    )

    mount_theme()
    hero()
    scope_note()

    metrics_data = load_json_if_exists(METRICS_PATH)
    file_metrics_data = load_json_if_exists(FILE_METRICS_PATH)
    manifest_data = load_json_if_exists(RUN_MANIFEST_PATH)
    s3_status = probe_s3_status()

    with st.sidebar:
        checkpoint_sha = compute_sha256(MODEL_PATH)[:16] if MODEL_PATH.exists() else None
        sidebar_cnn = metrics_data.get("cnn_1d", {}).get("test_3hp", {}) if metrics_data else {}
        sidebar_rf = metrics_data.get("random_forest_baseline", {}).get("test_3hp", {}) if metrics_data else {}
        sidebar_status(
            model_sha=checkpoint_sha,
            run_id=metrics_data.get("run_id", "N/A") if metrics_data else "N/A",
            epoch=metrics_data.get("selected_epoch", "N/A") if metrics_data else "N/A",
            cnn_f1=float(sidebar_cnn["macro_f1"]) if "macro_f1" in sidebar_cnn else None,
            rf_f1=float(sidebar_rf["macro_f1"]) if "macro_f1" in sidebar_rf else None,
            s3_connected=bool(s3_status["connected"]),
            s3_message=str(s3_status["message"]),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            agreement_threshold=DEFAULT_AGREEMENT_THRESHOLD,
        )

    tab_infer, tab_eval, tab_arch = st.tabs(
        [
            "01  /  Diagnose",
            "02  /  Evaluate",
            "03  /  Architecture",
        ]
    )

    with tab_infer:
        section("01 / LIVE INFERENCE", "What is your motor telling you?", "Choose a verified CWRU recording or upload a compatible single-channel vibration signal.")
        col_in1, col_in2 = st.columns([1.2, 1.0])
        with col_in1:
            st.markdown("##### Upload a vibration recording")
            uploaded_file = st.file_uploader(
                "Upload a CWRU MATLAB (.mat with `*_DE_time`) or single-column numeric CSV (>= 2,048 samples, max 25 MiB):",
                type=["mat", "csv"],
            )
        with col_in2:
            st.markdown("##### Select a verified 3 HP sample")
            demo_choice = st.selectbox(
                "Inspect a locked 3 HP test recording (never used in training or model selection):",
                options=list(HELD_OUT_DEMO_FILES.keys()),
                index=2,
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
                if demo_path is not None:
                    try:
                        with st.spinner(f"Loading & verifying official CWRU recording `{demo_path.name}`..."):
                            raw_bytes = fetch_official_cwru_demo_bytes(
                                file_id=int(demo_info["file_id"]),
                                url=str(demo_info["url"]),
                                dest_path_str=str(demo_path),
                                expected_sha256=str(demo_info["sha256"]),
                            )
                        source_name = demo_path.name
                        demo_meta = demo_info
                    except Exception as fetch_err:
                        st.warning(
                            f"Could not automatically retrieve `{demo_path.name}` from CWRU ({fetch_err}). "
                            "Please upload a `.mat` or `.csv` file directly."
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
                    model, ckpt_meta = get_cached_model(str(MODEL_PATH))
                    pred_result = predict_windows(model, windows, device="cpu")

                    section("DIAGNOSTIC OUTPUT", "Signal analysis complete", "Predictions shown below were computed from the loaded, unchanged TinyFaultCNN checkpoint.")
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

                    result_hero(
                        predicted_display=pred_result["predicted_display"],
                        source_file=source_name,
                        manual_review=bool(pred_result["manual_review_required"]),
                        note="3 HP held-out CWRU recording" if demo_meta is not None else "User-supplied recording",
                    )
                    metrics_row([
                        ("Top model score", f"{pred_result['top_uncalibrated_score']:.4f}", "Uncalibrated softmax"),
                        ("Segment agreement", f"{pred_result['segment_agreement']:.1%}", "Window-level consistency"),
                        ("Analyzed windows", str(pred_result["num_windows"]), f"{WINDOW_SIZE:,} samples per window"),
                        ("Operating load", f"{demo_meta['load_hp']} HP" if demo_meta else "Not specified", "Held-out test" if demo_meta else "Uploaded recording"),
                    ], featured=0)

                    score_col, meta_col = st.columns([1.2, 1.0])
                    with score_col:
                        section("MODEL OUTPUT", "Four-class score comparison", "Scores are not calibrated probabilities.")
                        score_fig = score_plot(pred_result["mean_scores"], CLASS_DISPLAY_NAMES, CLASS_NAMES)
                        st.pyplot(score_fig, clear_figure=True)
                        plt.close(score_fig)
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

                    with meta_col:
                        section("SIGNAL PROVENANCE", "Recording & preprocessing metadata", "Verified raw data, channel and window parameters.")
                        provenance_card(
                            source_file=source_name,
                            channel_key=str(sig_meta["channel_key"]),
                            sampling_hz=int(sig_meta["sampling_hz_nominal"]),
                            window_size=WINDOW_SIZE,
                            num_windows=int(seg_meta["num_windows"]),
                            manual_review=bool(pred_result["manual_review_required"]),
                        )
                        with st.expander("Inspect full provenance & preprocessing JSON"):
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

                    section("SIGNAL FORENSICS", "Vibration & frequency analysis", "Three complementary views of the same recording. Nominal sample rate: 12 kHz.")
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
        section("02 / MODEL EVALUATION", "Performance under load shift", "Training: 0 & 1 HP · Validation: 2 HP · Final held-out test: 3 HP. Scores come from frozen evaluation artifacts.")
        if metrics_data is None or file_metrics_data is None:
            st.info("No evaluation artifacts found yet. Run `python train.py` to generate `outputs/metrics.json`.")
        else:
            cnn_test = metrics_data["cnn_1d"]["test_3hp"]
            rf_test = metrics_data["random_forest_baseline"]["test_3hp"]
            cnn_val = metrics_data["cnn_1d"]["val_2hp"]
            rf_val = metrics_data["random_forest_baseline"]["val_2hp"]

            metrics_row([
                ("CNN · Validation macro-F1", f"{cnn_val['macro_f1']:.4f}", "2 HP unseen validation"),
                ("CNN · Test macro-F1", f"{cnn_test['macro_f1']:.4f}", "3 HP locked test"),
                ("RF · Validation macro-F1", f"{rf_val['macro_f1']:.4f}", "2 HP unseen validation"),
                ("RF · Test macro-F1", f"{rf_test['macro_f1']:.4f}", "3 HP locked test"),
            ], featured=1)
            eval_limitation_note()

            section("RECORDING AUDIT", "Four held-out test recordings", "The same four recordings were never included in training or validation selection.")
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

            section("DIAGNOSTIC EVIDENCE", "Confusion matrices", "Actual versus predicted windows, on the locked 3 HP test set.")
            confusion_fig = confusion_plot(
                cnn_test["confusion_matrix"], rf_test["confusion_matrix"], cnn_test["labels"]
            )
            st.pyplot(confusion_fig, clear_figure=True)
            plt.close(confusion_fig)

            history_data = load_json_if_exists(Path("outputs/training_history.json"))
            if history_data:
                section("TRAINING HISTORY", "Learning dynamics", "Validation data were used for model selection; final test data were held out.")
                history_fig = training_plot(history_data)
                st.pyplot(history_fig, clear_figure=True)
                plt.close(history_fig)


    with tab_arch:
        section("03 / SYSTEM ARCHITECTURE", "From raw signal to diagnosis", "One reproducible pipeline, two purpose-built deployment environments.")
        architecture()
        arch_img = Path("docs/architecture.png")
        if arch_img.exists():
            with st.expander("View original technical architecture figure"):
                st.image(str(arch_img), caption="Original AWS end-to-end architecture")

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
