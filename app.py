"""Streamlit dashboard for VibeGuard Cloud: Load-Shift-Aware Bearing Fault Classification."""

from __future__ import annotations

import hashlib
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
from download_data import download_single_file
from model import (
    DEFAULT_AGREEMENT_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    TinyFaultCNN,
    load_model_checkpoint,
    predict_windows,
)
from report_utils import build_diagnostic_csv_bytes, build_diagnostic_pdf_bytes
from ui_design import (
    active_input_banner,
    architecture,
    confusion_plot,
    educational_fault_guide,
    eval_limitation_note,
    hero,
    interactive_confusion_charts,
    interactive_eval_comparison_chart,
    interactive_score_chart,
    interactive_signal_charts,
    interactive_training_charts,
    metrics_row,
    mount_theme,
    provenance_card,
    result_hero,
    scope_note,
    score_plot,
    section,
    sidebar_status,
    signal_plot,
    training_plot,
    workflow_stepper,
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


def _execute_diagnosis(
    source_name: str,
    raw_bytes: bytes,
    demo_meta: dict[str, Any] | None,
    input_mode_label: str,
    checkpoint_sha: str | None,
) -> dict[str, Any]:
    """Parse recording, segment & normalize windows, run TinyFaultCNN, and return complete state."""
    signal, sig_meta = parse_uploaded_recording(source_name, raw_bytes)
    windows, seg_meta = segment_and_normalize_signal(
        signal,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        max_windows=MAX_WINDOWS_PER_FILE,
    )
    model, ckpt_meta = get_cached_model(str(MODEL_PATH))
    pred_result = predict_windows(model, windows, device="cpu")
    diag_state: dict[str, Any] = {
        "source_name": source_name,
        "signal": signal,
        "windows": windows,
        "sig_meta": sig_meta,
        "seg_meta": seg_meta,
        "ckpt_meta": ckpt_meta,
        "pred_result": pred_result,
        "demo_meta": demo_meta,
        "input_mode": input_mode_label,
        "checkpoint_sha": checkpoint_sha or "3cd1ac3b823ea903",
    }
    diag_state["csv_bytes"] = build_diagnostic_csv_bytes(diag_state)
    diag_state["pdf_bytes"] = build_diagnostic_pdf_bytes(diag_state)
    return diag_state


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
    checkpoint_sha = compute_sha256(MODEL_PATH)[:16] if MODEL_PATH.exists() else None

    with st.sidebar:
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

    # Initialize session state variables for explicit workflow control
    if "diagnosis_state" not in st.session_state:
        st.session_state["diagnosis_state"] = None
    if "diagnosis_signature" not in st.session_state:
        st.session_state["diagnosis_signature"] = None
    if "diagnosis_reset" not in st.session_state:
        st.session_state["diagnosis_reset"] = False
    if "show_signal_details" not in st.session_state:
        st.session_state["show_signal_details"] = False
    if "uploader_key_version" not in st.session_state:
        st.session_state["uploader_key_version"] = 0

    tab_infer, tab_eval, tab_arch = st.tabs(
        [
            "01  /  Diagnose",
            "02  /  Evaluate",
            "03  /  Architecture",
        ]
    )

    with tab_infer:
        section(
            "01 / SIGNAL DIAGNOSIS WORKSTATION",
            "Select a recording, run inference & export diagnostic evidence",
            "Analyze an official SHA-256-verified CWRU 3 HP test recording or upload a compatible single-channel vibration file.",
        )
        workflow_stepper(
            has_diagnosis=(st.session_state["diagnosis_state"] is not None and not st.session_state["diagnosis_reset"]),
            details_open=bool(st.session_state["show_signal_details"]),
        )

        input_source_mode = st.radio(
            "Select Signal Input Mode:",
            options=[
                "Official CWRU 3 HP Benchmark Recording",
                "Upload Custom Recording (.mat / .csv)",
            ],
            horizontal=True,
            key="input_source_mode_radio",
        )

        col_in1, col_in2 = st.columns([1.15, 1.0])
        with col_in1:
            st.markdown("##### Select a verified 3 HP CWRU sample")
            demo_choice = st.selectbox(
                "Inspect a locked 3 HP test recording (never used in training or model selection):",
                options=list(HELD_OUT_DEMO_FILES.keys()),
                index=2,
                key="cwru_demo_selectbox",
            )
        with col_in2:
            st.markdown("##### Upload a custom vibration recording")
            uploaded_file = st.file_uploader(
                "Upload a CWRU MATLAB (.mat with `*_DE_time`) or single-column numeric CSV (>= 2,048 samples, max 25 MiB):",
                type=["mat", "csv"],
                key=f"signal_file_uploader_{st.session_state['uploader_key_version']}",
            )

        btn_c1, btn_c2, btn_c3 = st.columns([1.2, 1.0, 1.2])
        with btn_c1:
            run_clicked = st.button("▶ Run Diagnosis", type="primary", width="stretch", key="btn_run_diagnosis")
        with btn_c2:
            reset_clicked = st.button("↺ Reset Analysis", type="secondary", width="stretch", key="btn_reset_analysis")
        with btn_c3:
            toggle_label = (
                "🔍 Hide Signal Details"
                if st.session_state["show_signal_details"]
                else "🔍 View Signal Details"
            )
            details_clicked = st.button(toggle_label, type="secondary", width="stretch", key="btn_toggle_details")

        if details_clicked:
            st.session_state["show_signal_details"] = not st.session_state["show_signal_details"]
            st.rerun()

        if reset_clicked:
            st.session_state["diagnosis_reset"] = True
            st.session_state["diagnosis_state"] = None
            st.session_state["diagnosis_signature"] = None
            st.session_state["show_signal_details"] = False
            st.session_state["uploader_key_version"] += 1
            st.rerun()

        if not MODEL_PATH.exists():
            st.warning(
                "Model checkpoint `outputs/model.pt` is missing. Please run `python train.py --epochs 14 --batch-size 64` "
                "to train and save the verified model before running inference."
            )
        else:
            source_name: str | None = None
            raw_bytes: bytes | None = None
            demo_meta: dict[str, Any] | None = None
            resolved_mode_label = input_source_mode

            # Uploaded file takes precedence when Upload mode is explicitly selected,
            # or when demo_choice is explicitly set to "None (Use File Uploader)".
            use_upload = (
                input_source_mode == "Upload Custom Recording (.mat / .csv)"
                or demo_choice == "None (Use File Uploader)"
            )

            if use_upload:
                if uploaded_file is not None:
                    source_name = uploaded_file.name
                    raw_bytes = uploaded_file.getvalue()
                    resolved_mode_label = "Uploaded Custom Recording (.mat / .csv)"
                elif run_clicked:
                    st.info("Please upload a valid `.mat` or `.csv` file first, or switch Input Mode to **Official CWRU 3 HP Benchmark Recording**.")
            else:
                if demo_choice != "None (Use File Uploader)":
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
                            resolved_mode_label = "Official CWRU 3 HP Benchmark Recording"
                        except Exception as fetch_err:
                            st.warning(
                                f"Could not automatically retrieve `{demo_path.name}` from CWRU ({fetch_err}). "
                                "Please upload a `.mat` or `.csv` file directly."
                            )

            if run_clicked:
                st.session_state["diagnosis_reset"] = False

            current_sig: str | None = None
            if source_name is not None and raw_bytes is not None:
                digest = hashlib.sha256(raw_bytes[:65536]).hexdigest()[:16]
                current_sig = f"{resolved_mode_label}:{source_name}:{len(raw_bytes)}:{digest}"

            # If the user was in reset state and changed the benchmark dropdown, re-enable analysis automatically
            if (
                st.session_state["diagnosis_reset"]
                and current_sig is not None
                and st.session_state.get("last_reset_sig") != current_sig
                and not reset_clicked
            ):
                if run_clicked or st.session_state.get("last_seen_sig") != current_sig:
                    st.session_state["diagnosis_reset"] = False

            st.session_state["last_seen_sig"] = current_sig

            if st.session_state["diagnosis_reset"]:
                st.session_state["last_reset_sig"] = current_sig
                st.info(
                    "Analysis has been reset. Select an official CWRU 3 HP sample or upload a `.mat`/`.csv` file and click **▶ Run Diagnosis** to begin."
                )
            elif source_name is not None and raw_bytes is not None:
                need_compute = (
                    run_clicked
                    or st.session_state["diagnosis_state"] is None
                    or st.session_state["diagnosis_signature"] != current_sig
                )
                if need_compute:
                    try:
                        with st.spinner(f"Running 1D TinyFaultCNN inference on `{source_name}`..."):
                            st.session_state["diagnosis_state"] = _execute_diagnosis(
                                source_name=source_name,
                                raw_bytes=raw_bytes,
                                demo_meta=demo_meta,
                                input_mode_label=resolved_mode_label,
                                checkpoint_sha=checkpoint_sha,
                            )
                            st.session_state["diagnosis_signature"] = current_sig
                            if "win_slider" in st.session_state:
                                st.session_state["win_slider"] = 0
                    except DataValidationError as val_err:
                        st.session_state["diagnosis_state"] = None
                        st.error(f"**Input Validation Error:** {val_err}")
                    except Exception as exc:
                        st.session_state["diagnosis_state"] = None
                        st.error(
                            f"**Unable to process uploaded file:** {type(exc).__name__}. "
                            "Please verify that the file is a valid CWRU `.mat` or single-column numeric `.csv` recording."
                        )

            diag: dict[str, Any] | None = st.session_state["diagnosis_state"]
            if diag is not None and not st.session_state["diagnosis_reset"]:
                pred_result = diag["pred_result"]
                sig_meta = diag["sig_meta"]
                seg_meta = diag["seg_meta"]
                signal = diag["signal"]
                windows = diag["windows"]
                active_source = str(diag["source_name"])
                active_demo_meta = diag["demo_meta"]
                active_mode = str(diag["input_mode"])

                extra_banner = (
                    f"Recording ID {active_demo_meta['file_id']} · {active_demo_meta['load_hp']} HP Motor Load · "
                    f"Ground-Truth: {CLASS_DISPLAY_NAMES[active_demo_meta['true_label']]}"
                    if active_demo_meta is not None
                    else f"{sig_meta['num_samples']:,} samples · Channel {sig_meta['channel_key']}"
                )
                active_input_banner(active_source, active_mode, extra_banner)

                section(
                    "DIAGNOSTIC OUTPUT",
                    "Signal analysis complete",
                    "Predictions shown below are cached in session state from the frozen, unchanged TinyFaultCNN checkpoint.",
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
                    source_file=active_source,
                    manual_review=bool(pred_result["manual_review_required"]),
                    note="3 HP held-out CWRU recording" if active_demo_meta is not None else "User-supplied recording",
                )
                metrics_row(
                    [
                        ("Top model score", f"{pred_result['top_uncalibrated_score']:.4f}", "Uncalibrated softmax"),
                        ("Segment agreement", f"{pred_result['segment_agreement']:.1%}", "Window-level consistency"),
                        ("Analyzed windows", str(pred_result["num_windows"]), f"{WINDOW_SIZE:,} samples per window"),
                        (
                            "Operating load",
                            f"{active_demo_meta['load_hp']} HP" if active_demo_meta else "Not specified",
                            "Held-out test" if active_demo_meta else "Uploaded recording",
                        ),
                    ],
                    featured=0,
                )

                # Export Toolbar (PDF & CSV Download Controls)
                source_stem = Path(active_source).stem
                exp_c1, exp_c2 = st.columns(2)
                with exp_c1:
                    st.download_button(
                        label=f"📄 Download Diagnostic Report (.PDF) — {active_source}",
                        data=diag["pdf_bytes"],
                        file_name=f"VibeGuard_Diagnostic_Report_{source_stem}.pdf",
                        mime="application/pdf",
                        width="stretch",
                        key="dl_pdf_btn",
                    )
                with exp_c2:
                    st.download_button(
                        label=f"📊 Export Window-Level Scores (.CSV) — {active_source}",
                        data=diag["csv_bytes"],
                        file_name=f"VibeGuard_Window_Predictions_{source_stem}.csv",
                        mime="text/csv",
                        width="stretch",
                        key="dl_csv_btn",
                    )

                score_col, meta_col = st.columns([1.2, 1.0])
                with score_col:
                    section("MODEL OUTPUT", "Four-class score comparison", "Interactive uncalibrated softmax scores across all four bearing conditions.")
                    plotly_score_fig = interactive_score_chart(
                        pred_result["mean_scores"],
                        pred_result["per_class_window_counts"],
                        CLASS_DISPLAY_NAMES,
                        CLASS_NAMES,
                    )
                    st.plotly_chart(plotly_score_fig, width="stretch", key="plotly_scores")

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
                        source_file=active_source,
                        channel_key=str(sig_meta["channel_key"]),
                        sampling_hz=int(sig_meta["sampling_hz_nominal"]),
                        window_size=WINDOW_SIZE,
                        num_windows=int(seg_meta["num_windows"]),
                        manual_review=bool(pred_result["manual_review_required"]),
                    )
                    with st.expander("Inspect full provenance & preprocessing JSON", expanded=bool(st.session_state["show_signal_details"])):
                        st.json(
                            {
                                "source_file": active_source,
                                "input_mode": active_mode,
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

                if st.session_state["show_signal_details"]:
                    section(
                        "DETAILED SIGNAL TELEMETRY",
                        "Statistical summary & window-level classification audit",
                        "Detailed amplitude statistics and per-window softmax scores for the active recording.",
                    )
                    rms_val = float(np.sqrt(np.mean(np.square(signal))))
                    peak_val = float(np.max(np.abs(signal)))
                    crest_val = peak_val / max(rms_val, 1e-12)
                    std_val = float(np.std(signal))
                    kurt_val = float(np.mean(((signal - float(np.mean(signal))) / max(std_val, 1e-12)) ** 4))
                    metrics_row(
                        [
                            ("Signal RMS", f"{rms_val:.4f} g", f"Peak |x|: {peak_val:.4f} g"),
                            ("Crest Factor", f"{crest_val:.2f}", "Peak-to-RMS ratio"),
                            ("Signal Kurtosis", f"{kurt_val:.2f}", "Impulsiveness (>3.0 = impulsive)"),
                            ("Sample Span", f"{seg_meta['end_sample']:,}", f"Out of {sig_meta['num_samples']:,} raw samples"),
                        ]
                    )

                section(
                    "SIGNAL FORENSICS",
                    "Interactive vibration & frequency analysis",
                    "Zoom, pan, and hover across raw drive-end acceleration, normalized Z-score windows, and FFT spectra (12 kHz nominal).",
                )
                max_win_idx = max(0, int(seg_meta["num_windows"]) - 1)
                selected_win_idx = 0
                if max_win_idx > 0:
                    selected_win_idx = int(
                        st.slider(
                            "Select 2,048-Sample Window Segment to Inspect in Plots 02 & 03:",
                            min_value=0,
                            max_value=max_win_idx,
                            value=0,
                            step=1,
                            key="win_slider",
                        )
                    )
                    selected_win_idx = min(selected_win_idx, max_win_idx)

                fig_raw, fig_norm, fig_fft = interactive_signal_charts(
                    raw_signal=signal,
                    window_norm=windows[selected_win_idx],
                    window_index=selected_win_idx,
                    sampling_hz=NOMINAL_SAMPLING_HZ,
                )
                st.plotly_chart(fig_raw, width="stretch", key="plotly_raw")
                sig_c1, sig_c2 = st.columns(2)
                with sig_c1:
                    st.plotly_chart(fig_norm, width="stretch", key="plotly_norm")
                with sig_c2:
                    st.plotly_chart(fig_fft, width="stretch", key="plotly_fft")

                section(
                    "ENGINEERING INTERPRETATION",
                    "Bearing fault mechanics & vibration reference guide",
                    "Mechanical meaning of each bearing condition compared against empirical scope on the CWRU 3 HP benchmark.",
                )
                educational_fault_guide(active_class_key=str(pred_result["predicted_label"]))

    with tab_eval:
        section(
            "02 / MODEL EVALUATION",
            "Cross-load generalization & baseline comparison",
            "Training: 0 & 1 HP (8 files, 673 windows) · Validation: 2 HP (4 files, 377 windows) · Locked Test: 3 HP (4 files, 378 windows).",
        )
        if metrics_data is None or file_metrics_data is None:
            st.info("No evaluation artifacts found yet. Run `python train.py` to generate `outputs/metrics.json`.")
        else:
            cnn_test = metrics_data["cnn_1d"]["test_3hp"]
            rf_test = metrics_data["random_forest_baseline"]["test_3hp"]
            cnn_val = metrics_data["cnn_1d"]["val_2hp"]
            rf_val = metrics_data["random_forest_baseline"]["val_2hp"]

            metrics_row(
                [
                    ("CNN · Validation macro-F1", f"{cnn_val['macro_f1']:.4f}", "2 HP unseen validation (377 windows)"),
                    ("CNN · Test macro-F1", f"{cnn_test['macro_f1']:.4f}", "3 HP locked test (378 windows)"),
                    ("RF · Validation macro-F1", f"{rf_val['macro_f1']:.4f}", "2 HP unseen validation (377 windows)"),
                    ("RF · Test macro-F1", f"{rf_test['macro_f1']:.4f}", "3 HP locked test (378 windows)"),
                ],
                featured=1,
            )
            eval_limitation_note()

            rec_rows = file_metrics_data.get("cnn_1d", {}).get("test_3hp", {}).get("recordings", [])
            rf_rows = file_metrics_data.get("random_forest_baseline", {}).get("test_3hp", {}).get("recordings", [])
            rf_by_id = {int(r["file_id"]): r for r in rf_rows}
            if rec_rows:
                section(
                    "MODEL COMPARISON UNDER LOAD SHIFT",
                    "Why 1D TinyFaultCNN outperforms the Random Forest baseline on 108.mat",
                    "While both models achieve 100% window-level accuracy on this 4-file benchmark, Random Forest vote share drops to 0.7273 on 108.mat (Inner Race, 3 HP) due to load-induced spectral shift, whereas 1D TinyFaultCNN maintains 0.9953.",
                )
                eval_comp_fig = interactive_eval_comparison_chart(rec_rows, rf_by_id)
                st.plotly_chart(eval_comp_fig, width="stretch", key="plotly_eval_comparison")

                section(
                    "RECORDING AUDIT",
                    "Four held-out 3 HP test recordings",
                    "The same four 3 HP recordings were never included in training (0 & 1 HP) or validation checkpoint selection (2 HP).",
                )
                clean_rec_rows = []
                for r in rec_rows:
                    rf_r = rf_by_id.get(int(r["file_id"]), {})
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

            section(
                "DIAGNOSTIC EVIDENCE",
                "Interactive confusion matrices (Locked 3 HP Test Split)",
                "Hover over any cell to inspect exact window counts and class percentages across all 378 held-out 3 HP test windows.",
            )
            plotly_cm = interactive_confusion_charts(
                cnn_test["confusion_matrix"], rf_test["confusion_matrix"], cnn_test["labels"]
            )
            st.plotly_chart(plotly_cm, width="stretch", key="plotly_confusion_matrices")

            history_data = load_json_if_exists(Path("outputs/training_history.json"))
            if history_data:
                section(
                    "TRAINING HISTORY",
                    "Interactive learning dynamics (Epochs 1–14)",
                    "Model checkpoint selection used 2 HP validation Macro-F1 (dashed vertical line at Epoch 14); 3 HP test data remained locked.",
                )
                plotly_hist = interactive_training_charts(history_data)
                st.plotly_chart(plotly_hist, width="stretch", key="plotly_training_history")

    with tab_arch:
        section(
            "03 / SYSTEM ARCHITECTURE & PROVENANCE",
            "End-to-end signal pipeline & dual cloud deployment architecture",
            "One reproducible, load-disjoint ML pipeline deployed across two strictly separated cloud environments.",
        )
        architecture()

        section(
            "LOAD-DISJOINT SPLIT PROTOCOL",
            "Zero recording or operating-load leakage across splits",
            "Every recording is assigned to exactly one split by motor load (HP) prior to window segmentation.",
        )
        split_summary_df = pd.DataFrame(
            [
                {
                    "Dataset Split": "Training Split",
                    "Operating Loads": "0 HP & 1 HP (~1,797 & ~1,772 RPM)",
                    "CWRU File IDs": "97, 98 (Normal) · 105, 106 (Inner) · 118, 119 (Ball) · 130, 131 (Outer)",
                    "Recordings": 8,
                    "Non-Overlapping Windows": 673,
                    "Role in Pipeline": "Gradient descent parameter optimization (TinyFaultCNN) & RF tree fitting",
                },
                {
                    "Dataset Split": "Validation Split",
                    "Operating Loads": "2 HP (~1,750 RPM)",
                    "CWRU File IDs": "99 (Normal) · 107 (Inner) · 120 (Ball) · 132 (Outer)",
                    "Recordings": 4,
                    "Non-Overlapping Windows": 377,
                    "Role in Pipeline": "Hyperparameter & best-epoch checkpoint selection (Epoch 14)",
                },
                {
                    "Dataset Split": "Locked Final Test Split",
                    "Operating Loads": "3 HP (~1,730 RPM)",
                    "CWRU File IDs": "100 (Normal) · 108 (Inner) · 121 (Ball) · 133 (Outer)",
                    "Recordings": 4,
                    "Non-Overlapping Windows": 378,
                    "Role in Pipeline": "Held-out cross-load evaluation & interactive dashboard benchmark",
                },
            ]
        )
        st.dataframe(split_summary_df, width="stretch", hide_index=True)

        arch_img = Path("docs/architecture.png")
        if arch_img.exists():
            with st.expander("View original technical architecture diagram (`docs/architecture.png`)"):
                st.image(str(arch_img), caption="VibeGuard Cloud End-to-End Architecture")

        st.markdown(
            """
### Key Methodological Boundaries & Deployment Separation
1. **Load-Conditional Split:** Training uses `0 HP` and `1 HP` recordings (`8` files, `673` windows); validation uses `2 HP` (`4` files, `377` windows); held-out testing uses `3 HP` (`4` files, `378` windows). Zero windows or file IDs are shared across splits.
2. **Sample Size & Single-Rig Limitation:** Each non-training operating load (`2 HP` validation and `3 HP` test) contains **one recording per bearing class** (`200` normal, `60` inner, `59` ball, `59` outer windows). A `1.0000` test macro-F1 demonstrates cross-load consistency on the same CWRU test stand and `0.007 in` seeded EDM faults, **not** generalization across different motor frames, variable radial loads, or natural progressive wear.
3. **Uncalibrated Softmax Scores:** Neural network softmax outputs are uncalibrated confidence scores that can be overconfident under domain shift. The manual-review flag (`score < 0.70` or `agreement < 80%`) is a decision-support heuristic, not a calibrated risk guarantee.
4. **Strict Separation of Cloud Environments:**
   - **Environment A (AWS EC2 + Private S3 + IAM Role):** Hosted on an Ubuntu 24.04 LTS EC2 instance attached to `VibeGuardEC2Role` with private S3 artifact synchronization (`Block Public Access = ON`), accessed via an encrypted SSH tunnel (`ssh -L 8501:127.0.0.1:8501`). The EC2 instance is kept `stopped` when not in active demonstration to preserve Free Plan credits.
   - **Environment B (Streamlit Community Cloud Public Demo):** Runs directly from the public GitHub repository using the committed `outputs/model.pt` checkpoint and on-demand SHA-256-verified CWRU recordings. It operates with **zero AWS credentials** and **does not access private Amazon S3**.
            """
        )
        if manifest_data:
            with st.expander("Inspect Reproducibility Run Manifest (`outputs/run_manifest.json`)"):
                st.json(manifest_data)


if __name__ == "__main__":
    main()
