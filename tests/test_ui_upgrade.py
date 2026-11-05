"""Automated UI workflow, interactive chart, and report export tests for VibeGuard Cloud."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from app import _execute_diagnosis
from data_utils import CLASS_DISPLAY_NAMES, CLASS_NAMES
from report_utils import build_diagnostic_csv_bytes, build_diagnostic_pdf_bytes
from ui_design import (
    interactive_confusion_charts,
    interactive_eval_comparison_chart,
    interactive_score_chart,
    interactive_signal_charts,
    interactive_training_charts,
)


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_default_load_and_all_four_recordings() -> None:
    """Verify default app load (108.mat) and switching across all 4 CWRU 3 HP recordings."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception, f"Unexpected exception on initial load: {at.exception}"
    assert at.session_state["diagnosis_state"] is not None

    expected = {
        "100.mat": ("normal", "Normal Baseline", 0.986942, 200),
        "108.mat": ("inner", "Inner Race Fault (0.007 in)", 0.995320, 60),
        "121.mat": ("ball", "Ball Fault (0.007 in)", 0.989040, 59),
        "133.mat": ("outer", "Outer Race Fault @ 6:00 (0.007 in)", 0.985114, 59),
    }

    sb = at.selectbox(key="cwru_demo_selectbox")
    for opt in sb.options[1:]:
        sb.select(opt).run()
        assert not at.exception, f"Exception when selecting {opt}: {at.exception}"
        diag = at.session_state["diagnosis_state"]
        assert diag is not None
        src = diag["source_name"]
        exp_key, exp_disp, exp_score, exp_wins = expected[src]
        pred = diag["pred_result"]
        assert pred["predicted_label"] == exp_key
        assert pred["predicted_display"] == exp_disp
        assert abs(float(pred["top_uncalibrated_score"]) - exp_score) < 1e-4
        assert int(pred["per_class_window_counts"][exp_key]) == exp_wins


def test_ui_workflow_controls_reset_run_and_details() -> None:
    """Verify Reset Analysis, Run Diagnosis, View Signal Details, and Window Slider controls."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.session_state["diagnosis_state"] is not None
    assert at.session_state["show_signal_details"] is False

    # 1. Click 'View Signal Details' button
    at.button(key="btn_toggle_details").click().run()
    assert not at.exception
    assert at.session_state["show_signal_details"] is True

    # 2. Move Window Segment slider to inspect Window #5 without re-running model inference
    slider = at.slider(key="win_slider")
    slider.set_value(5).run()
    assert not at.exception
    assert at.session_state["diagnosis_state"]["source_name"] == "108.mat"

    # 3. Click 'Reset Analysis' button
    at.button(key="btn_reset_analysis").click().run()
    assert not at.exception
    assert at.session_state["diagnosis_reset"] is True
    assert at.session_state["diagnosis_state"] is None
    assert any("Analysis has been reset" in info.value for info in at.info)

    # 4. Click 'Run Diagnosis' button to restore analysis
    at.button(key="btn_run_diagnosis").click().run()
    assert not at.exception
    assert at.session_state["diagnosis_reset"] is False
    assert at.session_state["diagnosis_state"] is not None
    assert at.session_state["diagnosis_state"]["pred_result"]["predicted_label"] == "inner"


def test_pdf_and_csv_diagnostic_report_export() -> None:
    """Verify PDF and CSV report generation on synthetic/real signal data and absence of private cloud info."""
    t = np.linspace(0.0, 1.0, 4096, dtype=np.float64)
    sig = 0.6 * np.sin(2.0 * np.pi * 120.0 * t) + 0.25 * np.cos(2.0 * np.pi * 540.0 * t)
    csv_raw = "\n".join(f"{v:.6f}" for v in sig).encode("utf-8")

    diag = _execute_diagnosis(
        source_name="unit_test_signal.csv",
        raw_bytes=csv_raw,
        demo_meta=None,
        input_mode_label="Uploaded Custom Recording (.mat / .csv)",
        checkpoint_sha="3cd1ac3b823ea903",
    )

    pdf_bytes = build_diagnostic_pdf_bytes(diag)
    assert pdf_bytes.startswith(b"%PDF-"), "Generated PDF must have a valid %PDF- header"
    assert len(pdf_bytes) > 10000, "Generated PDF should include embedded chart and tables"
    assert b"810995308548" not in pdf_bytes, "PDF report must never expose private AWS account IDs"
    assert b"vibeguard-cloud-2026" not in pdf_bytes, "PDF report must never expose private S3 bucket names"

    csv_bytes = build_diagnostic_csv_bytes(diag)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(df) == 2
    for required_col in [
        "source_file",
        "window_index",
        "window_predicted_condition",
        "score_normal",
        "score_inner",
        "score_ball",
        "score_outer",
        "recording_predicted_condition",
        "manual_review_required",
    ]:
        assert required_col in df.columns


def test_interactive_plotly_figures() -> None:
    """Verify all interactive Plotly chart builders return valid figures with accurate traces and axes."""
    scores = {"normal": 0.001, "inner": 0.995, "ball": 0.002, "outer": 0.002}
    counts = {"normal": 0, "inner": 60, "ball": 0, "outer": 0}
    fig_score = interactive_score_chart(scores, counts, CLASS_DISPLAY_NAMES, CLASS_NAMES)
    assert isinstance(fig_score, go.Figure)
    assert len(fig_score.data) == 1

    raw_sig = np.sin(np.linspace(0, 20, 4096))
    win_norm = np.sin(np.linspace(0, 10, 2048))
    fig_raw, fig_norm, fig_fft = interactive_signal_charts(raw_sig, win_norm, window_index=0, sampling_hz=12000)
    assert isinstance(fig_raw, go.Figure)
    assert isinstance(fig_norm, go.Figure)
    assert isinstance(fig_fft, go.Figure)

    metrics = json.loads(Path("outputs/metrics.json").read_text(encoding="utf-8"))
    file_metrics = json.loads(Path("outputs/file_level_metrics.json").read_text(encoding="utf-8"))
    history = json.loads(Path("outputs/training_history.json").read_text(encoding="utf-8"))

    rec_rows = file_metrics["cnn_1d"]["test_3hp"]["recordings"]
    rf_rows = file_metrics["random_forest_baseline"]["test_3hp"]["recordings"]
    rf_by_id = {int(r["file_id"]): r for r in rf_rows}

    fig_eval = interactive_eval_comparison_chart(rec_rows, rf_by_id)
    assert isinstance(fig_eval, go.Figure)
    assert len(fig_eval.data) == 2

    cnn_cm = metrics["cnn_1d"]["test_3hp"]["confusion_matrix"]
    rf_cm = metrics["random_forest_baseline"]["test_3hp"]["confusion_matrix"]
    fig_cm = interactive_confusion_charts(cnn_cm, rf_cm, CLASS_NAMES)
    assert isinstance(fig_cm, go.Figure)
    assert len(fig_cm.data) == 2

    fig_hist = interactive_training_charts(history)
    assert isinstance(fig_hist, go.Figure)
    assert len(fig_hist.data) == 4


def test_invalid_uploads_rejected_cleanly() -> None:
    """Verify corrupted .mat, short .csv, and unsupported extensions raise DataValidationError."""
    import pytest
    from app import parse_uploaded_recording
    from data_utils import DataValidationError

    with pytest.raises(DataValidationError):
        parse_uploaded_recording("corrupted.mat", b"not-a-valid-matlab-binary")

    with pytest.raises(DataValidationError):
        parse_uploaded_recording("too_short.csv", b"0.1\n0.2\n0.3\n")

    with pytest.raises(DataValidationError):
        parse_uploaded_recording("unsupported.txt", b"1,2,3")

