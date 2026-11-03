"""Core unit and integration tests for VibeGuard Cloud preprocessing, model, uploads, and cloud sync."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch

import cloud_sync
from app import parse_uploaded_recording
from data_utils import (
    CLASS_NAMES,
    DataValidationError,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    WINDOW_SIZE,
    extract_de_signal_from_mat,
    extract_handcrafted_features,
    extract_signal_from_csv,
    normalize_window,
    segment_and_normalize_signal,
)
from model import (
    TinyFaultCNN,
    count_trainable_parameters,
    evaluate_manual_review_flag,
    load_model_checkpoint,
    predict_windows,
    save_model_checkpoint,
)


def make_mat_bytes(payload: dict[str, object]) -> bytes:
    """Serialize a dictionary into MATLAB v5 binary bytes."""
    buf = io.BytesIO()
    scipy.io.savemat(buf, payload)
    return buf.getvalue()


def test_label_ordering_and_mappings() -> None:
    assert CLASS_NAMES == ["normal", "inner", "ball", "outer"]
    for idx, name in enumerate(CLASS_NAMES):
        assert LABEL_TO_IDX[name] == idx
        assert IDX_TO_LABEL[idx] == name


def test_model_shapes_and_parameter_count() -> None:
    model = TinyFaultCNN(num_classes=4, dropout_p=0.2)
    assert count_trainable_parameters(model) > 0

    x_3d = torch.randn(5, 1, WINDOW_SIZE)
    out_3d = model(x_3d)
    assert out_3d.shape == (5, 4)

    x_2d = torch.randn(3, WINDOW_SIZE)
    out_2d = model(x_2d)
    assert out_2d.shape == (3, 4)


def test_deterministic_segmentation_and_clamping() -> None:
    rng = np.random.default_rng(42)
    sig = rng.normal(loc=3.5, scale=2.0, size=WINDOW_SIZE * 4 + 300).astype(np.float32)
    # Inject extreme spike to test [-10, 10] outlier clamping
    sig[10] = 500.0
    sig[20] = -500.0

    w1, meta1 = segment_and_normalize_signal(sig)
    w2, meta2 = segment_and_normalize_signal(sig)

    assert w1.shape == (4, WINDOW_SIZE)
    assert np.allclose(w1, w2)
    assert meta1 == meta2
    assert meta1["start_sample"] == 0
    assert meta1["end_sample"] == 4 * WINDOW_SIZE
    assert float(np.max(w1)) <= 10.0
    assert float(np.min(w1)) >= -10.0


def test_nan_and_inf_rejection() -> None:
    sig = np.ones(WINDOW_SIZE * 2, dtype=np.float32)
    sig[50] = np.nan
    with pytest.raises(DataValidationError, match="NaN or infinite"):
        segment_and_normalize_signal(sig)

    win = np.arange(WINDOW_SIZE, dtype=np.float32)
    win[0] = np.inf
    with pytest.raises(DataValidationError, match="NaN or infinite"):
        normalize_window(win)


def test_constant_signal_rejection() -> None:
    const_win = np.full(WINDOW_SIZE, 5.0, dtype=np.float32)
    with pytest.raises(DataValidationError, match="zero-variance"):
        normalize_window(const_win)


def test_case_sensitive_de_time_channel_validation() -> None:
    rng = np.random.default_rng(7)
    valid_sig = rng.standard_normal((WINDOW_SIZE * 2, 1))
    valid_bytes = make_mat_bytes({"X105_DE_time": valid_sig, "X105RPM": np.array([[1797]])})
    sig, meta = extract_de_signal_from_mat(valid_bytes)
    assert sig.shape == (WINDOW_SIZE * 2,)
    assert meta["channel_key"] == "X105_DE_time"
    assert meta["recorded_rpm"] == 1797.0

    # Wrong case (_de_time instead of _DE_time) must be rejected
    wrong_case_bytes = make_mat_bytes({"X105_de_time": valid_sig})
    with pytest.raises(DataValidationError, match="Required drive-end channel"):
        extract_de_signal_from_mat(wrong_case_bytes)

    # HTML error page saved as .mat must be rejected
    html_bytes = b"<!DOCTYPE html><html><body>403 Forbidden</body></html>"
    with pytest.raises(DataValidationError, match="HTML page"):
        extract_de_signal_from_mat(html_bytes)

    # Multi-key MAT file (like CWRU 99.mat containing both X098_DE_time and X099_DE_time)
    # must select X099_DE_time to prevent leaking 1 HP X098_DE_time into 2 HP validation.
    multi_bytes = make_mat_bytes(
        {
            "X098_DE_time": rng.standard_normal((WINDOW_SIZE, 1)),
            "X099_DE_time": rng.standard_normal((WINDOW_SIZE * 2, 1)),
        }
    )
    sig_99, meta_99 = extract_de_signal_from_mat(multi_bytes)
    assert meta_99["channel_key"] == "X099_DE_time"
    assert sig_99.shape == (WINDOW_SIZE * 2,)


def test_short_signal_and_file_size_guards() -> None:
    short_bytes = make_mat_bytes({"X097_DE_time": np.random.randn(1024, 1)})
    with pytest.raises(DataValidationError, match="shorter than minimum window size"):
        extract_de_signal_from_mat(short_bytes)

    valid_bytes = make_mat_bytes({"X097_DE_time": np.random.randn(WINDOW_SIZE, 1)})
    with pytest.raises(DataValidationError, match="exceeds maximum size limit"):
        extract_de_signal_from_mat(valid_bytes, max_bytes=128)


def test_csv_validation_and_upload_parser() -> None:
    rng = np.random.default_rng(99)
    vals = rng.standard_normal(WINDOW_SIZE + 100)
    csv_text = "acceleration_g\n" + "\n".join(f"{v:.6f}" for v in vals)
    sig, meta = parse_uploaded_recording("sample.csv", csv_text.encode("utf-8"))
    assert sig.shape == (WINDOW_SIZE + 100,)
    assert meta["channel_key"] == "csv_single_column"

    # Multi-column CSV must be rejected
    bad_csv = "a,b\n" + "\n".join("1.0,2.0" for _ in range(WINDOW_SIZE))
    with pytest.raises(DataValidationError, match="exactly 1 column"):
        extract_signal_from_csv(bad_csv.encode("utf-8"))

    # Unsupported extension must be rejected
    with pytest.raises(DataValidationError, match="Unsupported file format"):
        parse_uploaded_recording("bad_file.txt", b"12345")


def test_handcrafted_features_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    raw = rng.standard_normal(WINDOW_SIZE * 3).astype(np.float32)
    windows, _ = segment_and_normalize_signal(raw)

    feats = extract_handcrafted_features(windows)
    assert feats.shape == (3, 18)
    assert np.all(np.isfinite(feats))

    model = TinyFaultCNN(num_classes=4)
    ckpt_path = tmp_path / "model.pt"
    save_model_checkpoint(model, ckpt_path, metadata={"run_id": "unit-test-run"})

    loaded_model, meta = load_model_checkpoint(ckpt_path)
    assert meta["metadata"]["run_id"] == "unit-test-run"

    pred_a = predict_windows(model, windows)
    pred_b = predict_windows(loaded_model, windows)
    assert pred_a["predicted_label"] == pred_b["predicted_label"]
    assert np.allclose(pred_a["window_probabilities"], pred_b["window_probabilities"])


def test_manual_review_heuristic() -> None:
    # High score & high agreement -> no review required
    res_ok = evaluate_manual_review_flag(
        mean_scores=np.array([0.92, 0.03, 0.03, 0.02]),
        window_preds=np.array([0, 0, 0, 0, 0]),
    )
    assert res_ok["manual_review_required"] is False

    # Low top score -> review required
    res_low_conf = evaluate_manual_review_flag(
        mean_scores=np.array([0.45, 0.35, 0.10, 0.10]),
        window_preds=np.array([0, 0, 0, 0, 0]),
    )
    assert res_low_conf["manual_review_required"] is True

    # Inconsistent window predictions -> review required
    res_low_agree = evaluate_manual_review_flag(
        mean_scores=np.array([0.75, 0.15, 0.05, 0.05]),
        window_preds=np.array([0, 0, 1, 1, 2]),
    )
    assert res_low_agree["manual_review_required"] is True


def test_cloud_sync_offline_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIBEGUARD_S3_BUCKET", raising=False)
    status = cloud_sync.probe_s3_status()
    assert status["configured"] is False
    assert status["connected"] is False

    sample_file = tmp_path / "artifact.json"
    sample_file.write_text('{"ok": true}', encoding="utf-8")
    dry_report = cloud_sync.sync_directory_to_s3(
        bucket="vibeguard-dry-run-bucket",
        source_dir=tmp_path,
        prefix="artifacts",
        dry_run=True,
    )
    assert dry_report["num_files"] == 1
    assert dry_report["objects"][0]["s3_key"] == "artifacts/artifact.json"


def test_real_checkpoint_and_four_conditions_end_to_end() -> None:
    """Verify loaded outputs/model.pt matches outputs/file_level_metrics.json across all 4 conditions."""
    import json
    import pandas as pd
    import pyarrow as pa

    # Verify NumPy 1.26.4 + PyArrow 18.1.0 + Pandas DataFrame-to-Arrow table conversion used by Streamlit
    df_check = pd.DataFrame({"class": CLASS_NAMES, "score": [0.98, 0.01, 0.005, 0.005]})
    arrow_tbl = pa.Table.from_pandas(df_check)
    assert arrow_tbl.num_rows == 4

    model_path = Path("outputs/model.pt")
    file_metrics_path = Path("outputs/file_level_metrics.json")
    if not model_path.exists() or not file_metrics_path.exists():
        pytest.skip("Trained model artifacts not yet generated.")

    model, ckpt_meta = load_model_checkpoint(model_path, device="cpu")
    recorded = json.loads(file_metrics_path.read_text(encoding="utf-8"))
    assert ckpt_meta["metadata"]["run_id"] == recorded["run_id"]

    test_files = [
        ("normal", Path("data/raw/normal/100.mat"), 100),
        ("inner", Path("data/raw/inner/108.mat"), 108),
        ("ball", Path("data/raw/ball/121.mat"), 121),
        ("outer", Path("data/raw/outer/133.mat"), 133),
    ]
    rec_by_id = {
        r["file_id"]: r for r in recorded["cnn_1d"]["test_3hp"]["recordings"]
    }

    for expected_label, mat_path, fid in test_files:
        if not mat_path.exists():
            continue
        raw_bytes = mat_path.read_bytes()
        sig, _ = parse_uploaded_recording(mat_path.name, raw_bytes)
        windows, _ = segment_and_normalize_signal(sig)
        pred = predict_windows(model, windows, device="cpu")

        assert pred["predicted_label"] == rec_by_id[fid]["predicted_label"]
        assert pred["predicted_label"] == expected_label
        assert pred["num_windows"] == rec_by_id[fid]["num_windows"]
        for cls_name in CLASS_NAMES:
            assert abs(
                pred["mean_scores"][cls_name]
                - rec_by_id[fid]["mean_uncalibrated_scores"][cls_name]
            ) < 1e-5

