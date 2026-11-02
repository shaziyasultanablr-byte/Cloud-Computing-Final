"""Reproducible training and locked cross-load evaluation for VibeGuard Cloud (1D TinyFaultCNN + Random Forest baseline)."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import (
    CLAMP_MAX,
    CLAMP_MIN,
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    IDX_TO_LABEL,
    MAX_WINDOWS_PER_FILE,
    NOMINAL_SAMPLING_HZ,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    build_split_datasets,
    compute_sha256,
    extract_handcrafted_features,
)
from model import (
    DEFAULT_AGREEMENT_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    TinyFaultCNN,
    count_trainable_parameters,
    evaluate_manual_review_flag,
    load_model_checkpoint,
    save_model_checkpoint,
)
from scripts.validate_dataset import validate_all_recordings


def set_deterministic_seed(seed: int = 42) -> None:
    """Configure deterministic seeds across Python, NumPy, and PyTorch CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_git_sha() -> str:
    """Return current Git commit SHA or 'uncommitted' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out if out else "uncommitted"
    except Exception:
        return "uncommitted"


def compute_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute window-level accuracy, macro-F1, per-class precision/recall/F1/support, and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec, rec, f1s, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
    ).tolist()

    per_class: dict[str, dict[str, float | int]] = {}
    for idx, cls_name in enumerate(CLASS_NAMES):
        per_class[cls_name] = {
            "precision": round(float(prec[idx]), 6),
            "recall": round(float(rec[idx]), 6),
            "f1": round(float(f1s[idx]), 6),
            "support": int(sup[idx]),
        }

    return {
        "num_windows": int(len(y_true)),
        "accuracy": round(acc, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": CLASS_NAMES,
    }


def compute_recording_level_metrics(
    records: list[dict[str, Any]],
    file_ids: np.ndarray,
    window_probs: np.ndarray,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate window-level probabilities into recording-level predictions using mean softmax score."""
    per_file_results: list[dict[str, Any]] = []
    y_true_files: list[int] = []
    y_pred_files: list[int] = []

    for rec in records:
        fid = rec["file_id"]
        mask = file_ids == fid
        probs_f = window_probs[mask]
        preds_f = np.argmax(probs_f, axis=1)
        mean_scores = np.mean(probs_f, axis=0)
        pred_idx = int(np.argmax(mean_scores))
        true_idx = int(rec["class_index"])

        y_true_files.append(true_idx)
        y_pred_files.append(pred_idx)

        review = evaluate_manual_review_flag(
            mean_scores=mean_scores,
            window_preds=preds_f,
            confidence_threshold=confidence_threshold,
            agreement_threshold=agreement_threshold,
        )

        per_file_results.append(
            {
                "file_id": fid,
                "filename": rec["filename"],
                "split": rec["split"],
                "load_hp": rec["load_hp"],
                "true_label": rec["label"],
                "predicted_label": IDX_TO_LABEL[pred_idx],
                "correct": bool(pred_idx == true_idx),
                "num_windows": int(probs_f.shape[0]),
                "window_accuracy_within_file": round(float(np.mean(preds_f == true_idx)), 6),
                "mean_uncalibrated_scores": {
                    CLASS_NAMES[i]: round(float(mean_scores[i]), 6)
                    for i in range(len(CLASS_NAMES))
                },
                **review,
            }
        )

    correct_count = sum(1 for r in per_file_results if r["correct"])
    total_files = len(per_file_results)
    macro_f1 = float(
        f1_score(
            y_true_files,
            y_pred_files,
            labels=list(range(len(CLASS_NAMES))),
            average="macro",
            zero_division=0,
        )
    )
    return {
        "num_recordings": total_files,
        "correct_recordings": correct_count,
        "recording_accuracy": round(correct_count / max(total_files, 1), 6),
        "recording_macro_f1": round(macro_f1, 6),
        "recordings": per_file_results,
    }


@torch.no_grad()
def evaluate_cnn_split(
    model: TinyFaultCNN,
    X: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate TinyFaultCNN on a split, returning (mean_loss, window_probs, window_preds)."""
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(X).unsqueeze(1),
        torch.from_numpy(y).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_count = 0
    probs_all: list[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_all.append(probs)
        total_loss += float(loss.item()) * len(yb)
        total_count += len(yb)

    window_probs = np.concatenate(probs_all, axis=0)
    window_preds = np.argmax(window_probs, axis=1)
    mean_loss = total_loss / max(total_count, 1)
    return mean_loss, window_probs, window_preds


def plot_training_history(
    history: list[dict[str, Any]],
    best_epoch: int,
    out_path: Path,
) -> None:
    """Render training loss and validation macro-F1 curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_f1 = [h["train_macro_f1"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss (0 & 1 HP)", color="#1d4ed8")
    axes[0].plot(epochs, val_loss, marker="s", linewidth=2, label="Val Loss (2 HP)", color="#dc2626")
    axes[0].axvline(best_epoch, linestyle="--", color="#059669", label=f"Selected Epoch ({best_epoch})")
    axes[0].set_title("Cross-Entropy Loss vs. Epoch", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(frameon=True)

    axes[1].plot(epochs, train_f1, marker="o", linewidth=2, label="Train Macro-F1 (0 & 1 HP)", color="#1d4ed8")
    axes[1].plot(epochs, val_f1, marker="s", linewidth=2, label="Val Macro-F1 (2 HP)", color="#059669")
    axes[1].axvline(best_epoch, linestyle="--", color="#059669", label=f"Best Val Checkpoint ({best_epoch})")
    axes[1].set_title("Macro-F1 Score vs. Epoch (Model Selection on 2 HP)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True)

    fig.suptitle("VibeGuard Cloud — 1D TinyFaultCNN Training Dynamics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_and_comparison(
    cnn_test_metrics: dict[str, Any],
    rf_test_metrics: dict[str, Any],
    out_path: Path,
) -> None:
    """Render side-by-side 3 HP test confusion matrices and per-class F1 comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), dpi=160)

    cms = [
        ("1D TinyFaultCNN (3 HP Held-Out Test)", np.array(cnn_test_metrics["confusion_matrix"]), "Blues"),
        ("Random Forest Baseline (3 HP Held-Out Test)", np.array(rf_test_metrics["confusion_matrix"]), "Oranges"),
    ]

    for idx, (title, cm, cmap) in enumerate(cms):
        ax = axes[idx]
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES])
        ax.set_yticklabels([c.capitalize() for c in CLASS_NAMES])
        ax.set_xlabel("Predicted Condition")
        ax.set_ylabel("True Condition")

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bar chart comparing CNN vs RF per-class F1 on 3 HP test
    ax3 = axes[2]
    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    cnn_f1s = [cnn_test_metrics["per_class"][c]["f1"] for c in CLASS_NAMES]
    rf_f1s = [rf_test_metrics["per_class"][c]["f1"] for c in CLASS_NAMES]

    rects1 = ax3.bar(x - width / 2, cnn_f1s, width, label=f"1D CNN (Macro-F1={cnn_test_metrics['macro_f1']:.3f})", color="#1d4ed8")
    rects2 = ax3.bar(x + width / 2, rf_f1s, width, label=f"Random Forest (Macro-F1={rf_test_metrics['macro_f1']:.3f})", color="#ea580c")

    ax3.set_title("Per-Class F1 Comparison on Unseen 3 HP Load", fontsize=10.5, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.capitalize() for c in CLASS_NAMES])
    ax3.set_ylim(0.0, 1.15)
    ax3.set_ylabel("F1 Score")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", frameon=True, fontsize=8.5)

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        ax3.annotate(
            f"{h:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Held-Out Cross-Load Evaluation (Trained on 0 & 1 HP -> Tested on 3 HP)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_training_pipeline(
    raw_dir: str | Path = "data/raw",
    manifest_path: str | Path = "docs/DATASET_MANIFEST.csv",
    output_dir: str | Path = "outputs",
    checksums_csv: str | Path = "docs/dataset_checksums.csv",
    epochs: int = 14,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute full dataset validation, CNN + RF training, model selection on 2 HP, and locked evaluation on 3 HP."""
    set_deterministic_seed(seed)
    started_utc = datetime.now(timezone.utc)
    run_id = f"vibeguard-{started_utc.strftime('%Y%m%dT%H%M%SZ')}-s{seed}"
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Step 1: Validate all 16 CWRU recordings and update dataset_checksums.csv
    validated_rows = validate_all_recordings(
        manifest_path=manifest_path,
        raw_dir=raw_dir,
        output_csv=checksums_csv,
    )

    # Step 2: Build disjoint splits
    splits = build_split_datasets(
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        max_windows=MAX_WINDOWS_PER_FILE,
    )

    X_train, y_train = splits["train"]["X"], splits["train"]["y"]
    X_val, y_val = splits["val"]["X"], splits["val"]["y"]
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]

    # Step 3: Train 1D TinyFaultCNN on CPU and select checkpoint with highest 2 HP validation macro-F1
    device = "cpu"
    model = TinyFaultCNN(num_classes=len(CLASS_NAMES), dropout_p=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).unsqueeze(1),
        torch.from_numpy(y_train).long(),
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )

    history: list[dict[str, Any]] = []
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 1
    model_path = out_root / "model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_loss, _, train_preds = evaluate_cnn_split(
            model, X_train, y_train, criterion, batch_size=batch_size, device=device
        )
        val_loss, _, val_preds = evaluate_cnn_split(
            model, X_val, y_val, criterion, batch_size=batch_size, device=device
        )

        train_metrics_ep = compute_window_metrics(y_train, train_preds)
        val_metrics_ep = compute_window_metrics(y_val, val_preds)

        ep_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": train_metrics_ep["accuracy"],
            "train_macro_f1": train_metrics_ep["macro_f1"],
            "val_loss": round(val_loss, 6),
            "val_accuracy": val_metrics_ep["accuracy"],
            "val_macro_f1": val_metrics_ep["macro_f1"],
        }
        history.append(ep_record)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} train_f1={train_metrics_ep['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_metrics_ep['macro_f1']:.4f}"
        )

        # Select checkpoint strictly by 2 HP validation macro-F1 (tie-break by lower val_loss)
        is_better = (val_metrics_ep["macro_f1"] > best_val_macro_f1 + 1e-9) or (
            abs(val_metrics_ep["macro_f1"] - best_val_macro_f1) <= 1e-9
            and val_loss < best_val_loss
        )
        if is_better:
            best_val_macro_f1 = val_metrics_ep["macro_f1"]
            best_val_loss = val_loss
            best_epoch = epoch
            save_model_checkpoint(
                model,
                model_path,
                metadata={
                    "run_id": run_id,
                    "selected_epoch": best_epoch,
                    "val_macro_f1": best_val_macro_f1,
                    "val_loss": round(best_val_loss, 6),
                    "seed": seed,
                    "window_size": WINDOW_SIZE,
                    "stride": WINDOW_STRIDE,
                    "max_windows_per_file": MAX_WINDOWS_PER_FILE,
                    "clamp_range": [CLAMP_MIN, CLAMP_MAX],
                    "nominal_sampling_hz": NOMINAL_SAMPLING_HZ,
                    "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                    "agreement_threshold": DEFAULT_AGREEMENT_THRESHOLD,
                },
            )

    # Step 4: Load best checkpoint selected on 2 HP validation and run locked evaluation once on 3 HP test
    best_model, ckpt_meta = load_model_checkpoint(model_path, device=device)
    train_loss_final, train_probs, train_preds = evaluate_cnn_split(
        best_model, X_train, y_train, criterion, batch_size=batch_size, device=device
    )
    val_loss_final, val_probs, val_preds = evaluate_cnn_split(
        best_model, X_val, y_val, criterion, batch_size=batch_size, device=device
    )
    test_loss_final, test_probs, test_preds = evaluate_cnn_split(
        best_model, X_test, y_test, criterion, batch_size=batch_size, device=device
    )

    cnn_train_metrics = compute_window_metrics(y_train, train_preds)
    cnn_val_metrics = compute_window_metrics(y_val, val_preds)
    cnn_test_metrics = compute_window_metrics(y_test, test_preds)

    # Step 5: Train Random Forest baseline on identical 0 & 1 HP training windows
    X_train_feat = extract_handcrafted_features(X_train, sampling_hz=NOMINAL_SAMPLING_HZ)
    X_val_feat = extract_handcrafted_features(X_val, sampling_hz=NOMINAL_SAMPLING_HZ)
    X_test_feat = extract_handcrafted_features(X_test, sampling_hz=NOMINAL_SAMPLING_HZ)

    scaler = StandardScaler()
    X_train_feat_s = scaler.fit_transform(X_train_feat)
    X_val_feat_s = scaler.transform(X_val_feat)
    X_test_feat_s = scaler.transform(X_test_feat)

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=seed,
        n_jobs=1,
    )
    rf_clf.fit(X_train_feat_s, y_train)

    rf_train_preds = rf_clf.predict(X_train_feat_s)
    rf_val_preds = rf_clf.predict(X_val_feat_s)
    rf_test_preds = rf_clf.predict(X_test_feat_s)
    rf_val_probs = rf_clf.predict_proba(X_val_feat_s)
    rf_test_probs = rf_clf.predict_proba(X_test_feat_s)

    rf_train_metrics = compute_window_metrics(y_train, rf_train_preds)
    rf_val_metrics = compute_window_metrics(y_val, rf_val_preds)
    rf_test_metrics = compute_window_metrics(y_test, rf_test_preds)

    # Step 6: Compute recording-level metrics on 2 HP validation and 3 HP held-out test
    cnn_val_file_metrics = compute_recording_level_metrics(
        splits["val"]["records"], splits["val"]["file_ids"], val_probs
    )
    cnn_test_file_metrics = compute_recording_level_metrics(
        splits["test"]["records"], splits["test"]["file_ids"], test_probs
    )
    rf_val_file_metrics = compute_recording_level_metrics(
        splits["val"]["records"], splits["val"]["file_ids"], rf_val_probs
    )
    rf_test_file_metrics = compute_recording_level_metrics(
        splits["test"]["records"], splits["test"]["file_ids"], rf_test_probs
    )

    # Step 7: Save test_predictions.csv
    test_pred_csv_path = out_root / "test_predictions.csv"
    file_id_to_rec = {r["file_id"]: r for r in splits["test"]["records"]}
    with open(test_pred_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "window_index",
                "file_id",
                "filename",
                "load_hp",
                "true_label",
                "predicted_label",
                "rf_predicted_label",
                "correct",
                "score_normal",
                "score_inner",
                "score_ball",
                "score_outer",
                "top_uncalibrated_score",
            ],
        )
        writer.writeheader()
        for idx in range(len(y_test)):
            fid = int(splits["test"]["file_ids"][idx])
            rec = file_id_to_rec[fid]
            true_lbl = IDX_TO_LABEL[int(y_test[idx])]
            pred_lbl = IDX_TO_LABEL[int(test_preds[idx])]
            rf_lbl = IDX_TO_LABEL[int(rf_test_preds[idx])]
            probs_row = test_probs[idx]
            writer.writerow(
                {
                    "run_id": run_id,
                    "window_index": idx,
                    "file_id": fid,
                    "filename": rec["filename"],
                    "load_hp": rec["load_hp"],
                    "true_label": true_lbl,
                    "predicted_label": pred_lbl,
                    "rf_predicted_label": rf_lbl,
                    "correct": bool(true_lbl == pred_lbl),
                    "score_normal": round(float(probs_row[0]), 6),
                    "score_inner": round(float(probs_row[1]), 6),
                    "score_ball": round(float(probs_row[2]), 6),
                    "score_outer": round(float(probs_row[3]), 6),
                    "top_uncalibrated_score": round(float(np.max(probs_row)), 6),
                }
            )

    # Step 8: Generate figures
    loss_fig_path = out_root / "training_loss.png"
    cm_fig_path = out_root / "confusion_matrix.png"
    plot_training_history(history, best_epoch=best_epoch, out_path=loss_fig_path)
    plot_confusion_and_comparison(cnn_test_metrics, rf_test_metrics, out_path=cm_fig_path)

    # Step 9: Save metrics.json, file_level_metrics.json, training_history.json, run_manifest.json
    completed_utc = datetime.now(timezone.utc)
    metrics_payload = {
        "run_id": run_id,
        "timestamp_utc": completed_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "selected_epoch": best_epoch,
        "total_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "model_trainable_parameters": count_trainable_parameters(best_model),
        "split_protocol": {
            "train_loads_hp": [0, 1],
            "val_loads_hp": [2],
            "test_loads_hp": [3],
            "train_recordings": 8,
            "val_recordings": 4,
            "test_recordings": 4,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "max_windows_per_recording": MAX_WINDOWS_PER_FILE,
        },
        "cnn_1d": {
            "train": {**cnn_train_metrics, "loss": round(train_loss_final, 6)},
            "val_2hp": {**cnn_val_metrics, "loss": round(val_loss_final, 6)},
            "test_3hp": {**cnn_test_metrics, "loss": round(test_loss_final, 6)},
        },
        "random_forest_baseline": {
            "n_estimators": 200,
            "max_depth": 12,
            "num_handcrafted_features": 18,
            "train": rf_train_metrics,
            "val_2hp": rf_val_metrics,
            "test_3hp": rf_test_metrics,
        },
        "limitations_note": (
            "Validation (2 HP) and held-out test (3 HP) splits each contain 4 recordings "
            "(1 recording per class, 200 non-overlapping windows per recording). "
            "Softmax scores are uncalibrated confidence estimates."
        ),
    }

    file_metrics_payload = {
        "run_id": run_id,
        "timestamp_utc": completed_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aggregation_rule": "mean_window_softmax_score",
        "manual_review_heuristic": {
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "agreement_threshold": DEFAULT_AGREEMENT_THRESHOLD,
        },
        "cnn_1d": {
            "val_2hp": cnn_val_file_metrics,
            "test_3hp": cnn_test_file_metrics,
        },
        "random_forest_baseline": {
            "val_2hp": rf_val_file_metrics,
            "test_3hp": rf_test_file_metrics,
        },
    }

    history_payload = {
        "run_id": run_id,
        "selected_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "epochs": history,
    }

    metrics_path = out_root / "metrics.json"
    file_metrics_path = out_root / "file_level_metrics.json"
    history_path = out_root / "training_history.json"
    manifest_out_path = out_root / "run_manifest.json"

    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    file_metrics_path.write_text(json.dumps(file_metrics_payload, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    run_manifest_payload = {
        "run_id": run_id,
        "started_utc": started_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "completed_utc": completed_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": get_git_sha(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "sklearn_version": sklearn.__version__,
            "device": device,
        },
        "hyperparameters": {
            "seed": seed,
            "epochs": epochs,
            "selected_epoch": best_epoch,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "dropout_p": 0.2,
        },
        "preprocessing": {
            "sensor_channel_pattern": "*_DE_time",
            "nominal_sampling_hz": NOMINAL_SAMPLING_HZ,
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "max_windows_per_recording": MAX_WINDOWS_PER_FILE,
            "normalization": "per_window_zscore_zero_mean_unit_std",
            "clamp_range": [CLAMP_MIN, CLAMP_MAX],
        },
        "dataset_files": validated_rows,
        "artifact_checksums_sha256": {
            "outputs/model.pt": compute_sha256(model_path),
            "outputs/metrics.json": compute_sha256(metrics_path),
            "outputs/file_level_metrics.json": compute_sha256(file_metrics_path),
            "outputs/training_history.json": compute_sha256(history_path),
            "outputs/test_predictions.csv": compute_sha256(test_pred_csv_path),
            "outputs/training_loss.png": compute_sha256(loss_fig_path),
            "outputs/confusion_matrix.png": compute_sha256(cm_fig_path),
        },
    }
    manifest_out_path.write_text(json.dumps(run_manifest_payload, indent=2), encoding="utf-8")

    print("\n=== LOCKED EVALUATION SUMMARY (3 HP HELD-OUT TEST) ===")
    print(
        f"1D TinyFaultCNN (Epoch {best_epoch}): "
        f"Window Acc = {cnn_test_metrics['accuracy']:.4f}, "
        f"Window Macro-F1 = {cnn_test_metrics['macro_f1']:.4f}, "
        f"Recording Acc = {cnn_test_file_metrics['recording_accuracy']:.4f} "
        f"({cnn_test_file_metrics['correct_recordings']}/{cnn_test_file_metrics['num_recordings']})"
    )
    print(
        f"Random Forest Baseline:      "
        f"Window Acc = {rf_test_metrics['accuracy']:.4f}, "
        f"Window Macro-F1 = {rf_test_metrics['macro_f1']:.4f}, "
        f"Recording Acc = {rf_test_file_metrics['recording_accuracy']:.4f} "
        f"({rf_test_file_metrics['correct_recordings']}/{rf_test_file_metrics['num_recordings']})"
    )
    return metrics_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train 1D TinyFaultCNN and Random Forest baseline on CWRU bearing dataset."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--manifest", default="docs/DATASET_MANIFEST.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training_pipeline(
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
