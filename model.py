"""1D Convolutional Neural Network architecture and inference utilities for VibeGuard Cloud."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from data_utils import CLASS_DISPLAY_NAMES, CLASS_NAMES, IDX_TO_LABEL, WINDOW_SIZE

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.70
DEFAULT_AGREEMENT_THRESHOLD: float = 0.80


class TinyFaultCNN(nn.Module):
    """Lightweight 1D CNN for 4-class bearing fault classification from 2,048-sample windows.

    Architecture (per AGENT_MASTER_PLAN.md §7):
    - Block 1: Conv1d(1 -> 16, kernel_size=15, padding=7) + BatchNorm1d + ReLU + MaxPool1d(4)
    - Block 2: Conv1d(16 -> 32, kernel_size=9, padding=4) + BatchNorm1d + ReLU + MaxPool1d(4)
    - Block 3: Conv1d(32 -> 64, kernel_size=5, padding=2) + BatchNorm1d + ReLU + AdaptiveAvgPool1d(1)
    - Head: Flatten + Dropout(0.2) + Linear(64 -> num_classes)
    """

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(16, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass accepting shape (B, 1, L) or (B, L)."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_checkpoint(
    model: TinyFaultCNN,
    checkpoint_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model state_dict and structured preprocessing/evaluation metadata."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_class": "TinyFaultCNN",
        "num_classes": model.num_classes,
        "dropout_p": model.dropout_p,
        "class_names": CLASS_NAMES,
        "window_size": WINDOW_SIZE,
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TinyFaultCNN, dict[str, Any]]:
    """Safely load a trusted local TinyFaultCNN checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)

    if isinstance(payload, dict) and "state_dict" in payload:
        num_classes = int(payload.get("num_classes", len(CLASS_NAMES)))
        dropout_p = float(payload.get("dropout_p", 0.2))
        model = TinyFaultCNN(num_classes=num_classes, dropout_p=dropout_p)
        model.load_state_dict(payload["state_dict"])
        meta = {k: v for k, v in payload.items() if k != "state_dict"}
    elif isinstance(payload, dict):
        model = TinyFaultCNN(num_classes=len(CLASS_NAMES))
        model.load_state_dict(payload)
        meta = {"class_names": CLASS_NAMES, "window_size": WINDOW_SIZE}
    else:
        raise ValueError("Unsupported model checkpoint structure.")

    model.to(device)
    model.eval()
    return model, meta


def evaluate_manual_review_flag(
    mean_scores: np.ndarray,
    window_preds: np.ndarray,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate pre-specified heuristic rules to flag uncertain or inconsistent predictions."""
    pred_idx = int(np.argmax(mean_scores))
    top_score = float(mean_scores[pred_idx])
    sorted_scores = np.sort(mean_scores)[::-1]
    score_margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else top_score
    agreement = float(np.mean(window_preds == pred_idx)) if len(window_preds) > 0 else 0.0

    reasons: list[str] = []
    if top_score < confidence_threshold:
        reasons.append(
            f"Mean uncalibrated softmax score ({top_score:.3f}) is below threshold ({confidence_threshold:.2f})."
        )
    if agreement < agreement_threshold:
        reasons.append(
            f"Window-level segment agreement ({agreement:.1%}) is below threshold ({agreement_threshold:.0%})."
        )

    return {
        "manual_review_required": len(reasons) > 0,
        "reasons": reasons,
        "top_uncalibrated_score": round(top_score, 6),
        "second_best_margin": round(score_margin, 6),
        "segment_agreement": round(agreement, 6),
        "confidence_threshold": confidence_threshold,
        "agreement_threshold": agreement_threshold,
        "calibration_note": (
            "Softmax outputs are uncalibrated model scores, not true posterior probabilities."
        ),
    }


@torch.no_grad()
def predict_windows(
    model: TinyFaultCNN,
    windows: np.ndarray,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> dict[str, Any]:
    """Run inference over normalized 2,048-sample windows and aggregate to recording-level prediction."""
    model.eval()
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != WINDOW_SIZE:
        raise ValueError(f"Expected window length {WINDOW_SIZE}, got {arr.shape[1]}.")

    probs_list: list[np.ndarray] = []
    for start in range(0, arr.shape[0], batch_size):
        batch = torch.from_numpy(arr[start : start + batch_size]).unsqueeze(1).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_list.append(probs)

    window_probs = np.concatenate(probs_list, axis=0)
    window_preds = np.argmax(window_probs, axis=1)
    mean_scores = np.mean(window_probs, axis=0)
    pred_idx = int(np.argmax(mean_scores))
    pred_label = IDX_TO_LABEL[pred_idx]

    review_info = evaluate_manual_review_flag(
        mean_scores=mean_scores,
        window_preds=window_preds,
        confidence_threshold=confidence_threshold,
        agreement_threshold=agreement_threshold,
    )

    per_class_window_counts = {
        CLASS_NAMES[i]: int(np.sum(window_preds == i)) for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_index": pred_idx,
        "predicted_label": pred_label,
        "predicted_display": CLASS_DISPLAY_NAMES[pred_label],
        "mean_scores": {
            CLASS_NAMES[i]: round(float(mean_scores[i]), 6) for i in range(len(CLASS_NAMES))
        },
        "num_windows": int(arr.shape[0]),
        "per_class_window_counts": per_class_window_counts,
        "window_predictions": window_preds.tolist(),
        "window_probabilities": window_probs,
        **review_info,
    }
