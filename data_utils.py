"""Data loading, validation, window segmentation, normalization, and feature extraction for VibeGuard Cloud."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import scipy.io
import scipy.stats

CLASS_NAMES: list[str] = ["normal", "inner", "ball", "outer"]
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "normal": "Normal Baseline",
    "inner": "Inner Race Fault (0.007 in)",
    "ball": "Ball Fault (0.007 in)",
    "outer": "Outer Race Fault @ 6:00 (0.007 in)",
}
LABEL_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

WINDOW_SIZE: int = 2048
WINDOW_STRIDE: int = 2048
MAX_WINDOWS_PER_FILE: int = 200
CLAMP_MIN: float = -10.0
CLAMP_MAX: float = 10.0
NOMINAL_SAMPLING_HZ: int = 12000
MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024  # 25 MiB guard

EXPECTED_SPLIT_LOADS: dict[str, set[int]] = {
    "train": {0, 1},
    "val": {2},
    "test": {3},
}


class DataValidationError(ValueError):
    """Raised when an input vibration recording or manifest fails validation."""


def compute_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dataset_manifest(
    manifest_path: str | Path = "docs/DATASET_MANIFEST.csv",
) -> list[dict[str, Any]]:
    """Load and parse the 16-file CWRU dataset manifest."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found at: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "file_id": int(row["file_id"]),
                    "filename": row["filename"].strip(),
                    "label": row["label"].strip(),
                    "class_index": int(row["class_index"]),
                    "fault_diameter_in": float(row["fault_diameter_in"]),
                    "load_hp": int(row["load_hp"]),
                    "approx_rpm": int(row["approx_rpm"]),
                    "split": row["split"].strip(),
                    "expected_subdir": row["expected_subdir"].strip(),
                    "sampling_hz": int(row["sampling_hz"]),
                    "original_url": row["original_url"].strip(),
                }
            )
    verify_split_disjointness(rows)
    return rows


def verify_split_disjointness(manifest_rows: list[dict[str, Any]]) -> None:
    """Programmatically verify recording-level and load-level split disjointness."""
    if len(manifest_rows) != 16:
        raise DataValidationError(
            f"Expected exactly 16 recordings in manifest, found {len(manifest_rows)}."
        )

    file_ids = [r["file_id"] for r in manifest_rows]
    if len(set(file_ids)) != 16:
        raise DataValidationError("Duplicate file_id detected in dataset manifest.")

    filenames = [r["filename"] for r in manifest_rows]
    if len(set(filenames)) != 16:
        raise DataValidationError("Duplicate filename detected in dataset manifest.")

    split_files: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    split_loads: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    split_class_counts: dict[str, dict[str, int]] = {
        "train": {c: 0 for c in CLASS_NAMES},
        "val": {c: 0 for c in CLASS_NAMES},
        "test": {c: 0 for c in CLASS_NAMES},
    }

    for row in manifest_rows:
        split = row["split"]
        label = row["label"]
        load_hp = row["load_hp"]
        fid = row["file_id"]

        if split not in EXPECTED_SPLIT_LOADS:
            raise DataValidationError(f"Unexpected split '{split}' for file {fid}.")
        if label not in LABEL_TO_IDX:
            raise DataValidationError(f"Unexpected label '{label}' for file {fid}.")
        if LABEL_TO_IDX[label] != row["class_index"]:
            raise DataValidationError(
                f"Class index mismatch for file {fid}: {label} vs {row['class_index']}."
            )
        if load_hp not in EXPECTED_SPLIT_LOADS[split]:
            raise DataValidationError(
                f"Load leakage detected: file {fid} has load {load_hp} HP in split '{split}' "
                f"(allowed: {sorted(EXPECTED_SPLIT_LOADS[split])})."
            )

        split_files[split].add(fid)
        split_loads[split].add(load_hp)
        split_class_counts[split][label] += 1

    # Verify zero recording or load overlap across splits
    if split_files["train"] & split_files["val"]:
        raise DataValidationError("Train and val splits share file IDs.")
    if split_files["train"] & split_files["test"]:
        raise DataValidationError("Train and test splits share file IDs.")
    if split_files["val"] & split_files["test"]:
        raise DataValidationError("Val and test splits share file IDs.")

    if split_loads["train"] & split_loads["val"]:
        raise DataValidationError("Train and val splits share motor loads.")
    if split_loads["train"] & split_loads["test"]:
        raise DataValidationError("Train and test splits share motor loads.")
    if split_loads["val"] & split_loads["test"]:
        raise DataValidationError("Val and test splits share motor loads.")

    # Verify exact class balance per split (2 per class in train, 1 per class in val/test)
    for c in CLASS_NAMES:
        if split_class_counts["train"][c] != 2:
            raise DataValidationError(f"Train split must have 2 files for class '{c}'.")
        if split_class_counts["val"][c] != 1:
            raise DataValidationError(f"Val split must have 1 file for class '{c}'.")
        if split_class_counts["test"][c] != 1:
            raise DataValidationError(f"Test split must have 1 file for class '{c}'.")


def extract_de_signal_from_mat(
    source: str | Path | bytes | BinaryIO,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract the drive-end (*_DE_time) vibration channel and metadata from a MATLAB .mat file."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataValidationError(f"MAT file does not exist: {path}")
        size_bytes = path.stat().st_size
        if size_bytes == 0:
            raise DataValidationError(f"MAT file is empty: {path}")
        if size_bytes > max_bytes:
            raise DataValidationError(
                f"MAT file exceeds maximum size limit ({size_bytes} > {max_bytes} bytes)."
            )
        raw_bytes = path.read_bytes()
    elif isinstance(source, bytes):
        if len(source) == 0:
            raise DataValidationError("Provided MAT byte stream is empty.")
        if len(source) > max_bytes:
            raise DataValidationError(
                f"MAT payload exceeds maximum size limit ({len(source)} > {max_bytes} bytes)."
            )
        raw_bytes = source
    else:
        raw_bytes = source.read(max_bytes + 1)
        if len(raw_bytes) == 0:
            raise DataValidationError("Uploaded MAT file is empty.")
        if len(raw_bytes) > max_bytes:
            raise DataValidationError(
                f"Uploaded MAT file exceeds maximum size limit ({max_bytes} bytes)."
            )

    # Guard against HTML error pages saved as .mat
    head_ascii = raw_bytes[:128].lstrip().lower()
    if head_ascii.startswith(b"<!doctype html") or head_ascii.startswith(b"<html"):
        raise DataValidationError("File appears to be an HTML page rather than a valid MATLAB .mat binary.")

    try:
        mat_dict = scipy.io.loadmat(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise DataValidationError(f"Failed to parse MATLAB .mat file: {exc}") from exc

    # Case-sensitive search for drive-end channel (*_DE_time)
    de_keys = sorted(k for k in mat_dict.keys() if not k.startswith("__") and k.endswith("_DE_time"))
    if not de_keys:
        available = [k for k in mat_dict.keys() if not k.startswith("__")]
        raise DataValidationError(
            f"Required drive-end channel matching '*_DE_time' not found. Available keys: {available}"
        )

    # Note: Official CWRU 99.mat contains both 'X098_DE_time' (from the prior workspace)
    # and 'X099_DE_time'. Match the filename ID when available, or select the latest key
    # (de_keys[-1]) so 'X099_DE_time' is chosen rather than leaking 'X098_DE_time'.
    channel_key = de_keys[-1]
    if isinstance(source, (str, Path)):
        stem = Path(source).stem
        if stem.isdigit():
            expected_key = f"X{int(stem):03d}_DE_time"
            if expected_key in de_keys:
                channel_key = expected_key
    raw_arr = np.asarray(mat_dict[channel_key])
    if raw_arr.size == 0 or not np.issubdtype(raw_arr.dtype, np.number):
        raise DataValidationError(f"Channel '{channel_key}' does not contain a valid numeric array.")

    signal = raw_arr.astype(np.float64).reshape(-1)
    if not np.all(np.isfinite(signal)):
        raise DataValidationError(f"Channel '{channel_key}' contains NaN or infinite values.")

    if signal.shape[0] < WINDOW_SIZE:
        raise DataValidationError(
            f"Signal length ({signal.shape[0]}) is shorter than minimum window size ({WINDOW_SIZE})."
        )

    rpm_keys = sorted(k for k in mat_dict.keys() if not k.startswith("__") and k.endswith("RPM"))
    recorded_rpm: float | None = None
    if rpm_keys:
        try:
            recorded_rpm = float(np.asarray(mat_dict[rpm_keys[0]]).reshape(-1)[0])
        except (ValueError, IndexError, TypeError):
            recorded_rpm = None

    metadata = {
        "channel_key": channel_key,
        "num_samples": int(signal.shape[0]),
        "recorded_rpm": recorded_rpm,
        "available_keys": sorted(k for k in mat_dict.keys() if not k.startswith("__")),
        "sampling_hz_nominal": NOMINAL_SAMPLING_HZ,
        "duration_sec_at_12khz": round(float(signal.shape[0]) / NOMINAL_SAMPLING_HZ, 4),
    }
    return signal.astype(np.float32), metadata


def extract_signal_from_csv(
    source: str | Path | bytes | BinaryIO,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse and validate a single-column numeric CSV vibration recording."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataValidationError(f"CSV file does not exist: {path}")
        raw_bytes = path.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    else:
        raw_bytes = source.read(max_bytes + 1)

    if len(raw_bytes) == 0:
        raise DataValidationError("CSV file is empty.")
    if len(raw_bytes) > max_bytes:
        raise DataValidationError(
            f"CSV file exceeds maximum size limit ({len(raw_bytes)} > {max_bytes} bytes)."
        )

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataValidationError("CSV file is not valid UTF-8 text.") from exc

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise DataValidationError("CSV file contains no non-empty rows.")

    values: list[float] = []
    for idx, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 1:
            raise DataValidationError(
                f"CSV must contain exactly 1 column (found {len(parts)} columns on line {idx + 1})."
            )
        token = parts[0]
        if idx == 0:
            try:
                val = float(token)
                values.append(val)
            except ValueError:
                # Allow a single header row on line 1
                continue
        else:
            try:
                val = float(token)
                values.append(val)
            except ValueError as exc:
                raise DataValidationError(
                    f"Non-numeric value '{token}' at CSV line {idx + 1}."
                ) from exc

    if not values:
        raise DataValidationError("CSV file contains no numeric signal samples.")

    signal = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(signal)):
        raise DataValidationError("CSV signal contains NaN or infinite values.")

    if signal.shape[0] < WINDOW_SIZE:
        raise DataValidationError(
            f"CSV signal length ({signal.shape[0]}) is shorter than minimum window size ({WINDOW_SIZE})."
        )

    metadata = {
        "channel_key": "csv_single_column",
        "num_samples": int(signal.shape[0]),
        "recorded_rpm": None,
        "available_keys": ["csv_single_column"],
        "sampling_hz_nominal": NOMINAL_SAMPLING_HZ,
        "duration_sec_at_12khz": round(float(signal.shape[0]) / NOMINAL_SAMPLING_HZ, 4),
    }
    return signal.astype(np.float32), metadata


def normalize_window(
    window: np.ndarray,
    clamp_min: float = CLAMP_MIN,
    clamp_max: float = CLAMP_MAX,
) -> np.ndarray:
    """Standardize a single 1D window to zero mean and unit variance, then clamp to [clamp_min, clamp_max]."""
    w = np.asarray(window, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(w)):
        raise DataValidationError("Window contains NaN or infinite values.")
    mean = float(np.mean(w))
    std = float(np.std(w))
    if std < 1e-12:
        raise DataValidationError("Constant/zero-variance signal window cannot be z-score normalized.")
    normed = (w - mean) / (std + 1e-8)
    clamped = np.clip(normed, clamp_min, clamp_max)
    return clamped.astype(np.float32)


def segment_and_normalize_signal(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    max_windows: int = MAX_WINDOWS_PER_FILE,
    clamp_min: float = CLAMP_MIN,
    clamp_max: float = CLAMP_MAX,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically segment a 1D signal into non-overlapping windows and standardize each window."""
    sig = np.asarray(signal, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(sig)):
        raise DataValidationError("Input signal contains NaN or infinite values.")
    total_len = int(sig.shape[0])
    if total_len < window_size:
        raise DataValidationError(
            f"Signal length {total_len} is smaller than window_size {window_size}."
        )

    possible_windows = 1 + (total_len - window_size) // stride
    num_windows = min(possible_windows, max_windows)

    windows = np.empty((num_windows, window_size), dtype=np.float32)
    index_ranges: list[tuple[int, int]] = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows[i] = normalize_window(sig[start:end], clamp_min=clamp_min, clamp_max=clamp_max)
        index_ranges.append((start, end))

    seg_meta = {
        "raw_num_samples": total_len,
        "window_size": window_size,
        "stride": stride,
        "max_windows": max_windows,
        "num_windows": num_windows,
        "start_sample": index_ranges[0][0],
        "end_sample": index_ranges[-1][1],
        "clamp_range": [clamp_min, clamp_max],
    }
    return windows, seg_meta


def extract_handcrafted_features(
    windows: np.ndarray,
    sampling_hz: int = NOMINAL_SAMPLING_HZ,
) -> np.ndarray:
    """Compute 18 time- and frequency-domain features per window for the Random Forest baseline."""
    arr = np.asarray(windows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n_windows, w_len = arr.shape
    freqs = np.fft.rfftfreq(w_len, d=1.0 / sampling_hz)
    nyquist = sampling_hz / 2.0
    band_edges = np.linspace(0.0, nyquist, 5)

    features = np.zeros((n_windows, 18), dtype=np.float32)
    for i in range(n_windows):
        x = arr[i]
        abs_x = np.abs(x)
        mean_abs = float(np.mean(abs_x)) + 1e-12
        rms = float(np.sqrt(np.mean(x**2))) + 1e-12
        peak = float(np.max(abs_x))
        p2p = float(np.max(x) - np.min(x))
        var = float(np.var(x))
        crest = peak / rms
        shape = rms / mean_abs
        impulse = peak / mean_abs
        sqrt_mean = float(np.mean(np.sqrt(abs_x))) ** 2 + 1e-12
        clearance = peak / sqrt_mean
        kurt = float(scipy.stats.kurtosis(x, fisher=True, bias=False))
        skew = float(scipy.stats.skew(x, bias=False))
        zcr = float(np.mean(np.diff(np.signbit(x)).astype(np.float64)))

        # Frequency-domain features
        mag = np.abs(np.fft.rfft(x))
        power = mag**2
        total_power = float(np.sum(power)) + 1e-12
        prob = power / total_power

        spec_centroid = float(np.sum(freqs * prob))
        spec_spread = float(np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * prob)))
        spec_entropy = float(-np.sum(prob * np.log2(prob + 1e-12)) / np.log2(len(prob)))
        dom_freq = float(freqs[int(np.argmax(power))])

        band_ratios: list[float] = []
        for b in range(4):
            mask = (freqs >= band_edges[b]) & (
                freqs <= band_edges[b + 1] if b == 3 else freqs < band_edges[b + 1]
            )
            band_ratios.append(float(np.sum(power[mask]) / total_power))

        features[i] = np.array(
            [
                rms,
                peak,
                p2p,
                var,
                mean_abs,
                crest,
                shape,
                impulse,
                clearance,
                kurt,
                skew,
                zcr,
                spec_centroid,
                spec_spread,
                spec_entropy,
                dom_freq,
                band_ratios[0],
                band_ratios[1] + band_ratios[2] + band_ratios[3] - band_ratios[0],
            ],
            dtype=np.float32,
        )
        # Store exact 4 band ratios in last 4 slots by overwriting last 2 slots with band 1, 2, 3
        features[i, 14] = band_ratios[0]
        features[i, 15] = band_ratios[1]
        features[i, 16] = band_ratios[2]
        features[i, 17] = band_ratios[3]

    return features


def build_split_datasets(
    raw_dir: str | Path = "data/raw",
    manifest_path: str | Path = "docs/DATASET_MANIFEST.csv",
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    max_windows: int = MAX_WINDOWS_PER_FILE,
) -> dict[str, dict[str, Any]]:
    """Load all 16 validated CWRU files and construct disjoint train, val, and test datasets."""
    manifest = load_dataset_manifest(manifest_path)
    raw_root = Path(raw_dir)

    splits: dict[str, dict[str, Any]] = {
        s: {"X": [], "y": [], "file_ids": [], "loads": [], "records": []}
        for s in ("train", "val", "test")
    }

    for row in manifest:
        mat_path = raw_root / row["expected_subdir"] / row["filename"]
        signal, mat_meta = extract_de_signal_from_mat(mat_path)
        windows, seg_meta = segment_and_normalize_signal(
            signal,
            window_size=window_size,
            stride=stride,
            max_windows=max_windows,
        )
        n_win = windows.shape[0]
        split = row["split"]
        cls_idx = row["class_index"]
        fid = row["file_id"]
        load_hp = row["load_hp"]

        splits[split]["X"].append(windows)
        splits[split]["y"].append(np.full(n_win, cls_idx, dtype=np.int64))
        splits[split]["file_ids"].append(np.full(n_win, fid, dtype=np.int64))
        splits[split]["loads"].append(np.full(n_win, load_hp, dtype=np.int64))
        splits[split]["records"].append(
            {
                **row,
                "path": str(mat_path.as_posix()),
                "sha256": compute_sha256(mat_path),
                "channel_key": mat_meta["channel_key"],
                "recorded_rpm": mat_meta["recorded_rpm"],
                **seg_meta,
            }
        )

    result: dict[str, dict[str, Any]] = {}
    for s, data in splits.items():
        result[s] = {
            "X": np.concatenate(data["X"], axis=0),
            "y": np.concatenate(data["y"], axis=0),
            "file_ids": np.concatenate(data["file_ids"], axis=0),
            "loads": np.concatenate(data["loads"], axis=0),
            "records": data["records"],
        }
    return result
