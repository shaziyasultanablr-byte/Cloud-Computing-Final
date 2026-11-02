"""Validate all 16 CWRU MATLAB files against docs/DATASET_MANIFEST.csv and write docs/dataset_checksums.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path when invoked as python scripts/validate_dataset.py
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_utils import (  # noqa: E402
    DataValidationError,
    compute_sha256,
    extract_de_signal_from_mat,
    load_dataset_manifest,
    segment_and_normalize_signal,
)


def validate_all_recordings(
    manifest_path: str | Path = "docs/DATASET_MANIFEST.csv",
    raw_dir: str | Path = "data/raw",
    output_csv: str | Path = "docs/dataset_checksums.csv",
) -> list[dict[str, object]]:
    """Validate all 16 CWRU files and write checksums + provenance metadata."""
    manifest = load_dataset_manifest(manifest_path)
    raw_root = Path(raw_dir)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checked_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    validated_rows: list[dict[str, object]] = []

    for row in manifest:
        mat_path = raw_root / row["expected_subdir"] / row["filename"]
        if not mat_path.exists():
            raise DataValidationError(f"Missing required CWRU file: {mat_path}")

        signal, meta = extract_de_signal_from_mat(mat_path)
        windows, seg_meta = segment_and_normalize_signal(signal)

        expected_prefix = f"X{int(row['file_id']):03d}_DE_time"
        if meta["channel_key"] != expected_prefix:
            raise DataValidationError(
                f"Channel identity mismatch in {mat_path}: found '{meta['channel_key']}', "
                f"expected '{expected_prefix}'."
            )

        sha256 = compute_sha256(mat_path)
        size_bytes = mat_path.stat().st_size

        validated_rows.append(
            {
                "file": f"{row['expected_subdir']}/{row['filename']}",
                "file_id": row["file_id"],
                "label": row["label"],
                "load_hp": row["load_hp"],
                "split": row["split"],
                "channel_key": meta["channel_key"],
                "sampling_hz": row["sampling_hz"],
                "recorded_rpm": meta["recorded_rpm"] if meta["recorded_rpm"] is not None else row["approx_rpm"],
                "num_samples": meta["num_samples"],
                "num_windows": seg_meta["num_windows"],
                "window_range": f"0..{seg_meta['end_sample']}",
                "bytes": size_bytes,
                "sha256": sha256,
                "original_url": row["original_url"],
                "checked_utc": checked_utc,
            }
        )

    fieldnames = [
        "file",
        "file_id",
        "label",
        "load_hp",
        "split",
        "channel_key",
        "sampling_hz",
        "recorded_rpm",
        "num_samples",
        "num_windows",
        "window_range",
        "bytes",
        "sha256",
        "original_url",
        "checked_utc",
    ]
    with open(out_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(validated_rows)

    return validated_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate all 16 CWRU .mat recordings and generate docs/dataset_checksums.csv."
    )
    parser.add_argument("--manifest", default="docs/DATASET_MANIFEST.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="docs/dataset_checksums.csv")
    args = parser.parse_args()

    try:
        rows = validate_all_recordings(
            manifest_path=args.manifest,
            raw_dir=args.raw_dir,
            output_csv=args.output,
        )
    except Exception as exc:
        print(f"[VALIDATION FAILED] {exc}", file=sys.stderr)
        return 1

    print(f"Validated {len(rows)} CWRU recordings -> {args.output}")
    for r in rows:
        print(
            f"  - {r['file']:16s} | {r['label']:6s} | {r['load_hp']} HP | {r['split']:5s} | "
            f"{r['channel_key']:12s} | {r['num_samples']:7d} samples | {r['num_windows']:3d} windows | "
            f"sha256={str(r['sha256'])[:12]}..."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
