"""Download and verify the 16 official CWRU Bearing Data Center MATLAB recordings."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

from data_utils import (
    DataValidationError,
    compute_sha256,
    extract_de_signal_from_mat,
    load_dataset_manifest,
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def download_single_file(
    url: str,
    dest_path: Path,
    timeout: int = 45,
    retries: int = 3,
) -> tuple[bool, str]:
    """Download a single .mat file and validate its MATLAB drive-end channel."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already present and valid
    if dest_path.exists() and dest_path.stat().st_size > 2048:
        try:
            sig, meta = extract_de_signal_from_mat(dest_path)
            sha = compute_sha256(dest_path)
            return (
                True,
                f"cached ({meta['channel_key']}, {len(sig)} samples, sha256={sha[:12]}...)",
            )
        except DataValidationError:
            dest_path.unlink(missing_ok=True)

    last_err = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code} from {url}"
                time.sleep(1.0 * attempt)
                continue

            content = resp.content
            # Validate before saving permanently
            sig, meta = extract_de_signal_from_mat(content)
            dest_path.write_bytes(content)
            sha = compute_sha256(dest_path)
            return (
                True,
                f"downloaded ({meta['channel_key']}, {len(sig)} samples, sha256={sha[:12]}...)",
            )
        except Exception as exc:
            last_err = str(exc)
            time.sleep(1.0 * attempt)

    return False, last_err


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the 16 CWRU bearing recordings defined in docs/DATASET_MANIFEST.csv."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="docs/DATASET_MANIFEST.csv",
        help="Path to DATASET_MANIFEST.csv",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Root directory for raw .mat files",
    )
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.manifest)
    raw_root = Path(args.raw_dir)

    failures: list[tuple[int, str]] = []
    for row in manifest:
        fid = row["file_id"]
        dest = raw_root / row["expected_subdir"] / row["filename"]
        url = row["original_url"]
        ok, msg = download_single_file(url=url, dest_path=dest)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {row['filename']} ({row['label']}, {row['load_hp']} HP, {row['split']}): {msg}")
        if not ok:
            failures.append((fid, msg))

    if failures:
        print(
            f"\n{len(failures)} file(s) could not be downloaded automatically. "
            "See AGENT_MASTER_PLAN.md §5 for manual fallback instructions.",
            file=sys.stderr,
        )
        return 1

    print("\nAll 16 CWRU MATLAB recordings verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
