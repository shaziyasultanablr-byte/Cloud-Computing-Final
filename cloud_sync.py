"""AWS S3 synchronization and verification utility for VibeGuard Cloud (IAM Role / Credential-Free)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from data_utils import compute_sha256


def get_s3_client(region: str | None = None, s3_client: Any | None = None) -> Any:
    """Create a boto3 S3 client using the ambient IAM role or environment session (never hard-coded keys)."""
    if s3_client is not None:
        return s3_client
    resolved_region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    session = boto3.Session(region_name=resolved_region)
    return session.client("s3")


def sync_directory_to_s3(
    bucket: str,
    source_dir: str | Path,
    prefix: str,
    region: str | None = None,
    s3_client: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upload all files under source_dir to s3://bucket/prefix/ with SSE-S3 encryption and SHA-256 metadata."""
    src_root = Path(source_dir)
    if not src_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")

    clean_prefix = prefix.strip("/")
    files = sorted(p for p in src_root.rglob("*") if p.is_file() and not p.name.startswith("."))

    client = None if dry_run else get_s3_client(region=region, s3_client=s3_client)
    uploaded: list[dict[str, Any]] = []

    for file_path in files:
        rel_posix = file_path.relative_to(src_root).as_posix()
        s3_key = f"{clean_prefix}/{rel_posix}" if clean_prefix else rel_posix
        sha256 = compute_sha256(file_path)
        size_bytes = file_path.stat().st_size

        if not dry_run and client is not None:
            client.upload_file(
                str(file_path),
                bucket,
                s3_key,
                ExtraArgs={
                    "ServerSideEncryption": "AES256",
                    "Metadata": {
                        "sha256": sha256,
                        "project": "vibeguard-cloud",
                    },
                },
            )

        uploaded.append(
            {
                "local_path": str(file_path.as_posix()),
                "s3_uri": f"s3://{bucket}/{s3_key}",
                "s3_key": s3_key,
                "bytes": size_bytes,
                "sha256": sha256,
                "dry_run": dry_run,
            }
        )

    return {
        "bucket": bucket,
        "prefix": clean_prefix,
        "region": region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_files": len(uploaded),
        "objects": uploaded,
    }


def list_and_verify_s3_prefix(
    bucket: str,
    prefix: str = "",
    region: str | None = None,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """List objects under s3://bucket/prefix using list_objects_v2 and return summary metadata."""
    client = get_s3_client(region=region, s3_client=s3_client)
    clean_prefix = prefix.strip("/")
    kwargs: dict[str, Any] = {"Bucket": bucket}
    if clean_prefix:
        kwargs["Prefix"] = f"{clean_prefix}/"

    resp = client.list_objects_v2(**kwargs)
    contents = resp.get("Contents", [])
    items = [
        {
            "key": obj["Key"],
            "size_bytes": int(obj["Size"]),
            "etag": str(obj.get("ETag", "")).strip('"'),
            "last_modified": (
                obj["LastModified"].astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if hasattr(obj.get("LastModified"), "astimezone")
                else str(obj.get("LastModified", ""))
            ),
        }
        for obj in contents
    ]
    return {
        "bucket": bucket,
        "prefix": clean_prefix,
        "count": len(items),
        "objects": items,
        "verified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def probe_s3_status(
    bucket: str | None = None,
    region: str | None = None,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Non-throwing helper for the Streamlit dashboard to report private S3 status."""
    resolved_bucket = bucket or os.environ.get("VIBEGUARD_S3_BUCKET", "").strip()
    resolved_region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    if not resolved_bucket:
        return {
            "configured": False,
            "connected": False,
            "bucket": None,
            "region": resolved_region,
            "message": "Local/Offline Mode (set VIBEGUARD_S3_BUCKET on EC2 to enable live private S3 verification).",
        }

    try:
        summary = list_and_verify_s3_prefix(
            bucket=resolved_bucket,
            prefix="",
            region=resolved_region,
            s3_client=s3_client,
        )
        return {
            "configured": True,
            "connected": True,
            "bucket": resolved_bucket,
            "region": resolved_region,
            "object_count": summary["count"],
            "verified_utc": summary["verified_utc"],
            "message": f"Connected via IAM role to private bucket s3://{resolved_bucket} ({summary['count']} objects).",
        }
    except (NoCredentialsError, ClientError, BotoCoreError) as exc:
        return {
            "configured": True,
            "connected": False,
            "bucket": resolved_bucket,
            "region": resolved_region,
            "message": f"S3 verification unavailable ({type(exc).__name__}). Ensure VibeGuardEC2Role is attached.",
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize or verify VibeGuard Cloud datasets and artifacts with private Amazon S3."
    )
    parser.add_argument("--bucket", required=True, help="Target private S3 bucket name")
    parser.add_argument("--source", default="outputs", help="Local directory to upload")
    parser.add_argument("--prefix", default="artifacts", help="Target S3 key prefix (e.g., raw or artifacts)")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument(
        "--action",
        choices=["upload", "verify"],
        default="upload",
        help="Upload local directory or verify remote S3 prefix",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute hashes and keys without calling AWS")
    args = parser.parse_args()

    try:
        if args.action == "upload":
            report = sync_directory_to_s3(
                bucket=args.bucket,
                source_dir=args.source,
                prefix=args.prefix,
                region=args.region,
                dry_run=args.dry_run,
            )
            print(json.dumps(report, indent=2))
        else:
            report = list_and_verify_s3_prefix(
                bucket=args.bucket,
                prefix=args.prefix,
                region=args.region,
            )
            print(json.dumps(report, indent=2))
    except Exception as exc:
        print(f"[CLOUD_SYNC ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
