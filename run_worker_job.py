import argparse
import json
import sys

from src.integration.backend_client import BackendApiClient
from src.integration.job_runner import run_analysis_job
from src.integration.minio_store import MinioScanStore


def main():
    parser = argparse.ArgumentParser(description="Run one measurement worker job from MinIO objects")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--minio-endpoint", required=True, help="e.g. localhost:9000")
    parser.add_argument("--minio-access-key", required=True)
    parser.add_argument("--minio-secret-key", required=True)
    parser.add_argument("--minio-bucket", required=True)
    parser.add_argument("--minio-secure", action="store_true", help="Use HTTPS for MinIO")
    parser.add_argument("--object-key", action="append", default=[], help="MinIO object key (repeatable)")
    parser.add_argument("--object-prefix", default=None, help="List objects from prefix instead of explicit keys")
    parser.add_argument("--backend-base-url", default=None, help="If set, post /result callback")
    parser.add_argument("--backend-token", default=None, help="Optional bearer token for backend")
    parser.add_argument("--decimate-threshold", type=int, default=100000)
    parser.add_argument("--decimate-target", type=int, default=50000)
    parser.add_argument("--no-decimate", action="store_true")
    args = parser.parse_args()

    minio_store = MinioScanStore(
        endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        bucket=args.minio_bucket,
        secure=args.minio_secure,
    )
    backend_client = None
    if args.backend_base_url:
        backend_client = BackendApiClient(
            base_url=args.backend_base_url,
            bearer_token=args.backend_token,
        )

    try:
        result = run_analysis_job(
            job_id=args.job_id,
            scan_id=args.scan_id,
            minio_store=minio_store,
            object_keys=args.object_key or None,
            object_prefix=args.object_prefix,
            backend_client=backend_client,
            decimate_threshold=args.decimate_threshold,
            decimate_target=args.decimate_target,
            disable_decimation=args.no_decimate,
        )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"Worker job failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

