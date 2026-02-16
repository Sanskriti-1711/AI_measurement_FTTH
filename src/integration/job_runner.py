import os
import tempfile

from src.integration.backend_client import BackendApiClient
from src.integration.minio_store import MinioScanStore, select_primary_model_file
from src.processor import MeasurementProcessor


def run_analysis_job(
    *,
    job_id: str,
    scan_id: str,
    minio_store: MinioScanStore,
    object_keys: list[str] | None = None,
    object_prefix: str | None = None,
    backend_client: BackendApiClient | None = None,
    decimate_threshold: int = 100000,
    decimate_target: int = 50000,
    disable_decimation: bool = False,
) -> dict:
    """
    Worker entrypoint:
    1) download uploaded scan files from MinIO
    2) run measurement pipeline
    3) optionally callback result to backend API
    """
    if not object_keys and not object_prefix:
        raise ValueError("Either object_keys or object_prefix must be provided")

    if object_prefix:
        keys = minio_store.list_keys_by_prefix(object_prefix)
    else:
        keys = object_keys or []
    if not keys:
        raise FileNotFoundError("No objects found for scan analysis")

    with tempfile.TemporaryDirectory(prefix=f"scan_{scan_id}_") as tmp:
        download_dir = os.path.join(tmp, "input")
        out_dir = os.path.join(tmp, "out")
        local_paths = minio_store.download_keys(keys, download_dir)
        input_path = select_primary_model_file(local_paths)

        processor = MeasurementProcessor(
            input_path=input_path,
            out_dir=out_dir,
            decimate_threshold=decimate_threshold,
            decimate_target=decimate_target,
            disable_decimation=disable_decimation,
        )
        try:
            result = processor.process()
            status = "done"
            if result.get("status") == "error":
                status = "failed"
            if backend_client is not None:
                if status == "done":
                    backend_client.post_job_result(job_id=job_id, status="done", result=result)
                else:
                    backend_client.post_job_result(
                        job_id=job_id,
                        status="failed",
                        error={
                            "code": "PROCESSING_ERROR",
                            "message": result.get("message", "Unknown processing error"),
                        },
                    )
            return result
        except Exception as exc:
            if backend_client is not None:
                backend_client.post_job_result(
                    job_id=job_id,
                    status="failed",
                    error={"code": "PROCESSING_ERROR", "message": str(exc)},
                )
            raise

