import os
from typing import Iterable

class MinioScanStore:
    """Download uploaded scan objects from MinIO."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.bucket = bucket
        # Lazy import to keep local tests runnable even when MinIO SDK isn't installed.
        from minio import Minio
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def list_keys_by_prefix(self, prefix: str) -> list[str]:
        objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def download_keys(self, keys: Iterable[str], target_dir: str) -> list[str]:
        local_paths: list[str] = []
        os.makedirs(target_dir, exist_ok=True)
        for key in keys:
            local_path = os.path.join(target_dir, key.replace("/", os.sep))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.client.fget_object(self.bucket, key, local_path)
            local_paths.append(local_path)
        return local_paths


def select_primary_model_file(local_paths: list[str]) -> str:
    """
    Prefer OBJ when available because the current pipeline is strongest there,
    then fallback to PLY/GLB/GLTF/STL.
    """
    if not local_paths:
        raise FileNotFoundError("No downloaded files available for model selection")

    lower_map = {p.lower(): p for p in local_paths}
    priority = (".obj", ".ply", ".glb", ".gltf", ".stl")
    for ext in priority:
        for p_low, p_real in lower_map.items():
            if p_low.endswith(ext):
                return p_real
    raise FileNotFoundError("No supported model file found in downloaded keys")
