from src.integration.minio_store import select_primary_model_file


def test_select_primary_model_file_prefers_obj():
    paths = [
        r"C:\tmp\scan\model.ply",
        r"C:\tmp\scan\model.obj",
        r"C:\tmp\scan\readme.txt",
    ]
    assert select_primary_model_file(paths).lower().endswith(".obj")

