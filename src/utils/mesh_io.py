import os
import trimesh
import numpy as np

def find_model_file(path: str) -> str:
    """Find a model file in a directory or return the path if it's a file."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        exts = ('.glb', '.gltf', '.obj', '.ply', '.stl')
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.lower().endswith(exts):
                    return os.path.join(root, fname)
    raise FileNotFoundError(f"No supported model file found at {path}")

def parse_obj_header_for_metadata(obj_path: str):
    """Read top of OBJ file for Scaniverse metadata like '# Unit: meter'."""
    meta = {}
    try:
        with open(obj_path, 'r', encoding='utf8', errors='ignore') as f:
            for _ in range(60):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith('#'):
                    continue
                parts = line[1:].split(':', 1)
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip()
                    meta[k] = v
    except Exception:
        pass
    return meta

def parse_ply_header_for_metadata(ply_path: str):
    """Read PLY header comments for metadata such as unit annotations."""
    meta = {}
    try:
        with open(ply_path, 'rb') as f:
            # PLY header is ASCII even for binary PLY payloads.
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                text = line.decode('utf8', errors='ignore').strip()
                if text == 'end_header':
                    break
                # Examples:
                # comment Unit: meter
                # comment units=mm
                if text.lower().startswith('comment'):
                    body = text[len('comment'):].strip()
                    if ':' in body:
                        k, v = body.split(':', 1)
                        meta[k.strip()] = v.strip()
                    elif '=' in body:
                        k, v = body.split('=', 1)
                        meta[k.strip()] = v.strip()
    except Exception:
        pass
    return meta

def _filter_point_cloud_outliers(vertices: np.ndarray) -> np.ndarray:
    """
    Remove extreme coordinate outliers from raw point clouds using a robust IQR
    envelope. Only applies when spread looks clearly dominated by outliers.
    """
    if vertices is None or len(vertices) < 1000:
        return vertices

    med = np.median(vertices, axis=0)
    q1 = np.percentile(vertices, 25, axis=0)
    q3 = np.percentile(vertices, 75, axis=0)
    iqr = np.maximum(q3 - q1, 1e-9)

    raw_extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    # 20*IQR keeps most inliers while dropping very distant points.
    robust_lo = med - 20.0 * iqr
    robust_hi = med + 20.0 * iqr
    robust_mask = np.all((vertices >= robust_lo) & (vertices <= robust_hi), axis=1)
    robust_vertices = vertices[robust_mask]
    if len(robust_vertices) < 100:
        return vertices

    robust_extent = np.max(robust_vertices, axis=0) - np.min(robust_vertices, axis=0)
    raw_max = float(np.max(raw_extent))
    robust_max = float(np.max(robust_extent))

    # Only trim when outliers dominate scale significantly.
    if robust_max > 0 and raw_max / robust_max > 8.0:
        return robust_vertices
    return vertices

def load_mesh(
    model_path: str,
    decimate_threshold: int = 100000,
    decimate_target: int = 50000,
    disable_decimation: bool = False
) -> (trimesh.Trimesh, dict):
    """Load mesh and return it along with metadata."""
    metadata = {}
    if model_path.lower().endswith('.obj'):
        metadata = parse_obj_header_for_metadata(model_path)
    elif model_path.lower().endswith('.ply'):
        metadata = parse_ply_header_for_metadata(model_path)

    # Try mesh-first load path.
    try:
        loaded = trimesh.load(model_path, force='mesh', skip_materials=True, maintain_order=True)
    except Exception:
        loaded = trimesh.load(model_path, force='mesh')

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError("Loaded scene has no geometry")
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    elif isinstance(loaded, trimesh.points.PointCloud):
        # Normalize point clouds into a face-less Trimesh so downstream code can
        # use a single mesh-like interface.
        pc_vertices = np.asarray(loaded.vertices)
        pc_vertices = _filter_point_cloud_outliers(pc_vertices)
        mesh = trimesh.Trimesh(
            vertices=pc_vertices,
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
    else:
        mesh = loaded

    # Some PLY point-cloud files can become an empty Trimesh when loaded with
    # force='mesh' depending on trimesh version/plugins. Recover by reloading
    # without forcing mesh and converting point cloud vertices.
    if len(mesh.vertices) == 0 and model_path.lower().endswith('.ply'):
        fallback = trimesh.load(model_path, process=False)
        if isinstance(fallback, trimesh.points.PointCloud):
            pc_vertices = np.asarray(fallback.vertices)
            pc_vertices = _filter_point_cloud_outliers(pc_vertices)
            mesh = trimesh.Trimesh(
                vertices=pc_vertices,
                faces=np.empty((0, 3), dtype=np.int64),
                process=False,
            )
        elif isinstance(fallback, trimesh.Scene):
            if len(fallback.geometry) == 0:
                raise ValueError("Loaded PLY scene has no geometry")
            mesh = trimesh.util.concatenate(list(fallback.geometry.values()))
        elif hasattr(fallback, "vertices"):
            mesh = fallback

    if (
        not disable_decimation
        and decimate_threshold > 0
        and decimate_target > 0
        and decimate_target < len(mesh.vertices)
        and hasattr(mesh, 'faces')
        and len(mesh.faces) > 0
        and len(mesh.vertices) > decimate_threshold
    ):
        try:
            mesh = mesh.simplify_quadratic_decimation(decimate_target)
        except Exception:
            # Keep original mesh if decimation backend is unavailable.
            pass

    return mesh, metadata

def get_units(model_path: str, metadata: dict, mesh: trimesh.Trimesh) -> str:
    """Infer units from metadata or mesh extents."""
    raw_unit = (
        metadata.get('Unit')
        or metadata.get('unit')
        or metadata.get('Units')
        or metadata.get('units')
    )
    if raw_unit:
        u = str(raw_unit).strip().lower()
        if u.startswith(('mm', 'milli')):
            return 'mm'
        if u.startswith(('cm', 'centi')):
            return 'cm'
        if u.startswith(('m', 'meter', 'metre')):
            return 'm'

    if model_path.lower().endswith(('.glb', '.gltf')):
        return 'm'

    try:
        max_extent = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    except:
        max_extent = 1.0

    if max_extent > 1000: return 'mm'
    if max_extent > 10: return 'cm'
    return 'm'
