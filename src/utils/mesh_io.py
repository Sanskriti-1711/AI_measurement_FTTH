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

def load_mesh(model_path: str) -> (trimesh.Trimesh, dict):
    """Load mesh and return it along with metadata."""
    obj_meta = {}
    if model_path.lower().endswith('.obj'):
        obj_meta = parse_obj_header_for_metadata(model_path)

    # Try to load without materials to save memory
    try:
        loaded = trimesh.load(model_path, force='mesh', skip_materials=True, maintain_order=True)
    except:
        loaded = trimesh.load(model_path, force='mesh')

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError("Loaded scene has no geometry")
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded

    if len(mesh.vertices) > 100000:
        mesh = mesh.simplify_quadratic_decimation(50000)

    return mesh, obj_meta

def get_units(model_path: str, obj_meta: dict, mesh: trimesh.Trimesh) -> str:
    """Infer units from metadata or mesh extents."""
    if obj_meta.get('Unit'):
        u = obj_meta.get('Unit').lower()
        if u.startswith('m'): return 'm'
        if u.startswith('cm'): return 'cm'
        if u.startswith('mm'): return 'mm'

    if model_path.lower().endswith(('.glb', '.gltf')):
        return 'm'

    try:
        max_extent = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    except:
        max_extent = 1.0

    if max_extent > 1000: return 'mm'
    if max_extent > 10: return 'cm'
    return 'm'
