import os
import sys
import json
import numpy as np
import trimesh
from src.utils.mesh_io import load_mesh, get_units, find_model_file
from src.utils.geometry import find_ground_plane, align_mesh_to_plane
from src.measurers import measure_rectangular, measure_circular, classify_feature, validate_scale

class MeasurementProcessor:
    def __init__(self, input_path, out_dir='out'):
        self.input_path = input_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def process(self):
        """
        Execute the full measurement pipeline.

        Steps:
        1. Model Loading: Locate and load the 3D file, detecting real-world units.
        2. Plane Alignment: Use RANSAC to find the ground plane and align it to Z=0.
        3. Cavity Segmentation: Identify vertices that lie below the threshold (e.g. 1cm below ground).
        4. Cleaning: Split segmented points into connected components and keep the largest one (the feature).
        5. Classification: Determine if the feature is a Trench, Manhole, or Duct.
        6. Precision Measurement: Apply feature-specific geometric fitting algorithms.
        """
        print(f"--- Starting Pipeline for {self.input_path} ---")

        # 1. Load Model
        print("[1/6] Loading model...")
        model_file = find_model_file(self.input_path)
        mesh, meta = load_mesh(model_file)
        unit = get_units(model_file, meta, mesh)
        print(f"      Detected units: {unit}. Normalizing to meters...")

        if unit == 'cm':
            mesh.apply_scale(0.01)
        elif unit == 'mm':
            mesh.apply_scale(0.001)

        # 2. Alignment
        print("[2/6] Aligning mesh to ground plane using RANSAC...")
        origin, normal = find_ground_plane(mesh)
        mesh = align_mesh_to_plane(mesh, origin, normal)

        # 3. Segmentation
        print("[3/6] Segmenting cavity from ground...")
        # Use a slightly more aggressive threshold to avoid ground noise
        threshold = -0.03 # 3cm below ground
        cavity_mask = mesh.vertices[:, 2] < threshold

        if not np.any(cavity_mask):
            print("Error: No cavity detected below ground level.")
            return {"status": "error", "message": "No cavity detected below ground plane"}

        # 4. Cleaning & Sub-mesh Extraction
        print("[4/6] Extracting largest connected component...")
        vertex_indices = np.where(cavity_mask)[0]
        face_mask = np.all(np.isin(mesh.faces, vertex_indices), axis=1)

        if not np.any(face_mask):
             cavity_mesh = trimesh.Trimesh(vertices=mesh.vertices[cavity_mask])
        else:
             cavity_mesh = mesh.submesh([face_mask])[0]
             # Optionally remove tiny components
             components = cavity_mesh.split(only_watertight=False)
             if len(components) > 1:
                 # Keep components that have at least 15% of the total vertices
                 total_v = len(cavity_mesh.vertices)
                 keep = [c for c in components if len(c.vertices) > total_v * 0.15]
                 if keep:
                     cavity_mesh = trimesh.util.concatenate(keep)

        if len(cavity_mesh.vertices) < 50:
             print("Error: Detected cavity is too small and likely noise.")
             return {"status": "error", "message": "Detected cavity is too small"}

        # 5. Classification
        print("[5/6] Classifying feature type...")
        obj_type = classify_feature(cavity_mesh)
        print(f"      Classification: {obj_type}")

        # 6. Measurement
        print(f"[6/6] Applying precision measurement for {obj_type}...")
        if obj_type in ['trench', 'manhole']:
            dims = measure_rectangular(cavity_mesh)
        else:
            dims = measure_circular(cavity_mesh)

        # 7. Validation & Confidence
        is_valid, reason = validate_scale(obj_type, dims)

        results = {
            "file": model_file,
            "detected_type": obj_type,
            "unit": "meters",
            "scale_valid": is_valid,
            "validation_reason": reason,
            "measurements_m": dims,
            "metadata": meta
        }
        # Add mm and cm conversions for ease of use
        results["measurements_mm"] = {k: (v * 1000 if isinstance(v, float) and k != 'confidence' else v) for k, v in dims.items()}
        results["measurements_cm"] = {k: (v * 100 if isinstance(v, float) and k != 'confidence' else v) for k, v in dims.items()}

        try:
            self.generate_debug_image(mesh, cavity_mesh, results)
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")

        self.save_results(results)
        return results

    def generate_debug_image(self, mesh, cavity_mesh, results):
        import cv2 as cv
        ext = mesh.extents
        bounds = mesh.bounds
        res = 512
        width = ext[0] if ext[0] > 0 else 1
        height = ext[1] if ext[1] > 0 else 1
        scale = (res - 1) / max(width, height)
        nx = int(width * scale) + 1
        ny = int(height * scale) + 1

        grid = np.zeros((ny, nx), dtype=np.float32)
        pts = mesh.vertices
        ix = np.clip(((pts[:, 0] - bounds[0, 0]) * scale).astype(int), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - bounds[0, 1]) * scale).astype(int), 0, ny - 1)
        grid[iy, ix] = pts[:, 2]

        grid_norm = cv.normalize(grid, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        img = cv.applyColorMap(grid_norm, cv.COLORMAP_VIRIDIS)

        cv.putText(img, f"Type: {results['detected_type']}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y = 60
        for k, v in results['measurements_m'].items():
            if k == 'confidence':
                line = f"Conf: {v:.2f}"
            else:
                line = f"{k}: {v:.3f} m"
            cv.putText(img, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y += 25
        cv.imwrite(os.path.join(self.out_dir, 'debug_view.png'), img)

    def save_results(self, results):
        out_json = os.path.join(self.out_dir, 'measurements.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=4)
        import pandas as pd
        pd.json_normalize(results).to_csv(os.path.join(self.out_dir, 'measurements.csv'), index=False)
