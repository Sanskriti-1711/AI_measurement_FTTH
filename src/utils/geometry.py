import numpy as np
import trimesh
from sklearn.linear_model import RANSACRegressor

def find_ground_plane(mesh: trimesh.Trimesh):
    """
    Find the dominant plane in the mesh (assumed to be the ground).
    Returns (plane_origin, plane_normal).

    Approach:
    1. Detect the "Up" axis by looking at vertex normals.
    2. Use RANSAC to fit a plane perpendicular to that axis.
    """
    print("Finding ground plane...")

    vertices = np.asarray(mesh.vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices are invalid for plane fitting")

    valid_vertices = np.isfinite(vertices).all(axis=1)
    vertices = vertices[valid_vertices]
    if len(vertices) < 3:
        raise ValueError("Mesh has too few valid vertices for plane fitting")

    # 1. Determine which axis is "up" (has most vertical normals), with fallback.
    normals = None
    if hasattr(mesh, 'vertex_normals'):
        try:
            normals = np.asarray(mesh.vertex_normals)
            if normals.shape[0] != np.asarray(mesh.vertices).shape[0]:
                normals = None
        except Exception:
            normals = None

    if normals is not None and len(normals) > 0:
        normals = normals[valid_vertices]
        normals_ok = np.isfinite(normals).all(axis=1)
        normals = normals[normals_ok]
        if len(normals) > 0:
            axis_counts = [int(np.sum(np.abs(normals[:, i]) > 0.8)) for i in range(3)]
            up_axis = int(np.argmax(axis_counts)) if max(axis_counts) > 0 else int(np.argmin(mesh.extents))
        else:
            up_axis = int(np.argmin(mesh.extents))
    else:
        up_axis = int(np.argmin(mesh.extents))

    print(f"      [Debug] Detected Up axis: {['X','Y','Z'][up_axis]}")

    # 2. Filter points with normals roughly along the Up axis
    if normals is not None and len(normals) == len(vertices):
        up_mask = np.abs(normals[:, up_axis]) > 0.7
        points = vertices[up_mask]
    else:
        points = vertices

    if len(points) < 10:
        points = vertices
    if len(points) < 3:
        raise ValueError("Not enough points for plane fitting after filtering")

    # Sample for RANSAC
    if len(points) > 20000:
        # Fixed seed for deterministic point sampling
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), 20000, replace=False)
        points = points[idx]

    # 3. Fit plane: Dependent variable is the Up axis
    other_axes = [i for i in range(3) if i != up_axis]
    X = points[:, other_axes]
    y = points[:, up_axis]
    if len(X) < 3:
        raise ValueError("Not enough samples for RANSAC plane fit")

    # Fixed random_state for deterministic RANSAC
    try:
        ransac = RANSACRegressor(residual_threshold=0.01, random_state=42)
        ransac.fit(X, y)
        coef = ransac.estimator_.coef_
        intercept = ransac.estimator_.intercept_
    except Exception:
        # Fallback: deterministic least squares plane fit.
        A = np.column_stack([X, np.ones(len(X))])
        coef_full, *_ = np.linalg.lstsq(A, y, rcond=None)
        coef = coef_full[:2]
        intercept = coef_full[2]

    # Construct normal
    normal = np.zeros(3)
    normal[up_axis] = -1.0
    normal[other_axes[0]] = coef[0]
    normal[other_axes[1]] = coef[1]
    normal /= np.linalg.norm(normal)

    # Ensure normal points in the "positive" direction of the detected axis
    if normal[up_axis] < 0:
        normal = -normal

    origin = np.zeros(3)
    origin[up_axis] = intercept

    return origin, normal

def align_mesh_to_plane(mesh: trimesh.Trimesh, origin, normal):
    """
    Rotate and translate the mesh so that the ground plane is perfectly aligned with Z=0.

    Approach:
    1. Compute the rotation matrix that aligns the detected plane normal with the world Z-axis [0,0,1].
    2. Translate the mesh so the median height of the ground points is at Z=0.
    """
    target_normal = np.array([0, 0, 1])
    rotation_matrix = trimesh.geometry.align_vectors(normal, target_normal)
    mesh.apply_transform(rotation_matrix)

    # Translate so the fitted plane origin lands on Z=0 after rotation.
    origin_h = np.array([origin[0], origin[1], origin[2], 1.0], dtype=np.float64)
    rotated_origin = rotation_matrix @ origin_h
    z_offset = rotated_origin[2]

    translation = np.eye(4)
    translation[2, 3] = -z_offset
    mesh.apply_transform(translation)

    return mesh
