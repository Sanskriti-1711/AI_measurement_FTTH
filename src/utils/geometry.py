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

    # 1. Determine which axis is "up" (has most vertical normals)
    if hasattr(mesh, 'vertex_normals'):
        # Check X, Y, Z axes
        axis_counts = []
        for i in range(3):
            count = np.sum(np.abs(mesh.vertex_normals[:, i]) > 0.8)
            axis_counts.append(count)
        up_axis = np.argmax(axis_counts)
    else:
        # Fallback: check extents (usually depth is the smallest extent in a scan)
        up_axis = np.argmin(mesh.extents)

    print(f"      [Debug] Detected Up axis: {['X','Y','Z'][up_axis]}")

    # 2. Filter points with normals roughly along the Up axis
    if hasattr(mesh, 'vertex_normals'):
        up_mask = np.abs(mesh.vertex_normals[:, up_axis]) > 0.7
        points = mesh.vertices[up_mask]
    else:
        points = mesh.vertices

    if len(points) < 10:
        points = mesh.vertices

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

    # Fixed random_state for deterministic RANSAC
    ransac = RANSACRegressor(residual_threshold=0.01, random_state=42)
    ransac.fit(X, y)

    coef = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_

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

    new_z = mesh.vertices[:, 2]
    z_offset = np.median(new_z)

    translation = np.eye(4)
    translation[2, 3] = -z_offset
    mesh.apply_transform(translation)

    return mesh
