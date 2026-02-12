import numpy as np
import trimesh
from sklearn.linear_model import RANSACRegressor

def find_ground_plane(mesh: trimesh.Trimesh):
    """
    Find the dominant plane in the mesh (assumed to be the ground).
    Returns (plane_origin, plane_normal).

    Approach:
    1. Filter vertices to keep only those with normals pointing roughly 'up'.
    2. Use RANSAC (Random Sample Consensus) to robustly fit a plane to the vertices.
       RANSAC is ideal here as it ignores 'outliers' like the cavity itself.
    3. The plane equation is used to determine the ground level.
    """
    print("Finding ground plane...")
    if hasattr(mesh, 'vertex_normals'):
        normals = mesh.vertex_normals
        up_mask = np.abs(normals[:, 2]) > 0.7
        points = mesh.vertices[up_mask]
    else:
        points = mesh.vertices

    if len(points) < 10:
        points = mesh.vertices

    if len(points) > 20000:
        idx = np.random.choice(len(points), 20000, replace=False)
        points = points[idx]

    X = points[:, :2]
    y = points[:, 2]

    ransac = RANSACRegressor(residual_threshold=0.01)
    ransac.fit(X, y)

    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    normal = np.array([a, b, -1.0])
    normal /= np.linalg.norm(normal)

    if normal[2] < 0:
        normal = -normal

    origin = np.array([0, 0, c])

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
