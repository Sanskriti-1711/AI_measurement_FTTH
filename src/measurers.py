import numpy as np
import trimesh

def measure_rectangular(mesh: trimesh.Trimesh):
    """
    Measure length, width, and depth of a rectangular cavity (Trench/Manhole).

    Approach:
    1. Project the 3D cavity points onto the 2D XY plane.
    2. Use the 'minimum rotated rectangle' algorithm (via Shapely) to find the
       tightest 2D bounding box (OBB) for the footprint.
    3. Calculate depth as the distance from the ground level (Z=0) to the
       bottom of the cavity (using the 5th percentile Z to be robust against noise).
    """
    points_2d = mesh.vertices[:, :2]

    from shapely.geometry import MultiPoint
    # Find the smallest oriented rectangle that encloses the 2D points
    points_shapely = MultiPoint(points_2d)
    rect = points_shapely.minimum_rotated_rectangle

    if rect.geom_type == 'Polygon':
        # Extract the corners of the rectangle
        x, y = rect.exterior.coords.xy
        # Compute distances between adjacent corners to get side lengths
        d1 = np.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)
        d2 = np.sqrt((x[2]-x[1])**2 + (y[2]-y[1])**2)
        width, length = sorted([d1, d2])
    else:
        # Fallback to Axis-Aligned Bounding Box if OBB fails
        min_xy = np.min(points_2d, axis=0)
        max_xy = np.max(points_2d, axis=0)
        diff = max_xy - min_xy
        width, length = sorted([diff[0], diff[1]])

    # Depth is measured from ground (Z=0) down into the cavity
    depth = np.abs(np.percentile(mesh.vertices[:, 2], 5))

    return {
        'length': float(length),
        'width': float(width),
        'depth': float(depth)
    }

def measure_circular(mesh: trimesh.Trimesh):
    """
    Measure diameter, circumference, and depth of a circular cavity (Duct/Manhole).

    Approach:
    1. Project the 3D cavity points onto the 2D XY plane.
    2. Filter out outliers using median distance to the center.
    3. Perform a robust Least Squares circle fit to find the radius and center.
    4. Calculate circumference as 2 * pi * radius.
    """
    points_2d = mesh.vertices[:, :2]
    from scipy.optimize import least_squares

    def calc_R(c, x, y):
        return np.sqrt((x - c[0])**2 + (y - c[1])**2)

    def f(c, x, y):
        ri = calc_R(c, x, y)
        return ri - ri.mean()

    center_rough = np.median(points_2d, axis=0)
    dists_rough = np.linalg.norm(points_2d - center_rough, axis=1)
    mask = dists_rough < np.percentile(dists_rough, 95)
    points_filtered = points_2d[mask]

    x = points_filtered[:, 0]
    y = points_filtered[:, 1]
    center_estimate = np.mean(points_filtered, axis=0)

    res = least_squares(f, center_estimate, args=(x, y), loss='soft_l1')
    center = res.x
    ri = calc_R(center, x, y)
    radius = np.median(ri)
    diameter = radius * 2
    circumference = np.pi * diameter
    depth = np.abs(np.percentile(mesh.vertices[:, 2], 5))

    return {'diameter': diameter, 'circumference': circumference, 'depth': depth}

def classify_feature(mesh: trimesh.Trimesh):
    """
    Classify the cavity type based on geometric properties.

    Heuristics:
    - If the XY aspect ratio is high (> 3.0), it's classified as a 'trench'.
    - If the area of the 2D Convex Hull is significantly smaller than the
      Oriented Bounding Box (ratio < 0.85), it suggests a circular shape,
      classified as 'circular_manhole'.
    - Otherwise, it's classified as a 'manhole' (rectangular).
    """
    extents = mesh.bounding_box_oriented.extents
    xy_ratio = max(extents[0], extents[1]) / min(extents[0], extents[1])
    if xy_ratio > 3.0:
        return 'trench'

    points_2d = mesh.vertices[:, :2]
    from scipy.spatial import ConvexHull
    try:
        hull_2d = ConvexHull(points_2d)
        area_hull = hull_2d.volume # for 2D hull, volume is the area
    except:
        area_hull = 0
    area_obb = extents[0] * extents[1]
    ratio = area_hull / area_obb
    if ratio < 0.85:
         return 'circular_manhole'

    return 'manhole'
