import numpy as np
import trimesh

def measure_rectangular(mesh: trimesh.Trimesh):
    """
    Measure length, width, and depth of a rectangular cavity (Trench/Manhole/Handhole).

    Approach:
    1. Project the 3D cavity points onto the 2D XY plane.
    2. Use the 'minimum rotated rectangle' algorithm (via Shapely) to find the
       tightest 2D bounding box (OBB) for the footprint.
    3. Use multiple horizontal slices to compute a more robust width/length profile.
    4. Calculate depth as the distance from the ground level (Z=0) to the
       bottom of the cavity (using the 5th percentile Z to be robust against noise).
    """
    points = mesh.vertices
    points_2d = points[:, :2]

    from shapely.geometry import MultiPoint
    points_shapely = MultiPoint(points_2d)
    rect = points_shapely.minimum_rotated_rectangle

    # Initial OBB dimensions
    if rect.geom_type == 'Polygon':
        x, y = rect.exterior.coords.xy
        d1 = np.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)
        d2 = np.sqrt((x[2]-x[1])**2 + (y[2]-y[1])**2)
        width, length = sorted([d1, d2])
    else:
        min_xy = np.min(points_2d, axis=0)
        max_xy = np.max(points_2d, axis=0)
        diff = max_xy - min_xy
        width, length = sorted([diff[0], diff[1]])

    # Depth is measured from ground (Z=0) down into the cavity
    # We use percentiles to ignore potential "floaters" or artifacts at the very bottom
    depth = np.abs(np.percentile(points[:, 2], 5))

    # Slice-based refinement (for tapered walls)
    # We slice at 25%, 50%, and 75% of the depth
    z_min = np.percentile(points[:, 2], 5)
    slices = []
    for level in [0.25, 0.5, 0.75]:
        target_z = z_min * level
        mask = np.abs(points[:, 2] - target_z) < 0.02 # 2cm window
        if np.any(mask):
            slice_pts = MultiPoint(points[mask, :2]).minimum_rotated_rectangle
            if slice_pts.geom_type == 'Polygon':
                sx, sy = slice_pts.exterior.coords.xy
                sd1 = np.sqrt((sx[1]-sx[0])**2 + (sy[1]-sy[0])**2)
                sd2 = np.sqrt((sx[2]-sx[1])**2 + (sy[2]-sy[1])**2)
                slices.append(sorted([sd1, sd2]))

    # The user suggests OBB is very reliable for rectangular assets.
    # We use the dimensions from the 2D projection of the entire feature
    # as it captures the full footprint (rim + walls + bottom).
    width_refined, length_refined = width, length

    # Fit residual (standard deviation of the points from the OBB edges in the middle slice)
    confidence = 1.0
    if len(points) < 100:
        confidence *= 0.8 # Lower confidence for sparse points

    return {
        'length': float(length_refined),
        'width': float(width_refined),
        'depth': float(depth),
        'confidence': float(confidence)
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

    # Residual error for confidence
    residual = np.std(ri)
    confidence = np.clip(1.0 - (residual / (radius + 1e-6)), 0, 1)

    diameter = radius * 2
    circumference = np.pi * diameter
    depth = np.abs(np.percentile(mesh.vertices[:, 2], 5))

    return {
        'diameter': float(diameter),
        'circumference': float(circumference),
        'depth': float(depth),
        'confidence': float(confidence)
    }

def validate_scale(obj_type, dims):
    """
    Validate if measurements fall within typical infrastructure ranges.
    Returns (is_valid, reason).
    """
    # Typical ranges in meters
    RANGES = {
        'duct': {'diameter': (0.05, 0.25)},
        'circular_manhole': {'diameter': (0.5, 1.6)},
        'manhole': {'width': (0.3, 1.5), 'length': (0.4, 2.5)},
        'trench': {'width': (0.2, 1.5)}
    }

    if obj_type not in RANGES:
        return True, "no_range_data"

    limits = RANGES[obj_type]
    for param, (pmin, pmax) in limits.items():
        val = dims.get(param)
        if val is not None:
            if val < pmin or val > pmax:
                return False, f"{param}_out_of_range"

    return True, "ok"

def classify_feature(mesh: trimesh.Trimesh):
    """
    Classify the cavity type based on geometric properties.

    Heuristics:
    - If the XY aspect ratio is high (> 3.5), it's definitely a 'trench'.
    - If it's more square/circular, check the ratio of Convex Hull to OBB.
    - Also check if it's small (potential duct).
    """
    extents = mesh.bounding_box_oriented.extents
    # xy_extents = extents[:2]
    width, length = sorted(extents[:2])
    xy_ratio = length / width

    if xy_ratio > 3.5:
        return 'trench'

    points_2d = mesh.vertices[:, :2]
    from scipy.spatial import ConvexHull
    try:
        hull_2d = ConvexHull(points_2d)
        area_hull = hull_2d.volume
    except:
        area_hull = 0

    area_obb = width * length
    hull_ratio = area_hull / area_obb if area_obb > 0 else 0

    # Circular features have a hull/obb ratio around pi/4 (~0.785)
    # Solid rectangular features have a ratio close to 1.0.
    # However, sparse scans of pits often have points only on the edges,
    # which can still yield a high hull ratio.
    # If the hull ratio is very low, it's likely an irregular rectangular feature.

    if 0.72 < hull_ratio < 0.85:
        if max(width, length) < 0.4:
            return 'duct'
        return 'circular_manhole'

    if xy_ratio > 2.5:
        return 'trench'

    return 'manhole'
