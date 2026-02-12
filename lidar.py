import os
import argparse
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image, ExifTags

# ArUco for scaling
ARUCO_DICT = cv.aruco.DICT_4X4_50
MARKER_SIZE_MM = 100.0  # Adjust if different
CALIB_PATH = "MultiMatrix.npz"


# ---------------------------
# EXIF GPS (if present)
# ---------------------------
def _dms_to_deg(dms, ref):
    deg = dms[0][0] / dms[0][1]
    minute = dms[1][0] / dms[1][1]
    sec = dms[2][0] / dms[2][1]
    val = deg + minute / 60.0 + sec / 3600.0
    return -val if ref in ("S", "W") else val

def extract_gps_from_exif(img_path: str):
    try:
        img = Image.open(img_path)
        exif = img.getexif()
        if not exif:
            return None
        gps_info = None
        for k, v in exif.items():
            if ExifTags.TAGS.get(k) == "GPSInfo":
                gps_info = v
                break
        if not gps_info:
            return None
        gps_tag_map = {}
        for key, val in gps_info.items():
            name = ExifTags.GPSTAGS.get(key, key)
            gps_tag_map[name] = val

        needed = ["GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef"]
        if not all(x in gps_tag_map for x in needed):
            return None

        lat = _dms_to_deg(gps_tag_map["GPSLatitude"], gps_tag_map["GPSLatitudeRef"])
        lon = _dms_to_deg(gps_tag_map["GPSLongitude"], gps_tag_map["GPSLongitudeRef"])
        return {"lat": lat, "lon": lon}
    except Exception:
        return None


def load_calibration(npz_path: str):
    data = np.load(npz_path)
    cam_mat = data["camMatrix"]
    dist_coef = data["distCoef"]
    return cam_mat, dist_coef


def make_aruco_detector(dict_id: int):
    aruco = cv.aruco

    if hasattr(aruco, "ArucoDetector"):
        marker_dict = aruco.getPredefinedDictionary(dict_id)
        params = aruco.DetectorParameters()

        # Tuning for detection
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 0.5
        params.minDistanceToBorder = 3

        detector = aruco.ArucoDetector(marker_dict, params)

        def detect(gray):
            return detector.detectMarkers(gray)

        return detect

    # Old API fallback
    marker_dict = aruco.Dictionary_get(dict_id)
    params = aruco.DetectorParameters_create()

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 0.5
    params.minDistanceToBorder = 3

    def detect(gray):
        corners, ids, rejected = aruco.detectMarkers(gray, marker_dict, parameters=params)
        return corners, ids, rejected

    return detect


def detect_aruco_and_get_scale(bgr: np.ndarray, marker_size_mm: float):
    """
    Detect ArUco marker and compute pixels per mm.
    Returns tuple (px_per_mm, corners, ids) or (None, None, None) if not found.
    """
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    detect = make_aruco_detector(ARUCO_DICT)
    corners, ids, rejected = detect(gray)

    if ids is None or len(corners) == 0:
        return None, None, None

    # Assume first marker
    marker_corners = corners[0][0]  # 4 corners

    # Compute side length in pixels (average of two sides)
    side1 = np.linalg.norm(marker_corners[0] - marker_corners[1])
    side2 = np.linalg.norm(marker_corners[1] - marker_corners[2])
    side_px = (side1 + side2) / 2.0

    px_per_mm = side_px / marker_size_mm
    return px_per_mm, corners, ids


def interactive_measure_px_per_mm(img_path: str):
    """Show image and let user click two points to measure pixel distance, then ask for real-world mm."""
    img = cv.imread(img_path)
    if img is None:
        print("Could not read image for interactive measurement:", img_path)
        return None

    pts = []

    def on_mouse(ev, x, y, flags, param):
        if ev == cv.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    win = "measure"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.imshow(win, img)
    cv.setMouseCallback(win, on_mouse)

    print("Click two points on the image (left button). Press 'q' to finish.")
    while True:
        vis = img.copy()
        for p in pts:
            cv.circle(vis, p, 4, (0, 255, 0), -1)
        if len(pts) >= 2:
            cv.line(vis, pts[0], pts[1], (0, 255, 0), 2)
        cv.imshow(win, vis)
        k = cv.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        if len(pts) >= 2:
            # allow immediate break after two clicks
            break

    cv.destroyWindow(win)

    if len(pts) < 2:
        print("Not enough points clicked.")
        return None

    (x1, y1), (x2, y2) = pts[0], pts[1]
    px_dist = float(np.hypot(x2 - x1, y2 - y1))
    try:
        mm = float(input("Enter real-world distance between points in millimeters: "))
    except Exception:
        print("Invalid input")
        return None

    if mm <= 0:
        print("Distance must be positive")
        return None

    px_per_mm = px_dist / mm
    print(f"Measured pixel distance = {px_dist:.2f}px -> {px_per_mm:.6f} px/mm")
    return px_per_mm


# ---------------------------
# Segmentation from "blue" depth/mask views
# ---------------------------
def segment_trench_blue(bgr: np.ndarray):
    """
    Designed for LiDAR View style: trench highlighted cyan/blue/green against dark/blue background.
    Tune HSV thresholds if needed.
    """
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)

    # Two ranges: cyan/blue and light-cyan/greenish
    mask1 = cv.inRange(hsv, (85, 40, 30), (140, 255, 255))   # blue-ish
    mask2 = cv.inRange(hsv, (55, 30, 30), (95, 255, 255))    # cyan/green-ish
    mask = cv.bitwise_or(mask1, mask2)

    # Clean up
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

    return mask


def largest_component(mask: np.ndarray):
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv.contourArea)
    if cv.contourArea(cnt) < 1000:
        return None
    return cnt


def detect_pit(mask: np.ndarray):
    """Detect a pit (hole) in the mask. Returns contour, length_px, width_px, rect or None."""
    # Use hierarchy to find contours with holes (children)
    cnts, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    if not cnts or hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    pit_candidates = []
    for i, h in enumerate(hierarchy):
        # h: [next, prev, first_child, parent]
        child = int(h[2])
        parent = int(h[3])
        # a contour with a parent is a hole (inner contour)
        if parent >= 0:
            # area of the hole contour
            a = abs(cv.contourArea(cnts[i]))
            if a > 100:  # filter tiny holes
                pit_candidates.append((a, cnts[i]))

    if not pit_candidates:
        return None

    # choose largest hole by area
    pit_cnt = max(pit_candidates, key=lambda x: x[0])[1]
    rect = cv.minAreaRect(pit_cnt)
    (cx, cy), (w, h), angle = rect
    length_px = float(max(w, h))
    width_px = float(min(w, h))
    return pit_cnt, length_px, width_px, float(angle), rect


def min_area_rect_dims(cnt):
    rect = cv.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
    (cx, cy), (w, h), angle = rect
    length_px = float(max(w, h))
    width_px = float(min(w, h))
    return (cx, cy), length_px, width_px, float(angle), rect


def width_profile_along_trench(mask: np.ndarray, bins: int = 60):
    """
    Estimate width profile by slicing the mask along its principal axis.
    Returns list of (t_0_1, width_px_at_t).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return []

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])

    # PCA for main axis
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    main = eigvecs[:, np.argmax(eigvals)]  # (2,)
    main = main / (np.linalg.norm(main) + 1e-9)
    perp = np.array([-main[1], main[0]], dtype=np.float32)

    # Project points onto main axis
    t = centered @ main
    tmin, tmax = float(t.min()), float(t.max())
    if abs(tmax - tmin) < 1e-6:
        return []

    profile = []
    for i in range(bins):
        a = tmin + (tmax - tmin) * (i / bins)
        b = tmin + (tmax - tmin) * ((i + 1) / bins)
        sel = (t >= a) & (t < b)
        if sel.sum() < 20:
            continue
        # width is spread along perpendicular axis
        w = (centered[sel] @ perp)
        width_px = float(w.max() - w.min())
        t01 = float((0.5 * (a + b) - tmin) / (tmax - tmin))
        profile.append((t01, width_px))

    return profile


def relative_depth_stats_from_colormap(depth_bgr: np.ndarray, trench_mask: np.ndarray):
    """
    Screenshot depth maps are not raw depth. This returns relative indices only.
    """
    gray = cv.cvtColor(depth_bgr, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    vals = gray[trench_mask > 0]
    if vals.size == 0:
        return None
    return {
        "depth_rel_median_0_1": float(np.median(vals)),
        "depth_rel_p10_0_1": float(np.quantile(vals, 0.10)),
        "depth_rel_p90_0_1": float(np.quantile(vals, 0.90)),
    }


def draw_debug(bgr, mask, cnt, rect, out_path):
    vis = bgr.copy()
    vis_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    vis = cv.addWeighted(vis, 0.85, vis_mask, 0.35, 0)

    if cnt is not None:
        cv.drawContours(vis, [cnt], -1, (0, 255, 255), 2, cv.LINE_AA)

    if rect is not None:
        box = cv.boxPoints(rect).astype(np.int32)
        cv.polylines(vis, [box], True, (0, 255, 0), 2, cv.LINE_AA)

    cv.imwrite(out_path, vis)


# ---------------------------
# MAIN
# ---------------------------
def process_pair(rgb_path, mask_or_depth_path, depth_path=None, out_dir="out", px_per_mm_override: float = None, marker_image: str = None, model_json: str = None):
    os.makedirs(out_dir, exist_ok=True)

    rgb = cv.imread(rgb_path)
    md = cv.imread(mask_or_depth_path)
    if rgb is None or md is None:
        raise RuntimeError("Could not read one of the images.")

    gps = extract_gps_from_exif(rgb_path)  # often None

    # Determine scale: prefer override, then optional marker image, then try rgb image
    scale_px_per_mm = px_per_mm_override
    marker_debug_path = None

    if scale_px_per_mm is None and marker_image:
        mimg = cv.imread(marker_image)
        if mimg is not None:
            s, corners, ids = detect_aruco_and_get_scale(mimg, MARKER_SIZE_MM)
            if s is not None:
                scale_px_per_mm = s
                marker_debug_path = os.path.join(out_dir, "marker_detect_from_image.png")
                vis = mimg.copy()
                if ids is not None:
                    try:
                        cv.aruco.drawDetectedMarkers(vis, corners, ids)
                    except Exception:
                        pass
                cv.imwrite(marker_debug_path, vis)

    if scale_px_per_mm is None:
        s, corners, ids = detect_aruco_and_get_scale(rgb, MARKER_SIZE_MM)
        if s is not None:
            scale_px_per_mm = s
            marker_debug_path = os.path.join(out_dir, "marker_detect_in_rgb.png")
            vis = rgb.copy()
            try:
                cv.aruco.drawDetectedMarkers(vis, corners, ids)
            except Exception:
                pass
            cv.imwrite(marker_debug_path, vis)

    print(f"Scale detected (px/mm): {scale_px_per_mm}")

    mask = segment_trench_blue(md)
    cnt = largest_component(mask)
    # detect pit (hole) if present
    pit_info = detect_pit(mask)
    if pit_info is not None:
        pit_cnt, pit_length_px, pit_width_px, pit_angle, pit_rect = pit_info
    else:
        pit_cnt = None
        pit_length_px = None
        pit_width_px = None
        pit_angle = None
        pit_rect = None

    if cnt is None:
        debug_path = os.path.join(out_dir, "debug_no_trench.png")
        cv.imwrite(debug_path, md)
        return {
            "status": "no_trench_found",
            "rgb": rgb_path,
            "mask_or_depth": mask_or_depth_path,
            "gps": gps,
            "debug": debug_path,
        }

    center, length_px, width_px, angle, rect = min_area_rect_dims(cnt)
    profile = width_profile_along_trench(mask, bins=70)

    # If still no scale, but user provided a model JSON, infer px/mm by matching
    # the measured trench length (px) to the model's largest extent (mm).
    if scale_px_per_mm is None and model_json is not None:
        try:
            import json
            with open(model_json, 'r', encoding='utf8') as f:
                mj = json.load(f)
            model_extents_mm = mj.get('extents_mm') or mj.get('extents_m')
            if model_extents_mm:
                if mj.get('extents_m') and not mj.get('extents_mm'):
                    model_extents_mm = [e * 1000.0 for e in mj['extents_m']]

                model_max_mm = float(max(model_extents_mm))
                if model_max_mm > 0:
                    scale_px_per_mm = length_px / model_max_mm
                    print(
                        f"Inferred scale from model_json: {scale_px_per_mm} px/mm "
                        f"(using model max extent {model_max_mm} mm)"
                    )
        except Exception as e:
            print('Failed to use model_json for scale inference:', e)

    depth_stats = None
    if depth_path:
        depth_img = cv.imread(depth_path)
        if depth_img is not None:
            depth_stats = relative_depth_stats_from_colormap(depth_img, mask)

    debug_path = os.path.join(out_dir, "debug_overlay.png")
    draw_debug(md, mask, cnt, rect, debug_path)
    # save pit debug overlay if found
    if pit_cnt is not None:
        pit_debug = os.path.join(out_dir, "debug_pit.png")
        vis2 = md.copy()
        try:
            cv.drawContours(vis2, [pit_cnt], -1, (0, 0, 255), 2, cv.LINE_AA)
            box = cv.boxPoints(pit_rect).astype(np.int32)
            cv.polylines(vis2, [box], True, (255, 0, 0), 2, cv.LINE_AA)
        except Exception:
            pass
        cv.imwrite(pit_debug, vis2)
    else:
        pit_debug = None

    # Convert to real units if scale available
    conversions = {}
    if scale_px_per_mm is not None:
        mm_per_px = 1.0 / scale_px_per_mm
        length_mm = length_px * mm_per_px
        width_mm = width_px * mm_per_px
        conversions = {
            "length_mm": length_mm,
            "width_mm": width_mm,
            "length_m": length_mm / 1000.0,
            "width_m": width_mm / 1000.0,
            "length_cm": length_mm / 10.0,
            "width_cm": width_mm / 10.0,
            "length_in": length_mm / 25.4,
            "width_in": width_mm / 25.4,
        }
        # Convert profile widths
        profile_converted = [(t, w * mm_per_px) for t, w in profile]
    else:
        profile_converted = profile

    # Pit conversions
    pit_conversions = {}
    if pit_length_px is not None and scale_px_per_mm is not None:
        pit_length_mm = pit_length_px * (1.0 / scale_px_per_mm)
        pit_width_mm = pit_width_px * (1.0 / scale_px_per_mm)
        pit_conversions = {
            'pit_length_mm': pit_length_mm,
            'pit_width_mm': pit_width_mm,
            'pit_length_cm': pit_length_mm / 10.0,
            'pit_width_cm': pit_width_mm / 10.0,
            'pit_length_m': pit_length_mm / 1000.0,
            'pit_width_m': pit_width_mm / 1000.0,
            'pit_length_in': pit_length_mm / 25.4,
            'pit_width_in': pit_width_mm / 25.4,
        }

    return {
        "status": "ok",
        "rgb": rgb_path,
        "mask_or_depth": mask_or_depth_path,
        "gps": gps,
        "center_px_x": center[0],
        "center_px_y": center[1],
        "length_px": length_px,
        "width_px": width_px,
        "angle_deg": angle,
        "depth_stats_rel": depth_stats,
        "width_profile": profile,  # original in px
        "width_profile_mm": profile_converted if scale_px_per_mm else None,
        "debug": debug_path,
        "pit_debug": pit_debug,
        "pit_found": bool(pit_cnt is not None),
        "pit_length_px": pit_length_px,
        "pit_width_px": pit_width_px,
        "pit_angle_deg": pit_angle,
        "scale_px_per_mm": scale_px_per_mm,
        **conversions,
        **pit_conversions,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", default="v6.jpeg", help="RGB image path")
    p.add_argument("--mask", default="v2.jpeg", help="mask/depth image path")
    p.add_argument("--depth", default="v3.jpeg", help="optional colored depth image")
    p.add_argument("--out", default="out", help="output directory")
    p.add_argument("--px-per-mm", type=float, default=None, help="Manual scale: pixels per millimeter")
    p.add_argument("--marker-image", default=None, help="Optional separate image to detect ArUco marker from")
    p.add_argument("--model-json", default=None, help="Path to model JSON from model_dims.py to infer scale automatically")
    p.add_argument("--measure", action="store_true", help="Interactively measure two points to compute px/mm")
    args = p.parse_args()

    res = process_pair(
        rgb_path=args.rgb,
        mask_or_depth_path=args.mask,
        depth_path=args.depth,
        out_dir=args.out,
        px_per_mm_override=args.px_per_mm,
        marker_image=args.marker_image,
        model_json=args.model_json,
    )

    # If user requested interactive measurement and no px/mm override provided
    if args.measure and args.px_per_mm is None:
        mm_scale = interactive_measure_px_per_mm(args.rgb)
        if mm_scale is not None:
            print("Re-running with measured scale...")
            res = process_pair(
                rgb_path=args.rgb,
                mask_or_depth_path=args.mask,
                depth_path=args.depth,
                out_dir=args.out,
                px_per_mm_override=mm_scale,
                marker_image=None,
            )

    # Save JSON + CSV-friendly outputs
    # Flatten width profile to a CSV
    if res["status"] == "ok":
        prof = pd.DataFrame(res["width_profile"], columns=["t_0_1", "width_px"])
        prof.to_csv("out/width_profile.csv", index=False)

        if res.get("width_profile_mm"):
            prof_mm = pd.DataFrame(res["width_profile_mm"], columns=["t_0_1", "width_mm"])
            prof_mm.to_csv("out/width_profile_mm.csv", index=False)

    pd.DataFrame([{
        k: v for k, v in res.items() if k not in ("width_profile", "width_profile_mm")
    }]).to_csv("out/summary.csv", index=False)

    files_saved = ["out/summary.csv", "out/width_profile.csv"]
    if res.get("width_profile_mm"):
        files_saved.append("out/width_profile_mm.csv")
    files_saved.append(res.get("debug"))
    print("Saved:", ", ".join(files_saved))
