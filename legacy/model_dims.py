#!/usr/bin/env python3
"""
model_dims.py

Load a 3D model (OBJ/GLB/GLTF/PLY) and report axis-aligned bounding-box dimensions.

Heuristics attempt to infer units automatically; for glTF/GLB files we assume meters by default
because glTF conventionally uses meters.

Usage:
  python model_dims.py --input "path/to/model.glb"
  python model_dims.py --input "path/to/folder"  # finds first supported file

Requires: trimesh
  pip install trimesh[all]

"""
import argparse
import json
import os
import sys
from typing import Optional
import numpy as np
import cv2 as cv

def find_model_file(path: str) -> Optional[str]:
    """Find a model file. If path is a file return it. If folder, search recursively
    and return the first supported file found."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        exts = ('.glb', '.gltf', '.obj', '.ply', '.stl')
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.lower().endswith(exts):
                    return os.path.join(root, fname)
    return None


def parse_obj_header_for_metadata(obj_path: str):
    """Read top of OBJ file for lines like '# Unit: meter' and return dict."""
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
                # format: '# Key: Value'
                parts = line[1:].split(':', 1)
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip()
                    meta[k] = v
    except Exception:
        pass
    return meta

def infer_units_from_extent(max_extent: float) -> (str, float):
    """
    Return (assumed_unit, confidence_score).
    Heuristic mapping:
      - ext > 1000 -> assume millimeters (mm)
      - ext > 10 -> assume centimeters (cm)
      - ext >= 0.01 and ext <= 10 -> assume meters (m)
      - otherwise fallback to meters
    """
    if max_extent > 1000:
        return 'mm', 0.6
    if max_extent > 10:
        return 'cm', 0.6
    if max_extent >= 0.01 and max_extent <= 10:
        return 'm', 0.9
    return 'm', 0.5

def convert_extents(extents: list, src_unit: str):
    # extents is [x,y,z]
    # Normalize to meters first
    if src_unit == 'm':
        meters = [float(e) for e in extents]
    elif src_unit == 'cm':
        meters = [float(e) / 100.0 for e in extents]
    elif src_unit == 'mm':
        meters = [float(e) / 1000.0 for e in extents]
    else:
        meters = [float(e) for e in extents]

    mm = [m * 1000.0 for m in meters]
    cm = [m * 100.0 for m in meters]
    inch = [m * 39.3700787402 for m in meters]
    return {
        'm': meters,
        'mm': mm,
        'cm': cm,
        'in': inch,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Model file or folder')
    p.add_argument('--assume-units', choices=['auto','m','cm','mm'], default='auto')
    p.add_argument('--out-json', default=None, help='Write results to JSON file')
    p.add_argument('--out-csv', default=None, help='Write results to CSV file')
    args = p.parse_args()

    model_path = find_model_file(args.input)
    if model_path is None:
        print('No supported model file found at', args.input)
        sys.exit(2)

    try:
        import trimesh
    except Exception as e:
        print('This tool requires trimesh. Install with: pip install trimesh[all]')
        raise

    mesh = None
    obj_meta = {}
    if model_path.lower().endswith('.obj'):
        obj_meta = parse_obj_header_for_metadata(model_path)

    # If OBJ header contains Unit, trust it
    if obj_meta.get('Unit'):
        assumed_unit = obj_meta.get('Unit').lower()
        # normalize common names
        if assumed_unit.startswith('m'):
            assumed_unit = 'm'
        elif assumed_unit.startswith('cm'):
            assumed_unit = 'cm'
        elif assumed_unit.startswith('mm'):
            assumed_unit = 'mm'
        else:
            assumed_unit = obj_meta.get('Unit')
        confidence = 0.99
    try:
        loaded = trimesh.load(model_path, force='mesh')
        # trimesh may return Scene; convert if needed
        if isinstance(loaded, trimesh.Scene):
            # merge geometry
            mesh = trimesh.util.concatenate(loaded.dump(concatenate=True))
        else:
            mesh = loaded
    except Exception as e:
        print('Failed to load model with trimesh:', e)
        sys.exit(3)

    bounds = mesh.bounds.tolist()  # [[minx,miny,minz],[maxx,maxy,maxz]]
    mins, maxs = bounds[0], bounds[1]
    extents = [maxs[i] - mins[i] for i in range(3)]
    max_extent = max(extents)

    assumed_unit = None
    confidence = 0.0
    if args.assume_units != 'auto':
        assumed_unit = args.assume_units
        confidence = 0.99
    else:
        # If glTF/GLB assume meters by convention
        if model_path.lower().endswith(('.glb', '.gltf')):
            assumed_unit = 'm'
            confidence = 0.9
        else:
            assumed_unit, confidence = infer_units_from_extent(max_extent)

    conv = convert_extents(extents, assumed_unit)

    # Oriented bounding box (OBB)
    obb = None
    try:
        obb_box = mesh.bounding_box_oriented
        obb_extents = obb_box.extents.tolist()
        # transform matrix
        obb_transform = obb_box.primitive.transform.tolist() if hasattr(obb_box, 'primitive') else None
        obb = {
            'obb_extents_m': obb_extents,
            'obb_transform': obb_transform,
        }
    except Exception:
        obb = None

    # Heightmap rasterization for pit detection
    pit_info = None
    try:
        # grid resolution (default 512 on longer side)
        GRID = 512
        minx, miny, minz = mins
        maxx, maxy, maxz = maxs
        width = maxx - minx
        height = maxy - miny
        if width <= 0 or height <= 0:
            raise RuntimeError('Degenerate XY extents')

        if width >= height:
            nx = GRID
            ny = max(4, int(round(GRID * (height / width))))
        else:
            ny = GRID
            nx = max(4, int(round(GRID * (width / height))))

        verts = mesh.vertices
        xs = verts[:, 0]
        ys = verts[:, 1]
        zs = verts[:, 2]

        ix = np.clip(((xs - minx) / width * (nx - 1)).astype(int), 0, nx - 1)
        iy = np.clip(((ys - miny) / height * (ny - 1)).astype(int), 0, ny - 1)

        idx = iy * nx + ix
        grid_max = np.full(nx * ny, -np.inf, dtype=np.float64)
        np.maximum.at(grid_max, idx, zs)
        grid = grid_max.reshape((ny, nx))

        # fill empty cells (-inf) by iterative neighbor max propagation
        invalid = (grid == -np.inf)
        iter_count = 0
        while invalid.any() and iter_count < 200:
            # pad and compute neighbor max
            pad = np.pad(grid, ((1, 1), (1, 1)), mode='constant', constant_values=-np.inf)
            neigh_max = np.full_like(grid, -np.inf)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neigh = pad[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
                    neigh_max = np.maximum(neigh_max, neigh)
            fill_mask = (grid == -np.inf) & (neigh_max > -np.inf)
            if not fill_mask.any():
                break
            grid[fill_mask] = neigh_max[fill_mask]
            invalid = (grid == -np.inf)
            iter_count += 1

        # if still invalid, set to min z
        if (grid == -np.inf).any():
            grid[grid == -np.inf] = np.nanmin(grid[grid != -np.inf])

        # smooth
        g = grid.astype(np.float32)
        g_blur = cv.GaussianBlur(g, (9, 9), 0)

        # Use DL model for pit detection
        import torch
        import torchvision.transforms as T
        from torchvision.models.segmentation import fcn_resnet50

        # Load pretrained FCN and replace classifier
        model = fcn_resnet50(weights=None)
        model.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 256, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 1, 1)
        )
        # Load trained weights
        model_path = os.path.join(os.path.dirname(__file__), 'out', 'fcn_pit_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.aux_classifier = None
        model.eval()

        # Prepare input image (heightmap)
        norm = cv.normalize(g_blur, None, 0, 255, cv.NORM_MINMAX)
        img = cv.cvtColor(norm.astype(np.uint8), cv.COLOR_GRAY2RGB)
        tf = T.Compose([T.ToPILImage(), T.Resize((256, 256)), T.ToTensor()])
        inp = tf(img).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)['out'].squeeze(0).squeeze(0).cpu().numpy()
        prob = 1 / (1 + np.exp(-out))
        mask_256 = cv.resize((~np.isnan(g)).astype(np.uint8), (256, 256), interpolation=cv.INTER_NEAREST)
        prob = prob * mask_256.astype(np.float32)
        pit_mask = (prob > 0.2).astype('uint8') * 255

        # Resize mask back to grid size
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(pit_mask.astype('uint8'))
        mask_pil = mask_pil.resize((nx, ny), resample=PILImage.NEAREST)
        pit_mask = np.array(mask_pil) > 127

        # find connected components
        contours, _ = cv.findContours(pit_mask.astype(np.uint8) * 255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Fallback to original heightmap thresholding
            print('DL model did not detect pit, falling back to original thresholding')
            local_mean = cv.blur(g_blur, (31, 31))
            diff = local_mean - g_blur
            dmean = float(np.nanmean(diff))
            dstd = float(np.nanstd(diff))
            thresh = dmean + 1.0 * dstd
            pit_mask = (diff > thresh).astype('uint8') * 255
            contours, _ = cv.findContours(pit_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            # choose largest area in pixels
            areas = [cv.contourArea(c) for c in contours]
            max_i = int(np.argmax(areas))
            pit_cnt = contours[max_i]
            # create binary mask for chosen pit
            pm = np.zeros_like(pit_mask, dtype=np.uint8)
            cv.drawContours(pm, [pit_cnt], -1, 255, -1)
            mask_bool = pm.astype(bool)
            # choose largest area in pixels
            areas = [cv.contourArea(c) for c in contours]
            max_i = int(np.argmax(areas))
            pit_cnt = contours[max_i]
            # create binary mask for chosen pit
            pm = np.zeros_like(pit_mask)
            cv.drawContours(pm, [pit_cnt], -1, 255, -1)
            mask_bool = pm.astype(bool)

            bottom = float(np.nanmin(g[mask_bool]))
            # rim: dilate mask and take ring
            k = np.ones((25, 25), dtype=np.uint8)
            dil = cv.dilate(pm, k, iterations=1)
            rim = (dil.astype(bool)) & (~mask_bool)
            g_valid = g[~np.isnan(g)]
            global_median = float(np.median(g_valid)) if len(g_valid) > 0 else 0.0
            if rim.any():
                rim_vals = g[rim]
                rim_vals = rim_vals[~np.isnan(rim_vals)]
                if len(rim_vals) > 0:
                    rim_height = float(np.median(rim_vals))
                else:
                    rim_height = global_median
            else:
                rim_height = global_median

            depth_m = rim_height - bottom
            # area in m^2: area_pixels * (res_x * res_y)
            res_x = width / (nx - 1)
            res_y = height / (ny - 1)
            area_px = np.sum(mask_bool)
            area_m2 = area_px * res_x * res_y
            # approximate equivalent diameter
            equiv_diam_m = 2.0 * np.sqrt(area_m2 / np.pi)

            # compute more accurate max width across pit using hull points
            ys_idx, xs_idx = np.where(pm == 255)
            if len(xs_idx) >= 2:
                # convert to world coordinates (x = minx + ix*res_x, y = miny + iy*res_y)
                pts_x = minx + (xs_idx.astype(np.float64) * res_x)
                pts_y = miny + (ys_idx.astype(np.float64) * res_y)
                pts2 = np.column_stack([pts_x, pts_y])
                # convex hull of 2D points
                try:
                    hull_idx = cv.convexHull(pts2.astype(np.float32), returnPoints=False)
                    hull_pts = pts2[hull_idx.squeeze()]
                except Exception:
                    hull_pts = pts2

                n_h = hull_pts.shape[0]
                max_d = 0.0
                if n_h > 1:
                    # brute-force pairwise distances (acceptable for moderate hull sizes)
                    diffs_x = hull_pts[:, 0][:, None] - hull_pts[:, 0][None, :]
                    diffs_y = hull_pts[:, 1][:, None] - hull_pts[:, 1][None, :]
                    dists = np.sqrt(diffs_x * diffs_x + diffs_y * diffs_y)
                    max_d = float(dists.max())
                else:
                    max_d = 0.0
                pit_max_width_m = max_d
            else:
                pit_max_width_m = equiv_diam_m

            if np.isnan(depth_m):
                pit_info = None
            else:
                pit_info = {
                    'pit_bottom_m': bottom,
                    'pit_rim_m': rim_height,
                    'pit_depth_m': depth_m,
                    'pit_area_m2': area_m2,
                    'pit_equiv_diam_m': equiv_diam_m,
                    'pit_max_width_m': pit_max_width_m,
                    'pit_mask_pixels': int(area_px),
                }

            # create overlay image
            norm = cv.normalize(g, None, 0, 255, cv.NORM_MINMAX)
            cmap = cv.applyColorMap(norm.astype(np.uint8), cv.COLORMAP_TURBO)
            # draw contour
            cv.drawContours(cmap, [pit_cnt], -1, (0, 0, 255), 2)
            # annotate depth/width with multiple lines
            M = cv.moments(pit_cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 10, 10
            txt1 = f"Depth: {pit_info['pit_depth_m']*1000:.1f} mm"
            txt2 = f"Width(eq): {pit_info['pit_equiv_diam_m']*1000:.1f} mm"
            txt3 = f"Width(max): {pit_info['pit_max_width_m']*1000:.1f} mm"
            txt4 = f"Rim: {pit_info['pit_rim_m']*1000:.1f} mm | Bottom: {pit_info['pit_bottom_m']*1000:.1f} mm"
            cv.putText(cmap, txt1, (max(5, cx-150), max(20, cy-40)), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv.LINE_AA)
            cv.putText(cmap, txt2, (max(5, cx-150), max(20, cy-25)), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv.LINE_AA)
            cv.putText(cmap, txt3, (max(5, cx-150), max(20, cy-10)), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv.LINE_AA)
            cv.putText(cmap, txt4, (max(5, cx-150), max(20, cy+5)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv.LINE_AA)
            pit_image_path = os.path.join('out', 'scaniverse_pit.png')
            os.makedirs(os.path.dirname(pit_image_path), exist_ok=True)
            cv.imwrite(pit_image_path, cmap)
        else:
            pit_info = None
            pit_image_path = None
    except Exception as e:
        pit_info = None
        pit_image_path = None

    out = {
        'file': model_path,
        'obj_metadata': obj_meta,
        'bounds_min': mins,
        'bounds_max': maxs,
        'extents_raw': extents,
        'assumed_unit': assumed_unit,
        'assumption_confidence': confidence,
        'extents_m': conv['m'],
        'extents_mm': conv['mm'],
        'extents_cm': conv['cm'],
        'extents_in': conv['in'],
    }
    if obb is not None:
        out.update(obb)
    if pit_info is not None:
        out['pit'] = pit_info
        out['pit_image'] = pit_image_path

    print(json.dumps(out, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, 'w', encoding='utf8') as f:
            json.dump(out, f, indent=2)

    if args.out_csv:
        # Write CSV with only length, width, depth
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        import csv
        with open(args.out_csv, 'w', newline='', encoding='utf8') as f:
            writer = csv.writer(f)
            header = ['length_cm', 'width_cm', 'depth_cm']
            writer.writerow(header)
            row = []
            if out.get('pit'):
                p = out['pit']
                row = [
                    f"{p.get('pit_max_width_m', 0)*100:.1f}",
                    f"{p.get('pit_equiv_diam_m', 0)*100:.1f}",
                    f"{p.get('pit_depth_m', 0)*100:.1f}",
                ]
            else:
                row = ['N/A', 'N/A', 'N/A']
            writer.writerow(row)

if __name__ == '__main__':
    main()
