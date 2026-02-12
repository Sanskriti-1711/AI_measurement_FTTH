# Scaniverse Pit Measurement Tool

This tool processes 3D scans from Scaniverse exports to automatically measure pit (cavity) dimensions, including depth, width, and area, in real-world units (meters, centimeters, millimeters). It uses deep learning for pit detection and provides outputs in JSON, CSV, and annotated images.

## Architecture

The pipeline consists of two main scripts:

1. **`run_scaniverse.py`**: 
   - Locates the OBJ model file in the provided Scaniverse export folder.
   - Calls `model_dims.py` to process the mesh and generate outputs.
   - Supports manual input via `--input` flag.

2. **`model_dims.py`**:
   - Loads the 3D mesh (OBJ/GLB/GLTF/PLY) using trimesh.
   - Infers units from OBJ metadata (e.g., "Unit: meter") or heuristics.
   - Computes axis-aligned bounding box (AABB) and oriented bounding box (OBB) extents.
   - Rasterizes the mesh into a heightmap grid.
   - Uses a fine-tuned FCN ResNet50 segmentation model (trained on synthetic pit masks) to detect pits in the heightmap.
   - Extracts pit contours, computes depth (rim - bottom Z), equivalent diameter, max width (via convex hull), and area.
   - Generates an annotated colormap image with measurements overlaid.
   - Outputs results to JSON and CSV files.

## What's Implemented

- **Automatic Unit Inference**: Reads Scaniverse OBJ headers for unit metadata; falls back to heuristic-based inference.
- **Deep Learning Pit Detection**: Pretrained FCN ResNet50 model fine-tuned on synthetic heightmap masks for pit segmentation.
- **Comprehensive Measurements**: Pit depth, rim height, bottom height, area, equivalent diameter, max width in m/cm/mm.
- **Outputs**:
  - `scaniverse_dims.json`: Full metadata, extents, and pit details.
  - `scaniverse_dims.csv`: Tabular data with per-axis extents and pit metrics.
  - `scaniverse_pit.png`: Annotated heightmap image with pit contours and text labels.
- **Robust Path Handling**: Supports nested Scaniverse folders and special characters.
- **Dependencies**: Listed in `requirements.txt` (trimesh, torch, torchvision, opencv, numpy, pillow).

## Usage

### Prerequisites
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

### Running the Tool
Process a Scaniverse export folder:

```bash
python run_scaniverse.py --input "path/to/Scaniverse_export_folder"
```

Or directly on a model file:

```bash
python model_dims.py --input "path/to/model.obj" --out-json results.json --out-csv results.csv
```

### Outputs
- Files saved to `out/` directory in the workspace.
- JSON: Detailed dictionary with bounds, extents, OBB, and pit metrics.
- CSV: Row with file info, units, extents in multiple units, and pit measurements.
- Image: Colormap of heightmap with pit annotation (depth, width, rim/bottom in mm).

## Future Enhancements

- **Multi-Scan Dataset Integration**: Process multiple Scaniverse ZIPs or folders, aggregating measurements into a master CSV dataset. Each row could include:
  - Location (latitude/longitude from OBJ metadata).
  - Measurements (depth, width, area in cm).
  - File path or ID for traceability.
  - This would enable batch processing and dataset building for analysis or ML training.
- **Improved Accuracy**: Fine-tune DL model on more diverse/real pit data, add multi-pit detection, or integrate pose estimation for photo overlays.
- **Web/CLI Enhancements**: Add batch processing flags, progress bars, or a simple web interface for uploads.

## Notes

- Assumes Scaniverse exports contain OBJ with "Unit: meter" metadata for accurate units.
- DL model trained on synthetic data; accuracy may vary — retrain with real labeled pits for better results.
- For glTF/GLB, assumes meters; for others, uses heuristic inference with confidence score.
- Force units with `--assume-units m|cm|mm` if needed.
