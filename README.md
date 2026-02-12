# AI-Powered 3D Measurement Pipeline (FTTH)

This tool automatically derives high-precision measurements (mm-level accuracy) from 3D scans (OBJ, GLB, etc.) of field infrastructure such as trenches, ducts, manholes, and handholes.

## Methodology

To achieve mm-level accuracy, the pipeline uses a **Direct Geometric Fitting** approach rather than relying solely on deep learning on 2D projections.

### 1. Automatic Plane Alignment (RANSAC)
Most 3D scans are not perfectly aligned with the world axes. We use **RANSAC (Random Sample Consensus)** to identify the dominant plane in the mesh (the ground). The model is then automatically rotated and translated so that the ground is perfectly flat at `Z=0`. This ensures that "depth" always refers to the vertical distance into the ground.

### 2. Feature Segmentation & Cleaning
Once aligned, the tool identifies "negative space"—points that lie below the ground level. We use connected component analysis to isolate the primary feature (e.g., the trench) from noise or peripheral objects in the scan.

### 3. Automated Classification
The tool analyzes the 2D footprint of the segmented feature to classify it:
- **Trench**: High aspect ratio (long and narrow).
- **Circular Manhole/Duct**: High circularity (based on Convex Hull to OBB area ratio).
- **Rectangular Manhole/Handhole**: Low aspect ratio, rectangular footprint.

### 4. Scale Validation
Dimensions are cross-referenced against standard infrastructure ranges (e.g., manholes between 60cm and 150cm). If a measurement is wildly outside these bounds, it is flagged for review.

### 5. Precision Measurement
- **Rectangular Features**: Uses a **2D Oriented Bounding Box (OBB)** algorithm on the XY projection to find the true length and width, regardless of the trench's orientation in the scan.
- **Circular Features**: Uses a **Robust Least Squares Circle Fit** to determine the exact diameter and circumference.
- **Depth**: Calculated using high-percentile statistics from the Z-axis distribution to ignore noise at the bottom of the cavity.

## Project Structure

- `main.py`: Primary CLI entry point.
- `src/`: Core source code.
  - `processor.py`: Orchestrates the measurement pipeline.
  - `measurers.py`: Implementation of OBB and Circle fitting algorithms.
  - `utils/geometry.py`: RANSAC plane detection and mesh alignment.
  - `utils/mesh_io.py`: Robust 3D file loading and unit inference.
- `out/`: Default output directory for JSON/CSV results and debug visualizations.

## Usage

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Tool
Process a Scaniverse export folder or a single model file:
```bash
python main.py --input "path/to/scan_folder"
```

## Outputs
- `measurements.json`: Full metadata and dimensions in meters, centimeters, and millimeters.
- `measurements.csv`: Tabular summary for easy integration.
- `debug_view.png`: A top-down heightmap visualization with detected dimensions overlaid for verification.

## Advantages of this Approach
- **Accuracy**: Works directly with high-resolution vertex data.
- **Automation**: Zero manual intervention; no need to click points or define axes.
- **Robustness**: RANSAC and robust fitting handle noisy scans or irregular ground surfaces.
