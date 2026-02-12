# Project Report: Automated 3D Measurement Pipeline (FTTH)

## 1. Executive Summary
This project provides a production-ready, automated system for deriving high-precision measurements from 3D scans. Designed specifically for Fiber-to-the-Home (FTTH) infrastructure, the tool identifies and measures trenches, ducts, manholes, and handholes with millimeter-level accuracy. Unlike previous versions that relied on deep learning on 2D images, this pipeline works directly with 3D vertex data to ensure maximum precision.

## 2. Technical Architecture
The system is built as a modular Python application, allowing for easy integration with future APIs or mobile apps.

*   **Core Logic (`src/processor.py`)**: The "brain" of the system that orchestrates the entire measurement flow.
*   **Geometric Engine (`src/measurers.py`)**: Contains the specialized algorithms for measuring different shapes (Rectangles vs. Circles).
*   **Auto-Alignment (`src/utils/geometry.py`)**: Uses mathematical algorithms to find the ground and level the model.
*   **Input/Output (`src/utils/mesh_io.py`)**: Handles loading 3D files (OBJ, GLB) and inferring real-world units.

## 3. The Measurement Methodology
The pipeline follows a 6-step automated process:

### Step 1: Automatic Axis & Ground Alignment (RANSAC)
Most scans are captured at an angle, and different devices use different "Up" axes (e.g., Y-up vs Z-up). The tool automatically detects the orientation of the scan by analyzing vertex normals. It then uses **RANSAC (Random Sample Consensus)** to find the flat ground surface and levels the model so that ground is exactly at $Z=0$. This ensures depth is always measured vertically.

### Step 2: Cavity Segmentation
The tool identifies the "negative space"—the part of the scan that goes into the ground. It isolates the trench or pit from the surrounding environment (like the street or grass).

### Step 3: Feature Classification
The system analyzes the "footprint" of the object to determine what it is:
*   **Trench**: Long and narrow.
*   **Manhole/Handhole**: Square or slightly rounded box.
*   **Duct/Circular Manhole**: Circular shape.

### Step 4: Precision Geometric Fitting (Slice-based Median)
To achieve maximum accuracy and handle slanted or irregular walls:
*   **Horizontal Slicing**: The tool takes multiple horizontal slices through the cavity at different depths.
*   **Median Fitting**: It fits an **Oriented Bounding Box (OBB)** or Circle to each slice and takes the **median** dimension. This ignores noise from the rim or debris at the bottom, providing the most stable "true" dimensions.

### Step 5: Scale Validation
The tool has "sanity checks" built-in. It knows a typical manhole isn't 10 meters wide. If the measurements are outside standard infrastructure ranges, the tool flags the result as "Invalid" for human review.

### Step 6: Confidence Scoring
Every measurement comes with a confidence score (0% to 100%). This is based on how many 3D points were available and how perfectly they fit the expected shape.

## 4. Output & Integration
The tool produces three main outputs:
1.  **JSON Data**: A clean, machine-readable file for API integration.
2.  **CSV Summary**: A simple spreadsheet-ready file.
3.  **Debug View**: A top-down image with the measurements drawn on top for quick visual verification.

## 5. Usage Instructions
To run the tool, simply point it at a Scaniverse folder:
```bash
python main.py -i "path/to/scan"
```

## 6. Future Roadmap
*   **Multi-Feature Detection**: Enhancing the code to measure multiple different ducts or pits in a single large scan.
*   **Texture Recognition**: Using the photos from the scan to identify material types (Concrete vs. Asphalt).
*   **API Wrapper**: Wrapping this logic in a FastAPI or Flask service for cloud processing.

---
*Prepared by Jules (Software Engineer)*
