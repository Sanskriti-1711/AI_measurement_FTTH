Approach 2: Scaniverse Mesh + FCN Pit Detection

Summary:
- Loads Scaniverse OBJ exports, builds heightmap, applies FCN segmentation, and extracts pit dimensions.

Primary files:
- model_dims.py
- run_scaniverse.py
- inspect_scaniverse.py
- lidar.py
- README_model_dims.md

Inputs:
- Scaniverse exports (OBJ/MTL/JPG), zip archives, or folder paths.
- Optional pretrained model weights: out\fcn_pit_model.pth

Outputs:
- out\scaniverse_dims.json
- out\scaniverse_dims.csv
- out\scaniverse_pit.png
