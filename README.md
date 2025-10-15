# RealSense Dual-Camera Extrinsic Calibration

A robust system for calibrating the extrinsic parameters between two Intel RealSense cameras using 3D-to-3D point cloud alignment with ChArUco board detection.

## 🎯 Overview

This calibration system uses the **Kabsch algorithm** to find the optimal rigid body transformation between two RealSense cameras by:
1. Detecting ChArUco board corners in both cameras
2. Converting 2D corners to 3D points using depth information
3. Finding the optimal rotation and translation using 3D-to-3D alignment
4. Validating the calibration with real-time visual feedback

## 📋 Requirements

### Hardware
- 2x Intel RealSense cameras with depth capability (tested with D435/D455)
- ChArUco calibration board (recommended size: A4, 5x7 squares)

### Software
```bash
pip install pyrealsense2 opencv-python numpy
```

## 🚀 Quick Start

### Step 1: Generate ChArUco Board Configuration
First, you need a ChArUco board configuration file. Create `charuco_board_config.npz` with:
```python
import numpy as np

# ChArUco board parameters
squares_x = 5          # Number of squares in X direction
squares_y = 7          # Number of squares in Y direction  
square_size_mm = 30.0  # Square size in millimeters
marker_size_mm = 24.0  # ArUco marker size in millimeters
aruco_dict_id = 0      # DICT_4X4_50

np.savez('charuco_board_config.npz',
         squares_x=squares_x,
         squares_y=squares_y,
         square_size_mm=square_size_mm,
         marker_size_mm=marker_size_mm,
         aruco_dict_id=aruco_dict_id)
```

### Step 2: Update Camera Serial Numbers
Edit both files and update the camera serial numbers:
```python
PC_CAMERA_SN = "YOUR_CAMERA_1_SERIAL"   # Primary camera
RGB_CAMERA_SN = "YOUR_CAMERA_2_SERIAL"  # Secondary camera
```

### Step 3: Perform Calibration
```bash
python calculate_extrinsics.py
```

**Instructions:**
- Position the ChArUco board so it's visible in both cameras
- Press **'c'** to capture a view when the board is clearly detected
- Capture **5-10 views** from different angles and distances
- Press **'q'** when done to calculate extrinsics

### Step 4: Validate Calibration
```bash
python check_inference.py
```

**What to look for:**
- **Green dots** = Points detected directly in the RGB camera
- **Red dots** = Points projected from the PC camera using calibration
- **Good calibration** = Green and red dots overlap closely
- Press **'q'** to quit

## 📁 Files

### `calculate_extrinsics.py`
- **Purpose**: Performs the actual calibration
- **Method**: 3D-to-3D rigid body transformation using Kabsch algorithm
- **Input**: ChArUco board views from both cameras
- **Output**: `extrinsics.npz` containing transformation parameters

### `check_inference.py`  
- **Purpose**: Real-time validation of calibration quality
- **Method**: Projects points between cameras and visualizes alignment
- **Input**: `extrinsics.npz` from calibration
- **Output**: Visual feedback on calibration accuracy

### `extrinsics.npz`
Contains the calibration results:
- `rvec`: Rotation vector (OpenCV format)
- `tvec`: Translation vector in millimeters
- `R`: Rotation matrix (3x3)
- `t`: Translation vector in meters
- `mean_3d_error`: Average 3D transformation error (mm)
- `calibration_method`: "3D_to_3D_Kabsch"

## 🔧 Coordinate Systems

The system uses **color-aligned coordinate systems** for consistency:
- **Calibration**: Uses color intrinsics for 3D point calculation
- **Validation**: Uses the same coordinate system for projection
- **Transformation**: PC camera → RGB camera coordinate system

## 📊 Quality Assessment

| Error Range | Quality | Recommendation |
|-------------|---------|----------------|
| < 5mm       | ✅ Excellent | Ready for production |
| 5-15mm      | ✅ Good | Acceptable for most applications |
| 15-30mm     | ⚠️ Acceptable | Consider recalibration |
| > 30mm      | ❌ Poor | Recalibration required |

## 🛠️ Troubleshooting

### High Calibration Error
- **Cause**: Poor data quality, insufficient views, or coordinate system issues
- **Solution**: 
  - Capture more views from diverse angles
  - Ensure good lighting and avoid glare
  - Check that both cameras have valid depth at calibration points

### Green/Red Dots Don't Align
- **Cause**: Calibration accuracy issues or coordinate system mismatch
- **Solution**:
  - Recalibrate with better data
  - Verify camera serial numbers are correct
  - Check that both cameras are detecting the board simultaneously

### "Not enough common corners" Error
- **Cause**: Board not visible in both cameras or poor detection
- **Solution**:
  - Reposition board to be clearly visible in both camera views
  - Improve lighting conditions
  - Ensure board is at appropriate distance (0.5-1.5m)

## 🔬 Technical Details

### Kabsch Algorithm
The system uses the Kabsch algorithm for optimal 3D-to-3D alignment:
1. **Centroid Calculation**: Find centroids of both point clouds
2. **Cross-Covariance**: Compute H = P_centered^T @ Q_centered  
3. **SVD**: Perform Singular Value Decomposition on H
4. **Rotation**: R = V^T @ U^T (ensuring det(R) = 1)
5. **Translation**: t = centroid_Q - R @ centroid_P

### Coordinate System Consistency
- Uses `rs.align(rs.stream.color)` to align depth to color frames
- All 3D calculations use color intrinsics for consistency
- Handles unit conversion properly (meters ↔ millimeters)

## 📋 Camera Setup Notes

- Both cameras must have depth capability enabled
- Recommended baseline: 20cm - 1m between cameras
- Avoid USB bandwidth issues by using separate USB controllers if possible
- Ensure stable mounting to prevent movement during calibration

## 🎯 Best Practices

1. **Data Collection**:
   - Capture 8-12 views from different positions
   - Include various board orientations and distances
   - Ensure board fills ~20-60% of image area

2. **Environment**:
   - Use consistent, diffuse lighting
   - Avoid reflective surfaces near the board
   - Minimize motion blur during capture

3. **Validation**:
   - Always run `check_inference.py` after calibration
   - Test with board at various positions
   - Verify alignment across the entire field of view

---
**Last Updated**: October 2025