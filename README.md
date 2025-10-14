# RealSense Camera Intrinsic Calibration

This project provides tools for calibrating Intel RealSense camera intrinsics using ChArUco boards.

## 📁 Project Structure

```
pnp_based_approach/
├── camera_intrinsic_calibration.py  # Main calibration script (START HERE)
├── generate_charuco.py              # Generate ChArUco calibration board
├── calculate_extrinsics.py          # Calculate camera extrinsics (optional)
├── find_camera.py                   # Find connected RealSense cameras
├── check.py                         # Verify calibration results
└── charuco_board_A4.png             # Generated calibration board (print this)
```

## 🚀 Quick Start

### 1. Generate Calibration Board
```bash
python generate_charuco.py
```
This creates `charuco_board_A4.png` - **print this on A4 paper**.

### 2. Run Camera Calibration
```bash
python camera_intrinsic_calibration.py
```

**Steps:**
1. Hold the printed ChArUco board in front of the camera
2. Press 'c' to capture frames (aim for 15-30 frames)
3. Press 'q' to start calibration
4. Results saved to `intrinsics_charuco_<SERIAL>.npz`

### 3. Verify Results (Optional)
```bash
python check.py
```

## ⚙️ Configuration

Edit `camera_intrinsic_calibration.py`:
```python
TARGET_SERIAL_NUMBER = "832112070255"  # Change to your camera's serial
```

Find your camera serial:
```bash
python find_camera.py
```

## 📊 Expected Results

**Good Calibration:**
- Reprojection error: < 0.5 pixels (excellent)
- Focal length symmetry: fx ≈ fy
- Stable distortion coefficients

## 🔧 Technical Details

- **Board Type:** ChArUco (combines checkerboard + ArUco markers)
- **Calibration Method:** OpenCV with `CALIB_FIX_K3` flag for stability
- **Units:** Millimeters (consistent throughout)
- **Resolution:** 640x480 (configurable)

## 📝 Output Files

- `intrinsics_charuco_<SERIAL>.npz` - Camera calibration data
  - `mtx` - Camera matrix (3x3)
  - `dist` - Distortion coefficients (1x5)
  - `reprojection_error` - Quality metric (pixels)

## 🎯 Quality Guidelines

| Error Range | Quality | Action |
|-------------|---------|--------|
| < 0.5 px    | Excellent | ✅ Ready to use |
| 0.5-1.0 px  | Good     | ✅ Acceptable |
| 1.0-2.0 px  | Fair     | ⚠️ Consider recalibrating |
| > 2.0 px    | Poor     | ❌ Recalibrate required |

## 🛠️ Troubleshooting

**Camera not found:**
- Check USB connection
- Verify serial number with `find_camera.py`
- Close other applications using the camera

**Low corner detection:**
- Ensure good lighting
- Hold board flat and steady
- Try different angles and distances

**High reprojection error:**
- Capture more frames (20-30)
- Ensure varied poses (angles, distances)
- Check board is printed correctly (no distortion)

## 📋 Requirements

- Intel RealSense camera
- Python 3.x
- OpenCV 4.x
- pyrealsense2
- NumPy

Install dependencies:
```bash
pip install opencv-python pyrealsense2 numpy
```
