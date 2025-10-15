import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- 1. SCRIPT CONFIGURATION ---
TARGET_SERIAL_NUMBER = "213622078112"  # Change this to your camera's serial number!
CONFIG_FILE = "charuco_board_config.npz"

# --- 2. LOAD CHARUCO BOARD CONFIGURATION ---
if not os.path.exists(CONFIG_FILE):
    print(f"Error: Configuration file '{CONFIG_FILE}' not found!")
    print("Please run generate_charuco.py first to create the board configuration.")
    import sys
    sys.exit(1)

config_data = np.load(CONFIG_FILE)
SQUARES_X = int(config_data['squares_x'])
SQUARES_Y = int(config_data['squares_y'])
SQUARE_SIZE_MM = float(config_data['square_size_mm'])
MARKER_SIZE_MM = float(config_data['marker_size_mm'])
ARUCO_DICT_ID = int(config_data['aruco_dict_id'])
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE_MM, MARKER_SIZE_MM, ARUCO_DICT)

print("--- ChArUco Board Configuration Loaded ---")
print(f"Board Dimensions: {SQUARES_X}x{SQUARES_Y}")
print(f"Square Size: {SQUARE_SIZE_MM} mm")
print(f"Marker Size: {MARKER_SIZE_MM} mm")
print("------------------------------------------")

# --- 3. CREATE CHARUCO DETECTOR ---
detector = cv2.aruco.CharucoDetector(board)

# --- 4. REALSENSE CAMERA SETUP ---
pipeline = rs.pipeline()
config = rs.config()

try:
    config.enable_device(TARGET_SERIAL_NUMBER)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    print(f"Camera {TARGET_SERIAL_NUMBER} initialized successfully")
except Exception as e:
    print(f"Error initializing camera {TARGET_SERIAL_NUMBER}: {e}")
    print("Please check:")
    print("1. Camera is connected")
    print("2. Serial number is correct")
    print("3. No other applications are using the camera")
    import sys
    sys.exit(1)

# Set camera parameters for consistent capture conditions
color_sensor = profile.get_device().first_color_sensor()
if color_sensor.supports(rs.option.enable_auto_exposure):
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    print("Auto Exposure Disabled")
if color_sensor.supports(rs.option.enable_auto_white_balance):
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
    print("Auto White Balance Disabled")

manual_exposure_value = 150
if color_sensor.supports(rs.option.exposure):
    color_sensor.set_option(rs.option.exposure, manual_exposure_value)
    print(f"Manual exposure set to: {manual_exposure_value}")

# --- 5. CALIBRATION DATA COLLECTION ---
all_charuco_corners = []
all_charuco_ids = []

print("\n=== CALIBRATION DATA COLLECTION ===")
print("Position the ChArUco board at different angles and distances")
print("Press 'c' to capture a frame when the board is clearly visible")
print("Press 'q' to finish collection and start calibration")
print("Recommendation: Capture 15-30 high-quality frames")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_copy = img.copy()

        # Detect ChArUco corners
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        # Draw detected corners
        if charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(img_copy, charuco_corners, charuco_ids)
            
            # Add corner count and quality indicator
            corner_count = len(charuco_ids)
            color = (0, 255, 0) if corner_count >= 10 else (0, 255, 255)
            cv2.putText(img_copy, f"Corners: {corner_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display capture count and instructions
        cv2.putText(img_copy, f"Captured: {len(all_charuco_corners)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_copy, "Press 'c' to capture, 'q' to calibrate", (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Camera Calibration', img_copy)
        key = cv2.waitKey(1)

        if key == ord('q'):
            if len(all_charuco_corners) < 10:
                print(f"Warning: Only {len(all_charuco_corners)} frames captured.")
                print("At least 10-15 frames are recommended for good calibration.")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    continue
            break
        
        if key == ord('c'):
            if charuco_ids is not None and len(charuco_ids) >= 6:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                print(f"Frame captured! Total: {len(all_charuco_corners)} (corners: {len(charuco_ids)})")
            else:
                print("Capture failed: Need at least 6 ChArUco corners detected")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

if len(all_charuco_corners) == 0:
    print("No frames were captured. Exiting.")
    sys.exit(1)

# --- 6. PERFORM CAMERA CALIBRATION ---
print(f"\n=== CAMERA CALIBRATION ===")
print(f"Processing {len(all_charuco_corners)} captured frames...")

h, w = gray.shape

# Prepare object points and image points for calibration
objpoints = []
imgpoints = []

all_board_corners = board.getChessboardCorners()

for i, ids in enumerate(all_charuco_ids):
    # Get 3D object points for detected corners (in millimeters)
    objp_for_frame = np.array([all_board_corners[id[0]] for id in ids], dtype=np.float32)
    objpoints.append(objp_for_frame)
    
    # Get 2D image points (reshape if needed)
    corners_2d = all_charuco_corners[i]
    if corners_2d.ndim == 3 and corners_2d.shape[1] == 1:
        corners_2d = corners_2d.reshape(-1, 2)
    imgpoints.append(corners_2d.astype(np.float32))

# CRITICAL: Use CALIB_FIX_K3 flag for numerical stability
# This prevents overfitting and ensures the calibration is robust
flags = cv2.CALIB_FIX_K3

print("Calibrating camera (k3 fixed to 0 for stability)...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    (w, h),
    None,
    None,
    flags=flags)


if not ret:
    print("❌ Calibration failed!")
    sys.exit(1)

# --- 7. CALCULATE REPROJECTION ERROR ---
print("Calculating reprojection error...")
total_error = 0
total_points = 0
frame_errors = []

for i in range(len(objpoints)):
    # Project 3D points back to 2D using calibration
    projected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    projected_points = projected_points.reshape(-1, 2)
    
    # Calculate error for this frame
    error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
    frame_errors.append(error)
    total_error += error * len(projected_points)
    total_points += len(projected_points)

mean_error = total_error / total_points

# --- 8. SAVE CALIBRATION RESULTS ---
filename = f"intrinsics_charuco_{TARGET_SERIAL_NUMBER}.npz"
np.savez(filename, 
         mtx=mtx, 
         dist=dist, 
         reprojection_error=mean_error,
         num_images=len(all_charuco_corners),
         image_size=(w, h),
         calibration_flags="CALIB_FIX_K3")

# --- 9. DISPLAY RESULTS ---
print("\n" + "="*50)
print("🎉 CALIBRATION COMPLETED SUCCESSFULLY!")
print("="*50)

print(f"\nCalibration Quality Assessment:")
print(f"📊 Mean reprojection error: {mean_error:.3f} pixels")

if mean_error < 0.5:
    print("✅ EXCELLENT calibration quality!")
elif mean_error < 1.0:
    print("✅ GOOD calibration quality")
elif mean_error < 2.0:
    print("⚠️  ACCEPTABLE calibration quality")
else:
    print("❌ POOR calibration quality - consider recalibrating")

print(f"\n📈 Error Statistics:")
print(f"   Min error: {min(frame_errors):.3f} pixels")
print(f"   Max error: {max(frame_errors):.3f} pixels")
print(f"   Std dev:   {np.std(frame_errors):.3f} pixels")

print(f"\n📷 Camera Parameters:")
fx, fy = mtx[0, 0], mtx[1, 1]
cx, cy = mtx[0, 2], mtx[1, 2]
print(f"   Focal length (fx, fy): ({fx:.2f}, {fy:.2f}) pixels")
print(f"   Principal point (cx, cy): ({cx:.2f}, {cy:.2f}) pixels")
print(f"   Focal length symmetry: {fx/fy:.4f} (ideal: 1.0000)")

print(f"\n🔧 Distortion Coefficients:")
k1, k2, p1, p2, k3 = dist[0]
print(f"   k1 (radial): {k1:.6f}")
print(f"   k2 (radial): {k2:.6f}")
print(f"   p1 (tangential): {p1:.6f}")
print(f"   p2 (tangential): {p2:.6f}")
print(f"   k3 (radial): {k3:.6f} (fixed)")

print(f"\n💾 Results saved to: {filename}")
print(f"   - Camera matrix (mtx)")
print(f"   - Distortion coefficients (dist)")
print(f"   - Reprojection error: {mean_error:.3f} px")
print(f"   - Images used: {len(all_charuco_corners)}")
print(f"   - Calibration flags: CALIB_FIX_K3")

print(f"\n🔍 Next Steps:")
print(f"1. Visually inspect undistorted images for straight lines")
print(f"2. Use this calibration for stereo calibration or 3D applications")
print(f"3. The k3=0 constraint ensures numerical stability")

print("\n" + "="*50)
print("Calibration complete! 🎯")
print("="*50)
