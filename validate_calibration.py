import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- CONFIGURATION ---
TARGET_SERIAL_NUMBER = "832112070255"
CALIBRATION_FILE = f"intrinsics_charuco_{TARGET_SERIAL_NUMBER}.npz"
CONFIG_FILE = "charuco_board_config.npz"

def load_calibration_data():
    """Load camera calibration data"""
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Error: Calibration file '{CALIBRATION_FILE}' not found!")
        print("Please run take_camera_intrinsic.py first.")
        return None, None
    
    calib_data = np.load(CALIBRATION_FILE)
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    
    print("--- Calibration Data Loaded ---")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients: {dist}")
    if 'reprojection_error' in calib_data:
        print(f"Original Reprojection Error: {calib_data['reprojection_error']:.3f} pixels")
    print("--------------------------------")
    
    return mtx, dist

def capture_test_image():
    """Capture a test image from the camera"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        config.enable_device(TARGET_SERIAL_NUMBER)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        
        print("Camera initialized. Capturing test image...")
        
        # Warm up the camera
        for _ in range(30):
            pipeline.wait_for_frames()
        
        # Capture the image
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if color_frame:
            img = np.asanyarray(color_frame.get_data())
            cv2.imwrite("test_image_original.jpg", img)
            print("Test image saved as 'test_image_original.jpg'")
            return img
        else:
            print("Failed to capture test image")
            return None
            
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None
    finally:
        pipeline.stop()

def test_undistortion(img, mtx, dist):
    """Test 1: Undistort image and check for straight lines"""
    print("\n=== TEST 1: UNDISTORTION TEST ===")
    
    h, w = img.shape[:2]
    
    # Get optimal new camera matrix
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # Undistort the image
    undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)
    
    # Crop the image based on ROI
    x, y, w_roi, h_roi = roi
    undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]
    
    # Save results
    cv2.imwrite("test_image_undistorted.jpg", undistorted)
    cv2.imwrite("test_image_undistorted_cropped.jpg", undistorted_cropped)
    
    print("✓ Undistorted images saved:")
    print("  - test_image_undistorted.jpg (full)")
    print("  - test_image_undistorted_cropped.jpg (cropped to ROI)")
    print("✓ MANUAL CHECK REQUIRED:")
    print("  - Look for straight lines (building edges, table edges, etc.)")
    print("  - Straight lines should remain straight after undistortion")
    print("  - If lines are wavy or curved, calibration may be poor")
    
    return undistorted, new_mtx

def test_reprojection_error_analysis():
    """Test 2: Detailed reprojection error analysis"""
    print("\n=== TEST 2: REPROJECTION ERROR ANALYSIS ===")
    
    # We'll need to recapture some ChArUco data for this test
    print("This test requires ChArUco board detection...")
    print("Please show the ChArUco board to the camera and press 'c' to capture frames for analysis")
    print("Press 'q' when you have captured 5-10 frames")
    
    # Load board configuration
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found!")
        return
    
    config_data = np.load(CONFIG_FILE)
    SQUARES_X = int(config_data['squares_x'])
    SQUARES_Y = int(config_data['squares_y'])
    SQUARE_SIZE_MM = float(config_data['square_size_mm'])
    MARKER_SIZE_MM = float(config_data['marker_size_mm'])
    ARUCO_DICT_ID = int(config_data['aruco_dict_id'])
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE_MM, MARKER_SIZE_MM, ARUCO_DICT)
    detector = cv2.aruco.CharucoDetector(board)
    
    # Initialize camera
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        config.enable_device(TARGET_SERIAL_NUMBER)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        
        all_charuco_corners = []
        all_charuco_ids = []
        
        print("Capturing frames for reprojection error analysis...")
        
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_copy = img.copy()

            charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

            if charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(img_copy, charuco_corners, charuco_ids)
            
            cv2.putText(img_copy, f"Captures: {len(all_charuco_corners)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_copy, "Press 'c' to capture, 'q' to analyze", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Reprojection Test', img_copy)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            
            if key == ord('c'):
                if charuco_ids is not None and len(charuco_ids) > 5:  # Require at least 5 corners
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    print(f"Frame captured! Total captures: {len(all_charuco_corners)}")
                else:
                    print("Capture failed: Need at least 5 ChArUco corners detected.")
                    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
    if len(all_charuco_corners) == 0:
        print("No frames captured for analysis")
        return
    
    # Load calibration data
    mtx, dist = load_calibration_data()
    if mtx is None:
        return
    
    # Test current calibration
    print(f"\n--- Current Calibration Performance ---")
    errors = []
    total_error = 0
    total_points = 0
    
    for i in range(len(all_charuco_corners)):
            # Get 3D points for this frame
        ids = all_charuco_ids[i].flatten()
        obj_points = board.getChessboardCorners()[ids]
        img_points = all_charuco_corners[i].reshape(-1, 2)
        
        # CRITICAL FIX: Use proper PnP to get rvec, tvec instead of zeros
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
        
        if success:
            # Project points using solved pose
            projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, mtx, dist)
            projected_points = projected_points.reshape(-1, 2)
            
            # Calculate error for this frame
            frame_error = cv2.norm(img_points, projected_points, cv2.NORM_L2) / len(projected_points)
            errors.append(frame_error)
            total_error += frame_error * len(projected_points)
            total_points += len(projected_points)
        else:
            # Skip this frame if PnP fails
            print(f"Frame {i+1}: PnP failed")
            continue
    
    if total_points > 0:
        mean_error = total_error / total_points
    else:
        print("❌ No successful PnP solutions found!")
        return
    
    print(f"Mean reprojection error: {mean_error:.3f} pixels")
    print(f"Individual frame errors: {[f'{e:.3f}' for e in errors]}")
    
    # Evaluate quality
    if mean_error < 0.5:
        print("✓ EXCELLENT calibration (< 0.5 px)")
    elif mean_error < 1.0:
        print("✓ ACCEPTABLE calibration (0.5-1.0 px)")
    else:
        print("⚠ PROBLEMATIC calibration (> 1.0 px) - Consider recalibrating")

def test_k3_sensitivity():
    """Test 3: Test sensitivity to k3 parameter"""
    print("\n=== TEST 3: K3 SENSITIVITY TEST ===")
    
    # This would require recalibration, so we'll provide instructions
    print("To test k3 sensitivity:")
    print("1. Modify the calibration flags in take_camera_intrinsic.py")
    print("2. Add this line before cv2.calibrateCamera():")
    print("   flags = cv2.CALIB_FIX_K3")
    print("3. Change the calibrateCamera call to:")
    print("   ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(")
    print("       objpoints, imgpoints, (w, h), None, None, flags=flags)")
    print("4. Compare reprojection errors with and without k3 fixed")
    print("5. If errors are similar, k3 might be overfitting")

def visualize_corners_on_undistorted():
    """Test 4: Visualize corners on undistorted images"""
    print("\n=== TEST 4: CORNER VISUALIZATION ON UNDISTORTED IMAGE ===")
    
    mtx, dist = load_calibration_data()
    if mtx is None:
        return
    
    print("Show ChArUco board to camera and press 'c' to test corner visualization")
    print("Press 'q' to finish")
    
    # Load board configuration
    config_data = np.load(CONFIG_FILE)
    SQUARES_X = int(config_data['squares_x'])
    SQUARES_Y = int(config_data['squares_y'])
    SQUARE_SIZE_MM = float(config_data['square_size_mm'])
    MARKER_SIZE_MM = float(config_data['marker_size_mm'])
    ARUCO_DICT_ID = int(config_data['aruco_dict_id'])
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE_MM, MARKER_SIZE_MM, ARUCO_DICT)
    detector = cv2.aruco.CharucoDetector(board)
    
    # Initialize camera
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        config.enable_device(TARGET_SERIAL_NUMBER)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Undistort the image
            h, w = img.shape[:2]
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)
            undistorted_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            
            # Detect corners on undistorted image
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(undistorted_gray)
            
            # Draw corners
            img_display = undistorted.copy()
            if charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(img_display, charuco_corners, charuco_ids)
                
                # Check grid alignment
                if len(charuco_corners) > 4:
                    corners_2d = charuco_corners.reshape(-1, 2)
                    
                    # Draw grid lines to visualize alignment
                    for i in range(len(corners_2d)):
                        cv2.circle(img_display, tuple(corners_2d[i].astype(int)), 3, (0, 255, 0), -1)
            
            cv2.putText(img_display, "Undistorted image with corners", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_display, "Press 'c' to save, 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Corner Visualization Test', img_display)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            
            if key == ord('c'):
                cv2.imwrite("corner_visualization_undistorted.jpg", img_display)
                print("✓ Corner visualization saved as 'corner_visualization_undistorted.jpg'")
                print("✓ Check that corners form a clean, regular grid pattern")
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def main():
    print("=== CAMERA CALIBRATION VALIDATION SUITE ===")
    print("This script will run comprehensive tests on your camera calibration")
    print("=" * 50)
    
    # Load calibration data
    mtx, dist = load_calibration_data()
    if mtx is None:
        return
    
    # Test 1: Capture and undistort test image
    test_img = capture_test_image()
    if test_img is not None:
        test_undistortion(test_img, mtx, dist)
    
    # Test 2: Reprojection error analysis
    test_reprojection_error_analysis()
    
    # Test 3: K3 sensitivity (instructions only)
    test_k3_sensitivity()
    
    # Test 4: Corner visualization
    visualize_corners_on_undistorted()
    
    print("\n=== VALIDATION SUMMARY ===")
    print("Check the generated images and console output:")
    print("1. test_image_undistorted.jpg - straight lines should be straight")
    print("2. Reprojection error should be < 1.0 pixels (preferably < 0.5)")
    print("3. corner_visualization_undistorted.jpg - corners should form clean grid")
    print("4. Consider k3 sensitivity test if reprojection error is high")

if __name__ == "__main__":
    main()
