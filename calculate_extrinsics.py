#!/usr/bin/env python3
"""
RealSense Dual-Camera Extrinsic Calibration
===========================================

This script performs 3D-to-3D extrinsic calibration between two RealSense cameras
using ChArUco board detection and the Kabsch algorithm for rigid body transformation.

Author: Multi-Camera Calibration System
Date: 2025
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- CONFIGURATION ---
PC_CAMERA_SN = "832112070255"  # Primary camera serial number
RGB_CAMERA_SN = "213622078112"  # Secondary camera serial number
CONFIG_FILE = "charuco_board_config.npz"
OUTPUT_FILE = "extrinsics.npz"

# Validate required files
if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: Missing required file: '{CONFIG_FILE}'")
    print("Please generate ChArUco board configuration first.")
    exit(1)

# Load ChArUco board configuration
config_data = np.load(CONFIG_FILE)
ARUCO_DICT_ID = int(config_data['aruco_dict_id'])
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
board = cv2.aruco.CharucoBoard(
    (int(config_data['squares_x']), int(config_data['squares_y'])),
    float(config_data['square_size_mm']),
    float(config_data['marker_size_mm']),
    ARUCO_DICT
)
detector = cv2.aruco.CharucoDetector(board)

print("🎯 RealSense Dual-Camera Extrinsic Calibration")
print("=" * 50)
print(f"Primary Camera: {PC_CAMERA_SN}")
print(f"Secondary Camera: {RGB_CAMERA_SN}")
print(f"ChArUco Board: {config_data['squares_x']}x{config_data['squares_y']}")

# --- REALSENSE SETUP ---
def setup_cameras():
    """Initialize and configure RealSense cameras."""
    print("\n🔧 Initializing cameras...")
    
    # Setup primary camera
    pipeline_pc = rs.pipeline()
    config_pc = rs.config()
    config_pc.enable_device(PC_CAMERA_SN)
    config_pc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_pc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Setup secondary camera
    pipeline_rgb = rs.pipeline()
    config_rgb = rs.config()
    config_rgb.enable_device(RGB_CAMERA_SN)
    config_rgb.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_rgb.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile_pc = pipeline_pc.start(config_pc)
    profile_rgb = pipeline_rgb.start(config_rgb)
    
    # Get intrinsics
    color_intrinsics_pc = profile_pc.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    color_intrinsics_rgb = profile_rgb.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    # Setup alignment
    align_pc = rs.align(rs.stream.color)
    align_rgb = rs.align(rs.stream.color)
    
    return (pipeline_pc, pipeline_rgb, color_intrinsics_pc, color_intrinsics_rgb, 
            align_pc, align_rgb)

# Initialize cameras
(pipeline_pc, pipeline_rgb, color_intrinsics_pc, color_intrinsics_rgb, 
 align_pc, align_rgb) = setup_cameras()

print("✅ Cameras initialized successfully")
print(f"Primary Camera Color - fx:{color_intrinsics_pc.fx:.2f}, fy:{color_intrinsics_pc.fy:.2f}")
print(f"Secondary Camera Color - fx:{color_intrinsics_rgb.fx:.2f}, fy:{color_intrinsics_rgb.fy:.2f}")
print("📌 Using COLOR intrinsics for consistent 3D coordinate system")


# --- DATA COLLECTION ---
def collect_calibration_data():
    """Collect 3D point pairs from both cameras for calibration."""
    all_points_3d_pc = []
    all_points_3d_rgb = []
    
    print("\n📸 DATA COLLECTION PHASE")
    print("=" * 30)
    print("Instructions:")
    print("• Position ChArUco board visible to both cameras")
    print("• Press 'c' to capture a view (aim for 8-15 different positions)")
    print("• Move board to different angles and distances")
    print("• Press 'q' when done collecting data")
    print("\nStarting live preview...")

try:
    while True:
        # Get and align frames for BOTH cameras
        frames_pc = pipeline_pc.wait_for_frames()
        frames_rgb = pipeline_rgb.wait_for_frames()

        aligned_frames_pc = align_pc.process(frames_pc)
        depth_frame_pc = aligned_frames_pc.get_depth_frame()
        color_frame_pc = aligned_frames_pc.get_color_frame()
        
        aligned_frames_rgb = align_rgb.process(frames_rgb)
        depth_frame_rgb = aligned_frames_rgb.get_depth_frame()
        color_frame_rgb = aligned_frames_rgb.get_color_frame()

        if not all([depth_frame_pc, color_frame_pc, depth_frame_rgb, color_frame_rgb]):
            continue

        img_pc = np.asanyarray(color_frame_pc.get_data())
        img_rgb = np.asanyarray(color_frame_rgb.get_data())
        gray_pc = cv2.cvtColor(img_pc, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        corners_pc, ids_pc, _, _ = detector.detectBoard(gray_pc)
        corners_rgb, ids_rgb, _, _ = detector.detectBoard(gray_rgb)

        if ids_pc is not None: cv2.aruco.drawDetectedCornersCharuco(img_pc, corners_pc, ids_pc)
        if ids_rgb is not None: cv2.aruco.drawDetectedCornersCharuco(img_rgb, corners_rgb, ids_rgb)

        cv2.imshow('PC Camera', img_pc)
        cv2.imshow('RGB Camera', img_rgb)

        key = cv2.waitKey(1)
        if key == ord('q'): break
        
        # --- 3D-to-3D CALIBRATION LOGIC (3D from PC camera -> 3D from RGB camera) ---
        if key == ord('c') and ids_pc is not None and ids_rgb is not None:
            common_ids = np.intersect1d(ids_pc.flatten(), ids_rgb.flatten())
            if len(common_ids) < 4:
                print("Not enough common corners found. Reposition the board.")
                continue

            points_3d_pc_capture = []
            points_3d_rgb_capture = []

            for id_val in common_ids:
                idx_pc = np.where(ids_pc.flatten() == id_val)[0][0]
                idx_rgb = np.where(ids_rgb.flatten() == id_val)[0][0]
                
                # Get 3D point from PC camera
                corner_pc = corners_pc[idx_pc][0]
                u_pc, v_pc = int(corner_pc[0]), int(corner_pc[1])
                depth_pc = depth_frame_pc.get_distance(u_pc, v_pc)
                
                # Get 3D point from RGB camera
                corner_rgb = corners_rgb[idx_rgb][0]
                u_rgb, v_rgb = int(corner_rgb[0]), int(corner_rgb[1])
                depth_rgb = depth_frame_rgb.get_distance(u_rgb, v_rgb)

                # Only use points where BOTH cameras have valid depth
                if depth_pc > 0.1 and depth_rgb > 0.1:
                    # CRITICAL FIX: Since we're using color-aligned frames, we need to use
                    # the color intrinsics to get correct 3D points in the aligned coordinate system
                    point_3d_pc = rs.rs2_deproject_pixel_to_point(color_intrinsics_pc, [u_pc, v_pc], depth_pc)
                    point_3d_rgb = rs.rs2_deproject_pixel_to_point(color_intrinsics_rgb, [u_rgb, v_rgb], depth_rgb)
                    
                    points_3d_pc_capture.append(point_3d_pc)
                    points_3d_rgb_capture.append(point_3d_rgb)
            
            if len(points_3d_pc_capture) < 3:
                print("Could not get enough valid 3D-3D point pairs. Check for glare or distance.")
                continue

            all_points_3d_pc.extend(points_3d_pc_capture)
            all_points_3d_rgb.extend(points_3d_rgb_capture)
            print(f"Capture successful. Total 3D-3D pairs: {len(all_points_3d_pc)}")

finally:
    pipeline_pc.stop()
    pipeline_rgb.stop()
    cv2.destroyAllWindows()

# Helper functions for converting RealSense intrinsics to OpenCV format
def rs_intrinsics_to_cv_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                     [0, intrinsics.fy, intrinsics.ppy],
                     [0, 0, 1]], dtype=np.float32)

def rs_intrinsics_to_cv_dist(intrinsics):
    return np.array([intrinsics.coeffs[0], intrinsics.coeffs[1], 
                     intrinsics.coeffs[2], intrinsics.coeffs[3], 
                     intrinsics.coeffs[4]], dtype=np.float32)

# Kabsch algorithm for 3D-to-3D rigid body transformation
def kabsch_algorithm(P, Q):
    """
    Calculate the optimal rotation matrix and translation vector to align point clouds P and Q.
    P: source points (Nx3)
    Q: target points (Nx3)  
    Returns: R (3x3 rotation matrix), t (3x1 translation vector)
    """
    # Center the point clouds
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute the cross-covariance matrix
    H = P_centered.T @ Q_centered
    
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_Q - R @ centroid_P
    
    return R, t

# --- 3D-to-3D CALIBRATION (Kabsch Algorithm) ---
if len(all_points_3d_pc) >= 3:
    print(f"\nCalculating extrinsics using 3D-to-3D calibration from {len(all_points_3d_pc)} point pairs...")
    
    # Convert lists to numpy arrays
    points_3d_pc = np.array(all_points_3d_pc, dtype=np.float32)
    points_3d_rgb = np.array(all_points_3d_rgb, dtype=np.float32)
    
    print(f"PC 3D points shape: {points_3d_pc.shape}")
    print(f"RGB 3D points shape: {points_3d_rgb.shape}")

    try:
        # Apply Kabsch algorithm to find transformation from PC to RGB coordinate system
        R, t = kabsch_algorithm(points_3d_pc, points_3d_rgb)

        print("\n3D-to-3D calibration successful!")
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (t) in meters:\n", t)
        
        # Convert to OpenCV format - KEEP CONSISTENT UNITS
        rvec, _ = cv2.Rodrigues(R)
        # Store both formats for compatibility
        tvec_mm = t.reshape(-1, 1) * 1000  # For OpenCV functions that expect mm
        tvec_m = t.reshape(-1, 1)          # For RealSense functions that work in meters
        
        print("Rotation Vector (rvec):\n", rvec)
        print("Translation Vector (tvec) in mm:\n", tvec_mm)
        print("Translation Vector (t) in meters:\n", tvec_m)
        
        # Verify the calibration by transforming PC points and comparing with RGB points
        points_3d_transformed = (R @ points_3d_pc.T + t.reshape(-1, 1)).T
        
        # Calculate 3D transformation error
        errors_3d = np.linalg.norm(points_3d_rgb - points_3d_transformed, axis=1)
        mean_error_3d = np.mean(errors_3d) * 1000  # Convert to mm
        max_error_3d = np.max(errors_3d) * 1000
        
        print(f"\n3D Transformation Error Analysis:")
        print(f"Mean error: {mean_error_3d:.2f} mm")
        print(f"Max error: {max_error_3d:.2f} mm")
        print(f"RMS error: {np.sqrt(np.mean(errors_3d**2)) * 1000:.2f} mm")
        
        if mean_error_3d < 5.0:
            print("✅ EXCELLENT calibration quality!")
        elif mean_error_3d < 15.0:
            print("✅ GOOD calibration quality")
        elif mean_error_3d < 30.0:
            print("✅ ACCEPTABLE calibration quality")
        else:
            print("⚠️ Consider recalibrating - high transformation error")
        
        # Save results with proper unit documentation
        np.savez("extrinsics.npz", 
                rvec=rvec, 
                tvec=tvec_mm,      # Translation in millimeters (OpenCV standard)
                R=R,
                t=tvec_m.flatten(), # Translation in meters (RealSense standard)
                mean_3d_error=mean_error_3d,
                max_3d_error=max_error_3d,
                calibration_method="3D_to_3D_Kabsch")
        print("3D-to-3D extrinsics saved to extrinsics.npz")
        
    except Exception as e:
        print(f"3D-to-3D calibration failed: {e}")
            
else:
    print(f"Not enough valid 3D-3D point pairs captured ({len(all_points_3d_pc)} found, need at least 3). Extrinsics not calculated.")