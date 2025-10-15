import pyrealsense2 as rs
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation

# --- 1. INITIAL CONFIGURATION ---
POINT_CLOUD_CAMERA_SN = "832112070255" 
RGB_CAMERA_SN = "213622078112"      

CONFIG_FILE = "charuco_board_config.npz"
INTRINSICS_PC_FILE = f"intrinsics_charuco_{POINT_CLOUD_CAMERA_SN}.npz"
INTRINSICS_RGB_FILE = f"intrinsics_charuco_{RGB_CAMERA_SN}.npz"

for f in [CONFIG_FILE, INTRINSICS_PC_FILE, INTRINSICS_RGB_FILE]:
    if not os.path.exists(f):
        print(f"Error: Missing required file: '{f}'")
        exit()

config_data = np.load(CONFIG_FILE)
SQUARES_X = int(config_data['squares_x'])
SQUARES_Y = int(config_data['squares_y'])
SQUARE_SIZE_MM = float(config_data['square_size_mm'])
MARKER_SIZE_MM = float(config_data['marker_size_mm'])
ARUCO_DICT_ID = int(config_data['aruco_dict_id'])
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE_MM, MARKER_SIZE_MM, ARUCO_DICT)

# Load intrinsic parameters
with np.load(INTRINSICS_PC_FILE) as data:
    pc_cam_mtx = data['mtx']
    pc_cam_dist = data['dist']

with np.load(INTRINSICS_RGB_FILE) as data:
    rgb_cam_mtx = data['mtx']
    rgb_cam_dist = data['dist']

detector = cv2.aruco.CharucoDetector(board)

# --- 2. REALSENSE PIPELINES SETUP ---
pipeline_pc = rs.pipeline()
config_pc = rs.config()
config_pc.enable_device(POINT_CLOUD_CAMERA_SN)
config_pc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_pc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_rgb = rs.pipeline()
config_rgb = rs.config()
config_rgb.enable_device(RGB_CAMERA_SN)
# NOTE: This script does not require depth from the second camera, but the verification script does.
config_rgb.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile_pc = pipeline_pc.start(config_pc)
pipeline_rgb.start(config_rgb)

depth_intrinsics = profile_pc.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
align = rs.align(rs.stream.color)

# Lists to store corresponding 3D point pairs from multiple captures
all_points_3d_pc = []
all_points_3d_rgb = []

print("Position the board to be visible in both cameras.")
print("Press 'c' to capture a view. Aim for 5-10 views from different angles.")
print("Press 'q' to quit and calculate extrinsics.")

try:
    while True:
        frames_pc = pipeline_pc.wait_for_frames()
        frames_rgb = pipeline_rgb.wait_for_frames()

        aligned_frames = align.process(frames_pc)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame_pc = aligned_frames.get_color_frame()
        color_frame_rgb = frames_rgb.get_color_frame()

        if not depth_frame or not color_frame_pc or not color_frame_rgb:
            continue

        img_pc = np.asanyarray(color_frame_pc.get_data())
        img_rgb = np.asanyarray(color_frame_rgb.get_data())
        gray_pc = cv2.cvtColor(img_pc, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        charuco_corners_pc, charuco_ids_pc, _, _ = detector.detectBoard(gray_pc)
        charuco_corners_rgb, charuco_ids_rgb, _, _ = detector.detectBoard(gray_rgb)

        if charuco_ids_pc is not None: cv2.aruco.drawDetectedCornersCharuco(img_pc, charuco_corners_pc, charuco_ids_pc)
        if charuco_ids_rgb is not None: cv2.aruco.drawDetectedCornersCharuco(img_rgb, charuco_corners_rgb, charuco_ids_rgb)

        cv2.imshow('Point Cloud Camera', img_pc)
        cv2.imshow('RGB Camera', img_rgb)

        key = cv2.waitKey(1)
        if key == ord('q'): break
        
        # --- 3D-to-3D CALIBRATION LOGIC ---
        if key == ord('c') and charuco_ids_pc is not None and charuco_ids_rgb is not None:
            common_ids = np.intersect1d(charuco_ids_pc.flatten(), charuco_ids_rgb.flatten())
            if len(common_ids) < 4:
                print("Not enough common corners found. Reposition the board.")
                continue
            
            # --- Estimate 3D points for the RGB Camera (Method: SolvePnP) ---
            obj_points_ideal = []
            img_points_for_pnp = []
            for id_val in common_ids:
                idx_rgb = np.where(charuco_ids_rgb.flatten() == id_val)[0][0]
                # The ID itself is the index into the ideal board definition
                obj_points_ideal.append(board.getChessboardCorners()[id_val])
                img_points_for_pnp.append(charuco_corners_rgb[idx_rgb])

            obj_points_ideal = np.array(obj_points_ideal, dtype=np.float32)
            img_points_for_pnp = np.array(img_points_for_pnp, dtype=np.float32)
            
            success_pnp, rvec_board_to_rgb, tvec_board_to_rgb = cv2.solvePnP(
                obj_points_ideal, img_points_for_pnp, rgb_cam_mtx, rgb_cam_dist
            )
            
            if not success_pnp:
                print("PnP failed for RGB camera on this view. Try another angle.")
                continue
                
            R_board_to_rgb, _ = cv2.Rodrigues(rvec_board_to_rgb)

            # --- Now collect corresponding 3D point pairs ---
            points_3d_pc_capture = []
            points_3d_rgb_capture = []

            for id_val in common_ids:
                idx_pc = np.where(charuco_ids_pc.flatten() == id_val)[0][0]
                
                # Get measured 3D point from the PC camera using its depth sensor
                u_pc, v_pc = map(int, charuco_corners_pc[idx_pc][0])
                depth = depth_frame.get_distance(u_pc, v_pc)

                if depth > 0.1: # If depth is valid...
                    # Point A: Measured 3D point from PC camera
                    point_3d_pc = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u_pc, v_pc], depth)
                    
                    # Point B: Estimated 3D point from RGB camera
                    ideal_corner = board.getChessboardCorners()[id_val].flatten()
                    point_3d_rgb = R_board_to_rgb @ ideal_corner + tvec_board_to_rgb.flatten()
                    
                    points_3d_pc_capture.append(point_3d_pc)
                    points_3d_rgb_capture.append(point_3d_rgb)

            if len(points_3d_pc_capture) < 4:
                print("Could not get enough valid 3D point pairs. Check for glare or distance.")
                continue
            
            all_points_3d_pc.extend(points_3d_pc_capture)
            all_points_3d_rgb.extend(points_3d_rgb_capture)
            print(f"Capture successful. Total 3D point pairs: {len(all_points_3d_pc)}")

finally:
    pipeline_pc.stop()
    pipeline_rgb.stop()
    cv2.destroyAllWindows()

# --- 3D-to-3D TRANSFORMATION CALCULATION ---
if len(all_points_3d_pc) >= 3:
    print(f"\nCalculating 3D-to-3D transformation from {len(all_points_3d_pc)} point pairs...")
    
    # Convert lists to numpy arrays and ensure units are consistent (meters)
    points_pc_m = np.array(all_points_3d_pc, dtype=np.float32)
    # The estimated points were in mm, so convert to meters
    points_rgb_m = np.array(all_points_3d_rgb, dtype=np.float32) * 0.001

    # Using Kabsch algorithm (SVD method) for rigid transformation
    # This is often more stable than estimateAffine3D for this specific problem
    try:
        # Center the point clouds
        centroid_pc = np.mean(points_pc_m, axis=0)
        centroid_rgb = np.mean(points_rgb_m, axis=0)
        
        centered_pc = points_pc_m - centroid_pc
        centered_rgb = points_rgb_m - centroid_rgb
        
        # SVD for rotation estimation
        H = centered_pc.T @ centered_rgb
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = centroid_pc - R @ centroid_rgb
        
        # Convert results to OpenCV format and desired units (mm for tvec)
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1) * 1000 # Reshape and convert to mm
        R_final = R
        tvec_final = tvec

        print("\n3D-to-3D calibration successful (SVD method)!")
        print("Rotation Vector (rvec):\n", rvec)
        print("Translation Vector (tvec) in mm:\n", tvec_final)
        
        # Save results
        np.savez("extrinsics.npz", 
                rvec=rvec, 
                tvec=tvec_final,
                R=R_final)
        print("3D-to-3D extrinsics saved to extrinsics.npz")
        
    except Exception as e:
        print(f"3D-to-3D transformation estimation failed: {e}")
            
else:
    print(f"Not enough valid 3D point pairs captured ({len(all_points_3d_pc)} found). Extrinsics not calculated.")