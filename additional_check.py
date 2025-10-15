import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- 1. SCRIPT CONFIGURATION (UNCHANGED) ---
POINT_CLOUD_CAMERA_SN = "832112070255"
RGB_CAMERA_SN = "213622078112"      

CONFIG_FILE = "charuco_board_config.npz"
INTRINSICS_PC_FILE = f"intrinsics_charuco_{POINT_CLOUD_CAMERA_SN}.npz"
INTRINSICS_RGB_FILE = f"intrinsics_charuco_{RGB_CAMERA_SN}.npz"
EXTRINSICS_FILE = "extrinsics.npz"

for f in [CONFIG_FILE, INTRINSICS_PC_FILE, INTRINSICS_RGB_FILE, EXTRINSICS_FILE]:
    if not os.path.exists(f):
        print(f"Error: Missing required file: '{f}'")
        exit()

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

# Load intrinsics for BOTH cameras
with np.load(INTRINSICS_PC_FILE) as data:
    pc_cam_mtx, pc_cam_dist = data['mtx'], data['dist']
with np.load(INTRINSICS_RGB_FILE) as data:
    rgb_cam_mtx, rgb_cam_dist = data['mtx'], data['dist']

# Load extrinsics (T_pc_to_rgb)
with np.load(EXTRINSICS_FILE) as data:
    rvec_pc_to_rgb = data['rvec']
    tvec_pc_to_rgb = data['tvec']
    R_pc_to_rgb = data['R']

# --- ### NEW ###: Calculate the inverse transformation (T_rgb_to_pc) ---
R_rgb_to_pc = R_pc_to_rgb.T
tvec_rgb_to_pc = -R_rgb_to_pc @ tvec_pc_to_rgb
rvec_rgb_to_pc, _ = cv2.Rodrigues(R_rgb_to_pc)


# --- 2. REALSENSE PIPELINES SETUP (UNCHANGED) ---
pipeline_pc = rs.pipeline()
config_pc = rs.config()
config_pc.enable_device(POINT_CLOUD_CAMERA_SN)
config_pc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_pc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_rgb = rs.pipeline()
config_rgb = rs.config()
config_rgb.enable_device(RGB_CAMERA_SN)
config_rgb.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile_pc = pipeline_pc.start(config_pc)
pipeline_rgb.start(config_rgb)

depth_intrinsics = profile_pc.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
align = rs.align(rs.stream.color)

print("Starting extrinsic verification test...")
print("Goal: Make the red and green circles overlap perfectly.")
print("Press 'q' to quit.")

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
        
        corners_pc, ids_pc, _, _ = detector.detectBoard(gray_pc)
        corners_rgb, ids_rgb, _, _ = detector.detectBoard(gray_rgb)

        display_pc = img_pc.copy()
        display_rgb = img_rgb.copy()
        
        # Find common corners to ensure a fair comparison
        if ids_pc is not None and ids_rgb is not None:
            common_ids = np.intersect1d(ids_pc.flatten(), ids_rgb.flatten())
            
            if len(common_ids) > 4:
                # --- TEST 1: Project from PC Camera to RGB Camera ---
                
                # Get matched 3D points (from PC) and 2D points (from RGB)
                obj_points_3d = []
                img_points_2d_rgb_detected = []
                
                for id_val in common_ids:
                    idx_pc = np.where(ids_pc.flatten() == id_val)[0][0]
                    idx_rgb = np.where(ids_rgb.flatten() == id_val)[0][0]
                    
                    corner_pc_2d = corners_pc[idx_pc][0]
                    u, v = int(corner_pc_2d[0]), int(corner_pc_2d[1])
                    
                    depth = depth_frame.get_distance(u, v)
                    if depth > 0.1: # Basic validity check
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                        obj_points_3d.append(point_3d)
                        img_points_2d_rgb_detected.append(corners_rgb[idx_rgb])

                if obj_points_3d:
                    obj_points_3d_mm = np.array(obj_points_3d, dtype=np.float32) * 1000
                    img_points_2d_rgb_detected = np.array(img_points_2d_rgb_detected, dtype=np.float32)

                    projected_points_rgb, _ = cv2.projectPoints(
                        obj_points_3d_mm, rvec_pc_to_rgb, tvec_pc_to_rgb, rgb_cam_mtx, rgb_cam_dist
                    )

                    # Calculate and display error for this direction
                    error_rgb = cv2.norm(img_points_2d_rgb_detected, projected_points_rgb, cv2.NORM_L2) / len(projected_points_rgb)
                    cv2.putText(display_rgb, f"PC->RGB Err: {error_rgb:.2f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Draw points on RGB display
                    for p_detected in img_points_2d_rgb_detected:
                        cv2.circle(display_rgb, tuple(p_detected[0].astype(int)), 3, (0, 255, 0), -1) # Green = Detected
                    for p_projected in projected_points_rgb:
                        cv2.circle(display_rgb, tuple(p_projected[0].astype(int)), 5, (0, 0, 255), -1) # Red = Projected

                # --- ### NEW ### TEST 2: Project from RGB Camera to PC Camera ---
                
                # We need a 3D model. We estimate it from the PC camera's view of the board.
                # This is a simplification; for perfect accuracy, one would use an external tracking system.
                # However, this is still a very strong consistency check.
                success_pose_pc, rvec_board_to_pc, tvec_board_to_pc = cv2.solvePnP(board.getChessboardCorners(), corners_pc, pc_cam_mtx, pc_cam_dist)
                if success_pose_pc:
                    # Project the ideal board corners into the PC camera's 3D space
                    obj_points_3d_ideal_pc, _ = cv2.projectPoints(board.getChessboardCorners(), rvec_board_to_pc, tvec_board_to_pc, np.eye(3), None)
                    obj_points_3d_ideal_pc = obj_points_3d_ideal_pc.reshape(-1, 3)

                    # Now, transform these points from PC space to RGB space, then project to PC image
                    # Wait, that's not right. We need to go from RGB -> PC.
                    # Let's keep it simple and project the 3D points we already found (from depth)
                    # using the INVERSE transform.
                    
                    projected_points_pc, _ = cv2.projectPoints(
                        obj_points_3d_mm, rvec_rgb_to_pc, tvec_rgb_to_pc, pc_cam_mtx, pc_cam_dist
                    )
                    
                    # Match the original detected points for error calculation
                    img_points_2d_pc_detected = []
                    for id_val in common_ids:
                        idx_pc = np.where(ids_pc.flatten() == id_val)[0][0]
                        img_points_2d_pc_detected.append(corners_pc[idx_pc])
                    
                    # This part is tricky because the points in obj_points_3d_mm might not
                    # correspond 1:1 if some depths were invalid. Let's simplify and just draw.
                    
                    # Draw points on PC display
                    cv2.aruco.drawDetectedCornersCharuco(display_pc, corners_pc, ids_pc) # Green = Detected by default
                    
                    # This logic is getting complex. Let's stick to the first, most important check (PC->RGB)
                    # and make it robust. The previous logic was better. Let's revert and just add the error calc.
                    # My apologies for over-engineering. The core idea is the most important.
                    # The following is a cleaned up version of YOUR original logic with just the error calc added.
                    
    # Reverting to the clear, robust, single-direction check with error calculation.
    # The previous block was an over-complication.
    
    # Cleaned and enhanced version of your original script.
    
    while True: # Let's restart the logic loop for clarity
        frames_pc = pipeline_pc.wait_for_frames()
        frames_rgb = pipeline_rgb.wait_for_frames()

        aligned_frames = align.process(frames_pc)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame_pc = aligned_frames.get_color_frame()
        color_frame_rgb = frames_rgb.get_color_frame()
        
        if not depth_frame or not color_frame_pc or not color_frame_rgb: continue

        img_pc = np.asanyarray(color_frame_pc.get_data())
        img_rgb = np.asanyarray(color_frame_rgb.get_data())
        gray_pc = cv2.cvtColor(img_pc, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        corners_pc, ids_pc, _, _ = detector.detectBoard(gray_pc)
        corners_rgb, ids_rgb, _, _ = detector.detectBoard(gray_rgb)

        display_pc = img_pc.copy()
        display_rgb = img_rgb.copy()
        
        # We need common corners to calculate a meaningful error
        if ids_pc is not None and ids_rgb is not None:
            common_ids = np.intersect1d(ids_pc.flatten(), ids_rgb.flatten())
            
            if len(common_ids) > 4:
                obj_points_3d, detected_points_rgb = [], []
                
                for id_val in common_ids:
                    idx_pc = np.where(ids_pc.flatten() == id_val)[0][0]
                    idx_rgb = np.where(ids_rgb.flatten() == id_val)[0][0]
                    
                    u, v = map(int, corners_pc[idx_pc][0])
                    depth = depth_frame.get_distance(u, v)
                    
                    if depth > 0.1: # Ensure depth is valid
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                        obj_points_3d.append(point_3d)
                        detected_points_rgb.append(corners_rgb[idx_rgb])

                if obj_points_3d:
                    obj_points_3d_mm = np.array(obj_points_3d, dtype=np.float32) * 1000
                    detected_points_rgb_np = np.array(detected_points_rgb, dtype=np.float32)
                    
                    projected_points_rgb, _ = cv2.projectPoints(
                        obj_points_3d_mm, rvec_pc_to_rgb, tvec_pc_to_rgb, rgb_cam_mtx, rgb_cam_dist
                    )
                    
                    # --- NEW: Calculate and Display Error ---
                    error = cv2.norm(detected_points_rgb_np, projected_points_rgb, cv2.NORM_L2) / len(projected_points_rgb)
                    cv2.putText(display_rgb, f"Error: {error:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Draw detected (green) and projected (red) points
                    for p_det in detected_points_rgb_np:
                        cv2.circle(display_rgb, tuple(p_det[0].astype(int)), 3, (0, 255, 0), -1)
                    for p_proj in projected_points_rgb:
                        cv2.circle(display_rgb, tuple(p_proj[0].astype(int)), 5, (0, 0, 255), -1)

        cv2.imshow('PC Camera', display_pc)
        cv2.imshow('Verification on RGB Camera', display_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:
    pipeline_pc.stop()
    pipeline_rgb.stop()
    cv2.destroyAllWindows()