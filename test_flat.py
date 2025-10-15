import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- INITIAL CONFIGURATION (UNCHANGED) ---
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

obj_points_ideal = board.getChessboardCorners()

with np.load(INTRINSICS_PC_FILE) as data:
    pc_cam_mtx, pc_cam_dist = data['mtx'], data['dist']
with np.load(INTRINSICS_RGB_FILE) as data:
    rgb_cam_mtx, rgb_cam_dist = data['mtx'], data['dist']

detector = cv2.aruco.CharucoDetector(board)

# --- REALSENSE PIPELINES SETUP (UNCHANGED) ---
pipeline_pc = rs.pipeline()
config_pc = rs.config()
config_pc.enable_device(POINT_CLOUD_CAMERA_SN)
config_pc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_rgb = rs.pipeline()
config_rgb = rs.config()
config_rgb.enable_device(RGB_CAMERA_SN)
config_rgb.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_pc.start(config_pc)
pipeline_rgb.start(config_rgb)

# --- ### CHANGE ###: Lists to store results from multiple captures ---
all_rvecs = []
all_tvecs = []

# --- ### CHANGE ###: Updated instructions for multi-view capture ---
print("Position the board in different orientations where it's visible to both cameras.")
print("Press 'c' to capture a view. Aim for 5-10 good views.")
print("Press 'q' to quit and calculate the final extrinsics.")

try:
    while True:
        frames_pc = pipeline_pc.wait_for_frames()
        frames_rgb = pipeline_rgb.wait_for_frames()

        color_frame_pc = frames_pc.get_color_frame()
        color_frame_rgb = frames_rgb.get_color_frame()

        if not color_frame_pc or not color_frame_rgb:
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
        if key == ord('q'):
            break

        # --- ### CHANGE ###: Collect an estimate from each 'c' press ---
        if key == ord('c') and charuco_ids_pc is not None and charuco_ids_rgb is not None:
            common_ids = np.intersect1d(charuco_ids_pc.flatten(), charuco_ids_rgb.flatten())
            if len(common_ids) < 6: # Require a decent number of corners
                print("Not enough common corners visible. Reposition the board.")
                continue

            matched_obj_points, matched_corners_pc, matched_corners_rgb = [], [], []
            for id_val in common_ids:
                idx_pc = np.where(charuco_ids_pc.flatten() == id_val)[0][0]
                idx_rgb = np.where(charuco_ids_rgb.flatten() == id_val)[0][0]
                matched_obj_points.append(obj_points_ideal[id_val])
                matched_corners_pc.append(charuco_corners_pc[idx_pc])
                matched_corners_rgb.append(charuco_corners_rgb[idx_rgb])

            matched_obj_points = np.array(matched_obj_points, dtype=np.float32)
            matched_corners_pc = np.array(matched_corners_pc, dtype=np.float32)
            matched_corners_rgb = np.array(matched_corners_rgb, dtype=np.float32)
            
            success1, rvec1, tvec1 = cv2.solvePnP(matched_obj_points, matched_corners_pc, pc_cam_mtx, pc_cam_dist, flags=cv2.SOLVEPNP_IPPE)
            success2, rvec2, tvec2 = cv2.solvePnP(matched_obj_points, matched_corners_rgb, rgb_cam_mtx, rgb_cam_dist, flags=cv2.SOLVEPNP_IPPE)

            if success1 and success2:
                R1, _ = cv2.Rodrigues(rvec1)
                R2, _ = cv2.Rodrigues(rvec2)
                
                # Calculate the transformation for this specific view
                R_capture = R2 @ R1.T
                tvec_capture = tvec2 - R_capture @ tvec1
                rvec_capture, _ = cv2.Rodrigues(R_capture)

                # Add the successful estimate to our lists
                all_rvecs.append(rvec_capture)
                all_tvecs.append(tvec_capture)
                print(f"Capture #{len(all_rvecs)} successful.")
            else:
                print("Pose estimation failed for this view. Try a different angle.")

finally:
    pipeline_pc.stop()
    pipeline_rgb.stop()
    cv2.destroyAllWindows()

# --- ### CHANGE ###: Calculate the final result by averaging all captures ---
if len(all_rvecs) > 0:
    print(f"\nCalculating final extrinsics from {len(all_rvecs)} captured views...")

    # Convert lists to numpy arrays
    all_rvecs_np = np.array(all_rvecs).reshape(-1, 3)
    all_tvecs_np = np.array(all_tvecs).reshape(-1, 3)

    # Calculate the median to get a robust average, resistant to outliers
    final_rvec = np.median(all_rvecs_np, axis=0)
    final_tvec = np.median(all_tvecs_np, axis=0)
    
    # Convert the final averaged rvec back to a rotation matrix
    final_R, _ = cv2.Rodrigues(final_rvec)

    print("\n--- Final Averaged Extrinsics ---")
    print("Rotation Vector (rvec) from PC Cam to RGB Cam:\n", final_rvec)
    print("Translation Vector (tvec) from PC Cam to RGB Cam (in mm):\n", final_tvec)
    
    np.savez("extrinsics.npz", rvec=final_rvec, tvec=final_tvec, R=final_R)
    print("\nFinal extrinsics saved to extrinsics.npz")
else:
    print("\nNo valid views were captured. Extrinsics not calculated.")