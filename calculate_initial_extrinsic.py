import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- 1. INITIAL CONFIGURATION ---
PC_CAMERA_SN = "832112070255"
RGB_CAMERA_SN = "213622078112"      

CONFIG_FILE = "charuco_board_config.npz"
INTRINSICS_PC_FILE = f"intrinsics_charuco_{PC_CAMERA_SN}.npz"
INTRINSICS_RGB_FILE = f"intrinsics_charuco_{RGB_CAMERA_SN}.npz"

# The output file that will be the input for the refinement script
OUTPUT_FILE = "data_for_refinement.npz"

for f in [CONFIG_FILE, INTRINSICS_PC_FILE, INTRINSICS_RGB_FILE]:
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

# --- 2. REALSENSE PIPELINES SETUP (BOTH WITH DEPTH) ---
pipeline_pc = rs.pipeline()
config_pc = rs.config()
config_pc.enable_device(PC_CAMERA_SN)
config_pc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_pc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_rgb = rs.pipeline()
config_rgb = rs.config()
config_rgb.enable_device(RGB_CAMERA_SN)
config_rgb.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_rgb.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile_pc = pipeline_pc.start(config_pc)
intrinsics_pc = profile_pc.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
align_pc = rs.align(rs.stream.color)

profile_rgb = pipeline_rgb.start(config_rgb)
intrinsics_rgb = profile_rgb.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
align_rgb = rs.align(rs.stream.color)

# Lists to store corresponding, *measured* 3D point pairs
all_points_3d_pc = []
all_points_3d_rgb = []

print("--- Step 1: Data Collection for Extrinsic Calibration ---")
print("Position the board to be visible in both cameras.")
print("Press 'c' to capture a view. Aim for 15-20 views from different angles.")
print("Press 'q' to quit and calculate the initial transformation.")

try:
    while True:
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
        
        if key == ord('c') and ids_pc is not None and ids_rgb is not None:
            common_ids = np.intersect1d(ids_pc.flatten(), ids_rgb.flatten())
            if len(common_ids) < 4:
                print("Not enough common corners found. Reposition the board.")
                continue

            points_3d_pc_capture, points_3d_rgb_capture = [], []
            for id_val in common_ids:
                idx_pc = np.where(ids_pc.flatten() == id_val)[0][0]
                idx_rgb = np.where(ids_rgb.flatten() == id_val)[0][0]
                
                u_pc, v_pc = map(int, corners_pc[idx_pc][0])
                depth_pc = depth_frame_pc.get_distance(u_pc, v_pc)
                
                u_rgb, v_rgb = map(int, corners_rgb[idx_rgb][0])
                depth_rgb = depth_frame_rgb.get_distance(u_rgb, v_rgb)

                if depth_pc > 0.1 and depth_rgb > 0.1:
                    points_3d_pc_capture.append(rs.rs2_deproject_pixel_to_point(intrinsics_pc, [u_pc, v_pc], depth_pc))
                    points_3d_rgb_capture.append(rs.rs2_deproject_pixel_to_point(intrinsics_rgb, [u_rgb, v_rgb], depth_rgb))
            
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

# --- 3D-to-3D TRANSFORMATION CALCULATION (INITIAL GUESS) ---
if len(all_points_3d_pc) >= 3:
    print(f"\nCalculating initial transformation from {len(all_points_3d_pc)} point pairs...")
    
    points_pc = np.array(all_points_3d_pc, dtype=np.float64)
    points_rgb = np.array(all_points_3d_rgb, dtype=np.float64)

    try:
        centroid_pc = np.mean(points_pc, axis=0)
        centroid_rgb = np.mean(points_rgb, axis=0)
        
        centered_pc = points_pc - centroid_pc
        centered_rgb = points_rgb - centroid_rgb
        
        H = centered_pc.T @ centered_rgb
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
           Vt[-1,:] *= -1
           R = Vt.T @ U.T
        
        t = centroid_rgb - R @ centroid_pc
        
        rvec, _ = cv2.Rodrigues(R)
        tvec_mm = t.reshape(3, 1) * 1000

        print("\nInitial guess calculation successful!")
        
        # Save the initial guess and the raw data for the refinement script
        np.savez(OUTPUT_FILE, 
                initial_rvec=rvec, 
                initial_tvec_mm=tvec_mm,
                initial_R=R,
                points_pc=points_pc,
                points_rgb=points_rgb)
        print(f"Initial guess and raw 3D points saved to '{OUTPUT_FILE}'")
        
    except Exception as e:
        print(f"Initial transformation estimation failed: {e}")
            
else:
    print(f"Not enough valid 3D point pairs captured ({len(all_points_3d_pc)} found).")