import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- 1. SCRIPT CONFIGURATION (MATCH YOUR CALIBRATION SCRIPT) ---
POINT_CLOUD_CAMERA_SN = "832112070255"
RGB_CAMERA_SN = "213622078112"      

# --- Load all configuration and calibration files ---
CONFIG_FILE = "charuco_board_config.npz"
INTRINSICS_PC_FILE = f"intrinsics_charuco_{POINT_CLOUD_CAMERA_SN}.npz"
INTRINSICS_RGB_FILE = f"intrinsics_charuco_{RGB_CAMERA_SN}.npz"
EXTRINSICS_FILE = "extrinsics.npz" # This will load the file from your new script

# Check if all required files exist
for f in [CONFIG_FILE, INTRINSICS_PC_FILE, INTRINSICS_RGB_FILE, EXTRINSICS_FILE]:
    if not os.path.exists(f):
        print(f"Error: Missing required file: '{f}'")
        exit()

# Load ChArUco board info
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

# Load intrinsics for the RGB camera (the projection target)
with np.load(INTRINSICS_RGB_FILE) as data:
    rgb_cam_mtx = data['mtx']
    rgb_cam_dist = data['dist']

# Load extrinsics (the transformation calculated by your 3D-to-3D script)
with np.load(EXTRINSICS_FILE) as data:
    rvec = data['rvec']
    tvec = data['tvec']

# --- 2. REALSENSE PIPELINES SETUP ---
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
print("A low pixel error (< 2.0) indicates a good calibration.")
print("Press 'q' to quit.")

try:
    while True:
        # Get frames from both cameras
        frames_pc = pipeline_pc.wait_for_frames()
        frames_rgb = pipeline_rgb.wait_for_frames()

        # Align depth to color for the point cloud camera
        aligned_frames = align.process(frames_pc)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame_pc = aligned_frames.get_color_frame()
        color_frame_rgb = frames_rgb.get_color_frame()

        if not depth_frame or not color_frame_pc or not color_frame_rgb:
            continue

        # Convert images to numpy arrays and grayscale
        img_pc = np.asanyarray(color_frame_pc.get_data())
        img_rgb = np.asanyarray(color_frame_rgb.get_data())
        gray_pc = cv2.cvtColor(img_pc, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Detect board in both camera views
        corners_pc, ids_pc, _, _ = detector.detectBoard(gray_pc)
        corners_rgb, ids_rgb, _, _ = detector.detectBoard(gray_rgb)

        # Create copies of images to draw on
        display_pc = img_pc.copy()
        display_rgb = img_rgb.copy()
        
        # We need common corners to calculate a meaningful error
        if ids_pc is not None and ids_rgb is not None:
            common_ids = np.intersect1d(ids_pc.flatten(), ids_rgb.flatten())
            
            if len(common_ids) > 4:
                # Lists to hold the matched points for comparison
                obj_points_3d = []
                detected_points_rgb = []
                
                # Find the corresponding 3D point (from PC camera) and 2D point (from RGB camera) for each common corner
                for id_val in common_ids:
                    idx_pc = np.where(ids_pc.flatten() == id_val)[0][0]
                    idx_rgb = np.where(ids_rgb.flatten() == id_val)[0][0]
                    
                    u, v = map(int, corners_pc[idx_pc][0])
                    depth = depth_frame.get_distance(u, v)
                    
                    if depth > 0.1: # Ensure depth is valid
                        # Convert 2D pixel + depth into a 3D point in the PC camera's coordinate system
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                        obj_points_3d.append(point_3d)
                        # Add the corresponding 2D point from the RGB camera
                        detected_points_rgb.append(corners_rgb[idx_rgb])

                # If we have valid matched points, proceed with projection and error calculation
                if obj_points_3d:
                    # Convert lists to numpy arrays in the correct format and units (mm)
                    obj_points_3d_mm = np.array(obj_points_3d, dtype=np.float32) * 1000
                    detected_points_rgb_np = np.array(detected_points_rgb, dtype=np.float32)
                    
                    # Project the 3D points from the PC camera's world onto the RGB camera's image plane
                    projected_points_rgb, _ = cv2.projectPoints(
                        obj_points_3d_mm, rvec, tvec, rgb_cam_mtx, rgb_cam_dist
                    )
                    
                    # --- Calculate and Display Quantitative Error ---
                    error = cv2.norm(detected_points_rgb_np, projected_points_rgb, cv2.NORM_L2) / len(projected_points_rgb)
                    cv2.putText(display_rgb, f"Error: {error:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Draw detected points (GREEN) and projected points (RED) for visual comparison
                    for p_det in detected_points_rgb_np:
                        cv2.circle(display_rgb, tuple(p_det[0].astype(int)), 3, (0, 255, 0), -1) # Green = Detected
                    for p_proj in projected_points_rgb:
                        cv2.circle(display_rgb, tuple(p_proj[0].astype(int)), 5, (0, 0, 255), -1) # Red = Projected

        # Draw detected corners on the source PC camera view
        if ids_pc is not None:
            cv2.aruco.drawDetectedCornersCharuco(display_pc, corners_pc, ids_pc)

        # Show the images
        cv2.imshow('PC Camera (Source)', display_pc)
        cv2.imshow('Verification on RGB Camera', display_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and close windows
    pipeline_pc.stop()
    pipeline_rgb.stop()
    cv2.destroyAllWindows()