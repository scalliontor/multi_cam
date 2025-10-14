import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- 1. SCRIPT CONFIGURATION (MATCH THE EXTRINSIC SCRIPT) ---
POINT_CLOUD_CAMERA_SN = "832112070255"
RGB_CAMERA_SN = "213622078112"      

# --- Load all configuration and calibration files ---
CONFIG_FILE = "charuco_board_config.npz"
INTRINSICS_PC_FILE = f"intrinsics_charuco_{POINT_CLOUD_CAMERA_SN}.npz"
INTRINSICS_RGB_FILE = f"intrinsics_charuco_{RGB_CAMERA_SN}.npz"
EXTRINSICS_FILE = "extrinsics.npz"

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

# Load extrinsics (the transformation)
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
print(" - Green Circles = Corners detected directly.")
print(" - Red Circles   = Corners projected from the other camera.")
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
        
        # --- MODIFICATION START ---
        # Convert both images to grayscale for detection
        gray_pc = cv2.cvtColor(img_pc, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) # Added this line
        
        # Detect board in both camera views
        corners_pc, ids_pc, _, _ = detector.detectBoard(gray_pc)
        corners_rgb, ids_rgb, _, _ = detector.detectBoard(gray_rgb) # Added this line

        # Create copies to draw on, so the original images are clean
        display_pc = img_pc.copy()
        display_rgb = img_rgb.copy()
        
        # --- END OF MODIFICATION ---

        # If the board is detected in the Point Cloud camera, project its corners
        if ids_pc is not None:
            cv2.aruco.drawDetectedCornersCharuco(display_pc, corners_pc, ids_pc) # Draw on the display copy
            
            # Get 3D coordinates from depth
            obj_points_3d = []
            for corner in corners_pc:
                u, v = int(corner[0][0]), int(corner[0][1])
                depth = depth_frame.get_distance(u, v)
                if depth > 0.1: # Basic validity check
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                    obj_points_3d.append(point_3d)

            if obj_points_3d:
                # Convert from meters to millimeters
                obj_points_3d = np.array(obj_points_3d, dtype=np.float32) * 1000
                
                # Project the 3D points onto the RGB camera's image plane
                projected_points, _ = cv2.projectPoints(
                    obj_points_3d, rvec, tvec, rgb_cam_mtx, rgb_cam_dist
                )

                # --- MODIFICATION START ---
                # Draw the PROJECTED points as RED circles on the RGB image
                for p in projected_points:
                    cv2.circle(display_rgb, tuple(p[0].astype(int)), 5, (0, 0, 255), -1) # RED, larger circle

        # --- NEW BLOCK ---
        # If the board is detected directly in the RGB camera, draw its corners
        if ids_rgb is not None:
             # Draw the DETECTED points as GREEN circles on the RGB image
            for corner in corners_rgb:
                cv2.circle(display_rgb, tuple(corner[0].astype(int)), 3, (0, 255, 0), -1) # GREEN, smaller circle
        # --- END OF NEW BLOCK ---

        # Add explanatory text to the display
        cv2.putText(display_rgb, "Green=Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_rgb, "Red=Projected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the images
        cv2.imshow('Source View (Point Cloud Camera)', display_pc) # Show the display copy
        cv2.imshow('Test View (RGB Camera)', display_rgb)           # Show the display copy

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline_pc.stop()
    pipeline_rgb.stop()
    cv2.destroyAllWindows()