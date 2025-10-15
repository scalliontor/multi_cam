import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- 1. CONFIGURATION ---
PC_CAMERA_SN = "832112070255"  # Source camera (has depth)
RGB_CAMERA_SN = "213622078112"  # Target camera

CONFIG_FILE = "charuco_board_config.npz"
EXTRINSICS_FILE = "extrinsics.npz"

# Check if required files exist
for f in [CONFIG_FILE, EXTRINSICS_FILE]:
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

# Load extrinsics (PC -> RGB transformation)
with np.load(EXTRINSICS_FILE) as data:
    rvec = data['rvec']
    tvec = data['tvec']  # in mm
    if 'R' in data:
        R = data['R']
    else:
        R, _ = cv2.Rodrigues(rvec)
    # For 3D-to-3D calibration, use the direct transformation vector
    if 't' in data:
        t_meters = data['t']  # Direct transformation in meters
    else:
        t_meters = tvec.flatten() * 0.001  # Convert mm to meters
    
    # CRITICAL FIX: Load metadata while still in context manager
    calibration_method = data.get('calibration_method', 'Unknown') if 'calibration_method' in data else 'Unknown'
    mean_3d_error = data['mean_3d_error'] if 'mean_3d_error' in data else None

print("Loaded extrinsics:")
print(f"Rotation vector: {rvec.flatten()}")
print(f"Translation vector (mm): {tvec.flatten()}")
print(f"Calibration method: {calibration_method}")
if mean_3d_error is not None:
    print(f"Original calibration error: {mean_3d_error:.2f} mm")
    if mean_3d_error > 20:
        print("⚠️  WARNING: High calibration error detected! Recommend recalibration.")

# --- 2. REALSENSE SETUP ---
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

# Start streaming and get SDK intrinsics
profile_pc = pipeline_pc.start(config_pc)
depth_intrinsics_pc = profile_pc.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrinsics_pc = profile_pc.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

profile_rgb = pipeline_rgb.start(config_rgb)
depth_intrinsics_rgb = profile_rgb.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrinsics_rgb = profile_rgb.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

align_pc = rs.align(rs.stream.color)
align_rgb = rs.align(rs.stream.color)

# Convert RealSense intrinsics to OpenCV format
def rs_intrinsics_to_cv_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                     [0, intrinsics.fy, intrinsics.ppy],
                     [0, 0, 1]], dtype=np.float32)

def rs_intrinsics_to_cv_dist(intrinsics):
    return np.array([intrinsics.coeffs[0], intrinsics.coeffs[1], 
                     intrinsics.coeffs[2], intrinsics.coeffs[3], 
                     intrinsics.coeffs[4]], dtype=np.float32)

# CRITICAL FIX: Use COLOR intrinsics for consistency with calibration
# Since calibration uses color-aligned frames, we must use color intrinsics
pc_cam_mtx = rs_intrinsics_to_cv_matrix(color_intrinsics_pc)
pc_cam_dist = rs_intrinsics_to_cv_dist(color_intrinsics_pc)
rgb_cam_mtx = rs_intrinsics_to_cv_matrix(color_intrinsics_rgb)
rgb_cam_dist = rs_intrinsics_to_cv_dist(color_intrinsics_rgb)

print("\nUsing SDK COLOR intrinsics for inference (consistent with updated calibration):")
print(f"PC Camera Matrix:\n{pc_cam_mtx}")
print(f"RGB Camera Matrix:\n{rgb_cam_mtx}")

print("\nCoordinate system verification:")
print(f"PC Depth intrinsics - fx:{depth_intrinsics_pc.fx:.2f}, fy:{depth_intrinsics_pc.fy:.2f}")
print(f"PC Color intrinsics - fx:{color_intrinsics_pc.fx:.2f}, fy:{color_intrinsics_pc.fy:.2f}")
print(f"RGB Depth intrinsics - fx:{depth_intrinsics_rgb.fx:.2f}, fy:{depth_intrinsics_rgb.fy:.2f}")
print(f"RGB Color intrinsics - fx:{color_intrinsics_rgb.fx:.2f}, fy:{color_intrinsics_rgb.fy:.2f}")

if abs(depth_intrinsics_pc.fx - color_intrinsics_pc.fx) > 10:
    print("⚠️  WARNING: Large difference between depth and color intrinsics detected!")
    print("   This may indicate alignment issues between depth and color streams.")

print("\n🎯 Starting calibration validation...")
print("Green dots = Direct detection | Red dots = Projected from other camera")
print("Press 'q' to quit.")

try:
    while True:
        # Get frames from both cameras
        frames_pc = pipeline_pc.wait_for_frames()
        frames_rgb = pipeline_rgb.wait_for_frames()

        # Align depth to color for both cameras
        aligned_frames_pc = align_pc.process(frames_pc)
        depth_frame_pc = aligned_frames_pc.get_depth_frame()
        color_frame_pc = aligned_frames_pc.get_color_frame()
        
        aligned_frames_rgb = align_rgb.process(frames_rgb)
        depth_frame_rgb = aligned_frames_rgb.get_depth_frame()
        color_frame_rgb = aligned_frames_rgb.get_color_frame()

        if not all([depth_frame_pc, color_frame_pc, depth_frame_rgb, color_frame_rgb]):
            continue

        # Convert to numpy arrays
        img_pc = np.asanyarray(color_frame_pc.get_data())
        img_rgb = np.asanyarray(color_frame_rgb.get_data())
        gray_pc = cv2.cvtColor(img_pc, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # Detect ChArUco board in both cameras
        corners_pc, ids_pc, _, _ = detector.detectBoard(gray_pc)
        corners_rgb, ids_rgb, _, _ = detector.detectBoard(gray_rgb)

        # Create display copies
        display_pc = img_pc.copy()
        display_rgb = img_rgb.copy()

        # Process if both cameras detect the board
        if ids_pc is not None and ids_rgb is not None:
            # Find common corner IDs
            common_ids = np.intersect1d(ids_pc.flatten(), ids_rgb.flatten())
            
            if len(common_ids) >= 3:
                # Get 3D points from BOTH cameras (same as calibration method)
                points_3d_pc = []
                points_3d_rgb = []
                points_2d_rgb_detected = []
                points_2d_pc = []
                
                valid_corners = 0
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
                    
                    # Only use points where BOTH cameras have valid depth (same as calibration)
                    if depth_pc > 0.1 and depth_rgb > 0.1:
                        # CRITICAL FIX: Use COLOR intrinsics to match calibration coordinate system
                        point_3d_pc = rs.rs2_deproject_pixel_to_point(color_intrinsics_pc, [u_pc, v_pc], depth_pc)
                        point_3d_rgb = rs.rs2_deproject_pixel_to_point(color_intrinsics_rgb, [u_rgb, v_rgb], depth_rgb)
                        
                        points_3d_pc.append(point_3d_pc)
                        points_3d_rgb.append(point_3d_rgb)
                        
                        # Get corresponding 2D points for visualization
                        points_2d_rgb_detected.append(corner_rgb)
                        points_2d_pc.append([u_pc, v_pc])
                        
                        valid_corners += 1
                        # Draw PC camera points in blue
                        cv2.circle(display_pc, (u_pc, v_pc), 5, (255, 0, 0), -1)

                if valid_corners >= 3:
                    # Convert to numpy arrays
                    points_3d_pc = np.array(points_3d_pc, dtype=np.float32)
                    points_3d_rgb = np.array(points_3d_rgb, dtype=np.float32)
                    points_2d_rgb_detected = np.array(points_2d_rgb_detected, dtype=np.float32)
                    
                    # Transform 3D points from PC to RGB coordinate system using 3D-to-3D calibration
                    # Apply transformation: P_rgb = R @ P_pc + t (where t is in meters)
                    points_3d_transformed = (R @ points_3d_pc.T + t_meters.reshape(-1, 1)).T
                    
                    # Calculate 3D-to-3D error (same method as calibration validation)
                    errors_3d = np.linalg.norm(points_3d_rgb - points_3d_transformed, axis=1) * 1000  # Convert to mm
                    mean_3d_error = np.mean(errors_3d)
                    max_3d_error = np.max(errors_3d)
                    
                    # CRITICAL FIX: Use COLOR intrinsics consistently (same as calibration)
                    # Since we're projecting to color-aligned coordinate system
                    
                    # Project transformed points to RGB image using COLOR intrinsics
                    points_3d_transformed_mm = points_3d_transformed * 1000  # Convert to mm for projectPoints
                    points_2d_rgb_projected, _ = cv2.projectPoints(
                        points_3d_transformed_mm,
                        np.zeros(3),  # No additional rotation
                        np.zeros(3),  # No additional translation
                        rgb_cam_mtx,  # Use RGB camera's COLOR intrinsics (consistent with calibration)
                        rgb_cam_dist  # Use RGB camera's COLOR distortion (consistent with calibration)
                    )
                    points_2d_rgb_projected = points_2d_rgb_projected.reshape(-1, 2)
                    
                    # Calculate 2D projection error (for visualization only)
                    projection_errors = np.linalg.norm(points_2d_rgb_detected - points_2d_rgb_projected, axis=1)
                    mean_projection_error = np.mean(projection_errors)
                    max_projection_error = np.max(projection_errors)
                    
                    # Draw only the dots - clean visualization
                    for i, (detected, projected) in enumerate(zip(points_2d_rgb_detected, points_2d_rgb_projected)):
                        # Green = detected directly in RGB camera
                        cv2.circle(display_rgb, tuple(detected.astype(int)), 4, (0, 255, 0), -1)
                        # Red = projected from PC camera via transformation
                        cv2.circle(display_rgb, tuple(projected.astype(int)), 6, (0, 0, 255), 2)

        # Draw detected corners
        if ids_pc is not None:
            cv2.aruco.drawDetectedCornersCharuco(display_pc, corners_pc, ids_pc)
        if ids_rgb is not None:
            cv2.aruco.drawDetectedCornersCharuco(display_rgb, corners_rgb, ids_rgb)

        # Add minimal labels only
        cv2.putText(display_rgb, "Green=Detected, Red=Projected", (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the images
        cv2.imshow('PC Camera (Source)', display_pc)
        cv2.imshow('RGB Camera (Inference Check)', display_rgb)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline_pc.stop()
    pipeline_rgb.stop()
    cv2.destroyAllWindows()
    print("Inference check completed.")
