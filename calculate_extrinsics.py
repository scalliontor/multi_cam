# --- CÁC PHẦN IMPORT VÀ CẤU HÌNH BAN ĐẦU GIỮ NGUYÊN ---
import pyrealsense2 as rs
import numpy as np
import cv2
import os

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
config_rgb.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile_pc = pipeline_pc.start(config_pc)
pipeline_rgb.start(config_rgb)

# Get depth sensor intrinsics for deprojection
depth_intrinsics = profile_pc.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Alignment for depth to color
align = rs.align(rs.stream.color)

# ### THAY ĐỔI ###: Khởi tạo list để lưu trữ điểm từ nhiều lần chụp
all_obj_points_3d = [] # Tập hợp các điểm 3D
all_img_points_2d_rgb = [] # Tập hợp các điểm 2D tương ứng

print("Position the board to be visible in both cameras.")
print("Press 'c' to capture a view. Aim for 5-10 views.")
print("Press 'q' to quit and calculate extrinsics.")

try:
    while True:
        # Get frames from both cameras
        frames_pc = pipeline_pc.wait_for_frames()
        frames_rgb = pipeline_rgb.wait_for_frames()

        # Align depth frame to color frame for the point cloud camera
        aligned_frames = align.process(frames_pc)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame_pc = aligned_frames.get_color_frame()
        color_frame_rgb = frames_rgb.get_color_frame()

        if not depth_frame or not color_frame_pc or not color_frame_rgb:
            continue

        # Convert images to numpy arrays
        img_pc = np.asanyarray(color_frame_pc.get_data())
        img_rgb = np.asanyarray(color_frame_rgb.get_data())
        gray_pc = cv2.cvtColor(img_pc, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # Detect board in both images
        charuco_corners_pc, charuco_ids_pc, _, _ = detector.detectBoard(gray_pc)
        charuco_corners_rgb, charuco_ids_rgb, _, _ = detector.detectBoard(gray_rgb)

        # Draw detections for visualization
        if charuco_ids_pc is not None: cv2.aruco.drawDetectedCornersCharuco(img_pc, charuco_corners_pc, charuco_ids_pc)
        if charuco_ids_rgb is not None: cv2.aruco.drawDetectedCornersCharuco(img_rgb, charuco_corners_rgb, charuco_ids_rgb)

        cv2.imshow('Point Cloud Camera', img_pc)
        cv2.imshow('RGB Camera', img_rgb)

        key = cv2.waitKey(1)
        if key == ord('q'): break
        
        # --- ### THAY ĐỔI ###: CALIBRATION LOGIC - CHỈ THU THẬP ĐIỂM ---
        if key == ord('c') and charuco_ids_pc is not None and charuco_ids_rgb is not None:
            print(f"Capturing frame #{len(all_obj_points_3d) + 1}...")

            # Match corners between the two views
            common_ids = np.intersect1d(charuco_ids_pc.flatten(), charuco_ids_rgb.flatten())
            if len(common_ids) < 4:
                print("Not enough common corners found. Reposition the board.")
                continue

            # Get the 3D points from the Point Cloud camera
            obj_points_3d_capture = []
            img_points_2d_rgb_capture = []

            for id_val in common_ids:
                idx_pc = np.where(charuco_ids_pc.flatten() == id_val)[0][0]
                idx_rgb = np.where(charuco_ids_rgb.flatten() == id_val)[0][0]
                
                corner_pc_2d = charuco_corners_pc[idx_pc][0]
                u, v = int(corner_pc_2d[0]), int(corner_pc_2d[1])

                depth = depth_frame.get_distance(u, v)
                if depth > 0:
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                    obj_points_3d_capture.append(point_3d)
                    img_points_2d_rgb_capture.append(charuco_corners_rgb[idx_rgb])

            if len(obj_points_3d_capture) < 4:
                print("Could not get enough valid 3D points. Check for glare or distance.")
                continue
            
            # Thêm các điểm của lần chụp này vào danh sách tổng
            all_obj_points_3d.extend(obj_points_3d_capture)
            all_img_points_2d_rgb.extend(img_points_2d_rgb_capture)
            print(f"Capture successful. Total captures: {len(all_obj_points_3d)}")

finally:
    pipeline_pc.stop()
    pipeline_rgb.stop()
    cv2.destroyAllWindows()

# --- ### THAY ĐỔI ###: TÍNH TOÁN EXTRINSICS SAU KHI THU THẬP XONG ---
if len(all_obj_points_3d) > 0 and len(all_img_points_2d_rgb) > 0:
    print("\nCalculating extrinsics from all captured views...")
    
    # Chuyển đổi list thành numpy arrays
    obj_points_3d = np.array(all_obj_points_3d, dtype=np.float32) * 1000 # convert meters to mm
    img_points_2d_rgb = np.array(all_img_points_2d_rgb, dtype=np.float32)

    # Sử dụng solvePnP để tìm rotation và translation
    success, rvec, tvec = cv2.solvePnP(
        obj_points_3d, img_points_2d_rgb, rgb_cam_mtx, rgb_cam_dist
    )

    if success:
        print("\nExtrinsic calibration successful!")
        print("Rotation Vector (rvec):\n", rvec)
        print("Translation Vector (tvec) in mm:\n", tvec)
        
        # Lưu kết quả
        np.savez("extrinsics.npz", rvec=rvec, tvec=tvec)
        print("Extrinsics saved to extrinsics.npz")
    else:
        print("solvePnP failed. Try to capture more varied views.")
else:
    print("No valid views were captured. Extrinsics not calculated.")