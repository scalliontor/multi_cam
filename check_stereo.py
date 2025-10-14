# check_stereo_projection.py
# Được sửa đổi từ check_extrinsic.py để xác thực kết quả từ file stereo_calibration.npz

import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- 1. CẤU HÌNH ---
# Sử dụng tên biến nhất quán với script hiệu chỉnh stereo
LEFT_CAMERA_SN = "832112070255"  # Camera có cảm biến chiều sâu
RIGHT_CAMERA_SN = "213622078112" # Camera chỉ có màu (hoặc camera mục tiêu)

# --- Các file cấu hình và file hiệu chỉnh mới ---
CONFIG_FILE = "charuco_board_config.npz"
STEREO_CALIB_FILE = "stereo_calibration.npz" # <<< THAY ĐỔI CHÍNH

# Kiểm tra các file cần thiết
for f in [CONFIG_FILE, STEREO_CALIB_FILE]:
    if not os.path.exists(f):
        print(f"Lỗi: Thiếu file bắt buộc: '{f}'")
        exit()

# Tải cấu hình bảng ChArUco
config_data = np.load(CONFIG_FILE)
board = cv2.aruco.CharucoBoard(
    (int(config_data['squares_x']), int(config_data['squares_y'])),
    float(config_data['square_size_mm']),
    float(config_data['marker_size_mm']),
    cv2.aruco.getPredefinedDictionary(int(config_data['aruco_dict_id']))
)
detector = cv2.aruco.CharucoDetector(board)

# --- THAY ĐỔI CHÍNH: Tải dữ liệu từ file hiệu chỉnh stereo ---
print(f"Đang tải dữ liệu hiệu chỉnh từ: {STEREO_CALIB_FILE}")
with np.load(STEREO_CALIB_FILE) as data:
    # Chúng ta chỉ cần thông số của camera phải để chiếu điểm lên đó
    mtx_right = data['mtx_right']
    dist_right = data['dist_right']
    
    # Lấy Ma trận xoay (R) và Vector tịnh tiến (T)
    R = data['R']
    T = data['T']

# --- Chuyển đổi Ma trận xoay R thành Vector xoay rvec ---
# Hàm cv2.projectPoints yêu cầu một vector xoay (rvec)
rvec, _ = cv2.Rodrigues(R)
tvec = T # Vector tịnh tiến có thể được sử dụng trực tiếp

# --- 2. THIẾT LẬP REALSENSE PIPELINES ---
pipeline_left = rs.pipeline()
config_left = rs.config()
config_left.enable_device(LEFT_CAMERA_SN)
config_left.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_left.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_right = rs.pipeline()
config_right = rs.config()
config_right.enable_device(RIGHT_CAMERA_SN)
config_right.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile_left = pipeline_left.start(config_left)
pipeline_right.start(config_right)

depth_intrinsics = profile_left.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
align = rs.align(rs.stream.color)

print("\nBắt đầu kiểm tra phép chiếu extrinsic...")
print(" - Vòng tròn XANH = Góc được phát hiện trực tiếp trên camera phải.")
print(" - Vòng tròn ĐỎ   = Góc được chiếu từ camera trái sang camera phải.")
print("Mục tiêu: Làm cho các vòng tròn màu đỏ và xanh lá chồng khít lên nhau.")
print("Nhấn 'q' để thoát.")

try:
    while True:
        frames_left = pipeline_left.wait_for_frames()
        frames_right = pipeline_right.wait_for_frames()

        aligned_frames = align.process(frames_left)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame_left = aligned_frames.get_color_frame()
        color_frame_right = frames_right.get_color_frame()

        if not depth_frame or not color_frame_left or not color_frame_right:
            continue

        img_left = np.asanyarray(color_frame_left.get_data())
        img_right = np.asanyarray(color_frame_right.get_data())
        
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        corners_left, ids_left, _, _ = detector.detectBoard(gray_left)
        corners_right, ids_right, _, _ = detector.detectBoard(gray_right)

        display_left = img_left.copy()
        display_right = img_right.copy()

        # Nếu phát hiện bảng trên camera trái, chiếu các góc của nó
        if ids_left is not None:
            cv2.aruco.drawDetectedCornersCharuco(display_left, corners_left, ids_left)
            
            # Lấy tọa độ 3D từ chiều sâu
            obj_points_3d = []
            for corner in corners_left:
                u, v = int(corner[0][0]), int(corner[0][1])
                depth = depth_frame.get_distance(u, v)
                if depth > 0.1:
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                    obj_points_3d.append(point_3d)

            if obj_points_3d:
                # Chuyển từ mét sang milimét
                obj_points_3d = np.array(obj_points_3d, dtype=np.float32) * 1000
                
                # Chiếu các điểm 3D lên mặt phẳng ảnh của camera phải
                # SỬ DỤNG CÁC THÔNG SỐ ĐÃ TẢI TỪ FILE HIỆU CHỈNH STEREO
                projected_points, _ = cv2.projectPoints(
                    obj_points_3d, rvec, tvec, mtx_right, dist_right
                )

                # Vẽ các điểm được CHIẾU thành các vòng tròn ĐỎ
                for p in projected_points:
                    cv2.circle(display_right, tuple(p[0].astype(int)), 5, (0, 0, 255), -1)

        # Nếu phát hiện bảng trực tiếp trên camera phải, vẽ các góc của nó
        if ids_right is not None:
            # Vẽ các điểm được PHÁT HIỆN thành các vòng tròn XANH
            for corner in corners_right:
                cv2.circle(display_right, tuple(corner[0].astype(int)), 3, (0, 255, 0), -1)

        # Thêm văn bản giải thích
        cv2.putText(display_right, "Green=Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_right, "Red=Projected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Hiển thị ảnh
        cv2.imshow('Left Camera View', display_left)
        cv2.imshow('Right Camera (Test View)', display_right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline_left.stop()
    pipeline_right.stop()
    cv2.destroyAllWindows()