# stereo_calibrate_live.py
# Nâng cấp từ calculate_extrinsics.py để sử dụng cv2.stereoCalibrate cho kết quả chính xác hơn.

import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- 1. CẤU HÌNH ---
# Đổi tên biến để rõ ràng hơn: "trái" và "phải" thay vì "point cloud" và "rgb"
LEFT_CAMERA_SN = "832112070255"
RIGHT_CAMERA_SN = "213622078112"      

CONFIG_FILE = "charuco_board_config.npz"
INTRINSICS_LEFT_FILE = f"intrinsics_charuco_{LEFT_CAMERA_SN}.npz"
INTRINSICS_RIGHT_FILE = f"intrinsics_charuco_{RIGHT_CAMERA_SN}.npz"
OUTPUT_FILE = "stereo_calibration.npz"

# Kiểm tra các file cần thiết
for f in [CONFIG_FILE, INTRINSICS_LEFT_FILE, INTRINSICS_RIGHT_FILE]:
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

# Tải thông số nội tại ban đầu (sẽ được tinh chỉnh lại)
with np.load(INTRINSICS_LEFT_FILE) as data:
    mtx_left_initial = data['mtx']
    dist_left_initial = data['dist']

with np.load(INTRINSICS_RIGHT_FILE) as data:
    mtx_right_initial = data['mtx']
    dist_right_initial = data['dist']


# --- 2. THIẾT LẬP REALSENSE PIPELINES (CHỈ CẦN LUỒNG MÀU) ---
pipeline_left = rs.pipeline()
config_left = rs.config()
config_left.enable_device(LEFT_CAMERA_SN)
config_left.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline_right = rs.pipeline()
config_right = rs.config()
config_right.enable_device(RIGHT_CAMERA_SN)
config_right.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Bắt đầu streaming
pipeline_left.start(config_left)
pipeline_right.start(config_right)

# --- 3. THU THẬP DỮ LIỆU ---
# Các list để lưu trữ điểm từ tất cả các lần chụp hợp lệ
all_obj_points = []   # Tọa độ 3D của các góc trên bảng (hệ tọa độ của bảng)
all_img_points_left = []  # Tọa độ 2D của các góc trên ảnh từ camera trái
all_img_points_right = [] # Tọa độ 2D của các góc trên ảnh từ camera phải

# Lấy tọa độ 3D lý tưởng của tất cả các góc trên bảng một lần duy nhất
all_board_corners_3d = board.getChessboardCorners()

print("Di chuyển bảng ChArUco đến nhiều vị trí và góc độ khác nhau.")
print("Nhấn 'c' để chụp một cặp ảnh. Cần ít nhất 15-20 cặp ảnh tốt.")
print("Nhấn 'q' để kết thúc và bắt đầu hiệu chỉnh.")

try:
    while True:
        frames_left = pipeline_left.wait_for_frames()
        frames_right = pipeline_right.wait_for_frames()
        
        color_frame_left = frames_left.get_color_frame()
        color_frame_right = frames_right.get_color_frame()

        if not color_frame_left or not color_frame_right:
            continue

        img_left = np.asanyarray(color_frame_left.get_data())
        img_right = np.asanyarray(color_frame_right.get_data())
        
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        corners_left, ids_left, _, _ = detector.detectBoard(gray_left)
        corners_right, ids_right, _, _ = detector.detectBoard(gray_right)

        if ids_left is not None: cv2.aruco.drawDetectedCornersCharuco(img_left, corners_left, ids_left)
        if ids_right is not None: cv2.aruco.drawDetectedCornersCharuco(img_right, corners_right, ids_right)
        
        cv2.putText(img_left, f"Captured: {len(all_obj_points)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Left Camera', img_left)
        cv2.imshow('Right Camera', img_right)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        if key == ord('c') and ids_left is not None and ids_right is not None:
            print(f"Attempting to capture frame #{len(all_obj_points) + 1}...")

            # --- LOGIC THU THẬP ĐIỂM ĐÃ SỬA ĐỔI CHO STEREO ---
            # Tìm các ID góc chung mà cả hai camera đều nhìn thấy
            common_ids = np.intersect1d(ids_left.flatten(), ids_right.flatten())
            
            if len(common_ids) > 6: # Cần ít nhất 6 góc chung để đảm bảo chất lượng
                objp_frame = []
                imgp_left_frame = []
                imgp_right_frame = []

                # Chỉ lặp qua các ID chung
                for id_val in common_ids:
                    # Lấy tọa độ 3D lý tưởng của góc từ định nghĩa của bảng
                    objp_frame.append(all_board_corners_3d[id_val])
                    
                    # Lấy tọa độ 2D tương ứng từ mỗi camera
                    idx_left = np.where(ids_left.flatten() == id_val)[0][0]
                    imgp_left_frame.append(corners_left[idx_left])
                    
                    idx_right = np.where(ids_right.flatten() == id_val)[0][0]
                    imgp_right_frame.append(corners_right[idx_right])

                # Thêm các cặp điểm đã được khớp hoàn hảo vào danh sách tổng
                all_obj_points.append(np.array(objp_frame, dtype=np.float32))
                all_img_points_left.append(np.array(imgp_left_frame, dtype=np.float32))
                all_img_points_right.append(np.array(imgp_right_frame, dtype=np.float32))
                
                print(f"Capture successful with {len(common_ids)} common corners.")
            else:
                print("Capture failed: Not enough common corners found. Reposition the board.")

finally:
    pipeline_left.stop()
    pipeline_right.stop()
    cv2.destroyAllWindows()

# --- 4. THỰC HIỆN HIỆU CHỈNH STEREO ---
if len(all_obj_points) < 15:
    print(f"\nCẢNH BÁO: Chỉ có {len(all_obj_points)} cặp ảnh được chụp. Kết quả hiệu chỉnh có thể không tốt.")
    if len(all_obj_points) < 5:
        print("Không đủ dữ liệu. Thoát.")
        exit()

print(f"\n=== BẮT ĐẦU HIỆU CHỈNH STEREO VỚI {len(all_obj_points)} CẶP ẢNH ===")
image_size = gray_left.shape[::-1] # (width, height)

# Sử dụng cờ CALIB_USE_INTRINSIC_GUESS để cho phép thuật toán tinh chỉnh thêm các thông số nội tại
flags = cv2.CALIB_USE_INTRINSIC_GUESS

ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    all_obj_points,
    all_img_points_left,
    all_img_points_right,
    mtx_left_initial,
    dist_left_initial,
    mtx_right_initial,
    dist_right_initial,
    image_size,
    flags=flags,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
)

if ret:
    print("\n" + "="*50)
    print("🎉 HIỆU CHỈNH STEREO THÀNH CÔNG!")
    print("="*50)

    print(f"\nChất lượng hiệu chỉnh:")
    print(f"📊 Lỗi chiếu lại trung bình (Reprojection Error): {ret:.4f} pixels")
    if ret < 1.0: print("✅ Chất lượng hiệu chỉnh tốt.")
    else: print("⚠️ Chất lượng hiệu chỉnh tệ.")
    
    # Lưu kết quả
    np.savez(OUTPUT_FILE,
             mtx_left=mtx_left, dist_left=dist_left,
             mtx_right=mtx_right, dist_right=dist_right,
             R=R, T=T, E=E, F=F,
             reprojection_error=ret, image_size=image_size
             )
    
    print(f"\n💾 Kết quả đã được lưu vào file: {OUTPUT_FILE}")
    print("\n--- THÔNG SỐ NGOẠI TẠI (Extrinsics) ---")
    print("(Mối quan hệ của Camera Phải so với Camera Trái)")
    print("\n🔄 Ma trận xoay (Rotation Matrix R):")
    print(R)
    print("\n📏 Vector tịnh tiến (Translation Vector T) theo mm:")
    print(T)
else:
    print("\n❌ Hiệu chỉnh Stereo thất bại. Hãy thử lại với nhiều ảnh đa dạng hơn.")