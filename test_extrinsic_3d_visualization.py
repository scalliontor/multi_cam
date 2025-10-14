import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
from matplotlib.animation import FuncAnimation

# --- CONFIGURATION ---
TARGET_SERIAL_NUMBER = "213622078112"  # Your camera serial
CALIBRATION_FILE = f"intrinsics_charuco_{TARGET_SERIAL_NUMBER}.npz"

# For stereo calibration, you'll need these files
STEREO_CALIBRATION_FILE = "/mnt/DA0054DE0054C365/linh_tinh/Share_tech/multi_camera_calibrate/stereo_calibration_rotation_fixed.yml"
CHARUCO_CONFIG_FILE = "charuco_board_config.npz"

class Camera3DVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.pipeline = None
        self.camera_mtx = None
        self.camera_dist = None
        self.stereo_data = None
        
        # For real-time updates
        self.current_points_3d = []
        self.current_camera_pose = None
        
    def load_calibration(self):
        """Load camera intrinsic calibration"""
        try:
            calib_data = np.load(CALIBRATION_FILE)
            self.camera_mtx = calib_data['mtx']
            self.camera_dist = calib_data['dist']
            print("✅ Camera intrinsic calibration loaded")
            return True
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False
    
    def load_stereo_calibration(self):
        """Load stereo calibration data if available"""
        try:
            import yaml
            with open(STEREO_CALIBRATION_FILE, 'r') as f:
                stereo_data = yaml.safe_load(f)
            
            # Convert lists to numpy arrays
            self.stereo_data = {
                'R': np.array(stereo_data['rotation_matrix']),
                'T': np.array(stereo_data['translation_vector']).reshape(-1, 1),
                'mtx1': np.array(stereo_data['camera_matrix_1']),
                'dist1': np.array(stereo_data['distortion_coeffs_1']),
                'mtx2': np.array(stereo_data['camera_matrix_2']),
                'dist2': np.array(stereo_data['distortion_coeffs_2']),
                'rms_error': stereo_data['rms_error'],
                'note': stereo_data.get('note', '')
            }
            
            print("✅ Stereo calibration loaded")
            print(f"   RMS Error: {self.stereo_data['rms_error']:.3f}")
            if 'note' in stereo_data:
                print(f"   Note: {stereo_data['note']}")
            return True
            
        except Exception as e:
            print(f"⚠️ Stereo calibration not available: {e}")
        
        return False
    
    def load_charuco_config(self):
        """Load ChArUco board configuration"""
        try:
            config = np.load(CHARUCO_CONFIG_FILE)
            self.charuco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self.charuco_board = cv2.aruco.CharucoBoard_create(
                config['squares_x'], config['squares_y'],
                config['square_length'], config['marker_length'],
                self.charuco_dict
            )
            print("✅ ChArUco board configuration loaded")
            return True
        except Exception as e:
            print(f"❌ Error loading ChArUco config: {e}")
            return False
    
    def detect_charuco_pose(self, image):
        """Detect ChArUco board and estimate camera pose"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.charuco_dict)
        
        if ids is not None and len(ids) > 3:
            # Interpolate ChArUco corners
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.charuco_board
            )
            
            if ret > 3:  # Need at least 4 corners
                # Estimate pose
                success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.charuco_board,
                    self.camera_mtx, self.camera_dist, None, None
                )
                
                if success:
                    return True, rvec, tvec, charuco_corners, charuco_ids
        
        return False, None, None, None, None
    
    def project_3d_points(self, rvec, tvec):
        """Generate 3D points for visualization"""
        # Create a 3D coordinate system at the board
        axis_length = 0.1  # 10cm axes
        axis_points = np.float32([
            [0, 0, 0],           # Origin
            [axis_length, 0, 0], # X-axis
            [0, axis_length, 0], # Y-axis
            [0, 0, axis_length]  # Z-axis
        ])
        
        # Transform to camera coordinate system
        R, _ = cv2.Rodrigues(rvec)
        points_3d = []
        
        for point in axis_points:
            # Transform from board coordinates to camera coordinates
            point_cam = R @ point + tvec.flatten()
            points_3d.append(point_cam)
        
        return np.array(points_3d)
    
    def get_camera_pose_from_board(self, rvec, tvec):
        """Get camera pose in world coordinates (board as origin)"""
        R, _ = cv2.Rodrigues(rvec)
        
        # Camera pose in world coordinates
        R_cam = R.T
        t_cam = -R.T @ tvec.flatten()
        
        return R_cam, t_cam
    
    def setup_3d_plot(self):
        """Setup the 3D plot"""
        self.ax.clear()
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_title('3D Camera Pose Visualization')
        
        # Set equal aspect ratio
        max_range = 0.5
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([0, max_range])
    
    def draw_camera_frustum(self, R, t, color='blue', label='Camera'):
        """Draw camera frustum in 3D"""
        # Camera frustum corners (in camera coordinate system)
        frustum_size = 0.1
        frustum_depth = 0.2
        
        frustum_points = np.array([
            [0, 0, 0],  # Camera center
            [-frustum_size, -frustum_size, frustum_depth],
            [frustum_size, -frustum_size, frustum_depth],
            [frustum_size, frustum_size, frustum_depth],
            [-frustum_size, frustum_size, frustum_depth],
        ])
        
        # Transform to world coordinates
        world_points = []
        for point in frustum_points:
            world_point = R @ point + t
            world_points.append(world_point)
        
        world_points = np.array(world_points)
        
        # Draw frustum lines
        connections = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # From center to corners
            (1, 2), (2, 3), (3, 4), (4, 1)   # Rectangle at far plane
        ]
        
        for start, end in connections:
            self.ax.plot3D(
                [world_points[start, 0], world_points[end, 0]],
                [world_points[start, 1], world_points[end, 1]],
                [world_points[start, 2], world_points[end, 2]],
                color=color, alpha=0.7
            )
        
        # Camera position
        self.ax.scatter(t[0], t[1], t[2], color=color, s=100, label=label)
    
    def draw_coordinate_axes(self, R, t, length=0.1):
        """Draw coordinate axes at given pose"""
        # Axis vectors
        axes = np.array([
            [length, 0, 0],  # X-axis (red)
            [0, length, 0],  # Y-axis (green)
            [0, 0, length]   # Z-axis (blue)
        ])
        
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        for i, (axis, color, label) in enumerate(zip(axes, colors, labels)):
            world_axis = R @ axis + t
            self.ax.plot3D(
                [t[0], world_axis[0]],
                [t[1], world_axis[1]],
                [t[2], world_axis[2]],
                color=color, linewidth=3, label=f'{label}-axis'
            )
    
    def update_visualization(self, rvec, tvec):
        """Update the 3D visualization with new pose data"""
        self.setup_3d_plot()
        
        # Get camera pose
        R_cam, t_cam = self.get_camera_pose_from_board(rvec, tvec)
        
        # Draw world coordinate system (ChArUco board)
        self.ax.scatter(0, 0, 0, color='black', s=200, label='Board Origin')
        self.draw_coordinate_axes(np.eye(3), np.zeros(3), length=0.15)
        
        # Draw camera
        self.draw_camera_frustum(R_cam, t_cam, color='blue', label='Camera')
        self.draw_coordinate_axes(R_cam, t_cam, length=0.1)
        
        # If stereo calibration is available, show second camera
        if self.stereo_data is not None:
            R_stereo = R_cam @ self.stereo_data['R']
            t_stereo = R_cam @ self.stereo_data['T'].flatten() + t_cam
            self.draw_camera_frustum(R_stereo, t_stereo, color='red', label='Camera 2')
        
        # Add text info
        info_text = f"Camera Position: ({t_cam[0]:.3f}, {t_cam[1]:.3f}, {t_cam[2]:.3f})"
        self.ax.text2D(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                      bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)
    
    def run_camera_test(self):
        """Run real-time camera pose estimation"""
        # Initialize camera
        pipeline = rs.pipeline()
        config = rs.config()
        
        try:
            config.enable_device(TARGET_SERIAL_NUMBER)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipeline.start(config)
            
            print("✅ Camera initialized")
            print("🎯 Point camera at ChArUco board to see 3D pose")
            print("📷 Press 'q' in camera window to quit")
            
            frame_count = 0
            last_pose_time = 0
            
            plt.ion()  # Interactive mode
            
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                frame_count += 1
                img = np.asanyarray(color_frame.get_data())
                
                # Detect ChArUco pose
                success, rvec, tvec, corners, ids = self.detect_charuco_pose(img)
                
                if success:
                    # Update 3D visualization (limit update rate)
                    current_time = time.time()
                    if current_time - last_pose_time > 0.1:  # Update at 10Hz max
                        self.update_visualization(rvec, tvec)
                        last_pose_time = current_time
                    
                    # Draw pose on image
                    cv2.aruco.drawDetectedCornersCharuco(img, corners, ids)
                    cv2.aruco.drawAxis(img, self.camera_mtx, self.camera_dist, 
                                     rvec, tvec, 0.1)
                    
                    # Add pose info to image
                    pose_text = f"Position: ({tvec[0,0]:.3f}, {tvec[1,0]:.3f}, {tvec[2,0]:.3f})"
                    cv2.putText(img, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "No ChArUco board detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show camera feed
                cv2.imshow('Camera Feed - 3D Pose Estimation', img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            plt.ioff()
    
    def run(self):
        """Main run function"""
        print("=== 3D CAMERA POSE VISUALIZATION ===")
        print("Testing camera extrinsic calibration with 3D visualization")
        print("=" * 50)
        
        # Load calibrations
        if not self.load_calibration():
            return
        
        if not self.load_charuco_config():
            print("⚠️ ChArUco config not found. Creating default...")
            self.create_default_charuco()
        
        # Try to load stereo calibration
        self.load_stereo_calibration()
        
        # Setup initial plot
        self.setup_3d_plot()
        plt.show(block=False)
        
        # Start camera test
        self.run_camera_test()
    
    def create_default_charuco(self):
        """Create default ChArUco configuration"""
        # Default 5x7 board with 36mm squares and 21.6mm markers
        config = {
            'squares_x': 5,
            'squares_y': 7,
            'square_length': 0.036,  # 36mm in meters
            'marker_length': 0.0216  # 21.6mm in meters
        }
        
        np.savez(CHARUCO_CONFIG_FILE, **config)
        
        self.charuco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard_create(
            config['squares_x'], config['squares_y'],
            config['square_length'], config['marker_length'],
            self.charuco_dict
        )
        
        print("✅ Default ChArUco configuration created")

def main():
    visualizer = Camera3DVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
