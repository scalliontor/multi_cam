import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# --- CONFIGURE YOUR CAMERA SERIALS ---
CAMERA_1_SN = "832112070255" # Corresponds to PC_CAMERA in other scripts
CAMERA_2_SN = "213622078112" # Corresponds to RGB_CAMERA in other scripts

def capture_point_cloud(pipeline, align):
    """Captures a frame and converts it to an Open3D point cloud."""
    try:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None

        # Create Open3D objects
        depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, convert_rgb_to_intensity=False
        )
        
        # Get intrinsics for point cloud creation
        profile = depth_frame.get_profile()
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, 
            intrinsics.fx, intrinsics.fy, 
            intrinsics.ppx, intrinsics.ppy
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, pinhole_camera_intrinsic
        )
        
        return pcd
    except Exception as e:
        print(f"Error capturing point cloud: {e}")
        return None

# Setup pipelines for both cameras
p1 = rs.pipeline()
c1 = rs.config()
c1.enable_device(CAMERA_1_SN)
c1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
c1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
p1.start(c1)
align1 = rs.align(rs.stream.color)

p2 = rs.pipeline()
c2 = rs.config()
c2.enable_device(CAMERA_2_SN)
c2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
c2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
p2.start(c2)
align2 = rs.align(rs.stream.color)

print("Pipelines started. Point a 3D object so both cameras can see it.")
print("Press Enter to capture point clouds...")
input()

# Capture from both cameras
print("Capturing from Camera 1...")
pcd1 = capture_point_cloud(p1, align1)
print("Capturing from Camera 2...")
pcd2 = capture_point_cloud(p2, align2)

p1.stop()
p2.stop()

if pcd1 is None or pcd2 is None:
    print("Failed to capture from one or both cameras. Exiting.")
    exit()

# --- ICP Registration ---
print("Performing ICP registration...")
# Preprocess point clouds
pcd1 = pcd1.voxel_down_sample(voxel_size=0.005) # 5mm
pcd2 = pcd2.voxel_down_sample(voxel_size=0.005)

# The registration algorithm
threshold = 0.02  # Max distance between corresponding points (2cm)
trans_init = np.identity(4) # Initial guess is no transformation

# This is the core ICP function
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

print("ICP Finished.")
print("Transformation matrix (from Camera 1 to Camera 2):")
print(reg_p2p.transformation)

# Extract R and t
transformation = reg_p2p.transformation
R_final = transformation[0:3, 0:3]
tvec_final_meters = transformation[0:3, 3]
tvec_final_mm = tvec_final_meters * 1000 # Convert to mm

# Convert rotation matrix to rotation vector for saving
rvec_final, _ = cv2.Rodrigues(R_final)

print("\nRotation Vector (rvec):\n", rvec_final)
print("Translation Vector (tvec) in mm:\n", tvec_final_mm)

# Save the result in the same format as the other script
np.savez("extrinsics_icp.npz", rvec=rvec_final, tvec=tvec_final_mm, R=R_final)
print("\nExtrinsics saved to extrinsics_icp.npz")

# Optional: Visualize the alignment
pcd1.transform(reg_p2p.transformation) # Apply the transformation to the source
o3d.visualization.draw_geometries([pcd1, pcd2], window_name="ICP Alignment")