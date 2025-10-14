import pyrealsense2 as rs
import numpy as np
import cv2
import os

# --- CONFIGURATION (MATCH CHECK.PY) ---
POINT_CLOUD_CAMERA_SN = "832112070255"  # Camera with depth for 3D points
RGB_CAMERA_SN = "213622078112"           # Target camera for projection

# YOLO Configuration - Using Ultralytics
YOLO_MODEL_PATH = "/mnt/DA0054DE0054C365/linh_tinh/Share_tech/multi_camera_calibrate/pnp_based_approach/best.pt"


INTRINSICS_PC_FILE = f"intrinsics_charuco_{POINT_CLOUD_CAMERA_SN}.npz"
INTRINSICS_RGB_FILE = f"intrinsics_charuco_{RGB_CAMERA_SN}.npz"
EXTRINSICS_FILE = "extrinsics.npz"

def load_calibration():
    """Load camera calibration data for both cameras and extrinsics"""
    # # Check if all required files exist
    # for f in [CONFIG_FILE, INTRINSICS_PC_FILE, INTRINSICS_RGB_FILE, EXTRINSICS_FILE]:
    #     if not os.path.exists(f):
    #         print(f"Error: Missing required file: '{f}'")
    #         return None, None, None, None
    
    # Load RGB camera intrinsics (for projection)
    with np.load(INTRINSICS_RGB_FILE) as data:
        rgb_cam_mtx = data['mtx']
        rgb_cam_dist = data['dist']
    
    # Load PC camera intrinsics (for depth processing)
    with np.load(INTRINSICS_PC_FILE) as data:
        pc_cam_mtx = data['mtx']
        pc_cam_dist = data['dist']
    
    # Load extrinsics (PC camera to RGB camera transformation)
    with np.load(EXTRINSICS_FILE) as data:
        rvec = data['rvec']
        tvec = data['tvec']
    
    print("✅ Camera calibration loaded successfully")
    print(f"   Point Cloud Camera: {POINT_CLOUD_CAMERA_SN}")
    print(f"   RGB Camera: {RGB_CAMERA_SN}")
    print("   Extrinsics loaded for camera-to-camera projection")
    
    return rgb_cam_mtx, rgb_cam_dist, rvec, tvec

def load_yolo_model():
    """Load YOLO model using Ultralytics"""
    try:
        # Check if model file exists
        if not os.path.exists(YOLO_MODEL_PATH):
            print(f"❌ YOLO model file not found: {YOLO_MODEL_PATH}")
            return None
        
        # Import ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            print("❌ Ultralytics not installed. Please install it:")
            print("   pip install ultralytics")
            return None
        
        # Load YOLO model
        model = YOLO(YOLO_MODEL_PATH)
        
        print(f"✅ YOLO model loaded successfully")
        print(f"   Model: {YOLO_MODEL_PATH}")
        print(f"   Classes: {len(model.names) if hasattr(model, 'names') else 'Unknown'}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return None

def detect_objects(model, image, confidence_threshold=0.5):
    """Detect objects using Ultralytics YOLO"""
    try:
        # Run inference
        results = model(image, conf=confidence_threshold, verbose=False)
        
        detections = []
        check = False
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to xywh format
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id] if hasattr(model, 'names') else f"class_{class_id}"
                    
                    detections.append({
                        'box': [x, y, w, h],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        return detections
        
    except Exception as e:
        print(f"⚠️ Detection error: {e}")
        return []

def draw_detections(image, detections):
    """Draw detection results on image"""
    colors = np.random.uniform(0, 255, size=(len(detections), 3))
    
    for i, detection in enumerate(detections):
        x, y, w, h = detection['box']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Draw bounding box
        color = colors[i % len(colors)]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

def project_segmentation_to_other_camera(detections, depth_frame, depth_intrinsics, rvec, tvec, rgb_cam_mtx, rgb_cam_dist):
    """Project segmentation results from PC camera to RGB camera using 3D coordinates with stability improvements"""
    projected_detections = []
    
    for detection in detections:
        x, y, w, h = detection['box']
        
        # Collect depth values for filtering
        valid_depths = []
        valid_points_3d = []
        
        # Sample more densely but filter out bad depth values
        step_x = max(1, w // 15)  # More samples for better stability
        step_y = max(1, h // 15)
        
        # Sample grid points across the bounding box
        for i in range(0, w, step_x):
            for j in range(0, h, step_y):
                u, v = x + i, y + j
                if 0 <= u < 640 and 0 <= v < 480:
                    depth = depth_frame.get_distance(u, v)
                    if depth > 0.1 and depth < 10.0:  # Filter out obviously bad depths (< 10cm or > 10m)
                        valid_depths.append(depth)
                        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                        valid_points_3d.append(point_3d)
        
        # Also sample the center point with higher weight
        center_u, center_v = x + w // 2, y + h // 2
        if 0 <= center_u < 640 and 0 <= center_v < 480:
            center_depth = depth_frame.get_distance(center_u, center_v)
            if center_depth > 0.1 and center_depth < 10.0:
                # Add center point multiple times for stability
                for _ in range(3):
                    valid_depths.append(center_depth)
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center_u, center_v], center_depth)
                    valid_points_3d.append(point_3d)
        
        if len(valid_points_3d) > 3:  # Need at least 4 points for stable projection
            valid_depths = np.array(valid_depths)
            valid_points_3d = np.array(valid_points_3d)
            
            # Filter out outliers using median absolute deviation
            median_depth = np.median(valid_depths)
            mad = np.median(np.abs(valid_depths - median_depth))
            
            # Keep points within 2 MAD of median (robust outlier removal)
            if mad > 0:
                outlier_mask = np.abs(valid_depths - median_depth) < (2 * mad)
                filtered_points_3d = valid_points_3d[outlier_mask]
            else:
                filtered_points_3d = valid_points_3d
            
            if len(filtered_points_3d) > 2:  # Still need enough points
                # Convert from meters to millimeters
                filtered_points_3d = filtered_points_3d.astype(np.float32) * 1000
                
                # Project the 3D points onto the RGB camera's image plane
                projected_points, _ = cv2.projectPoints(
                    filtered_points_3d, rvec, tvec, rgb_cam_mtx, rgb_cam_dist
                )
                
                # Calculate bounding box of projected points
                projected_points = projected_points.reshape(-1, 2)
                if len(projected_points) > 0:
                    # Use percentiles instead of min/max for more robust bounding box
                    min_x, min_y = np.percentile(projected_points, 10, axis=0)  # 10th percentile
                    max_x, max_y = np.percentile(projected_points, 90, axis=0)  # 90th percentile
                    
                    # Ensure minimum box size
                    box_w = max(10, int(max_x - min_x))
                    box_h = max(10, int(max_y - min_y))
                    
                    projected_detections.append({
                        'box': [int(min_x), int(min_y), box_w, box_h],
                        'confidence': detection['confidence'],
                        'class_name': detection['class_name'],
                        'projected_points': projected_points,
                        'depth_info': {
                            'median_depth': float(median_depth),
                            'num_points': len(filtered_points_3d),
                            'depth_std': float(np.std(valid_depths[outlier_mask] if mad > 0 else valid_depths))
                        }
                    })
    
    return projected_detections

def draw_projected_detections(image, projected_detections):
    """Draw projected detection results on RGB camera image"""
    colors = np.random.uniform(0, 255, size=(len(projected_detections), 3))
    
    for i, detection in enumerate(projected_detections):
        x, y, w, h = detection['box']
        confidence = detection['confidence']
        class_name = detection['class_name']
        projected_points = detection['projected_points']
        
        # Draw bounding box
        color = colors[i % len(colors)]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw projected points
        for point in projected_points:
            cv2.circle(image, tuple(point.astype(int)), 2, color, -1)
        
        # Draw label
        label = f"PROJECTED {class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

def smooth_detections(current_detections, previous_detections, smoothing_factor=0.3):
    """Apply temporal smoothing to reduce jitter in detections"""
    if not previous_detections:
        return current_detections
    
    smoothed_detections = []
    
    for curr_det in current_detections:
        curr_box = curr_det['box']
        curr_center = [curr_box[0] + curr_box[2]//2, curr_box[1] + curr_box[3]//2]
        
        # Find closest previous detection of same class
        best_match = None
        min_distance = float('inf')
        
        for prev_det in previous_detections:
            if prev_det['class_name'] == curr_det['class_name']:
                prev_box = prev_det['box']
                prev_center = [prev_box[0] + prev_box[2]//2, prev_box[1] + prev_box[3]//2]
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                
                if distance < min_distance and distance < 100:  # Within 100 pixels
                    min_distance = distance
                    best_match = prev_det
        
        if best_match:
            # Smooth the bounding box coordinates
            prev_box = best_match['box']
            smoothed_box = [
                int(curr_box[0] * smoothing_factor + prev_box[0] * (1 - smoothing_factor)),
                int(curr_box[1] * smoothing_factor + prev_box[1] * (1 - smoothing_factor)),
                int(curr_box[2] * smoothing_factor + prev_box[2] * (1 - smoothing_factor)),
                int(curr_box[3] * smoothing_factor + prev_box[3] * (1 - smoothing_factor))
            ]
            
            smoothed_det = curr_det.copy()
            smoothed_det['box'] = smoothed_box
            smoothed_detections.append(smoothed_det)
        else:
            smoothed_detections.append(curr_det)
    
    return smoothed_detections

def main():
    print("=== YOLO INSTANCE SEGMENTATION + DUAL CAMERA PROJECTION TEST ===")
    print("Running YOLO on Point Cloud camera and projecting to RGB camera")
    print("=" * 70)
    
    # Load camera calibration
    rgb_cam_mtx, rgb_cam_dist, rvec, tvec = load_calibration()
    if rgb_cam_mtx is None:
        return
    
    # Load YOLO model
    model = load_yolo_model()
    if model is None:
        print("\n💡 To use YOLO detection, please:")
        print("1. Install Ultralytics: pip install ultralytics")
        print("2. Make sure your model file exists at the specified path")
        print("\nFor now, running camera test without YOLO...")
        model = None
    
    # Initialize tracking variables for smoothing
    previous_projected_detections = []
    
    # --- REALSENSE PIPELINES SETUP ---
    pipeline_pc = rs.pipeline()
    config_pc = rs.config()
    config_pc.enable_device(POINT_CLOUD_CAMERA_SN)
    config_pc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_pc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline_rgb = rs.pipeline()
    config_rgb = rs.config()
    config_rgb.enable_device(RGB_CAMERA_SN)
    config_rgb.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile_pc = pipeline_pc.start(config_pc)
        pipeline_rgb.start(config_rgb)
        
        depth_intrinsics = profile_pc.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        align = rs.align(rs.stream.color)
        
        print(f"✅ Point Cloud Camera {POINT_CLOUD_CAMERA_SN} initialized")
        print(f"✅ RGB Camera {RGB_CAMERA_SN} initialized")
        
        print(f"\n🎥 Live Test - Press 'q' to quit, 's' to save frame")
        if model is not None:
            print(f"🤖 YOLO instance segmentation enabled")
            print(f"   - Detection on Point Cloud camera")
            print(f"   - 3D projection to RGB camera")
        else:
            print(f"📷 Camera test (no YOLO)")
        
        frame_count = 0
        
        while True:
            frames_pc = pipeline_pc.wait_for_frames()
            frames_rgb = pipeline_rgb.wait_for_frames()

            aligned_frames = align.process(frames_pc)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame_pc = aligned_frames.get_color_frame()
            color_frame_rgb = frames_rgb.get_color_frame()

            if not depth_frame or not color_frame_pc or not color_frame_rgb:
                continue

            frame_count += 1
            img_pc = np.asanyarray(color_frame_pc.get_data())
            img_rgb = np.asanyarray(color_frame_rgb.get_data())
            
            # Create display copies
            display_pc = img_pc.copy()
            display_rgb = img_rgb.copy()
            
            # Run YOLO detection on Point Cloud camera
            if model is not None and frame_count % 2 == 0:  # Run detection every 2nd frame for better performance
                detections = detect_objects(model, img_pc)
                if detections:
                    # Draw detections on Point Cloud camera view
                    display_pc = draw_detections(display_pc, detections)
                    
                    # Project detections to RGB camera using 3D coordinates
                    projected_detections = project_segmentation_to_other_camera(
                        detections, depth_frame, depth_intrinsics, rvec, tvec, rgb_cam_mtx, rgb_cam_dist
                    )
                    
                    # Apply temporal smoothing to reduce jitter
                    if projected_detections:
                        smoothed_projected_detections = smooth_detections(
                            projected_detections, previous_projected_detections, smoothing_factor=0.4
                        )
                        previous_projected_detections = smoothed_projected_detections
                        
                        # Draw projected detections on RGB camera view
                        display_rgb = draw_projected_detections(display_rgb, smoothed_projected_detections)
                    
                    # Print detection info with depth statistics
                    if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                        print(f"Frame {frame_count}: {len(detections)} objects detected, {len(projected_detections)} projected")
                        for i, det in enumerate(detections):
                            if i < len(projected_detections) and 'depth_info' in projected_detections[i]:
                                depth_info = projected_detections[i]['depth_info']
                                print(f"  - {det['class_name']}: {det['confidence']:.2f} | "
                                      f"Depth: {depth_info['median_depth']:.2f}m | "
                                      f"Points: {depth_info['num_points']} | "
                                      f"Std: {depth_info['depth_std']:.3f}")
                            else:
                                print(f"  - {det['class_name']}: {det['confidence']:.2f}")
            elif model is not None:
                # Use previous detections for frames where we don't run detection
                if previous_projected_detections:
                    display_rgb = draw_projected_detections(display_rgb, previous_projected_detections)
            
            # Add info overlay to both cameras
            info_text = [
                f"Frame: {frame_count}",
                f"YOLO: {'Active' if model is not None else 'Disabled'}",
                f"Projection: {'Active' if model is not None else 'Disabled'}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(display_pc, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_rgb, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add camera labels
            cv2.putText(display_pc, "POINT CLOUD CAMERA (Source)", (10, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_rgb, "RGB CAMERA (Projection Target)", (10, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(display_pc, "Press 'q' to quit, 's' to save", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Show both camera views
            cv2.imshow('Point Cloud Camera (YOLO Source)', display_pc)
            cv2.imshow('RGB Camera (Projection Target)', display_rgb)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename_pc = f"yolo_pc_frame_{frame_count}.jpg"
                filename_rgb = f"yolo_rgb_frame_{frame_count}.jpg"
                cv2.imwrite(filename_pc, display_pc)
                cv2.imwrite(filename_rgb, display_rgb)
                print(f"💾 Frames saved as {filename_pc} and {filename_rgb}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline_pc.stop()
        pipeline_rgb.stop()
        cv2.destroyAllWindows()
        print("\n✅ Test completed!")

if __name__ == "__main__":
    main()
