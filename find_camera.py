import pyrealsense2 as rs

context = rs.context()
devices = context.query_devices()

if len(devices) < 2:
    print("Please connect two RealSense cameras.")
    exit()

for dev in devices:
    print(f"Device: {dev.get_info(rs.camera_info.name)}, "
          f"Serial Number: {dev.get_info(rs.camera_info.serial_number)}")