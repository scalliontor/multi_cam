#!/usr/bin/env python3
"""
Setup script for RealSense Dual-Camera Calibration
Creates the necessary ChArUco board configuration file.
"""

import numpy as np
import cv2
import os

def create_charuco_config():
    """Create ChArUco board configuration file."""
    
    print("🎯 RealSense Dual-Camera Calibration Setup")
    print("=" * 50)
    
    # ChArUco board parameters - optimized for A4 paper
    squares_x = 7          # Number of squares in X direction
    squares_y = 5          # Number of squares in Y direction  
    square_size_mm = 34.0  # Square size in millimeters
    marker_size_mm = 21.0  # ArUco marker size in millimeters
    aruco_dict_id = 0      # DICT_4X4_50
    
    # Save configuration
    config_file = 'charuco_board_config.npz'
    np.savez(config_file,
             squares_x=squares_x,
             squares_y=squares_y,
             square_size_mm=square_size_mm,
             marker_size_mm=marker_size_mm,
             aruco_dict_id=aruco_dict_id)
    
    print(f"✅ ChArUco configuration saved to: {config_file}")
    print(f"   Board size: {squares_x}x{squares_y}")
    print(f"   Square size: {square_size_mm}mm")
    print(f"   Marker size: {marker_size_mm}mm")
    
    # Generate board image
    print("\n📋 Generating ChArUco board image...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_size_mm, marker_size_mm, aruco_dict)
    
    # Create image (A4 size at 300 DPI)
    img_size = (2480, 3508)  # A4 at 300 DPI
    board_img = board.generateImage(img_size)
    
    # Save board image
    board_file = 'charuco_board_A4.png'
    cv2.imwrite(board_file, board_img)
    print(f"✅ ChArUco board image saved to: {board_file}")
    print("   Print this image on A4 paper for calibration")
    
    return True

def main():
    """Main setup function."""
    try:
        create_charuco_config()
        print("\n🚀 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Print the charuco_board_A4.png on A4 paper")
        print("2. Update camera serial numbers in calculate_extrinsics.py")
        print("3. Run: python find_camera.py (to get camera serial numbers)")
        print("4. Run: python calculate_extrinsics.py (to calibrate)")
        print("5. Run: python check_inference.py (to validate)")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
