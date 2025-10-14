import cv2
import numpy as np
import os

# --- Configuration ---
# We will design the board to fit a landscape A4 paper.
# A4 paper dimensions in millimeters
A4_PAPER_WIDTH_MM = 297
A4_PAPER_HEIGHT_MM = 210

# Margin from the edge of the paper in millimeters
MARGIN_MM = 15

# Number of squares for the board. A good number for a balance of
# robustness and corner count is 11x8.
SQUARES_X = 7
SQUARES_Y = 5

# ArUco dictionary to use. DICT_5X5_250 is a common and robust choice.
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# Output filenames
IMAGE_FILE = "charuco_board_A4.png"
CONFIG_FILE = "charuco_board_config.npz"

# --- Calculations ---

# 1. Calculate the maximum size for each square to fit the paper with margins
printable_width_mm = A4_PAPER_WIDTH_MM - (2 * MARGIN_MM)
printable_height_mm = A4_PAPER_HEIGHT_MM - (2 * MARGIN_MM)

# To maintain the aspect ratio, the limiting dimension determines the square size
square_size_mm_width = printable_width_mm / SQUARES_X
square_size_mm_height = printable_height_mm / SQUARES_Y

# The actual square size must be the smaller of the two to fit
SQUARE_SIZE_MM = min(square_size_mm_width, square_size_mm_height)

# 2. Set the marker size as a ratio of the square size (a good ratio is 0.5 to 0.7)
MARKER_SIZE_MM = SQUARE_SIZE_MM * 0.6
SQUARE_SIZE_MM = 34.0
MARKER_SIZE_MM = 21.0
print("--- Board Configuration ---")
print(f"Paper Size: {A4_PAPER_WIDTH_MM}mm x {A4_PAPER_HEIGHT_MM}mm (Landscape)")
print(f"Board Dimensions: {SQUARES_X}x{SQUARES_Y} squares")
print(f"Calculated Square Size: {SQUARE_SIZE_MM:.2f} mm")
print(f"Calculated Marker Size: {MARKER_SIZE_MM:.2f} mm")
print("---------------------------")


# 3. Create the ChArUco board object in memory
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE_MM,
    MARKER_SIZE_MM,
    ARUCO_DICT)

# 4. Generate the image for printing. We need to convert mm to pixels.
# We'll use a high DPI (Dots Per Inch) for good print quality. 300 DPI is standard.
DPI = 300
INCH_TO_MM = 25.4
pixels_per_mm = DPI / INCH_TO_MM

# Get the total board dimensions in pixels
board_width_px = int(SQUARES_X * SQUARE_SIZE_MM * pixels_per_mm)
board_height_px = int(SQUARES_Y * SQUARE_SIZE_MM * pixels_per_mm)

# --- THIS IS THE CORRECTED LINE for OpenCV 4.8+ ---
# The 'draw()' method was renamed to 'generateImage()'
board_image = board.generateImage((board_width_px, board_height_px))
# ----------------------------------------------------

# 5. Create a blank A4-sized canvas and place the board in the center
canvas_width_px = int(A4_PAPER_WIDTH_MM * pixels_per_mm)
canvas_height_px = int(A4_PAPER_HEIGHT_MM * pixels_per_mm)

# Create a white canvas (255)
canvas = np.ones((canvas_height_px, canvas_width_px), dtype=np.uint8) * 255

# Calculate top-left corner position to center the board
x_offset = (canvas_width_px - board_width_px) // 2
y_offset = (canvas_height_px - board_height_px) // 2

# Paste the board onto the canvas
canvas[y_offset:y_offset + board_height_px, x_offset:x_offset + board_width_px] = board_image

# --- Save the Files ---

# Save the final image to be printed
try:
    cv2.imwrite(IMAGE_FILE, canvas)
    print(f"\nSuccessfully created printable board: '{IMAGE_FILE}'")
    print("IMPORTANT: When printing, ensure you use 'Actual Size' or '100% Scale'.")
except Exception as e:
    print(f"\nError: Could not save image file. {e}")

# Save the configuration data needed for calibration
aruco_dict_id = cv2.aruco.DICT_5X5_250

try:
    np.savez(CONFIG_FILE,
             squares_x=SQUARES_X,
             squares_y=SQUARES_Y,
             square_size_mm=SQUARE_SIZE_MM,
             marker_size_mm=MARKER_SIZE_MM,
             aruco_dict_id=aruco_dict_id)
    print(f"Successfully saved configuration: '{CONFIG_FILE}'")
except Exception as e:
    print(f"Error: Could not save config file. {e}")