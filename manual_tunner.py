# --- In manual_tuner.py ---
import numpy as np

EXTRINSICS_FILE = "extrinsics.npz"
with np.load(EXTRINSICS_FILE) as data:
    rvec_orig = data['rvec']
    tvec_orig = data['tvec']

print("--- Original Values ---")
print("rvec:\n", rvec_orig)
print("tvec:\n", tvec_orig)

# =================================================================
# === MANUAL TUNING AREA ===
# Change the values here and re-run the script to see the effect.
# =================================================================

# # We believe the rotation is good, so we will not change it.
# rvec = np.array([
#         [-1.10079845],
#         [-1.0303166 ],
#         [-1.20699798]
# ], dtype=np.float32)

# # Let's create a new tvec with our adjusted values
# tvec = np.array([
#     [45.71],      # Original was 42.71. Increased to move projection RIGHT.
#     [-605.93955112],    # Original was -606.93. Decreased to move projection UP.
#     [548.46737697]      # Leave Z the same for now.
# ], dtype=np.float32)

# =================================================================

print("\n--- Using Tuned Values ---")
print("rvec:\n", rvec_orig)
print("tvec:\n", tvec_orig)
# np.savez("extrinsics.npz", rvec=rvec, tvec=tvec)

# ... the rest of your check2.py code continues from here ...
# The code will now use the new 'rvec' and 'tvec' you defined.