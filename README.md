- use calculate_extrinsics.py to calibrate 3d-3d
- use check_stereo.py to verify

```python
np.savez(OUTPUT_FILE, 
        initial_rvec=rvec, 
        initial_tvec_mm=tvec_mm,
        initial_R=R,
        points_pc=points_pc,
        points_rgb=points_rgb)
```