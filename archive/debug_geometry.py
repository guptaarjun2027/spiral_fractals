import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Add repo root to path
repo_root = Path(os.getcwd())
sys.path.append(str(repo_root))

from src.geometry import (
    load_image_gray, binary_mask, extract_boundary, skeletonize_boundary,
    estimate_center, detect_arms_by_angle, fit_log_spiral_for_arm,
    polar_coords_from_center
)

image_path = repo_root / 'figures' / 'best' / 'tight_2arm_additive.png'
print(f"Analyzing {image_path}")

img = load_image_gray(image_path)
mask = binary_mask(img)
boundary = extract_boundary(mask)
skel = skeletonize_boundary(boundary)
center = estimate_center(mask)
print(f"Center: {center}")

# Debug detect_arms_by_angle internals
y_idxs, x_idxs = np.where(skel)
print(f"Skeleton pixels: {len(y_idxs)}")

points = np.column_stack((y_idxs, x_idxs))
r, theta = polar_coords_from_center(points, center)
print(f"R range: {r.min():.2f} to {r.max():.2f}")

r_max = np.max(r)
mask_r = (r >= 0.1 * r_max) & (r <= 0.9 * r_max)
print(f"Pixels in band: {np.sum(mask_r)}")

theta_band = theta[mask_r]
theta_band_2pi = np.mod(theta_band, 2 * np.pi)

hist, bin_edges = np.histogram(theta_band_2pi, bins=72, range=(0, 2 * np.pi))
print(f"Histogram max: {hist.max()}")

height_thresh = 0.05 * np.max(hist)
peaks, _ = signal.find_peaks(hist, height=height_thresh, distance=72//12)
print(f"Peaks found: {len(peaks)} at indices {peaks}")

arms = detect_arms_by_angle(skel, center)
print(f"Arms detected: {len(arms)}")

for i, arm in enumerate(arms):
    print(f"Arm {i}: {len(arm)} pixels")
    res = fit_log_spiral_for_arm(arm, center)
    print(f"  Fit: {res}")
