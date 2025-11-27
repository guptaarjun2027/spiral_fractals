import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

def create_notebook():
    nb = new_notebook()
    
    cells = []
    
    # 1. Imports and Setup
    cells.append(new_markdown_cell("""# Geometry Extraction Sanity Checks (v2)

This notebook validates the **updated** image analysis pipeline for Week 2.
Key improvements:
- **Arm Detection**: Angle clustering (histogram peaks) instead of connected components.
- **Spiral Fitting**: Robust linear regression on unwrapped angles per arm.
"""))
    
    cells.append(new_code_cell(r"""import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, color
from scipy import signal

# Add repo root to path
current_dir = Path(os.getcwd())
if current_dir.name == 'notebooks':
    repo_root = current_dir.parent
else:
    repo_root = current_dir

sys.path.append(str(repo_root))

from src.geometry import (
    load_image_gray, binary_mask, extract_boundary, skeletonize_boundary,
    estimate_center, detect_arms_by_angle, fit_log_spiral_for_arm,
    box_count_fractal_dimension, analyze_spiral_image, polar_coords_from_center
)

# Setup plotting
plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline

print(f"Repo root: {repo_root}")
"""))

    # 2. Load sample images
    cells.append(new_markdown_cell("""## 1. Load Sample Images
We'll load the high-quality showcase spirals.
"""))
    
    cells.append(new_code_cell(r"""image_dir = repo_root / 'figures' / 'best'
sample_filenames = [
    'tight_2arm_additive.png',
    'loose_2arm_additive.png',
    'tight_3arm_additive.png',
    'loose_3arm_additive.png',
    'dense_4arm_additive.png'
]

images = {}
for fname in sample_filenames:
    path = image_dir / fname
    if path.exists():
        images[fname] = load_image_gray(path)
        print(f"Loaded {fname}")
    else:
        print(f"Warning: {fname} not found")

# Visualize
fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
if len(images) == 1: axes = [axes]
for ax, (fname, img) in zip(axes, images.items()):
    ax.imshow(img, cmap='gray')
    ax.set_title(fname, fontsize=8)
    ax.axis('off')
plt.show()
"""))

    # 3. Preprocessing (Mask & Skeleton)
    cells.append(new_markdown_cell("""## 2. Preprocessing
Generate binary masks and skeletons.
"""))
    
    cells.append(new_code_cell(r"""skeletons = {}
centers = {}

fig, axes = plt.subplots(len(images), 2, figsize=(8, 3*len(images)))

for i, (fname, img) in enumerate(images.items()):
    # Pipeline steps
    mask = binary_mask(img)
    boundary = extract_boundary(mask)
    skel = skeletonize_boundary(boundary)
    center = estimate_center(mask)
    
    skeletons[fname] = skel
    centers[fname] = center
    
    # Plot Mask
    axes[i, 0].imshow(mask, cmap='gray')
    axes[i, 0].set_title(f"{fname}\nMask")
    axes[i, 0].axis('off')
    
    # Plot Skeleton + Center
    axes[i, 1].imshow(skel, cmap='gray')
    axes[i, 1].plot(center[0], center[1], 'r+', markersize=10, label='Center')
    axes[i, 1].set_title("Skeleton")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
"""))

    # 4. Arm Detection (Angle Clustering)
    cells.append(new_markdown_cell("""## 3. Arm Detection (Angle Clustering)
We detect arms by looking for peaks in the angular histogram of skeleton pixels.
"""))
    
    cells.append(new_code_cell(r"""for fname, skel in skeletons.items():
    center = centers[fname]
    
    # 1. Get polar coords of skeleton
    y, x = np.where(skel)
    points = np.column_stack((y, x))
    r, theta = polar_coords_from_center(points, center)
    
    # 2. Histogram
    # Filter mid-radius for cleaner histogram
    r_max = r.max()
    mask_band = (r > 0.2 * r_max) & (r < 0.8 * r_max)
    theta_band = np.mod(theta[mask_band], 2*np.pi)
    
    hist, bin_edges = np.histogram(theta_band, bins=72, range=(0, 2*np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks (using same logic as src/geometry.py)
    peaks, _ = signal.find_peaks(hist, height=0.05*hist.max(), distance=72//12)
    
    # Detect arms using the function
    arms = detect_arms_by_angle(skel, center)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax1.plot(bin_centers, hist)
    ax1.plot(bin_centers[peaks], hist[peaks], "x", color='r')
    ax1.set_title(f"{fname}: {len(peaks)} Peaks Detected")
    ax1.set_xlabel("Theta (rad)")
    
    # Colored Arms
    ax2.imshow(images[fname], cmap='gray', alpha=0.5)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(arms)))
    for arm_idx, arm_pixels in enumerate(arms):
        ax2.scatter(arm_pixels[:, 1], arm_pixels[:, 0], s=1, color=colors[arm_idx], label=f"Arm {arm_idx}")
    ax2.set_title(f"Detected Arms: {len(arms)}")
    ax2.legend()
    ax2.axis('off')
    
    plt.show()
"""))

    # 5. Log-Spiral Fitting
    cells.append(new_markdown_cell("""## 4. Log-Spiral Fitting
Fit $ \ln(r) = b \\theta + \ln(a) $ to each detected arm.
"""))
    
    cells.append(new_code_cell(r"""for fname, skel in skeletons.items():
    center = centers[fname]
    arms = detect_arms_by_angle(skel, center)
    
    print(f"--- {fname} ---")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, arm_pixels in enumerate(arms):
        res = fit_log_spiral_for_arm(arm_pixels, center)
        
        if res['valid']:
            print(f"Arm {i}: b={res['b']:.3f}, R2={res['r2']:.3f}, n={res['n_points']}")
            
            # Plot data
            r, theta = polar_coords_from_center(arm_pixels, center)
            # Sort for plotting
            idx = np.argsort(r)
            r = r[idx]
            theta = theta[idx]
            theta_unwrap = np.unwrap(theta)
            
            ax.plot(theta_unwrap, np.log(r), '.', markersize=2, label=f"Arm {i} Data")
            
            # Plot fit
            # y = b * theta + intercept
            # intercept is ln(a)
            fit_y = res['b'] * theta_unwrap + res['intercept']
            
            ax.plot(theta_unwrap, fit_y, '-', alpha=0.5, label=f"Arm {i} Fit")
            
    ax.set_xlabel("Theta (unwrapped)")
    ax.set_ylabel("ln(r)")
    ax.set_title(f"{fname} Log-Spiral Fits")
    ax.legend()
    plt.show()
"""))

    # 6. Full Pipeline & Summary
    cells.append(new_markdown_cell("""## 5. Full Pipeline Summary
Run `analyze_spiral_image` on all samples and report metrics.
"""))
    
    cells.append(new_code_cell(r"""results = []

for fname in sample_filenames:
    path = image_dir / fname
    if not path.exists(): continue
        
    metrics = analyze_spiral_image(path)
    metrics['image'] = fname
    results.append(metrics)

df = pd.DataFrame(results)
cols = ['image', 'arm_count', 'b_mean', 'r2_mean', 'fractal_dimension']
display(df[cols])

print("\nDetailed Metrics:")
display(df)
"""))

    nb['cells'] = cells
    
    output_path = 'notebooks/geometry_sanity_checks.ipynb'
    with open(output_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Created notebook at {output_path}")

if __name__ == "__main__":
    create_notebook()
