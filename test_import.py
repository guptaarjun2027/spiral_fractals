import sys
import os
from pathlib import Path

# Add repo root to path
repo_root = Path(os.getcwd())
sys.path.append(str(repo_root))

try:
    from src.geometry import (
        load_image_gray, binary_mask, extract_boundary, skeletonize_boundary,
        estimate_center, detect_arms_by_angle, fit_log_spiral_for_arm,
        box_count_fractal_dimension, analyze_spiral_image, polar_coords_from_center
    )
    print("Import successful!")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
