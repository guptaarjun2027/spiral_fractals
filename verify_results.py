import sys
import os
from pathlib import Path
import pandas as pd

# Add repo root to path
repo_root = Path(os.getcwd())
sys.path.append(str(repo_root))

from src.geometry import analyze_spiral_image

image_dir = repo_root / 'figures' / 'best'
sample_filenames = [
    'tight_2arm_additive.png',
    'loose_2arm_additive.png',
    'tight_3arm_additive.png',
    'loose_3arm_additive.png',
    'dense_4arm_additive.png'
]

results = []
for fname in sample_filenames:
    path = image_dir / fname
    if path.exists():
        metrics = analyze_spiral_image(path)
        metrics['image'] = fname
        results.append(metrics)

df = pd.DataFrame(results)
cols = ['image', 'arm_count', 'b_mean', 'r2_mean', 'fractal_dimension']
print(df[cols].to_string())
