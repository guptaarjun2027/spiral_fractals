"""Debug script to test controlled map iteration."""

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.controlled_map import iterate_controlled, ControlledConfig
from src.render import render_grid
from PIL import Image

# Test configuration
cfg = ControlledConfig(
    radial_mode="additive",
    delta_r=0.006,
    alpha=1.02,
    omega=0.25 * np.pi,
    phase_eps=0.02,
    k_arms=3,
    escape_radius=80.0
)

print("Testing controlled map iteration...")
print(f"Config: {cfg}")

# Test a single point
z0 = complex(0.1, 0.1)
c = complex(0, 0)
max_iter = 350

print(f"\nTesting single point: z0={z0}, c={c}")
traj = iterate_controlled(z0=z0, c=c, cfg=cfg, max_iter=max_iter)
print(f"Trajectory length: {len(traj)}")
if len(traj) > 0:
    print(f"First few points: {traj[:5]}")
    print(f"Last point: {traj[-1]}")
    print(f"Last |z|: {np.abs(traj[-1])}")

# Test step function
def step_fn(z0):
    traj = iterate_controlled(z0=z0, c=c, cfg=cfg, max_iter=max_iter)
    n = len(traj)
    last = traj[-1] if n > 0 else z0
    return n, last

print(f"\nTesting step_fn wrapper:")
n, last = step_fn(z0)
print(f"Returned: n={n}, last={last}")

# Test grid rendering
print("\nTesting grid rendering...")
print("Rendering 64x64 test grid...")

rgb = render_grid(
    step_fn=step_fn,
    xmin=-2.5, xmax=2.5,
    ymin=-2.5, ymax=2.5,
    width=64, height=64,
    max_iter=max_iter,
    escape_radius=cfg.escape_radius,
    color_mode="iters",
)

print(f"RGB shape: {rgb.shape}")
print(f"RGB dtype: {rgb.dtype}")
print(f"RGB min: {rgb.min()}, max: {rgb.max()}")
print(f"RGB mean: {rgb.mean():.2f}")

# Check if image is all black
if rgb.max() == 0:
    print("\n⚠️  WARNING: Image is completely black!")
elif rgb.max() == rgb.min():
    print(f"\n⚠️  WARNING: Image is uniform (value={rgb.max()})!")
else:
    print(f"\n✓ Image has variation (range: {rgb.min()} to {rgb.max()})")

# Save test image
test_path = ROOT / "test_controlled_debug.png"
Image.fromarray(rgb).save(test_path)
print(f"\nSaved test image to: {test_path}")
print("Open it to visually inspect.")

# Additional diagnostics: sample the grid
print("\n--- Grid Sampling Diagnostics ---")
test_points = [
    complex(-1, 0),
    complex(0, 1),
    complex(1, 0),
    complex(0, -1),
    complex(0.5, 0.5),
]

for pt in test_points:
    traj = iterate_controlled(z0=pt, c=c, cfg=cfg, max_iter=max_iter)
    print(f"z0={pt:>12}: len={len(traj):>3}, final_r={np.abs(traj[-1]) if len(traj) > 0 else 0:>8.2f}")
