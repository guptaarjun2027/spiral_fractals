"""Debug script with correct parameters matching sweep defaults."""

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.controlled_map import iterate_controlled, ControlledConfig
from src.render import render_grid
from PIL import Image

# Use parameters matching the current sweep defaults
escape_radius = 8.0  # NEW: much lower
max_iter = 600       # NEW: higher
delta_r = 0.015      # from small sweep range
phase_eps = 0.02

cfg = ControlledConfig(
    radial_mode="additive",
    delta_r=delta_r,
    alpha=1.02,
    omega=0.3 * np.pi,
    phase_eps=phase_eps,
    k_arms=3,
    escape_radius=escape_radius
)

print("Testing with CORRECTED parameters:")
print(f"  escape_radius: {escape_radius}")
print(f"  max_iter: {max_iter}")
print(f"  delta_r: {delta_r}")
print(f"  phase_eps: {phase_eps}")
print()

# Test a few points
test_points = [
    complex(0.1, 0.1),
    complex(1.0, 0.0),
    complex(0.0, 1.0),
    complex(-0.5, 0.5),
]

c = complex(0, 0)

print("Testing individual points:")
for z0 in test_points:
    traj = iterate_controlled(z0=z0, c=c, cfg=cfg, max_iter=max_iter)
    final_r = np.abs(traj[-1]) if len(traj) > 0 else 0
    print(f"  z0={z0:>12}: len={len(traj):>3}, final_r={final_r:>8.2f}, escaped={'YES' if len(traj) < max_iter else 'NO '}")

print()

# Render a test grid
def step_fn(z0):
    traj = iterate_controlled(z0=z0, c=c, cfg=cfg, max_iter=max_iter)
    n = len(traj)
    last = traj[-1] if n > 0 else z0
    return n, last

print("Rendering 128x128 test grid...")
rgb = render_grid(
    step_fn=step_fn,
    xmin=-2.5, xmax=2.5,
    ymin=-2.5, ymax=2.5,
    width=128, height=128,
    max_iter=max_iter,
    escape_radius=escape_radius,
    color_mode="iters",
)

print(f"RGB stats: min={rgb.min()}, max={rgb.max()}, mean={rgb.mean():.2f}")

unique_vals = np.unique(rgb[:,:,0])
print(f"Unique gray values: {len(unique_vals)} (showing first 10: {unique_vals[:10]})")

if rgb.max() == 0:
    print("\n⚠️  STILL BLACK!")
elif rgb.max() == rgb.min():
    print(f"\n⚠️  UNIFORM (value={rgb.max()})")
else:
    print(f"\n✓ Has variation!")

# Save
test_path = ROOT / "test_controlled_fixed.png"
Image.fromarray(rgb).save(test_path)
print(f"\nSaved to: {test_path}")
