#!/usr/bin/env python3
"""
Generate high-quality showcase spirals for README and documentation.

Reads configs/showcase_spirals.json and generates each spiral image
with the specified parameters, saving to figures/best/.
"""

import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controlled_map import ControlledConfig, iterate_controlled


def generate_spiral(config: dict, output_path: Path) -> None:
    """Generate a single spiral image from configuration."""
    
    # Extract parameters
    grid_size = config.get('grid_size', 1024)
    max_iter = config.get('max_iter', 400)
    escape_radius = config.get('escape_radius', 80.0)
    
    # Create ControlledConfig
    ctrl_cfg = ControlledConfig(
        core="quadratic",
        radial_mode=config['radial_mode'],
        delta_r=config.get('delta_r', 0.005),
        alpha=config.get('alpha', 1.02),
        omega=config['omega'],
        phase_eps=config['phase_eps'],
        k_arms=config['k_arms'],
        c_bias=complex(-0.12, 0.74),
        escape_radius=escape_radius
    )
    
    # Generate grid
    print(f"  Generating {grid_size}x{grid_size} grid...")
    
    # Create coordinate grid
    xmin, xmax = -2.5, 2.5
    ymin, ymax = -2.5, 2.5
    
    x = np.linspace(xmin, xmax, grid_size)
    y = np.linspace(ymin, ymax, grid_size)
    X, Y = np.meshgrid(x, y)
    Z0 = X + 1j * Y
    
    # Iterate each point
    iteration_counts = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    for i in range(grid_size):
        if i % 100 == 0:
            print(f"    Row {i}/{grid_size}")
        for j in range(grid_size):
            z0 = Z0[i, j]
            traj = iterate_controlled(z0, 0.0, ctrl_cfg, max_iter)
            iteration_counts[i, j] = len(traj)
    
    # Normalize and create image
    print(f"  Rendering image...")
    
    # Log scale for better visualization
    img_data = np.log1p(iteration_counts)
    img_data = (img_data / img_data.max() * 255).astype(np.uint8)
    
    # Create RGB image with color mapping
    # Use a nice color gradient
    from matplotlib import cm
    colormap = cm.get_cmap('twilight')
    img_colored = colormap(img_data / 255.0)
    img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Save image
    img = Image.fromarray(img_rgb)
    img.save(output_path)
    print(f"  ✓ Saved to {output_path}")


def main():
    """Main entry point."""
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'showcase_spirals.json'
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    spirals = config['spirals']
    output_dir = Path(__file__).parent.parent / 'figures' / 'best'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(spirals)} showcase spirals...")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate each spiral
    for idx, spiral_config in enumerate(spirals, 1):
        name = spiral_config['name']
        description = spiral_config['description']
        
        print(f"[{idx}/{len(spirals)}] {name}")
        print(f"  {description}")
        
        output_path = output_dir / f"{name}.png"
        
        try:
            generate_spiral(spiral_config, output_path)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
        
        print()
    
    print("=" * 60)
    print("✓ Showcase spiral generation complete!")
    print(f"✓ Generated {len(spirals)} images in {output_dir}")
    print()
    print("Parameter Summary:")
    print("-" * 60)
    print(f"{'Name':<25} {'Arms':<6} {'Mode':<10} {'δr/α':<8} {'ω':<8}")
    print("-" * 60)
    
    for spiral_config in spirals:
        name = spiral_config['name']
        k_arms = spiral_config['k_arms']
        mode = spiral_config['radial_mode']
        
        if mode == 'additive':
            param = f"{spiral_config['delta_r']:.3f}"
        else:
            param = f"{spiral_config['alpha']:.2f}"
        
        omega = f"{spiral_config['omega']:.3f}"
        
        print(f"{name:<25} {k_arms:<6} {mode:<10} {param:<8} {omega:<8}")
    
    print("-" * 60)


if __name__ == '__main__':
    main()
