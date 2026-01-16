#!/usr/bin/env python
"""
Stability experiment script.
Perturbs parameters around clean spirals to see how geometry changes.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.controlled_map import iterate_controlled, ControlledConfig
from src.geometry import analyze_spiral_image
# Try to import render_grid, fallback if needed
try:
    from src.render import render_grid
except ImportError:
    from src.engine import render_grid

def get_baseline_spirals():
    """
    Load metrics and pick 3 clean spirals.
    Returns list of dicts with params.
    """
    metrics_path = ROOT / "results" / "spiral_metrics_labeled.csv"
    if not metrics_path.exists():
        metrics_path = ROOT / "results" / "spiral_metrics.csv"
    
    if not metrics_path.exists():
        print("No metrics found. Using default baselines.")
        return get_default_baselines()
        
    df = pd.read_csv(metrics_path)
    
    # Filter for clean spirals
    if "quality_label" in df.columns:
        clean = df[df["quality_label"] == "clean"]
    else:
        # Heuristic
        clean = df[
            (df["arm_count"].isin([2, 3, 4])) & 
            (df["r2_mean"] >= 0.8)
        ]
        
    if len(clean) < 3:
        print(f"Only found {len(clean)} clean spirals. Using defaults.")
        return get_default_baselines()
        
    # We need parameters. Check if we have parameter file
    sweep_path = ROOT / "results" / "spiral_sweep.csv"
    if not sweep_path.exists():
        print("No sweep parameters found. Using defaults.")
        return get_default_baselines()
        
    sweep_df = pd.read_csv(sweep_path)
    
    # Join on image/id if possible. 
    # Assuming image column in metrics matches image_path in sweep or filename matches
    # Let's try to match by filename
    
    baselines = []
    # Pick 3 distinct ones if possible
    selected = clean.sample(min(3, len(clean)), random_state=42)
    
    for _, row in selected.iterrows():
        img_name = Path(row["image"]).name
        # Find in sweep_df
        # sweep_df image_path might be relative
        match = sweep_df[sweep_df["image_path"].apply(lambda x: Path(x).name == img_name)]
        
        if not match.empty:
            p = match.iloc[0]
            # Only support controlled map for now as we need those params
            if p["map_type"] == "controlled":
                # Handle missing k_arms in older CSVs
                if "k_arms" in p:
                    k = int(p["k_arms"])
                else:
                    # Fallback to arm_count from metrics if available, else default 3
                    k = int(row["arm_count"]) if "arm_count" in row and not pd.isna(row["arm_count"]) else 3
                    
                baselines.append({
                    "id": p["id"],
                    "delta_r": p["delta_r"],
                    "omega": p["omega"],
                    "phase_eps": p["phase_eps"],
                    "alpha": p["alpha"],
                    "k_arms": k,
                    "radial_mode": p["radial_mode"],
                    "escape_radius": p["escape_radius"],
                    "max_iter": int(p["max_iter"])
                })
    
    if len(baselines) < 3:
        print("Could not match enough clean spirals to parameters. Adding defaults.")
        defaults = get_default_baselines()
        baselines.extend(defaults[:3-len(baselines)])
        
    return baselines

def get_default_baselines():
    """Hardcoded baselines based on configs/controlled_best.json style."""
    return [
        {
            "id": "default_3arm",
            "delta_r": 0.006,
            "omega": 0.785, # pi/4
            "phase_eps": 0.01,
            "alpha": float("nan"),
            "k_arms": 3,
            "radial_mode": "additive",
            "escape_radius": 80.0,
            "max_iter": 350
        },
        {
            "id": "default_2arm",
            "delta_r": 0.015,
            "omega": 0.628, # pi/5
            "phase_eps": 0.00,
            "alpha": float("nan"),
            "k_arms": 2,
            "radial_mode": "additive",
            "escape_radius": 80.0,
            "max_iter": 350
        },
        {
            "id": "default_4arm",
            "delta_r": 0.003,
            "omega": 0.942, # 0.3*pi
            "phase_eps": 0.02,
            "alpha": float("nan"),
            "k_arms": 4,
            "radial_mode": "additive",
            "escape_radius": 80.0,
            "max_iter": 350
        }
    ]

def generate_image(params, out_path):
    """Generate image using controlled map."""
    cfg = ControlledConfig(
        core="quadratic",
        radial_mode=params["radial_mode"],
        delta_r=params["delta_r"],
        alpha=params["alpha"] if not np.isnan(params["alpha"]) else 1.02,
        omega=params["omega"],
        phase_eps=params["phase_eps"],
        k_arms=params["k_arms"],
        escape_radius=params["escape_radius"]
    )
    
    # c=0 for controlled map usually
    c = 0+0j
    
    def step_fn(z0):
        traj = iterate_controlled(z0=z0, c=c, cfg=cfg, max_iter=params["max_iter"])
        n = len(traj)
        last = traj[-1] if n > 0 else z0
        return n, last

    rgb = render_grid(
        step_fn=step_fn,
        xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
        width=256, height=256,
        max_iter=params["max_iter"],
        escape_radius=params["escape_radius"],
        color_mode="iters",
    )
    
    Image.fromarray(rgb).save(str(out_path))

def main():
    print("Starting Stability Experiment...")
    
    # 1. Choose baselines
    baselines = get_baseline_spirals()
    print(f"Selected {len(baselines)} baseline spirals.")
    
    results = []
    out_dir = ROOT / "figures" / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Perturb and Generate
    for i, base in enumerate(baselines):
        print(f"\nProcessing baseline {i+1}/{len(baselines)} (ID: {base.get('id', 'unknown')})")
        
        # Perturbations
        # delta_r' = delta_r * (1 + eps)
        # omega' = omega * (1 + eps_w)
        
        eps_vals = [-0.05, -0.02, 0, 0.02, 0.05]
        eps_w_vals = [-0.05, 0, 0.05]
        
        for eps in eps_vals:
            for eps_w in eps_w_vals:
                # Create perturbed params
                p = base.copy()
                
                # Handle nan delta_r (if power mode)
                if not np.isnan(p["delta_r"]):
                    p["delta_r"] = base["delta_r"] * (1.0 + eps)
                
                p["omega"] = base["omega"] * (1.0 + eps_w)
                
                # Generate image
                fname = f"stability_base{i}_dr{p['delta_r']:.4f}_om{p['omega']:.3f}.png"
                img_path = out_dir / fname
                
                if not img_path.exists():
                    generate_image(p, img_path)
                
                # Analyze
                try:
                    metrics = analyze_spiral_image(img_path)
                    
                    res_row = {
                        "image": str(img_path.relative_to(ROOT)),
                        "base_id": base.get("id", f"base_{i}"),
                        "base_idx": i,
                        "delta_r": p["delta_r"],
                        "omega": p["omega"],
                        "phase_eps": p["phase_eps"],
                        "eps_dr": eps,
                        "eps_om": eps_w,
                        "arm_count": metrics["arm_count"],
                        "b_mean": metrics["b_mean"],
                        "r2_mean": metrics["r2_mean"],
                        "fractal_dimension": metrics["fractal_dimension"]
                    }
                    results.append(res_row)
                except Exception as e:
                    print(f"Error analyzing {fname}: {e}")

    # 3. Save Results
    df = pd.DataFrame(results)
    csv_path = ROOT / "results" / "stability_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")
    
    # 4. Summary Plots
    print("Generating summary plots...")
    
    for i in range(len(baselines)):
        subset = df[df["base_idx"] == i]
        if subset.empty:
            continue
            
        # Filter to only delta_r changes (eps_om == 0) for clean line plots
        dr_subset = subset[subset["eps_om"] == 0].sort_values("delta_r")
        
        if not dr_subset.empty and not dr_subset["delta_r"].isna().all():
            # Plot delta_r vs fractal_dimension
            plt.figure(figsize=(8, 5))
            plt.plot(dr_subset["delta_r"], dr_subset["fractal_dimension"], 'o-')
            plt.title(f"Base {i}: Delta R vs Fractal Dimension")
            plt.xlabel("Delta R")
            plt.ylabel("Fractal Dimension")
            plt.grid(True)
            plt.savefig(out_dir / f"base{i}_delta_vs_dimension.png")
            plt.close()
            
            # Plot delta_r vs arm_count
            plt.figure(figsize=(8, 5))
            plt.plot(dr_subset["delta_r"], dr_subset["arm_count"], 'o-')
            plt.title(f"Base {i}: Delta R vs Arm Count")
            plt.xlabel("Delta R")
            plt.ylabel("Arm Count")
            plt.grid(True)
            plt.savefig(out_dir / f"base{i}_delta_vs_armcount.png")
            plt.close()
            
    print("Done.")

if __name__ == "__main__":
    main()
