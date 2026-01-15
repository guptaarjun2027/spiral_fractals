"""
Experiment: Convergence Analysis (Step 4.1)
Demonstrates numerical stability of spiral exponents (kappa, beta) under changing resolution.
STRICT TIERS as requested:
LOW:  grid=200,  iters=500,  R_esc=50
MED:  grid=500,  iters=1500, R_esc=100
HIGH: grid=1000, iters=4000, R_esc=200
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from src.iterators import iterate_map
from experiments.analysis_utils import fit_orbit_tail, estimate_pitch_stats, detect_scaling_window, bootstrap_scaling_beta

# --- CONFIGURATION ---
# "grid" likely means n_angles for sampling density. We will scale n_radii proportionally or keep fixed reasonable.
TIERS = {
    'LOW':  {'n_angles': 200,  'n_radii': 20, 'max_iter': 500,  'esc_rad': 50.0},
    'MED':  {'n_angles': 500,  'n_radii': 35, 'max_iter': 1500, 'esc_rad': 100.0},
    'HIGH': {'n_angles': 1000, 'n_radii': 50, 'max_iter': 4000, 'esc_rad': 200.0}
}

# Test Parameters (Representative Spiral - Slower Growth for sampling)
# start r=10. Fixed point ~ 1/|lam|. We want FP > 10 so it escapes? 
# No, if FP > 10, then below FP it might shrink? 
# Map is F(z) ~ lam*z^2. If |lam|*r^2 > r => |lam|*r > 1.
# At r=10, we need |lam| > 0.1.
# If |lam|=0.15, r=10 -> 15 -> 33 -> 160. Still fast.
# Try |lam|=0.11. 10 -> 11 -> 13.3 -> 19 -> 40 -> 170. ~5 steps.
# We need 20 steps? 
# Let's use Controlled Map? No, must use Theorem Map.
# We need to start closer to the repeller?
# Or reduce fit_min? 
# Let's set lam=0.105 and fit_min=5.0?
# And esc_rad is 50.
# Let's try lam=0.105.
PARAMS = {
    'theta': -0.2,
    'lam': 0.101 + 0.0j, # Very slow escape near r=10
    'eps': 0.01 + 0.0j, 
    'wedge_eta': 0.1,
    'crash_radius': 1e-6
}

OUTPUT_DIR = Path("figures/rigor")
RESULTS_FILE = Path("results/convergence_summary.csv")

def run_tier(tier_name, cfg, params):
    """Runs analysis for a single tier."""
    print(f"Running Tier: {tier_name}...")
    
    n_angles = cfg['n_angles']
    n_radii = cfg['n_radii']
    max_iter = cfg['max_iter']
    esc_rad = cfg['esc_rad']
    
    # Radii grid: range scaling with escape radius to ensure we cover the scaling window
    fit_min = 5.0 
    fit_max = esc_rad * 0.95 # stay within escape radius logic
    radii_sim = np.logspace(np.log10(fit_min), np.log10(fit_max), n_radii)
    
    orbit_kappas = []
    rhos = [] 
    n_escaped_list = []
    
    for r in radii_sim:
        z0s = r * np.exp(1j * np.random.uniform(-np.pi, np.pi, n_angles))
        escaped = 0
        
        for z0 in z0s:
            traj = iterate_map(
                z0, 0j, max_iter, mode="theorem_map",
                theta=params['theta'], lam=params['lam'], eps=params['eps'],
                crash_radius=params['crash_radius'], escape_radius=esc_rad
            )
            
            # Escape Check
            if len(traj) > 0 and np.abs(traj[-1]) > esc_rad:
                escaped += 1
                
                # Pitch Extraction (Tail)
                r_traj = np.abs(traj)
                # Tail window: consistent logical window (e.g. 5 to fit_max)
                # But actual trajectory goes to R_esc.
                mask = (r_traj >= 5.0) & (r_traj <= esc_rad)
                if np.sum(mask) >= 5: # relaxed count for Low tier 
                    k = fit_orbit_tail(
                        np.log(r_traj[mask]), 
                        np.unwrap(np.angle(traj))[mask], 
                        tail_fraction=0.4
                    )
                    if not np.isnan(k) and abs(k) < 20.0:
                        orbit_kappas.append(k)
                        
        rhos.append(escaped / n_angles)
        n_escaped_list.append(escaped)
        
    # Stats
    k_med, k_lo, k_hi = estimate_pitch_stats(orbit_kappas, bootstrap_B=200)
    
    y_vals = 1.0 - np.array(rhos)
    beta, win, _ = detect_scaling_window(radii_sim, y_vals, min_points=5)
    b_lo, b_hi = np.nan, np.nan
    if not np.isnan(beta) and win is not None:
         mask_y = (y_vals >= 1e-4) & (y_vals <= 0.3)
         if np.sum(mask_y) > 0:
             b_lo, b_hi = bootstrap_scaling_beta(n_angles, n_escaped_list, radii_sim, mask_y, win, bootstrap_B=200)
         
    return {
        'tier': tier_name,
        'grid': n_angles,
        'iters': max_iter,
        'R_escape': esc_rad,
        'kappa_med': k_med, 'kappa_lo': k_lo, 'kappa_hi': k_hi,
        'beta_med': beta, 'beta_lo': b_lo, 'beta_hi': b_hi
    }

def plot_convergence(df):
    """Robust plotting logic."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tier_order = ['LOW', 'MED', 'HIGH']
    
    # 1. Kappa
    ax = axes[0]
    valid_data_points = 0
    
    for i, tier in enumerate(tier_order):
        row = df[df['tier'] == tier]
        if row.empty: continue
        row = row.iloc[0]
        
        val = row['kappa_med']
        lo, hi = row['kappa_lo'], row['kappa_hi']
        
        if np.isnan(val) or np.isnan(lo) or np.isnan(hi):
            ax.text(i, 0, "Rejected", ha='center', va='bottom', color='red', rotation=90)
            continue
            
        valid_data_points += 1
        yerr = [[val - lo], [hi - val]]
        ax.errorbar(i, val, yerr=yerr, fmt='o', color='blue', capsize=5, lw=2)
        ax.plot(i, val, 'bo', markersize=8)
        
    ax.set_xticks(range(len(tier_order)))
    ax.set_xticklabels(tier_order)
    ax.set_title(r"Pitch ($\kappa$) Convergence")
    ax.set_ylabel("Kappa")
    ax.grid(True, alpha=0.3)
    
    # 2. Beta
    ax = axes[1]
    
    for i, tier in enumerate(tier_order):
        row = df[df['tier'] == tier]
        if row.empty: continue
        row = row.iloc[0]
        
        val = row['beta_med']
        lo, hi = row['beta_lo'], row['beta_hi']
        
        if np.isnan(val) or np.isnan(lo) or np.isnan(hi):
            ax.text(i, 0, "Rejected", ha='center', va='bottom', color='red', rotation=90)
            continue
            
        yerr = [[val - lo], [hi - val]]
        ax.errorbar(i, val, yerr=yerr, fmt='s', color='green', capsize=5, lw=2)
        ax.plot(i, val, 'gs', markersize=8)
        
    ax.set_xticks(range(len(tier_order)))
    ax.set_xticklabels(tier_order)
    ax.set_title(r"Scaling ($\beta$) Convergence")
    ax.set_ylabel("Beta")
    ax.grid(True, alpha=0.3)
    
    # Overall Annotation
    plt.suptitle("Numerical Convergence Check (Strict Tiers)", fontsize=14)
    if valid_data_points == 0:
        plt.figtext(0.5, 0.5, "WARNING: All tiers rejected / Did not converge", 
                   ha='center', va='center', color='red', fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
    
    plot_path = OUTPUT_DIR / "convergence_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    for tier in ['LOW', 'MED', 'HIGH']:
        res = run_tier(tier, TIERS[tier], PARAMS)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    # Enforce column order
    cols = ['tier', 'grid', 'iters', 'R_escape', 'kappa_med', 'kappa_lo', 'kappa_hi', 'beta_med', 'beta_lo', 'beta_hi']
    df = df[cols]
    
    df.to_csv(RESULTS_FILE, index=False)
    print(f"Saved results to {RESULTS_FILE}")
    
    plot_convergence(df)

if __name__ == "__main__":
    main()
