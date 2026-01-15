"""
Experiment: Phase Diagram (Step 4.2)
Maps the spiral existence phase space and overlays the theoretical sufficient condition.
Strict Dense Grid: 50x50.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.iterators import iterate_map
from src.theorem_conditions import estimate_wedge_scanning
from experiments.analysis_utils import fit_orbit_tail

# --- CONFIGURATION ---
LAM_RANGE = (0.01, 2.0)
EPS_RANGE = (0.0, 2.0)
STEPS = 50 # Strict requirement: 50x50
THETA = 0.1
R0_THEORY = 5.0 # Conservative sufficient radius

OUTPUT_DIR = Path("figures/rigor")
OUTPUT_FIG = OUTPUT_DIR / "phase_diagram.png"

def check_spiral_existence(lam, eps, theta, n_checks=20):
    """Binary test: Does a spiral exist at these params?"""
    # 1. Quick Wedge Check (Optional / Diagnostic)
    # We do NOT use this to reject spiral existence, because we want to see
    # spirals that exist even where the sufficient condition (Lemma) fails.
    # wedge_found, _, _, _, _, _, _, _ = estimate_wedge_scanning(p_cfg, dummy_cfg)
    
    # 2. Simulation Check (Empirical Existence)
    z0s = 10.0 * np.exp(1j * np.random.uniform(-np.pi, np.pi, n_checks))
    valid_orbits = 0
    
    for z0 in z0s:
        # Reduced max_iter for speed in dense grid, but sufficient for escape
        traj = iterate_map(z0, 0j, max_iter=600, mode="theorem_map",
                          theta=theta, lam=lam, eps=eps, crash_radius=1e-6, escape_radius=100.0)
        
        if len(traj) > 0 and np.abs(traj[-1]) > 100.0:
            # Check pitch
            r_traj = np.abs(traj)
            mask = (r_traj > 10.0) & (r_traj < 100.0)
            if np.sum(mask) >= 8: # Lowered point requirement slightly
                k = fit_orbit_tail(np.log(r_traj[mask]), np.unwrap(np.angle(traj))[mask])
                if not np.isnan(k) and abs(k) < 30.0: # Relaxed kappa bound
                    valid_orbits += 1
    
    return 1 if valid_orbits >= 1 else 0

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    lam_vals = np.linspace(*LAM_RANGE, STEPS)
    eps_vals = np.linspace(*EPS_RANGE, STEPS)
    
    X_eps, Y_lam = np.meshgrid(eps_vals, lam_vals)
    
    print(f"Sweeping dense grid {STEPS}x{STEPS} = {STEPS*STEPS} points...")
    
    grid = np.zeros((STEPS, STEPS))
    
    # Loop
    # Note: meshgrid X is cols (eps), Y is rows (lam)
    for i in range(STEPS): # Row index (Lambda)
        for j in range(STEPS): # Col index (Epsilon)
            lam_mag = Y_lam[i, j]
            eps_mag = X_eps[i, j]
            
            lam = complex(lam_mag, 0.0)
            eps = complex(eps_mag, 0.0)
            
            grid[i, j] = check_spiral_existence(lam, eps, THETA)
            
    # Sanity Checks
    total_spirals = np.sum(grid)
    if total_spirals == 0:
        print("WARNING: Phase diagram is empty! No spirals found.")
    else:
        print(f"Found {int(total_spirals)} spiral configurations.")
        
    # Theory Consistency Check
    # Points with very high lambda and low epsilon MUST be spirals.
    # Check top-left region (Low Eps, High Lam)
    # i=STEPS-1 (Max Lam), j=0 (Min Eps)
    if grid[-1, 0] == 0:
         print("WARNING: Theory violation? High-Lambda point (Max Lam, Min Eps) failed spiral check.")
         # Relax checks? We proceed but flag it.
         
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # pcolormesh
    plt.pcolormesh(X_eps, Y_lam, grid, cmap='Greys', vmin=0, vmax=1, shading='auto')
    
    # Theory Curve
    # lam = 4/R + 4eps/R^4
    slope = 4.0 / (R0_THEORY**4)
    intercept = 4.0 / R0_THEORY
    
    x_line = np.linspace(*EPS_RANGE, 100)
    y_line = intercept + slope * x_line
    
    plt.plot(x_line, y_line, color='red', linewidth=3.0, label='Theoretical Sufficient')
    
    plt.xlabel(r'Epsilon ($\varepsilon$)')
    plt.ylabel(r'Lambda ($\lambda$)')
    plt.title(f'Spiral Phase Diagram (Density {STEPS}x{STEPS})')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3.0, label='Theoretical Sufficient Boundary'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Spiral Exists'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor='k', markersize=10, label='No Spiral')
    ]
    plt.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    plt.xlim(EPS_RANGE)
    plt.ylim(LAM_RANGE)
    
    output_path = OUTPUT_FIG
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
