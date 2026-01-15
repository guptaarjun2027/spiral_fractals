"""
Experiment: Phase Diagram (Step 4.2)
Maps the spiral existence phase space and overlays the theoretical sufficient condition.

Uses growth-valid orbit logic and correct matrix orientation:
- Matrix phase[j, i] corresponds to (lam[j], eps[i])
- imshow with origin='lower' and extent=[eps_min, eps_max, lam_min, lam_max]
- Spiral classification: wedge_found AND kappa_estimated AND n_orbits_valid >= min_valid
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.iterators import iterate_map
from src.theorem_conditions import estimate_wedge_scanning
from experiments.analysis_utils import estimate_kappa_from_orbits_growth


def check_spiral_existence_growth(lam, eps, theta, config, orbit_cfg, pitch_cfg, wedge_cfg):
    """
    Check if a spiral exists at (lam, eps) using growth-valid logic.
    
    Returns:
        dict with keys: spiral_exists, wedge_found, wedge_valid_fraction,
                       kappa_hat, n_orbits_valid, escape_fraction, fail_reason
    """
    crash_radius = float(config.get('crash_radius', 1e-6))
    wedge_eta = float(config.get('wedge_eta', 0.3))
    
    phase_check = config.get('phase_check', {})
    num_test_orbits = int(phase_check.get('num_test_orbits', 30))
    max_iter = int(phase_check.get('max_iter_phase', 1000))
    escape_radius = float(phase_check.get('escape_radius_phase', 120.0))
    
    # 1. Wedge check
    p_cfg = {'theta': theta, 'lam': lam, 'eps': eps, 'wedge_eta': wedge_eta}
    config_wedge = {'wedge_scan': wedge_cfg}
    
    try:
        wedge_found, _, _, _, wedge_valid_frac, _, _, _ = estimate_wedge_scanning(p_cfg, config_wedge)
    except Exception:
        wedge_found = 0
        wedge_valid_frac = 0.0
    
    # 2. Simulate orbits
    init_r = 5.0
    init_angles = np.random.uniform(-np.pi, np.pi, num_test_orbits)
    z0s = init_r * np.exp(1j * init_angles)
    
    trajectories = []
    n_escaped = 0
    
    for z0 in z0s:
        traj = iterate_map(
            z0, 0j, max_iter, mode="theorem_map",
            theta=theta, lam=lam, eps=eps,
            crash_radius=crash_radius,
            escape_radius=escape_radius
        )
        
        if len(traj) > 0:
            trajectories.append(traj)
            if np.abs(traj[-1]) > escape_radius:
                n_escaped += 1
    
    escape_fraction = n_escaped / num_test_orbits if num_test_orbits > 0 else 0.0
    
    # 3. Estimate kappa
    r_growth_min = float(orbit_cfg.get('r_growth_min', 20.0))
    r_growth_max = orbit_cfg.get('r_growth_max', None)
    min_tail_points = int(orbit_cfg.get('min_tail_points', 30))
    min_valid_orbits = int(orbit_cfg.get('min_valid_orbits', 8))
    kappa_clip = float(pitch_cfg.get('kappa_clip', 10.0))
    bootstrap_B = int(pitch_cfg.get('bootstrap_B', 200))
    
    kappa_results = estimate_kappa_from_orbits_growth(
        trajectories,
        r_growth_min=r_growth_min,
        r_growth_max=r_growth_max,
        min_tail_points=min_tail_points,
        kappa_clip=kappa_clip,
        min_valid_orbits=min_valid_orbits,
        bootstrap_B=bootstrap_B
    )
    
    kappa_hat = kappa_results['kappa_hat']
    n_orbits_valid = kappa_results['n_orbits_valid']
    kappa_fail_reason = kappa_results['fail_reason']
    
    # 4. Spiral classification: wedge_found AND kappa_estimated AND n_orbits_valid >= 3
    # Note: For phase diagram, we use a lower threshold (3) than rigor analysis (min_valid_orbits)
    # because we're doing a broad sweep and want to detect spiral regions even with limited data.
    kappa_estimated = not np.isnan(kappa_hat)
    phase_min_valid = 3  # Minimum for phase diagram classification
    spiral_exists = (wedge_found == 1) and kappa_estimated and (n_orbits_valid >= phase_min_valid)
    
    # Determine fail_reason
    if not spiral_exists:
        if wedge_found == 0:
            fail_reason = "no_wedge"
        elif not kappa_estimated:
            fail_reason = kappa_fail_reason if kappa_fail_reason else "kappa_estimation_failed"
        else:
            fail_reason = f"insufficient_valid_orbits({n_orbits_valid}<{phase_min_valid})"
    else:
        fail_reason = None
    
    return {
        'spiral_exists': int(spiral_exists),
        'wedge_found': wedge_found,
        'wedge_valid_fraction': wedge_valid_frac,
        'kappa_hat': kappa_hat,
        'n_orbits_valid': n_orbits_valid,
        'escape_fraction': escape_fraction,
        'fail_reason': fail_reason
    }


def run_phase_diagram(config, output_csv='results/phase_map.csv', 
                      output_dir='figures/rigor', seed=0):
    """
    Main driver for phase diagram sweep.
    """
    np.random.seed(seed)
    
    # Extract config
    param = config.get('param', {})
    theta = float(param.get('theta', 0.1))
    wedge_cfg = config.get('wedge_scan', {})
    orbit_cfg = config.get('orbit_validation', {})
    pitch_cfg = config.get('pitch', {})
    
    phase_sweep = config.get('phase_sweep', {})
    eps_min = float(phase_sweep.get('eps_min', 0.0))
    eps_max = float(phase_sweep.get('eps_max', 2.0))
    eps_steps = int(phase_sweep.get('eps_steps', 20))
    
    lam_min = float(phase_sweep.get('lam_min', 0.01))
    lam_max = float(phase_sweep.get('lam_max', 2.0))
    lam_steps = int(phase_sweep.get('lam_steps', 20))
    
    theory_cfg = config.get('theory_boundary', {})
    R0 = float(theory_cfg.get('R0', 5.0))
    
    print(f"Running Phase Diagram Sweep")
    print(f"  Eps range: [{eps_min}, {eps_max}], steps={eps_steps}")
    print(f"  Lam range: [{lam_min}, {lam_max}], steps={lam_steps}")
    print(f"  Total points: {eps_steps * lam_steps}")
    
    # Generate grids
    eps_vals = np.linspace(eps_min, eps_max, eps_steps)
    lam_vals = np.linspace(lam_min, lam_max, lam_steps)
    
    # Matrix: phase[j, i] = result at (lam[j], eps[i])
    phase = np.zeros((lam_steps, eps_steps))
    results = []
    
    for j, lam_mag in enumerate(lam_vals):
        for i, eps_mag in enumerate(eps_vals):
            lam = complex(lam_mag, 0.0)
            eps = complex(eps_mag, 0.0)
            
            result = check_spiral_existence_growth(lam, eps, theta, param, 
                                                   orbit_cfg, pitch_cfg, wedge_cfg)
            
            phase[j, i] = result['spiral_exists']
            
            results.append({
                'theta': theta,
                'eps': eps_mag,
                'lam': lam_mag,
                'spiral_exists': result['spiral_exists'],
                'wedge_found': result['wedge_found'],
                'wedge_valid_fraction': result['wedge_valid_fraction'],
                'kappa_hat': result['kappa_hat'],
                'n_orbits_valid': result['n_orbits_valid'],
                'escape_fraction': result['escape_fraction'],
                'fail_reason': result['fail_reason']
            })
    
    # Save CSV
    df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved phase map to {output_csv}")
    
    # Statistics
    n_spirals = np.sum(phase)
    print(f"Found {int(n_spirals)} spiral points (out of {eps_steps * lam_steps})")
    
    # Plotting
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    plot_phase_diagram(phase, eps_vals, lam_vals, R0, n_spirals, output_dir_path)
    
    return df


def plot_phase_diagram(phase, eps_vals, lam_vals, R0, n_spirals, output_dir):
    """
    Plot phase diagram with correct orientation.
    
    Args:
        phase: Matrix [lam_steps, eps_steps] where phase[j,i] = result at (lam[j], eps[i])
        eps_vals: 1D array of epsilon values
        lam_vals: 1D array of lambda values
        R0: Theory boundary parameter
        n_spirals: Total number of spiral points
        output_dir: Output directory path
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    eps_min, eps_max = eps_vals[0], eps_vals[-1]
    lam_min, lam_max = lam_vals[0], lam_vals[-1]
    
    # imshow with origin='lower': bottom-left is (eps_min, lam_min)
    # extent=[left, right, bottom, top] = [eps_min, eps_max, lam_min, lam_max]
    # phase[j, i] where j=lam index, i=eps index
    # imshow displays rows (first index) along y-axis, cols (second index) along x-axis
    im = ax.imshow(phase, origin='lower', aspect='auto',
                   extent=[eps_min, eps_max, lam_min, lam_max],
                   cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    
    # Theory boundary: lam >= 4/R + 4*eps/R^4
    eps_theory = np.linspace(eps_min, eps_max, 200)
    lam_theory = 4.0/R0 + 4.0*eps_theory/(R0**4)
    
    ax.plot(eps_theory, lam_theory, 'r-', linewidth=2.5, label=f'Theoretical Sufficient (R={R0})')
    
    ax.set_xlabel(r'Epsilon ($\varepsilon$)', fontsize=13)
    ax.set_ylabel(r'Lambda ($\lambda$)', fontsize=13)
    ax.set_title(f'Spiral Phase Diagram ({len(lam_vals)}x{len(eps_vals)})', fontsize=14, fontweight='bold')
    
    # Annotation if no spirals
    if n_spirals == 0:
        ax.text(0.5, 0.5, "No spiral points under current existence criteria\n(wedge + kappa)",
               transform=ax.transAxes, ha='center', va='center', fontsize=13, color='red',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='red', linewidth=2))
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2.5, label='Theoretical Sufficient'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Spiral Exists'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='white', 
               markeredgecolor='k', markersize=10, label='No Spiral')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    ax.set_xlim(eps_min, eps_max)
    ax.set_ylim(lam_min, lam_max)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / "phase_diagram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved phase diagram to {output_dir / 'phase_diagram.png'}")


def main():
    parser = argparse.ArgumentParser(description="Phase Diagram Experiment (Step 4.2)")
    parser.add_argument('--config', required=True, help="Path to YAML config")
    parser.add_argument('--output-csv', default='results/phase_map.csv')
    parser.add_argument('--output-dir', default='figures/rigor')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    run_phase_diagram(config, args.output_csv, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
