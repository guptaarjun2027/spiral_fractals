"""
Experiment: Rigor Sensitivity (Step 4.1)
Tests numerical convergence of spiral metrics (kappa, beta) across LOW/MED/HIGH computational tiers.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from src.iterators import iterate_map
from src.theorem_conditions import estimate_wedge_scanning
from experiments.analysis_utils import estimate_pitch_stats, detect_scaling_window, bootstrap_scaling_beta, fit_orbit_tail

def run_rigor_sensitivity(config_path: str, out_csv: str, out_dir: str, seed: int):
    np.random.seed(seed)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    tiers = config.get('tiers', {})
    param_list = config.get('params', [])
    fit_cfg = config.get('fit', {})
    
    tail_fraction = float(fit_cfg.get('tail_fraction', 0.4))
    bootstrap_B = int(fit_cfg.get('bootstrap_B', 100))
    fit_r_min = float(fit_cfg.get('fit_window_r_min', 10.0))
    fit_r_max = float(fit_cfg.get('fit_window_r_max', 100.0))
    
    results = []
    
    # Iterate Parameters
    for p_idx, p_cfg in enumerate(param_list):
        pid = f"param_{p_idx}"
        print(f"Analyzing {pid}...")
        
        # Base params
        theta = float(p_cfg.get('theta', 0.0))
        lam = complex(p_cfg.get('lam', 1.0))
        eps = complex(p_cfg.get('eps', 0.0))
        wedge_eta = float(p_cfg.get('wedge_eta', 0.3))
        crash_radius = float(p_cfg.get('crash_radius', 1e-6))
        
        # Iterate Tiers
        for tier_name, tier_cfg in tiers.items():
            print(f"  Running Tier: {tier_name}")
            
            # Tier parameters
            n_angles = int(tier_cfg.get('num_angles', 128))
            n_radii = int(tier_cfg.get('num_radii', 12))
            max_iter = int(tier_cfg.get('max_iter', 400))
            esc_rad = float(tier_cfg.get('escape_radius', 60.0))
            
            # 1. Wedge Check (quick)
            wedge_found, _, _, _, wedge_valid_frac, _, _, _ = estimate_wedge_scanning(p_cfg, config)
            
            # 2. Simulation Loop (Scaling & Pitch style)
            # We need to simulate orbits at various radii to get rho(r) and also collect escaping orbits for pitch
            
            # Radii grid for this tier
            # We'll use a standard range, but density depends on n_radii
            r_sim_min = fit_r_min / 2.0 # start a bit earlier
            r_sim_max = fit_r_max * 2.0 # go a bit further
            radii = np.logspace(np.log10(r_sim_min), np.log10(r_sim_max), n_radii)
            
            rhos = []
            n_escaped_list = []
            
            orbit_kappas = []
            n_valid_orbits_pitch = 0
            
            for r in radii:
                phis = np.random.uniform(-np.pi, np.pi, n_angles)
                z0s = r * np.exp(1j * phis)
                
                escaped_count = 0
                
                for z0 in z0s:
                    traj = iterate_map(
                        z0, 0j, max_iter, mode="theorem_map",
                        theta=theta, lam=lam, eps=eps,
                        crash_radius=crash_radius,
                        escape_radius=esc_rad
                    )
                    
                    if len(traj) > 0 and np.abs(traj[-1]) > esc_rad:
                        escaped_count += 1
                        
                        # Pitch Analysis for this orbit
                        r_traj = np.abs(traj)
                        mask_window = (r_traj >= fit_r_min) & (r_traj <= fit_r_max)
                        
                        if np.sum(mask_window) >= 20:
                             phi_full = np.unwrap(np.angle(traj))
                             
                             p_mask = phi_full[mask_window]
                             r_mask = r_traj[mask_window]
                             lr_mask = np.log(r_mask)
                             
                             k_i = fit_orbit_tail(lr_mask, p_mask, tail_fraction=tail_fraction, min_points=20)
                             
                             if not np.isnan(k_i) and abs(k_i) <= 5.0:
                                 orbit_kappas.append(k_i)
                                 n_valid_orbits_pitch += 1
                                 
                rho_val = escaped_count / n_angles
                rhos.append(rho_val)
                n_escaped_list.append(escaped_count)
            
            # 3. Analyze Results
            
            # Kappa Stats
            med_kappa, k_lo, k_hi = estimate_pitch_stats(orbit_kappas, bootstrap_B=bootstrap_B)
            
            # Beta Stats
            # For beta, we need rho(r) to match scaling law 1-rho ~ r^-beta
            rhos = np.array(rhos)
            y_vals = 1.0 - rhos
            
            beta, win_idx, _ = detect_scaling_window(radii, y_vals)
            
            b_lo, b_hi = np.nan, np.nan
            n_valid_points_beta = 0
            if not np.isnan(beta) and win_idx is not None:
                # Need mask_y for bootstrapping func
                mask_y = (y_vals >= 1e-4) & (y_vals <= 0.3)
                b_lo, b_hi = bootstrap_scaling_beta(n_angles, n_escaped_list, radii, mask_y, win_idx, bootstrap_B=bootstrap_B)
                n_valid_points_beta = np.sum(mask_y) # Approximate metric of points in "scaling regime" roughly
            
            # Store
            res_row = {
                'param_id': pid,
                'tier': tier_name,
                'theta': theta,
                'lam': str(lam),
                'eps': str(eps),
                'wedge_eta': wedge_eta,
                'num_angles': n_angles,
                'num_radii': n_radii,
                'max_iter': max_iter,
                'escape_radius': esc_rad,
                'wedge_found': wedge_found,
                'wedge_valid_fraction': wedge_valid_frac,
                'median_kappa': med_kappa,
                'kappa_ci_low': k_lo,
                'kappa_ci_high': k_hi,
                'n_valid_orbits': n_valid_orbits_pitch,
                'beta': beta,
                'beta_ci_low': b_lo,
                'beta_ci_high': b_hi,
                'n_valid_points': n_valid_points_beta,
                'analysis_performed': True,
                'rejection_reason': 'None'
            }
            results.append(res_row)
            
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
    
    # Plotting
    if not df.empty:
        # Plot 1: Kappa Convergence
        # Group by param_id
        unique_params = df['param_id'].unique()
        
        # Order tiers
        tier_order = ['LOW', 'MED', 'HIGH']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # (a) Kappa
        ax = axes[0]
        for pid in unique_params:
            sub = df[df['param_id'] == pid].set_index('tier').reindex(tier_order)
            valid = sub.dropna(subset=['median_kappa'])
            if not valid.empty:
                y = valid['median_kappa']
                yerr = [y - valid['kappa_ci_low'], valid['kappa_ci_high'] - y]
                ax.errorbar(valid.index, y, yerr=yerr, fmt='o-', label=pid, capsize=5)
        ax.set_title("Convergence of Pitch (Kappa)")
        ax.set_ylabel("Median Kappa")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # (b) Beta
        ax = axes[1]
        for pid in unique_params:
            sub = df[df['param_id'] == pid].set_index('tier').reindex(tier_order)
            valid = sub.dropna(subset=['beta'])
            if not valid.empty:
                y = valid['beta']
                yerr = [y - valid['beta_ci_low'], valid['beta_ci_high'] - y]
                ax.errorbar(valid.index, y, yerr=yerr, fmt='s--', label=pid, capsize=5)
        ax.set_title("Convergence of Scaling Exponent (Beta)")
        ax.set_ylabel("Beta")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir_path / "rigor_sensitivity.png")
        plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outcsv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    run_rigor_sensitivity(args.config, args.outcsv, args.outdir, args.seed)
