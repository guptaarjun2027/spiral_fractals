"""
Experiment: Scaling Exponents (Step 3.2) - Updated for Theorem Map

Computes escape fraction rho(r) and fits scaling law 1 - rho(r) ~ r^-beta.
Also estimates pitch from escaping trajectories.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from src.iterators import iterate_map, ControlledConfig

def run_scaling_exponents(config_path: str, out_csv: str, out_dir: str, seed: int):
    np.random.seed(seed)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    n_steps = int(config.get('n_steps', 2000))
    escape_radius = float(config.get('escape_radius', 1e3))
    persist_steps = int(config.get('persist_steps', 50))
    mode = config.get('mode', 'theorem_map')
    
    # Grid of radii
    grid_cfg = config.get('grid', {})
    r_min = float(grid_cfg.get('r_min', 5.0))
    r_max = float(grid_cfg.get('r_max', 100.0))
    n_radii = int(grid_cfg.get('n_radii', 20))
    
    if 'radii' in config:
        r_cfg = config['radii']
        r_min = float(r_cfg.get('r_min', r_min))
        r_max = float(r_cfg.get('r_max', r_max))
        n_radii = int(r_cfg.get('num_r', n_radii))
    
    n_angles = int(config.get('num_angles', 500))
    n_angles = int(grid_cfg.get('n_angles', n_angles))
    
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
    
    fit_cfg = config.get('fit', {})
    if 'fit_window' in config:
         fit_cfg = config['fit_window']
         
    fit_window_min = float(fit_cfg.get('fit_window_min', fit_cfg.get('r_lo', 10.0)))
    fit_window_max = float(fit_cfg.get('fit_window_max', fit_cfg.get('r_hi', 80.0)))
    tail_fraction = float(fit_cfg.get('tail_fraction', 0.4))
    bootstrap_B = int(fit_cfg.get('bootstrap_B', config.get('bootstrap_B', 200)))
    
    param_list = config.get('params', [])
    
    from src.utils import extract_monotone_branch, filter_dominant_arm
    from src.theorem_conditions import estimate_wedge_scanning
    
    for i, p_cfg in enumerate(param_list):
        pid = f"param_{i}"
        
        # Theorem Map Params
        theta = float(p_cfg.get('theta', 0.0))
        lam = complex(p_cfg.get('lam', 1.0))
        eps = complex(p_cfg.get('eps', 0.0))
        crash_radius = float(p_cfg.get('crash_radius', 1e-6))
        
        # Params for controlled fallback
        omega = float(p_cfg.get('omega', 0.2))
        radial_mode = p_cfg.get('radial_mode', 'additive')
        
        # --- FIX 3: Wedge Consistency Check ---
        wedge_found, _, _, _, wedge_valid_frac, _, _, _ = estimate_wedge_scanning(p_cfg, config)
        
        if wedge_found == 0 or wedge_valid_frac < 0.3:
            # Skip analysis
            results.append({
                'param_id': pid,
                'theta': theta,
                'analysis_performed': False,
                'rejection_reason': 'no_wedge' if wedge_found==0 else 'insufficient_wedge_fraction',
                'beta_hat': np.nan, 'pitch_mean': np.nan,
                'ci_beta_low': np.nan, 'ci_beta_high': np.nan,
                'pitch_ci_low': np.nan, 'pitch_ci_high': np.nan
            })
            
            # Plot dummy
            plt.figure(figsize=(6,4))
            plt.text(0.5, 0.5, f"Analysis Skipped: {pid}\nReason: Wedge failure", 
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f"Skipped: {pid}")
            plt.savefig(out_dir_path / f"scaling_rho_r_{pid}.png")
            plt.close()
            continue
            
        rhos = []
        n_escaped_list = []
        
        # Per-Orbit Pitch Storage
        orbit_kappas = []
        orbit_points_logr = []
        orbit_points_phi = []
        
        for r_idx, r in enumerate(radii):
            phis = np.random.uniform(-np.pi, np.pi, n_angles)
            z0s = r * np.exp(1j * phis) # Start on circle
            
            escaped_count = 0
            
            for z0 in z0s:
                if mode == 'theorem_map':
                    traj = iterate_map(
                        z0, 0j, n_steps, mode="theorem_map",
                        theta=theta, lam=lam, eps=eps,
                        crash_radius=crash_radius,
                        escape_radius=escape_radius
                    )
                else:
                    traj = iterate_map(
                        z0, 0j, n_steps, mode="controlled",
                        escape_radius=escape_radius,
                        omega=omega,
                        radial_mode=radial_mode,
                        delta=float(p_cfg.get('delta', 0.01)),
                        alpha=float(p_cfg.get('alpha', 1.05)),
                        phase_eps=float(p_cfg.get('phase_eps', 0.0))
                    )
                
                # Check escape
                if len(traj) > 0 and np.abs(traj[-1]) > escape_radius:
                    # Fix 2: Persistence check
                    is_truly_escaped = True
                    if persist_steps > 0:
                        extra_traj = iterate_map(
                            z0=traj[-1], c=0j, max_iter=persist_steps,
                            mode=mode, theta=theta, lam=lam, eps=eps,
                            crash_radius=crash_radius,
                            escape_radius=1e20 # Infinite
                        )
                        if len(extra_traj) > 0 and np.any(np.abs(extra_traj) <= escape_radius):
                            is_truly_escaped = False
                            
                    if is_truly_escaped:
                        escaped_count += 1
                        
                        # --- FIX 1: Per-Orbit Pitch ---
                        r_traj = np.abs(traj)
                        mask = (r_traj >= fit_window_min) & (r_traj <= fit_window_max)
                        if np.sum(mask) >= 20: # Min points per orbit
                            phi_full = np.unwrap(np.angle(traj))
                            
                            p_mask = phi_full[mask]
                            r_mask = r_traj[mask]
                            lr_mask = np.log(r_mask)
                            
                            n_p = len(p_mask)
                            start_k = int((1.0 - tail_fraction) * n_p)
                            p_tail = p_mask[start_k:]
                            lr_tail = lr_mask[start_k:]
                            
                            if len(p_tail) >= 20:
                                # kappa_i = dphi / dlogr
                                s_i, _, _, _, _ = stats.linregress(lr_tail, p_tail)
                                
                                if abs(s_i) <= 5.0: # Clip outlier
                                    orbit_kappas.append(s_i)
                                    orbit_points_logr.append(lr_tail)
                                    orbit_points_phi.append(p_tail)
                
            rho_val = escaped_count / n_angles
            rhos.append(rho_val)
            n_escaped_list.append(escaped_count)
            
        # --- Aggregate Pitch ---
        pitch_median = np.nan
        pitch_ci_low = np.nan
        pitch_ci_high = np.nan
        
        if len(orbit_kappas) > 5:
            pitch_median = np.median(orbit_kappas)
            
            # Bootstrap orbits
            if bootstrap_B > 0:
                boot_k = []
                n_orb = len(orbit_kappas)
                orb_arr = np.array(orbit_kappas)
                for _ in range(bootstrap_B):
                    res = np.random.choice(orb_arr, n_orb, replace=True)
                    boot_k.append(np.median(res))
                
                pitch_ci_low = np.percentile(boot_k, 2.5)
                pitch_ci_high = np.percentile(boot_k, 97.5)
                
        # Plot Pitch
        plt.figure(figsize=(8,6))
        # Faint lines
        idx_show = np.random.choice(len(orbit_points_logr), min(100, len(orbit_points_logr)), replace=False)
        for idx_k in idx_show:
            plt.plot(orbit_points_logr[idx_k], orbit_points_phi[idx_k], 'k-', alpha=0.05, lw=1)
            
        if not np.isnan(pitch_median):
            # Plot median line
            # visual anchor
            all_lr = np.concatenate(orbit_points_logr)
            x_min, x_max = np.min(all_lr), np.max(all_lr)
            x_vals = np.array([x_min, x_max])
            y_mean_offset = np.mean(np.concatenate(orbit_points_phi))
            mid_x = (x_min + x_max)/2
            y_vals = pitch_median * (x_vals - mid_x) + y_mean_offset 
            plt.plot(x_vals, y_vals, 'r-', lw=2.5, label=f'Median Kappa={pitch_median:.3f}')
            
        plt.xlabel("log(r)")
        plt.ylabel("phi")
        plt.title(f"Pitch: {pid}\nKappa={pitch_median:.3f} [{pitch_ci_low:.3f}, {pitch_ci_high:.3f}]\nN_orbits={len(orbit_kappas)}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir_path / f"pitch_vs_r_{pid}.png")
        plt.close()

        # --- FIX 2: Asymptotic Scaling ---
        rhos = np.array(rhos)
        y_vals = 1.0 - rhos
        
        # Window Detection
        # 1. Restrict y
        mask_y = (y_vals >= 1e-4) & (y_vals <= 0.3)
        
        radii_valid = radii[mask_y]
        y_valid = y_vals[mask_y]
        
        beta_hat = np.nan
        ci_beta_low = np.nan
        ci_beta_high = np.nan
        fit_msg = "No asymptotic scaling regime"
        
        valid_window = None # (start_idx, end_idx) in 'radii_valid'
        
        if len(radii_valid) >= 5:
            log_r_v = np.log10(radii_valid) # Base 10 for decade check
            log_y_v = np.log(y_valid) # Natural log for slope
            
            # Slide window
            n_v = len(radii_valid)
            best_window = None
            max_len = 0
            
            for i in range(n_v):
                for j in range(i + 4, n_v): # at least 5 points
                    span = log_r_v[j] - log_r_v[i]
                    if span >= 0.5:
                        # Check fit
                        slope, _, r_val, _, _ = stats.linregress(log_r_v[i:j+1] * np.log(10), log_y_v[i:j+1]) 
                        if (r_val**2 >= 0.95) and (slope < 0):
                            if (j - i) > max_len:
                                max_len = j - i
                                best_window = (i, j)
                                beta_hat = -slope
            
            if best_window:
                i, j = best_window
                valid_window = (radii_valid[i], radii_valid[j])
                
                # Bootstrap on this window
                n_escaped_win = np.array(n_escaped_list)[mask_y][i:j+1]
                log_r_win = np.log(radii_valid[i:j+1])
                
                if bootstrap_B > 0:
                    boot_b = []
                    for _ in range(bootstrap_B):
                        res_esc = np.random.binomial(n_angles, n_escaped_win/n_angles)
                        res_y = 1.0 - res_esc/n_angles
                        valid_b = (res_y > 0)
                        if np.sum(valid_b) >= 3:
                             s, _, _, _, _ = stats.linregress(log_r_win[valid_b], np.log(res_y[valid_b]))
                             boot_b.append(-s)
                    if boot_b:
                        ci_beta_low = np.percentile(boot_b, 2.5)
                        ci_beta_high = np.percentile(boot_b, 97.5)
                        
                fit_msg = f"Beta={beta_hat:.3f}"
        
        # Plot Scaling
        plt.figure(figsize=(8, 6))
        plt.loglog(radii, y_vals, 'o-', color='gray', alpha=0.5, label='Raw 1-rho')
        
        if not np.isnan(beta_hat) and valid_window is not None:
             # Highlight window
             r_start, r_end = valid_window
             mask_w = (radii >= r_start) & (radii <= r_end)
             plt.loglog(radii[mask_w], y_vals[mask_w], 'bo', label='Scaling Window')
             
             # Line
             # Re-fit line for plotting to ensure it passes through points accurately
             rw = radii[mask_w]
             yw = y_vals[mask_w]
             m, c, _, _, _ = stats.linregress(np.log(rw), np.log(yw))
             fit_y = np.exp(c) * (rw ** m)
             plt.loglog(rw, fit_y, 'r--', lw=2, label=f'Fit beta={-m:.3f}')
             
             plt.title(f"Scaling Law: {pid}\n{fit_msg} [{ci_beta_low:.3f}, {ci_beta_high:.3f}]")
        else:
             plt.title(f"Scaling Law: {pid}\n{fit_msg}")

        plt.xlabel("Radius r")
        plt.ylabel("1 - rho(r)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.savefig(out_dir_path / f"scaling_rho_r_{pid}.png")
        plt.close()

        results.append({
            'param_id': pid,
            'theta': theta,
            'analysis_performed': True,
            'wedge_found': wedge_found,
            'beta_hat': beta_hat,
            'ci_beta_low': ci_beta_low,
            'ci_beta_high': ci_beta_high,
            'pitch_mean': pitch_median,
            'pitch_ci_low': pitch_ci_low,
            'pitch_ci_high': pitch_ci_high,
            'n_angles': n_angles
        })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outcsv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    run_scaling_exponents(args.config, args.outcsv, args.outdir, args.seed)
