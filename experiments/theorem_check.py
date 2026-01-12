"""
Experiment: Theorem Check (Step 3.1) - Updated for Theorem Map (Option A)

Validates the spiral structure of escaping orbits and checks consistency with predicted kappa.
Uses F(z) = e^{i\theta}z + \lambda z^2 + \varepsilon z^{-2}.
Includes honest wedge scanning and asymptotic tail fitting.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from src.iterators import iterate_map, ControlledConfig
from src.theorem_conditions import lemma_holds, estimate_wedge_scanning

def run_theorem_check(config_path: str, out_csv: str, out_dir: str, seed: int):
    np.random.seed(seed)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    n_steps = config.get('n_steps', 2000)
    escape_radius = float(config.get('escape_radius', 1e3))
    persist_steps = int(config.get('persist_steps', 50))
    mode = config.get('mode', 'theorem_map')
    
    sampling = config.get('sampling', {})
    r_min = float(sampling.get('r_min', 1.0))
    r_max = float(sampling.get('r_max', 10.0))
    num_points = int(sampling.get('num_points', 100))
    
    fit_cfg = config.get('fit', {})
    r_fit_min = float(fit_cfg.get('r_fit_min', 10.0))
    r_fit_max = float(fit_cfg.get('r_fit_max', 1e5))
    tail_fraction = float(fit_cfg.get('tail_fraction', 0.4))
    bootstrap_B = int(fit_cfg.get('bootstrap_B', 200))
    
    scan_cfg = config.get('wedge_scan', {})
    tau = float(scan_cfg.get('tau', 0.9))
    
    param_list = config.get('params', [])
    
    from src.utils import extract_monotone_branch, filter_dominant_arm
    
    for i, p_cfg in enumerate(param_list):
        # Extract params
        theta = float(p_cfg.get('theta', 0.0))
        lam = complex(p_cfg.get('lam', 1.0))
        eps = complex(p_cfg.get('eps', 0.0))
        crash_radius = float(p_cfg.get('crash_radius', 1e-6))
        
        # Wedge scanning
        pid = f"param_{i}"
        
        # --- 1) Estimated Wedge Scanning ---
        wedge_found, phi_lo, phi_hi, wedge_width, wedge_valid_frac, r_grid, phi_grid, valid_mask = estimate_wedge_scanning(p_cfg, config)
        
        # Plot Wedge Map
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(phi_grid, np.log10(r_grid), valid_mask.astype(int), cmap='Greys', vmin=0, vmax=1, shading='auto')
        plt.colorbar(label='Lemma Holds')
        
        # Only draw lines if wedge found
        if wedge_found == 1:
            plt.axvline(phi_lo, color='r', linestyle='--', label=f'Phi Lo: {phi_lo:.2f}')
            plt.axvline(phi_hi, color='r', linestyle='--', label=f'Phi Hi: {phi_hi:.2f}')
            
        plt.xlabel('Phi')
        plt.ylabel('log10(Radius)')
        
        title_str = f'Wedge Stability Scan: {pid}\nValid Fraction: {wedge_valid_frac:.2f}'
        if wedge_found == 0:
            title_str += f"\n(No valid wedge found at tau={tau})"
        elif wedge_width < 0.1:
            title_str += f"\n(Wedge too narrow: width={wedge_width:.3f})"
            
        plt.title(title_str)
        plt.legend()
        plt.savefig(out_dir_path / f"wedge_map_{pid}.png")
        plt.close()
        
        # --- 2) Sample initial points ---
        log_r = np.random.uniform(np.log(r_min), np.log(r_max), num_points)
        rs = np.exp(log_r)
        
        sampled_mode = "wedge"
        if wedge_found == 0 or wedge_width < 0.1:
             phis = np.random.uniform(-np.pi, np.pi, num_points)
             sampled_mode = "fallback_full_circle" if wedge_found==1 else "no_wedge_full_circle"
        else:
             phis = np.random.uniform(phi_lo, phi_hi, num_points)
        
        z0s = rs * np.exp(1j * phis)
        
        # Check fraction of sampled points that are valid
        valid_sample_count = sum(lemma_holds(r, p, p_cfg) for r, p in zip(rs, phis))
        sampled_valid_fraction = valid_sample_count / num_points if num_points > 0 else 0
        
        # --- 3) Simulate ---
        all_phi = []
        all_logr = []
        escaped_count = 0
        crashed_count = 0
        points_used = 0
        orbit_indices = [] 
        
        for idx, z0 in enumerate(z0s):
            if mode == 'theorem_map':
                traj = iterate_map(
                    z0=z0,
                    c=0j, 
                    max_iter=n_steps,
                    mode="theorem_map",
                    theta=theta,
                    lam=lam,
                    eps=eps,
                    crash_radius=crash_radius,
                    escape_radius=escape_radius
                )
            else:
                traj = iterate_map(
                    z0=z0, c=0j, max_iter=n_steps, mode="controlled",
                    escape_radius=escape_radius,
                    omega=float(p_cfg.get('omega', 0.2)),
                    radial_mode=p_cfg.get('radial_mode', 'additive'),
                    delta=float(p_cfg.get('delta', 0.01)),
                    alpha=float(p_cfg.get('alpha', 1.05)),
                    phase_eps=float(p_cfg.get('phase_eps', 0.0))
                )
            
            # Check status
            if len(traj) > 0:
                last_z = traj[-1]
                if np.abs(last_z) < crash_radius:
                    crashed_count += 1
                elif np.abs(last_z) > escape_radius:
                    # Persistence Check
                    is_truly_escaped = True
                    if persist_steps > 0:
                        extra_traj = iterate_map(
                            z0=last_z, c=0j, max_iter=persist_steps,
                            mode=mode, theta=theta, lam=lam, eps=eps,
                            crash_radius=crash_radius,
                            escape_radius=1e20
                        )
                        if len(extra_traj) > 0 and np.any(np.abs(extra_traj) <= escape_radius):
                            is_truly_escaped = False
                    
                    if is_truly_escaped:
                        escaped_count += 1
                        
                        r_traj = np.abs(traj)
                        mask = (r_traj >= r_fit_min) & (r_traj <= r_fit_max)
                        
                        if np.sum(mask) > 1:
                            # Full unwrap for safety
                            phi_full = np.unwrap(np.angle(traj))
                            
                            # Apply mask
                            phi_valid = phi_full[mask]
                            r_valid = r_traj[mask]
                            log_r_valid = np.log(r_valid)
                            
                            # Monotone Branch Isolation
                            phi_seg, logr_seg = extract_monotone_branch(phi_valid, log_r_valid, min_len=2)
                            
                            if len(phi_seg) > 0:
                                all_logr.append(logr_seg)
                                all_phi.append(phi_seg)
                                orbit_indices.extend([idx] * len(phi_seg))
                                points_used += len(phi_seg)
        
        escape_fraction = escaped_count / num_points if num_points > 0 else 0
        
        # --- 4) Fit ---
        k_hat = np.nan
        ci_low = np.nan
        ci_high = np.nan
        
        if len(all_logr) > 0:
            flat_logr = np.concatenate(all_logr)
            flat_phi = np.concatenate(all_phi)
            orbit_indices_arr = np.array(orbit_indices)
            
            # Dominant Arm Filter (Fix 1 Part 2)
            if len(flat_logr) > 10:
                res_temp = stats.linregress(flat_logr, flat_phi)
                psi = flat_phi - res_temp.slope * flat_logr
                psi_mod = np.mod(psi, 2 * np.pi)
                hist, bin_edges = np.histogram(psi_mod, bins=30, range=(0, 2*np.pi))
                best_bin = np.argmax(hist)
                center = (bin_edges[best_bin] + bin_edges[best_bin+1])/2
                hw = (bin_edges[1] - bin_edges[0]) * 2.5
                dist = np.abs(psi_mod - center)
                dist = np.minimum(dist, 2*np.pi - dist)
                mask_dom = dist < hw
                if np.sum(mask_dom) >= 5:
                    flat_logr = flat_logr[mask_dom]
                    flat_phi = flat_phi[mask_dom]
                    orbit_indices_arr = orbit_indices_arr[mask_dom]
            
            # Radius-Ordered Tail-Only Fitting (Fix 3)
            sort_idx = np.argsort(flat_logr)
            flat_logr = flat_logr[sort_idx]
            flat_phi = flat_phi[sort_idx]
            orbit_indices_arr = orbit_indices_arr[sort_idx]
            
            n_total = len(flat_logr)
            start_idx = int((1.0 - tail_fraction) * n_total)
            if start_idx < n_total - 5:
                flat_logr = flat_logr[start_idx:]
                flat_phi = flat_phi[start_idx:]
                orbit_indices_arr = orbit_indices_arr[start_idx:]
            
            if len(flat_logr) > 5:
                slope, intercept, r_val, p_val, std_err = stats.linregress(flat_logr, flat_phi)
                k_hat = slope
                
                # Bootstrap
                unique_orbits = np.unique(orbit_indices_arr)
                n_orbits_fit = len(unique_orbits)
                
                if n_orbits_fit > 2:
                    boot_slopes = []
                    orbit_map = {}
                    for oi, lr, ph in zip(orbit_indices_arr, flat_logr, flat_phi):
                        if oi not in orbit_map: orbit_map[oi] = ([], [])
                        orbit_map[oi][0].append(lr)
                        orbit_map[oi][1].append(ph)
                    
                    valid_orbit_ids = list(orbit_map.keys())
                    n_valid = len(valid_orbit_ids)
                    
                    for _ in range(bootstrap_B):
                        resample_ids = np.random.choice(valid_orbit_ids, n_valid, replace=True)
                        boot_logr = []
                        boot_phi = []
                        for rid in resample_ids:
                            boot_logr.extend(orbit_map[rid][0])
                            boot_phi.extend(orbit_map[rid][1])
                        
                        if len(boot_logr) > 2:
                            s, _, _, _, _ = stats.linregress(boot_logr, boot_phi)
                            boot_slopes.append(s)
                    
                    if boot_slopes:
                        ci_low = np.percentile(boot_slopes, 2.5)
                        ci_high = np.percentile(boot_slopes, 97.5)
            
            # Plot
            plt.figure(figsize=(8, 6))
            plot_indices = np.random.choice(len(flat_logr), min(2000, len(flat_logr)), replace=False)
            plt.scatter(flat_logr[plot_indices], flat_phi[plot_indices], alpha=0.3, s=5, label='Traj points (Filtered)')
            
            if not np.isnan(k_hat):
                x_vals = np.array([np.min(flat_logr), np.max(flat_logr)])
                y_vals = k_hat * x_vals + intercept
                plt.plot(x_vals, y_vals, 'r--', lw=2, label=f'Fit k={k_hat:.3f}')
            
            plt.title(f"Theorem Check: {pid}\nk_hat={k_hat:.3f} CI[{ci_low:.3f}, {ci_high:.3f}]")
            plt.xlabel("log(r)")
            plt.ylabel("phi")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir_path / f"theorem_phi_logr_{pid}.png")
            plt.close()

        results.append({
            'param_id': pid,
            'theta': theta,
            'lam': str(lam),
            'eps': str(eps),
            'wedge_found': wedge_found,
            'phi_lo': phi_lo,
            'phi_hi': phi_hi,
            'wedge_width': wedge_width,
            'wedge_valid_fraction': wedge_valid_frac,
            'sampled_valid_fraction': sampled_valid_fraction,
            'sampled_mode': sampled_mode,
            'k_hat': k_hat,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_orbits': escaped_count, 
            'n_crashed': crashed_count,
            'n_points_used': points_used,
            'escape_fraction': escape_fraction,
            'tail_fraction': tail_fraction,
            'wedge_tau': tau,
            'wedge_eta': float(p_cfg.get('wedge_eta', 0.3))
        })
    
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
    
    # Summary plot
    if len(results) > 1:
        plt.figure()
        valid_res = df.dropna(subset=['k_hat'])
        if not valid_res.empty:
            plt.errorbar(
                range(len(valid_res)), 
                valid_res['k_hat'], 
                yerr=[valid_res['k_hat']-valid_res['ci_low'], valid_res['ci_high']-valid_res['k_hat']], 
                fmt='o'
            )
            plt.xticks(range(len(valid_res)), valid_res['param_id'], rotation=45)
            plt.ylabel("Estimated Kappa (k_hat)")
            plt.title("Kappa Agreement")
            plt.tight_layout()
            plt.savefig(out_dir_path / "kappa_agreement.png")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outcsv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    run_theorem_check(args.config, args.outcsv, args.outdir, args.seed)
