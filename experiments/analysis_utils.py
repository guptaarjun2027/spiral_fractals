"""
Shared analysis utilities for spiral fractals experiments.
Contains reusable logic for pitch estimation, scaling law detection, and statistical bootstrapping.
"""

import numpy as np
from scipy import stats

def detect_scaling_window(radii, y_vals, min_points=5, min_span_decade=0.5, r_sq_thresh=0.95):
    """
    Detects the best power-law scaling window in 1 - rho(r) vs r.
    
    Args:
        radii: Array of radii.
        y_vals: Array of 1 - rho(r) values.
        min_points: Minimum number of points in a window.
        min_span_decade: Minimum span in log10(r) to consider.
        r_sq_thresh: Minimum R^2 to accept a fit.
        
    Returns:
        tuple: (beta_hat, best_window_indices, valid_window_r_range)
               beta_hat: The negative slope (scaling exponent). NaN if no window found.
               best_window_indices: Tuple (start_idx, end_idx). None if no window found.
               valid_window_r_range: Tuple (r_min, r_max). None if no window found.
    """
    # 1. Restrict y to valid scaling regime
    mask_y = (y_vals >= 1e-4) & (y_vals <= 0.3)
    radii_valid = radii[mask_y]
    y_valid = y_vals[mask_y]
    
    if len(radii_valid) < min_points:
        return np.nan, None, None
        
    log_r_v = np.log10(radii_valid) # Base 10 for decade check
    log_y_v = np.log(y_valid)       # Natural log for slope
    
    n_v = len(radii_valid)
    best_window = None
    max_len = 0
    beta_hat = np.nan
    
    # Slide window
    for i in range(n_v):
        for j in range(i + min_points - 1, n_v): 
            span = log_r_v[j] - log_r_v[i]
            if span >= min_span_decade:
                # Check fit
                slope, _, r_val, _, _ = stats.linregress(log_r_v[i:j+1] * np.log(10), log_y_v[i:j+1]) 
                if (r_val**2 >= r_sq_thresh) and (slope < 0):
                    if (j - i) > max_len:
                        max_len = j - i
                        best_window = (i, j)
                        beta_hat = -slope
    
    valid_window_r_range = None
    if best_window:
        i, j = best_window
        valid_window_r_range = (radii_valid[i], radii_valid[j])
        
    return beta_hat, best_window, valid_window_r_range

def bootstrap_scaling_beta(n_angles, n_escaped_list, radii, mask_y, best_window, bootstrap_B=200):
    """
    Bootstraps the scaling exponent beta.
    
    Args:
        n_angles: Total number of angles simulated.
        n_escaped_list: List/Array of escaped counts per radius.
        radii: Array of radii.
        mask_y: Boolean mask for valid y values (used in detection).
        best_window: (i, j) indices into the masked arrays.
        bootstrap_B: Number of bootstrap samples.
        
    Returns:
        tuple: (ci_low, ci_high) 95% confidence interval.
    """
    if best_window is None or bootstrap_B <= 0:
        return np.nan, np.nan
        
    i, j = best_window
    radii_valid = radii[mask_y]
    n_escaped_win = np.array(n_escaped_list)[mask_y][i:j+1]
    log_r_win = np.log(radii_valid[i:j+1])
    
    boot_b = []
    for _ in range(bootstrap_B):
        # Resample binomial counts
        res_esc = np.random.binomial(n_angles, n_escaped_win/n_angles)
        res_y = 1.0 - res_esc/n_angles
        
        # Valid points for log
        valid_b = (res_y > 0)
        if np.sum(valid_b) >= 3:
             s, _, _, _, _ = stats.linregress(log_r_win[valid_b], np.log(res_y[valid_b]))
             boot_b.append(-s)
             
    if not boot_b:
        return np.nan, np.nan
        
    return np.percentile(boot_b, 2.5), np.percentile(boot_b, 97.5)

def estimate_pitch_stats(orbit_kappas, bootstrap_B=200):
    """
    Computes median pitch and bootstrap CI.
    
    Args:
        orbit_kappas: List of kappa values from individual orbits.
        bootstrap_B: Number of bootstrap iterations.
        
    Returns:
        tuple: (median_kappa, ci_low, ci_high)
    """
    if len(orbit_kappas) < 5:
        return np.nan, np.nan, np.nan
        
    pitch_median = np.median(orbit_kappas)
    pitch_ci_low = np.nan
    pitch_ci_high = np.nan
    
    if bootstrap_B > 0:
        boot_k = []
        n_orb = len(orbit_kappas)
        orb_arr = np.array(orbit_kappas)
        for _ in range(bootstrap_B):
            res = np.random.choice(orb_arr, n_orb, replace=True)
            boot_k.append(np.median(res))
        
        if boot_k:
            pitch_ci_low = np.percentile(boot_k, 2.5)
            pitch_ci_high = np.percentile(boot_k, 97.5)
            
    return pitch_median, pitch_ci_low, pitch_ci_high

def fit_orbit_tail(log_r, phi, tail_fraction=0.4, min_points=20):
    """
    Fits the tail of a single orbit or aggregated points.
    
    Args:
        log_r: Array of log(r).
        phi: Array of phi (unwrapped).
        tail_fraction: Fraction of points to use from end.
        min_points: Minimum points required.
        
    Returns:
        float: Slope (kappa). NaN if invalid.
    """
    if len(log_r) < min_points:
        return np.nan
        
    n_p = len(log_r)
    start_k = int((1.0 - tail_fraction) * n_p)
    # Ensure we have enough points in tail
    if n_p - start_k < min_points:
        start_k = n_p - min_points
        
    p_tail = phi[start_k:]
    lr_tail = log_r[start_k:]
    
    slope, _, _, _, _ = stats.linregress(lr_tail, p_tail)
    return slope
