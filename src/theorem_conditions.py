"""
Mathematical conditions and lemmas for spiral stability.
Based on theoretical analysis in notebooks/theory.ipynb.
"""

import numpy as np

def lemma_holds(r: float, phi: float, params: dict) -> bool:
    """
    Check if the local stability condition (Lemma) holds at (r, phi).
    
    The stability condition derived in theory.ipynb is:
        delta / (omega * r) << 1
    
    which ensures that radial growth is small relative to angular rotation,
    allowing coherent spiral arms to form.
    
    For controlled map with arm modulation:
        delta_eff = delta * (1 + phase_eps * cos(k * phi))
        Condition: delta_eff / (omega * r) < threshold
        
    We use a conservative threshold of 0.5 (or user-configurable).
    
    Args:
        r: Radius
        phi: Angle (radians)
        params: Dictionary containing 'omega', 'delta', 'phase_eps', 'k_arms', 'radial_mode'
                and optionally 'stability_threshold' (default 0.5).
                For power mode, 'alpha' is used to estimate equivalent delta.
    
    Returns:
        True if condition holds, False otherwise.
    """
    
    # 1. Distinguish mode
    if 'radial_mode' in params:
        # --- Existing Controlled Map Logic ---
        omega = float(params.get('omega', 0.2))
        if omega == 0 or r == 0:
            return False
            
        radial_mode = params['radial_mode']
        phase_eps = float(params.get('phase_eps', 0.0))
        k_arms = int(params.get('k_arms', 3))
        threshold = float(params.get('stability_threshold', 0.5))
        
        arm_mod = 1.0 + phase_eps * np.cos(k_arms * phi)
        
        if radial_mode == 'additive':
            delta = float(params.get('delta', 0.01))
            val = (delta * arm_mod) / (r * omega)
            return val < threshold
            
        elif radial_mode == 'power':
            alpha = float(params.get('alpha', 1.05))
            r_next = (r ** alpha) * arm_mod
            delta_r = r_next - r
            val = abs(delta_r) / (r * omega)
            return val < threshold
            
        return False

    # --- Theorem Map Logic ---
    # F(z) = e^{i\theta}z + \lambda z^2 + \varepsilon z^{-2}
    # Params: theta, lam, eps, wedge_eta
    # Conditions:
    # A) Dominance: |lam|r^2 >= 4r + 4|eps|r^-2
    # B) Wedge Margin: cos(theta - phi - arg(lam)) >= wedge_eta
    
    theta = float(params.get('theta', 0.0))
    lam = complex(params.get('lam', 1.0))
    eps = complex(params.get('eps', 0.0))
    wedge_eta = float(params.get('wedge_eta', 0.3))
    
    # Precompute magnitudes
    norm_lam = abs(lam)
    norm_eps = abs(eps)
    
    # Safety
    if r <= 0 or norm_lam == 0:
        return False
        
    # A) Dominance Condition
    # |lam|r^2 >= 4r + 4|eps|r^-2
    lhs_dom = norm_lam * (r**2)
    rhs_dom = 4.0 * r + 4.0 * norm_eps * (r**-2)
    
    if lhs_dom < rhs_dom:
        return False
        
    # B) Wedge Condition
    # phase = theta - phi - arg(lam)
    arg_lam = np.angle(lam)
    phase = theta - phi - arg_lam
    cos_phase = np.cos(phase)
    
    
    if cos_phase < wedge_eta:
        return False
            
    return True

def estimate_wedge_scanning(params: dict, config: dict) -> tuple[int, float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerically scan (r, phi) space to find the 'valid wedge' where the lemma holds.
    
    Returns:
        wedge_found: 1 if wedge exists, 0 otherwise.
        phi_lo, phi_hi: The angular interval (NaN if not found).
        wedge_width: Width.
        valid_fraction: Global validity.
        r_grid, phi_grid, valid_mask: Grid data.
    """
    scan_cfg = config.get('wedge_scan', {})
    r_min = float(scan_cfg.get('r_scan_min', 5.0))
    r_max = float(scan_cfg.get('r_scan_max', 100.0))
    num_r = int(scan_cfg.get('num_r_scan', 20))
    
    phi_min = float(scan_cfg.get('phi_min', -np.pi))
    phi_max = float(scan_cfg.get('phi_max', np.pi))
    num_phi = int(scan_cfg.get('num_phi', 100))
    
    tau = float(scan_cfg.get('tau', 0.9)) # Validation threshold fraction
    
    rs = np.logspace(np.log10(r_min), np.log10(r_max), num_r)
    phis = np.linspace(phi_min, phi_max, num_phi)
    
    r_grid, phi_grid = np.meshgrid(rs, phis)
    valid_mask = np.zeros_like(r_grid, dtype=bool)
    
    for i in range(r_grid.shape[0]):
        for j in range(r_grid.shape[1]):
            valid_mask[i, j] = lemma_holds(r_grid[i, j], phi_grid[i, j], params)
            
    # Validity per angle: fraction of r-values where lemma holds
    valid_rate_per_phi = np.mean(valid_mask, axis=1) # Shape (num_phi,)
    
    valid_per_phi = valid_rate_per_phi >= tau
    valid_phis = phis[valid_per_phi]
    
    if len(valid_phis) == 0:
        wedge_found = 0
        phi_lo, phi_hi = np.nan, np.nan
        wedge_width = 0.0
    else:
        wedge_found = 1
        # Find largest contiguous interval
        indices = np.where(valid_per_phi)[0]
        
        # Check for wrapping? We assume single wedge for now.
        diffs = np.diff(indices)
        splits = np.where(diffs > 1)[0]
        
        if len(splits) == 0:
            phi_lo, phi_hi = phis[indices[0]], phis[indices[-1]]
        else:
            blocks = []
            start_idx = 0
            for split_idx in splits:
                block_len = indices[split_idx] - indices[start_idx]
                blocks.append((block_len, indices[start_idx], indices[split_idx]))
                start_idx = split_idx + 1
            blocks.append((indices[-1] - indices[start_idx], indices[start_idx], indices[-1]))
            
            best_block = max(blocks, key=lambda x: x[0])
            phi_lo = phis[best_block[1]]
            phi_hi = phis[best_block[2]]
        
        wedge_width = phi_hi - phi_lo
    
    valid_fraction = np.sum(valid_mask) / valid_mask.size
    
    return wedge_found, phi_lo, phi_hi, wedge_width, valid_fraction, r_grid, phi_grid, valid_mask
