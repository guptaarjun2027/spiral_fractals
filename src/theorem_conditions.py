"""
Mathematical conditions and lemmas for spiral stability.

This file supports two regimes:
1) Controlled Map (arm modulation)  -> legacy stability condition from theory.ipynb
2) Theorem Map: F(z) = e^{iθ} z + λ z^2 + ε z^{-2}
   -> Large-radius dominance + Option (2) "good-angle" condition:
      |F(re^{iφ})| >= |λ| r^2 - r + α

This aligns wedge estimation with the Lemma A2 / Option (2) proof path.
"""

import numpy as np


# -----------------------------
# Theorem map utilities
# -----------------------------

def F_theorem(z: complex, theta: float, lam: complex, eps: complex) -> complex:
    return np.exp(1j * theta) * z + lam * (z ** 2) + eps * (z ** -2)


def dominance_condition(r: float, lam: complex, eps: complex) -> bool:
    """
    Quadratic dominance condition:
        |λ| r^2 >= 4 r + 4 |ε| r^{-2}
    """
    if r <= 0:
        return False
    norm_lam = abs(lam)
    if norm_lam == 0:
        return False
    lhs = norm_lam * (r ** 2)
    rhs = 4.0 * r + 4.0 * abs(eps) * (r ** -2)
    return lhs >= rhs


def option2_threshold(r: float, lam: complex, alpha: float) -> float:
    """
    Option (2) magnitude threshold:
        |F| >= |λ| r^2 - (1 - α/r) r
             = |λ| r^2 - r + α
    """
    return abs(lam) * (r ** 2) - r + alpha


def theorem_map_lemma_holds(r: float, phi: float, params: dict) -> bool:
    """
    Theorem-map lemma (Option 2):

    Returns True if BOTH:
      A) dominance_condition holds at radius r
      B) |F(re^{iφ})| >= |λ| r^2 - r + α

    Required params:
      theta, lam, eps
    Optional:
      alpha (default 2.0)
    """
    theta = float(params.get("theta", 0.0))
    lam = complex(params.get("lam", 1.0))
    eps = complex(params.get("eps", 0.0))
    alpha = float(params.get("alpha", params.get("wedge_alpha", 2.0)))  # allow either key

    if not dominance_condition(r, lam, eps):
        return False

    z = r * np.exp(1j * phi)
    val = abs(F_theorem(z, theta, lam, eps))
    return val >= option2_threshold(r, lam, alpha)


# -----------------------------
# Public API
# -----------------------------

def lemma_holds(r: float, phi: float, params: dict) -> bool:
    """
    Dispatches to the correct lemma depending on params.

    Controlled map is detected by presence of 'radial_mode'.
    Otherwise uses theorem map lemma (Option 2).
    """

    # 1) Controlled map logic (UNCHANGED behavior)
    if "radial_mode" in params:
        omega = float(params.get("omega", 0.2))
        if omega == 0 or r == 0:
            return False

        radial_mode = params["radial_mode"]
        phase_eps = float(params.get("phase_eps", 0.0))
        k_arms = int(params.get("k_arms", 3))
        threshold = float(params.get("stability_threshold", 0.5))

        arm_mod = 1.0 + phase_eps * np.cos(k_arms * phi)

        if radial_mode == "additive":
            delta = float(params.get("delta", 0.01))
            val = (delta * arm_mod) / (r * omega)
            return val < threshold

        elif radial_mode == "power":
            alpha = float(params.get("alpha", 1.05))
            r_next = (r ** alpha) * arm_mod
            delta_r = r_next - r
            val = abs(delta_r) / (r * omega)
            return val < threshold

        return False

    # 2) Theorem map logic (UPDATED to Option 2)
    return theorem_map_lemma_holds(r, phi, params)


def estimate_wedge_scanning(params: dict, config: dict):
    """
    Numerically scan (r, phi) space to find the 'valid wedge' where lemma_holds is true.

    Interpretation:
      For each phi, compute fraction of radii r in [r_scan_min, r_scan_max]
      such that lemma_holds(r, phi, params) is True.
      Then a phi is considered "valid" if that fraction >= tau.

    Returns:
      wedge_found (0/1),
      phi_lo, phi_hi,
      wedge_width,
      valid_fraction_global,
      r_grid, phi_grid, valid_mask
    """
    scan_cfg = config.get("wedge_scan", {})

    r_min = float(scan_cfg.get("r_scan_min", 10.0))
    r_max = float(scan_cfg.get("r_scan_max", 200.0))
    num_r = int(scan_cfg.get("num_r_scan", 25))

    phi_min = float(scan_cfg.get("phi_min", -np.pi))
    phi_max = float(scan_cfg.get("phi_max", np.pi))
    num_phi = int(scan_cfg.get("num_phi", 400))

    tau = float(scan_cfg.get("tau", 0.9))

    # Allow alpha to be set in the scan config without repeating in every params entry
    if "alpha" in scan_cfg and ("alpha" not in params and "wedge_alpha" not in params):
        params = dict(params)
        params["alpha"] = float(scan_cfg["alpha"])

    rs = np.logspace(np.log10(r_min), np.log10(r_max), num_r)
    phis = np.linspace(phi_min, phi_max, num_phi)

    # meshgrid: (num_phi, num_r)
    r_grid, phi_grid = np.meshgrid(rs, phis)
    valid_mask = np.zeros_like(r_grid, dtype=bool)

    for i in range(r_grid.shape[0]):
        for j in range(r_grid.shape[1]):
            valid_mask[i, j] = lemma_holds(float(r_grid[i, j]), float(phi_grid[i, j]), params)

    # per-phi validity rate
    valid_rate_per_phi = np.mean(valid_mask, axis=1)
    valid_per_phi = valid_rate_per_phi >= tau

    if not np.any(valid_per_phi):
        wedge_found = 0
        phi_lo, phi_hi = np.nan, np.nan
        wedge_width = 0.0
    else:
        wedge_found = 1
        indices = np.where(valid_per_phi)[0]

        # find largest contiguous block
        diffs = np.diff(indices)
        splits = np.where(diffs > 1)[0]

        if len(splits) == 0:
            phi_lo, phi_hi = phis[indices[0]], phis[indices[-1]]
        else:
            blocks = []
            start = 0
            for s in splits:
                blocks.append((indices[s] - indices[start], indices[start], indices[s]))
                start = s + 1
            blocks.append((indices[-1] - indices[start], indices[start], indices[-1]))
            best = max(blocks, key=lambda x: x[0])
            phi_lo, phi_hi = phis[best[1]], phis[best[2]]

        wedge_width = float(phi_hi - phi_lo)

    valid_fraction_global = float(np.sum(valid_mask) / valid_mask.size)
    return wedge_found, phi_lo, phi_hi, wedge_width, valid_fraction_global, r_grid, phi_grid, valid_mask
