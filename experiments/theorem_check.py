"""
Experiment: Theorem Check (Step 3.1) - UPDATED (per-orbit theorem-check, tail-safe)

Key change:
- We separate "escape threshold" from "stop radius".

escape_radius = threshold at which we declare the orbit escaped (e.g. 1e3)
stop_radius   = much larger radius used ONLY to stop iteration so we collect a long tail
               (e.g. 1e12). This prevents "num_orbits_fit=0" due to too-short orbits.

This version:
1) Finds wedge using Option (2) + dominance.
2) Seeds inside wedge.
3) Iterates theorem map until stop_radius (or crash/nan/inf).
4) Declares escaped if orbit ever crosses escape_radius.
5) Fits each orbit tail: phi = kappa log r + b.
6) Theorem PASS per orbit: R^2 >= R2_min AND k_pred in CI.

Outputs:
- summary CSV per parameter set
- per-orbit CSV per parameter set
- wedge_map image
- per-orbit phi vs log r plots (a few)
- kappa_pred vs kappa_hat agreement plot
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from src.iterators import iterate_map


# ----------------------------
# Core map + theorem utilities
# ----------------------------

def F_map(z: complex, theta: float, lam: complex, eps: complex) -> complex:
    return np.exp(1j * theta) * z + lam * (z ** 2) + eps * (z ** -2)


def dominance_holds(r: float, lam: complex, eps: complex) -> bool:
    """
    Large-radius dominance:
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
    Option (2):
        |F| >= |λ| r^2 - r + α
    """
    return abs(lam) * (r ** 2) - r + alpha


def option2_condition_holds(r: float, phi: float, theta: float, lam: complex, eps: complex, alpha: float) -> bool:
    """
    "Good angle" test for wedge scanning:
    - dominance holds at r
    - |F(re^{iφ})| >= |λ| r^2 - r + α
    """
    if not dominance_holds(r, lam, eps):
        return False
    z = r * np.exp(1j * phi)
    val = abs(F_map(z, theta, lam, eps))
    return val >= option2_threshold(r, lam, alpha)


def estimate_wedge_option2(theta: float, lam: complex, eps: complex, alpha: float, scan_cfg: dict):
    """
    Scan (r, phi) grid and find a contiguous wedge of phi values
    such that Option (2) condition holds for >= tau fraction of radii.

    Returns:
      wedge_found (0/1), phi_lo, phi_hi, wedge_width,
      valid_fraction_global, r_grid, phi_grid, valid_mask
    """
    r_min = float(scan_cfg.get('r_scan_min', 10.0))
    r_max = float(scan_cfg.get('r_scan_max', 200.0))
    num_r = int(scan_cfg.get('num_r_scan', 25))

    phi_min = float(scan_cfg.get('phi_min', -np.pi))
    phi_max = float(scan_cfg.get('phi_max', np.pi))
    num_phi = int(scan_cfg.get('num_phi', 400))

    tau = float(scan_cfg.get('tau', 0.9))

    rs = np.logspace(np.log10(r_min), np.log10(r_max), num_r)
    phis = np.linspace(phi_min, phi_max, num_phi)

    # meshgrid with shapes (num_phi, num_r)
    r_grid, phi_grid = np.meshgrid(rs, phis)
    valid_mask = np.zeros_like(r_grid, dtype=bool)

    for i in range(r_grid.shape[0]):
        for j in range(r_grid.shape[1]):
            valid_mask[i, j] = option2_condition_holds(
                r=float(r_grid[i, j]),
                phi=float(phi_grid[i, j]),
                theta=theta,
                lam=lam,
                eps=eps,
                alpha=alpha
            )

    valid_rate_per_phi = np.mean(valid_mask, axis=1)  # fraction over r, for each phi
    valid_per_phi = valid_rate_per_phi >= tau

    if not np.any(valid_per_phi):
        wedge_found = 0
        phi_lo, phi_hi, wedge_width = np.nan, np.nan, 0.0
    else:
        wedge_found = 1
        indices = np.where(valid_per_phi)[0]
        diffs = np.diff(indices)
        splits = np.where(diffs > 1)[0]

        if len(splits) == 0:
            phi_lo, phi_hi = phis[indices[0]], phis[indices[-1]]
        else:
            # pick largest contiguous block
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


def kappa_pred(phi0: float, r0: float, lam: complex) -> float:
    """
    Orbit-dependent predicted spiral pitch from asymptotic doubling:
        κ_pred ≈ (φ0 + arg λ) / (log r0 + log |λ|)
    """
    denom = (np.log(r0) + np.log(abs(lam)))
    if denom == 0:
        return np.nan
    return (phi0 + np.angle(lam)) / denom


def fit_orbit_tail(logr: np.ndarray, phi: np.ndarray, tail_fraction: float):
    """
    Fit φ = κ log r + b on the last tail_fraction of points.
    Returns dict with k_hat, intercept, R2, CI, n_tail.
    """
    n = len(logr)
    if n < 3:
        return None

    start = int((1.0 - tail_fraction) * n)
    x = logr[start:]
    y = phi[start:]

    # Need at least 3 points for linear regression with confidence intervals
    if len(x) < 3:
        # Fall back to using all points if tail is too short
        x = logr
        y = phi

    if len(x) < 3:
        return None

    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    R2 = r_val ** 2

    df = len(x) - 2
    tcrit = stats.t.ppf(0.975, df) if df > 0 else 1.96
    ci_low = slope - tcrit * std_err
    ci_high = slope + tcrit * std_err

    return {
        "k_hat": float(slope),
        "intercept": float(intercept),
        "R2": float(R2),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "std_err": float(std_err),
        "n_tail": int(len(x))
    }


# ----------------------------
# Main experiment
# ----------------------------

def run_theorem_check(config_path: str, out_csv: str, out_dir: str, seed: int):
    rng = np.random.default_rng(seed)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    results_summary = []

    n_steps = int(config.get('n_steps', 2000))

    # Escape threshold (classification)
    escape_radius = float(config.get('escape_radius', 1e3))

    # Stop radius (data collection) - MUST be much larger than escape_radius
    stop_radius = float(config.get('stop_radius', escape_radius * 1e6))

    sampling = config.get('sampling', {})
    r_min = float(sampling.get('r_min', 10.0))
    r_max = float(sampling.get('r_max', 50.0))
    num_points = int(sampling.get('num_points', 50))
    margin_steps = int(sampling.get('wedge_margin_steps', 5))

    fit_cfg = config.get('fit', {})
    r_fit_min = float(fit_cfg.get('r_fit_min', 10.0))
    r_fit_max = float(fit_cfg.get('r_fit_max', stop_radius))
    tail_fraction = float(fit_cfg.get('tail_fraction', 0.4))
    R2_min = float(fit_cfg.get('R2_min', 0.99))

    scan_cfg = config.get('wedge_scan', {})
    alpha = float(scan_cfg.get('alpha', 2.0))

    params_list = config.get('params', [])

    from src.utils import extract_monotone_branch

    for i, p_cfg in enumerate(params_list):
        pid = f"param_{i}"

        theta = float(p_cfg.get('theta', 0.0))
        lam = complex(p_cfg.get('lam', 1.0))
        eps = complex(p_cfg.get('eps', 0.0))
        crash_radius = float(p_cfg.get('crash_radius', 1e-6))

        # ----------------------------
        # 1) Wedge scan
        # ----------------------------
        wedge_found, phi_lo, phi_hi, wedge_width, valid_frac, r_grid, phi_grid, valid_mask = estimate_wedge_option2(
            theta=theta, lam=lam, eps=eps, alpha=alpha, scan_cfg=scan_cfg
        )

        # plot wedge map
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(phi_grid, np.log10(r_grid), valid_mask.astype(int),
                       cmap='Greys', vmin=0, vmax=1, shading='auto')
        plt.colorbar(label='Option(2) holds')

        if wedge_found == 1:
            plt.axvline(phi_lo, color='r', linestyle='--', label=f'phi_lo={phi_lo:.2f}')
            plt.axvline(phi_hi, color='r', linestyle='--', label=f'phi_hi={phi_hi:.2f}')

        plt.xlabel('phi')
        plt.ylabel('log10(r)')
        plt.title(f'Wedge Scan (Option 2): {pid}\nvalid_frac={valid_frac:.3f}, width={wedge_width:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir_path / f"wedge_map_{pid}.png")
        plt.close()

        # ----------------------------
        # 2) seed angles in wedge
        # ----------------------------
        log_r = rng.uniform(np.log(r_min), np.log(r_max), num_points)
        rs = np.exp(log_r)

        sampled_mode = "wedge"
        if wedge_found == 0 or wedge_width < 1e-6:
            phis = rng.uniform(-np.pi, np.pi, num_points)
            sampled_mode = "fallback_full_circle"
        else:
            num_phi = int(scan_cfg.get('num_phi', 400))
            phi_step = (float(scan_cfg.get('phi_max', np.pi)) - float(scan_cfg.get('phi_min', -np.pi))) / max(1, (num_phi - 1))
            margin = margin_steps * phi_step
            lo = phi_lo + margin
            hi = phi_hi - margin
            if hi <= lo:
                lo, hi = phi_lo, phi_hi
            phis = rng.uniform(lo, hi, num_points)

        z0s = rs * np.exp(1j * phis)

        sampled_valid = 0
        for r0, ph0 in zip(rs, phis):
            if option2_condition_holds(r0, ph0, theta, lam, eps, alpha):
                sampled_valid += 1
        sampled_valid_fraction = sampled_valid / num_points if num_points > 0 else 0.0

        # ----------------------------
        # 3) simulate + collect per-orbit data
        # ----------------------------
        orbits = {}
        escaped_count = 0
        crashed_count = 0
        debug_counts = {"empty_traj": 0, "crashed": 0, "not_escaped": 0, "mask_fail": 0, "monotone_fail": 0, "success": 0}

        for idx, z0 in enumerate(z0s):
            traj = iterate_map(
                z0=z0,
                c=0j,
                max_iter=n_steps,
                mode="theorem_map",
                theta=theta,
                lam=lam,
                eps=eps,
                crash_radius=crash_radius,

                # IMPORTANT:
                # escape_radius = classification threshold (1e3)
                # stop_radius = hard stop for data collection (1e12)
                # stop_on_escape=False so we don't stop at escape_radius
                escape_radius=escape_radius,
                stop_radius=stop_radius,
                stop_on_escape=False
            )

            if len(traj) == 0:
                debug_counts["empty_traj"] += 1
                continue

            # Debug: show trajectory details for first few orbits
            if idx < 3:
                r_vals = np.abs(traj)
                print(f"  orbit {idx}: len={len(traj)}, r values: {[f'{r:.2e}' for r in r_vals[:10]]}")

            # crash detection
            if np.any(np.abs(traj) < crash_radius):
                crashed_count += 1
                debug_counts["crashed"] += 1
                continue

            # escape detection (classification threshold)
            did_escape = bool(np.any(np.abs(traj) > escape_radius))
            if not did_escape:
                debug_counts["not_escaped"] += 1
                continue

            escaped_count += 1

            r_traj = np.abs(traj)
            phi_full = np.unwrap(np.angle(traj))

            mask = (r_traj >= r_fit_min) & (r_traj <= r_fit_max)
            n_mask = np.sum(mask)
            if n_mask < 3:
                debug_counts["mask_fail"] += 1
                if idx == 0:  # Debug first orbit
                    print(f"DEBUG param {pid} orbit {idx}: traj_len={len(traj)}, r_range=[{r_traj.min():.2e}, {r_traj.max():.2e}], n_in_window={n_mask}")
                continue

            phi_valid = phi_full[mask]
            logr_valid = np.log(r_traj[mask])

            phi_seg, logr_seg = extract_monotone_branch(phi_valid, logr_valid, min_len=2)
            if len(phi_seg) < 3:
                debug_counts["monotone_fail"] += 1
                continue

            debug_counts["success"] += 1
            orbits[idx] = {
                "logr": np.array(logr_seg, dtype=float),
                "phi": np.array(phi_seg, dtype=float),
                "r0": float(abs(z0)),
                "phi0": float(np.angle(z0)),
            }

        escape_fraction = escaped_count / num_points if num_points > 0 else 0.0

        # Debug output
        print(f"\n{pid} debug counts: {debug_counts}")
        print(f"{pid}: escaped={escaped_count}, crashed={crashed_count}, orbits_collected={len(orbits)}")

        # ----------------------------
        # 4) per-orbit fits + pass
        # ----------------------------
        orbit_rows = []
        fit_fail_count = 0
        for oid, d in orbits.items():
            fit = fit_orbit_tail(d["logr"], d["phi"], tail_fraction=tail_fraction)
            if fit is None:
                fit_fail_count += 1
                if fit_fail_count <= 3:
                    print(f"  Fit failed for orbit {oid}: len={len(d['logr'])}")
                continue

            kp = kappa_pred(d["phi0"], d["r0"], lam)
            passed = (fit["R2"] >= R2_min) and (fit["ci_low"] <= kp <= fit["ci_high"])

            orbit_rows.append({
                "param_id": pid,
                "orbit_id": int(oid),
                "r0": d["r0"],
                "phi0": d["phi0"],
                "k_pred": float(kp),
                "k_hat": fit["k_hat"],
                "ci_low": fit["ci_low"],
                "ci_high": fit["ci_high"],
                "R2": fit["R2"],
                "n_tail": fit["n_tail"],
                "pass": int(passed),
            })

        df_orbits = pd.DataFrame(orbit_rows)
        orbit_csv_path = out_dir_path / f"theorem_check_orbits_{pid}.csv"
        df_orbits.to_csv(orbit_csv_path, index=False)

        pass_rate = float(df_orbits["pass"].mean()) if len(df_orbits) > 0 else np.nan

        # ----------------------------
        # 5) plots
        # ----------------------------
        if len(df_orbits) > 0:
            show_n = min(6, len(df_orbits))
            chosen_orbits = df_orbits.sample(n=show_n, random_state=seed)["orbit_id"].tolist()

            plt.figure(figsize=(10, 7))
            for oid in chosen_orbits:
                d = orbits[int(oid)]
                fit = fit_orbit_tail(d["logr"], d["phi"], tail_fraction=tail_fraction)
                if fit is None:
                    continue

                plt.plot(d["logr"], d["phi"], alpha=0.6, linewidth=1)

                x = d["logr"]
                n = len(x)
                start = int((1.0 - tail_fraction) * n)
                x_tail = x[start:]
                y_line = fit["k_hat"] * x_tail + fit["intercept"]
                plt.plot(x_tail, y_line, linestyle="--", linewidth=2)

            plt.title(f"Sample per-orbit spiral fits (phi vs log r): {pid}")
            plt.xlabel("log(r)")
            plt.ylabel("unwrapped phi")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir_path / f"orbits_phi_logr_fits_{pid}.png")
            plt.close()

            # k_pred vs k_hat
            plt.figure(figsize=(7, 6))
            x = df_orbits["k_pred"].values
            y = df_orbits["k_hat"].values
            yerr_low = y - df_orbits["ci_low"].values
            yerr_high = df_orbits["ci_high"].values - y
            plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", alpha=0.8)

            mn = np.nanmin(np.concatenate([x, y]))
            mx = np.nanmax(np.concatenate([x, y]))
            plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)

            plt.title(f"kappa agreement (per orbit): {pid}\npass_rate={pass_rate:.2f}")
            plt.xlabel("k_pred")
            plt.ylabel("k_hat")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir_path / f"kappa_pred_vs_hat_{pid}.png")
            plt.close()

        # ----------------------------
        # 6) summary row
        # ----------------------------
        results_summary.append({
            "param_id": pid,
            "theta": theta,
            "lam": str(lam),
            "eps": str(eps),
            "alpha": alpha,
            "wedge_found": wedge_found,
            "phi_lo": phi_lo,
            "phi_hi": phi_hi,
            "wedge_width": wedge_width,
            "wedge_valid_fraction": valid_frac,
            "sampled_valid_fraction": sampled_valid_fraction,
            "sampled_mode": sampled_mode,
            "num_points": num_points,
            "escaped_count": escaped_count,
            "crashed_count": crashed_count,
            "escape_fraction": escape_fraction,
            "num_orbits_fit": int(len(df_orbits)),
            "pass_rate": pass_rate,
            "R2_min": R2_min,
            "tail_fraction": tail_fraction,
            "r_fit_min": r_fit_min,
            "r_fit_max": r_fit_max,
            "escape_radius": escape_radius,
            "stop_radius": stop_radius,
            "orbit_csv": str(orbit_csv_path)
        })

    df_sum = pd.DataFrame(results_summary)
    df_sum.to_csv(out_csv, index=False)
    print(f"Saved summary to {out_csv}")
    print(f"Saved per-orbit CSVs in {out_dir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outcsv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_theorem_check(args.config, args.outcsv, args.outdir, args.seed)
