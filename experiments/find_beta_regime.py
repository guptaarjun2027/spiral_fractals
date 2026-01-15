# experiments/find_beta_regime.py
"""
Sweep (lambda, epsilon) to find a regime where beta (scaling exponent) is detectable.

Goal (Option B):
- Find TWO regimes:
  Regime A: known spiral parameter where beta FAILS (e.g., fully escaping: rho(r)=1 everywhere)
  Regime B: parameter where beta EXISTS (partial escape across radii so 1-rho(r) has a scaling window)

Why this script exists:
- Your current beta failure is often because rho(r)=1.0 everywhere (fully escaping),
  so 1-rho(r)=0 and there is nothing to fit.

This script searches for (lam, eps) where:
- rho(r) is NOT constant across radii
- some points fall inside a y-window for scaling detection
- detect_scaling_window can return beta_hat

Outputs:
- Prints top candidate (lam, eps) pairs ranked by "points_in_window" and nontrivial rho range.
- Optionally writes results CSV to results/find_beta_regime_candidates.csv
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- Make imports work when running as: python3 experiments/find_beta_regime.py ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.iterators import iterate_map  # noqa: E402
from experiments.analysis_utils import detect_scaling_window  # noqa: E402


def escaped_anytime(traj: np.ndarray, escape_radius: float) -> bool:
    """True iff orbit ever exceeds escape_radius."""
    if traj is None or len(traj) == 0:
        return False
    return float(np.max(np.abs(traj))) > float(escape_radius)


def estimate_rho_for_radius(
    r0: float,
    *,
    theta: float,
    lam: complex,
    eps: complex,
    max_iters: int,
    escape_radius: float,
    crash_radius: float,
    n_test_angles: int,
    rng: np.random.Generator,
) -> float:
    """Estimate rho(r0) = P(escape before cap) using n_test_angles random angles."""
    angles = rng.uniform(-np.pi, np.pi, int(n_test_angles))
    z0s = r0 * np.exp(1j * angles)

    n_esc = 0
    for z0 in z0s:
        traj = iterate_map(
            z0,
            0j,
            int(max_iters),
            mode="theorem_map",
            theta=float(theta),
            lam=lam,
            eps=eps,
            crash_radius=float(crash_radius),
            escape_radius=float(escape_radius),
        )
        if escaped_anytime(traj, escape_radius):
            n_esc += 1

    return n_esc / float(n_test_angles)


def main():
    # ---------------------------
    # Sweep knobs (edit here)
    # ---------------------------
    theta = 0.5
    crash_radius = 1e-6

    # Make escape "harder" so we can see partial escape (not rho=1 everywhere)
    escape_radius = 5000.0
    max_iters = 5000

    # Sweep ranges (smaller lambda reduces immediate blow-up)
    lam_min, lam_max, lam_steps = 0.01, 0.40, 18
    eps_min, eps_max, eps_steps = 0.0, 2.0, 18

    # Radii for rho(r)
    num_radii = 20
    r_min_beta, r_max_beta = 5.0, 2000.0

    # Monte Carlo angles per radius
    n_test_angles = 400

    # Scaling detection config
    min_points = 5
    min_span_decade = 0.5
    r_sq_thresh = 0.95
    y_min, y_max = 1e-3, 0.9  # window on y = 1 - rho(r)

    # RNG (stable)
    seed = 12345
    rng = np.random.default_rng(seed)

    # Output
    out_csv = Path("results/find_beta_regime_candidates.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Run sweep
    # ---------------------------
    lam_vals = np.linspace(lam_min, lam_max, lam_steps)
    eps_vals = np.linspace(eps_min, eps_max, eps_steps)
    radii = np.logspace(np.log10(r_min_beta), np.log10(r_max_beta), num_radii)

    print("\n=== find_beta_regime sweep ===")
    print(f"theta={theta}")
    print(f"lam in [{lam_min}, {lam_max}] ({lam_steps} steps)")
    print(f"eps in [{eps_min}, {eps_max}] ({eps_steps} steps)")
    print(f"radii: {num_radii} log-spaced in [{r_min_beta}, {r_max_beta}]")
    print(f"escape_radius={escape_radius}, max_iters={max_iters}, n_test_angles={n_test_angles}")
    print(f"y-window: [{y_min}, {y_max}]")
    print("detect_scaling_window: AVAILABLE (will compute beta_hat when possible)\n")

    rows = []
    total = lam_steps * eps_steps
    idx = 0

    for lam_mag in lam_vals:
        for eps_mag in eps_vals:
            idx += 1
            lam = complex(float(lam_mag), 0.0)
            eps = complex(float(eps_mag), 0.0)

            # For reproducibility but different per point
            # (cheap deterministic mixing)
            point_seed = int(seed + 100000 * lam_mag + 1000 * eps_mag) % (2**32 - 1)
            point_rng = np.random.default_rng(point_seed)

            rhos = []
            for r0 in radii:
                rho = estimate_rho_for_radius(
                    float(r0),
                    theta=theta,
                    lam=lam,
                    eps=eps,
                    max_iters=max_iters,
                    escape_radius=escape_radius,
                    crash_radius=crash_radius,
                    n_test_angles=n_test_angles,
                    rng=point_rng,
                )
                rhos.append(rho)

            rhos = np.array(rhos, dtype=float)
            y_vals = 1.0 - rhos

            # points in y-window
            mask_y = (y_vals >= y_min) & (y_vals <= y_max)
            points_in_window = int(np.sum(mask_y))

            beta_hat = np.nan
            best_window = None
            try:
                beta_hat, best_window, _ = detect_scaling_window(
                    radii,
                    y_vals,
                    min_points=min_points,
                    min_span_decade=min_span_decade,
                    r_sq_thresh=r_sq_thresh,
                    y_min=y_min,
                    y_max=y_max,
                )
            except TypeError:
                # If your detect_scaling_window doesn't accept y_min/y_max yet,
                # this keeps the script runnable; but you should update analysis_utils.py.
                beta_hat, best_window, _ = detect_scaling_window(
                    radii,
                    y_vals,
                    min_points=min_points,
                    min_span_decade=min_span_decade,
                    r_sq_thresh=r_sq_thresh,
                )

            rho_min = float(np.min(rhos))
            rho_max = float(np.max(rhos))
            rho_span = rho_max - rho_min

            rows.append(
                {
                    "theta": theta,
                    "lam": float(lam_mag),
                    "eps": float(eps_mag),
                    "rho_min": rho_min,
                    "rho_max": rho_max,
                    "rho_span": rho_span,
                    "points_in_window": points_in_window,
                    "beta_hat": (None if np.isnan(beta_hat) else float(beta_hat)),
                    "has_window": bool(best_window is not None),
                }
            )

    df = pd.DataFrame(rows)

    # Ranking heuristic:
    # 1) prioritize lots of points in window
    # 2) prioritize nontrivial rho span (avoid rho=1 everywhere)
    # 3) prioritize having a beta_hat
    df_ranked = df.copy()
    df_ranked["beta_ok"] = df_ranked["beta_hat"].apply(lambda x: 0 if x is None else 1)
    df_ranked = df_ranked.sort_values(
        by=["beta_ok", "points_in_window", "rho_span"],
        ascending=[False, False, False],
    )

    df_ranked.to_csv(out_csv, index=False)

    print("=== Top candidates (beta likely detectable) ===")
    top_n = min(15, len(df_ranked))
    for k in range(top_n):
        row = df_ranked.iloc[k]
        lam_mag = row["lam"]
        eps_mag = row["eps"]
        points = int(row["points_in_window"])
        rho_min = row["rho_min"]
        rho_max = row["rho_max"]
        beta_hat = row["beta_hat"]
        beta_str = "NA" if beta_hat is None else f"{beta_hat:.6f}"
        print(
            f"{k+1:02d}) lam={lam_mag:.6g}, eps={eps_mag:.6g} | "
            f"points_in_window={points}/{num_radii} | rho_range=[{rho_min:.2f}, {rho_max:.2f}] | beta_hat={beta_str}"
        )

    print("\nSaved full sweep table to:", str(out_csv))
    print("\nTip:")
    print("- Pick a top candidate with rho_range NOT [1.00, 1.00] and points_in_window >= 6.")
    print("- Then set that (lam, eps) into configs/rigor_sensitivity.yaml as your Regime B and rerun rigor_sensitivity.\n")


if __name__ == "__main__":
    main()
