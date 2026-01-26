"""
Experiment: Phase Diagram (Step 4) — stable rotation fraction via Δphi tail variance

Key fixes implemented:
1) Δphi stability computed once r is large enough (r >= r_fit_min), NOT tied to "full escape".
2) CSV includes debug columns:
      stable_count
      dphi_tail_var_mean
3) Plot color scale fixed to fractions: vmin=0.0, vmax=1.0

Output:
- phase_diagram.csv
- phase_diagram.png
- phase_diagram_with_theory_overlay.png
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.iterators import iterate_map


def safe_iterate_map(**kwargs):
    """
    Compatibility wrapper: your iterate_map signature may differ slightly.
    We try the most modern kwargs first; if it fails, retry without newer args.
    """
    try:
        return iterate_map(**kwargs)
    except TypeError:
        # fallback: drop stop_radius / stop_on_escape if older iterator
        kwargs.pop("stop_radius", None)
        kwargs.pop("stop_on_escape", None)
        return iterate_map(**kwargs)


def dominance_eps_bound_at_r(r: float, lam: complex) -> float:
    """
    From dominance condition style: |λ| r^2 >= 4 r + 4|ε| r^{-2}
    Solve for |ε|:
        4|ε| r^{-2} <= |λ| r^2 - 4r
        |ε| <= (|λ| r^2 - 4r) * r^2 / 4
    """
    a = abs(lam)
    rhs = (a * (r**2) - 4.0 * r)
    if rhs <= 0:
        return 0.0
    return rhs * (r**2) / 4.0


def compute_dphi_tail_var(traj: np.ndarray, r_fit_min: float, tail_fraction: float, min_tail_points: int):
    """
    Compute variance of Δphi on the tail of the indices where r >= r_fit_min.
    Returns: var (float) or None if insufficient data.
    """
    if traj is None or len(traj) < 5:
        return None

    r_vals = np.abs(traj)
    phi_vals = np.angle(traj)

    # unwrap angles once globally  ✅ REQUIRED FIX
    phi_unwrapped = np.unwrap(phi_vals)

    # indices where r is large enough
    valid = np.where(r_vals >= r_fit_min)[0]

    if len(valid) < min_tail_points:
        return None

    # tail indices among valid points
    start = int((1.0 - tail_fraction) * len(valid))
    tail_idx = valid[start:]

    if len(tail_idx) < 2:
        return None

    dphi_tail = np.diff(phi_unwrapped[tail_idx])
    if len(dphi_tail) == 0:
        return None

    return float(np.var(dphi_tail))


def run_phase_diagram(config_path: str, outcsv: str, outdir: str, seed: int):
    rng = np.random.default_rng(seed)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Config (generic)
    # -----------------------
    lam = complex(cfg.get("lam", 1.0 + 0j))
    alpha = float(cfg.get("alpha", 2.0))  # kept for CSV parity, even if not used in Δphi test

    # grid
    theta_min = float(cfg.get("theta_min", 0.0))
    theta_max = float(cfg.get("theta_max", 2*np.pi))
    num_theta = int(cfg.get("num_theta", 50))

    eps_abs_min = float(cfg.get("eps_abs_min", 0.0))
    eps_abs_max = float(cfg.get("eps_abs_max", 0.6))
    num_eps = int(cfg.get("num_eps", 25))

    # simulation
    n_steps = int(cfg.get("n_steps", 2000))
    escape_radius = float(cfg.get("escape_radius", 1000.0))
    stop_radius = float(cfg.get("stop_radius", 1e7))
    crash_radius = float(cfg.get("crash_radius", 1e-6))

    # sampling
    sampling = cfg.get("sampling", {})
    num_points = int(sampling.get("num_points", 60))
    r_min = float(sampling.get("r_min", 10.0))
    r_max = float(sampling.get("r_max", 50.0))

    # stability test  ✅ (from your requested yaml edits)
    dphi_var_thresh = float(cfg.get("dphi_var_thresh", 0.2))
    tail_fraction = float(cfg.get("tail_fraction", 0.4))
    min_tail_points = int(cfg.get("min_tail_points", 12))
    r_fit_min = float(cfg.get("r_fit_min", 25.0))

    # -----------------------
    # Build grid
    # -----------------------
    thetas = np.linspace(theta_min, theta_max, num_theta)
    eps_abs_grid = np.linspace(eps_abs_min, eps_abs_max, num_eps)

    # store for plotting
    Z = np.zeros((num_eps, num_theta), dtype=float)

    rows = []

    for i_eps, eps_abs in enumerate(eps_abs_grid):
        for i_th, theta in enumerate(thetas):
            # use eps as real magnitude along positive real axis unless config overrides
            eps = complex(eps_abs, 0.0)

            # sample seeds (log-uniform radii, uniform angles)
            rs = np.exp(rng.uniform(np.log(r_min), np.log(r_max), num_points))
            phis = rng.uniform(-np.pi, np.pi, num_points)
            z0s = rs * np.exp(1j * phis)

            dphi_vars = []
            escaped_count = 0
            crashed_count = 0

            for z0 in z0s:
                traj = safe_iterate_map(
                    z0=z0,
                    c=0j,
                    max_iter=n_steps,
                    mode="theorem_map",
                    theta=float(theta),
                    lam=lam,
                    eps=eps,
                    crash_radius=crash_radius,
                    escape_radius=escape_radius,
                    stop_radius=stop_radius,
                    stop_on_escape=False,   # keep collecting if iterator supports it
                )

                if traj is None or len(traj) == 0:
                    continue

                last = traj[-1]
                if np.isnan(last.real) or np.isnan(last.imag) or np.isinf(last.real) or np.isinf(last.imag):
                    continue

                if abs(last) < crash_radius:
                    crashed_count += 1
                    continue

                if abs(last) > escape_radius:
                    escaped_count += 1

                # ✅ key: compute Δphi tail var purely based on r >= r_fit_min
                v = compute_dphi_tail_var(
                    traj=np.array(traj, dtype=np.complex128),
                    r_fit_min=r_fit_min,
                    tail_fraction=tail_fraction,
                    min_tail_points=min_tail_points,
                )
                if v is not None:
                    dphi_vars.append(v)

            # stable count + mean variance ✅ debug columns
            stable_count = int(np.sum(np.array(dphi_vars) <= dphi_var_thresh)) if len(dphi_vars) > 0 else 0
            dphi_tail_var_mean = float(np.mean(dphi_vars)) if len(dphi_vars) > 0 else np.nan

            # define fraction over orbits where we successfully computed a Δphi variance
            denom = len(dphi_vars)
            stable_rotation_fraction = (stable_count / denom) if denom > 0 else 0.0

            Z[i_eps, i_th] = stable_rotation_fraction

            rows.append([
                float(theta),
                float(eps_abs),
                float(stable_rotation_fraction),
                int(stable_count),
                int(escaped_count),
                int(crashed_count),
                dphi_tail_var_mean,
                str(lam),
                float(alpha),
                # Below are placeholders to match your existing header style.
                # If your older pipeline expects wedge stats, keep them as NaNs.
                1,           # wedge_found (kept as 1 for compatibility; adjust if you compute wedge here)
                np.nan,      # phi_lo
                np.nan,      # phi_hi
                np.nan,      # wedge_width
                np.nan,      # wedge_valid_fraction
                "full_circle",  # sampled_mode
                int(num_points),
                float(escaped_count / num_points) if num_points > 0 else 0.0,
                float(dphi_var_thresh),
                float(r_fit_min),
                float(tail_fraction),
                int(min_tail_points),
                float(escape_radius),
                float(stop_radius),
            ])

    # -----------------------
    # Write CSV
    # -----------------------
    header = [
        "theta","eps_abs","stable_rotation_fraction","stable_count","escaped_count","crashed_count","dphi_tail_var_mean",
        "lam","alpha","wedge_found","phi_lo","phi_hi","wedge_width","wedge_valid_fraction",
        "sampled_mode","num_points","escape_fraction","dphi_var_thresh","r_fit_min","tail_fraction","min_tail_points",
        "escape_radius","stop_radius"
    ]
    df = pd.DataFrame(rows, columns=header)
    Path(outcsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outcsv, index=False)
    print(f"Saved phase diagram CSV to {outcsv}")

    # -----------------------
    # Plot (fraction scale fixed) ✅ REQUIRED FIX
    # -----------------------
    plt.figure(figsize=(10, 6))
    plt.imshow(
        Z,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
        extent=[theta_min, theta_max, eps_abs_min, eps_abs_max]
    )
    plt.colorbar(label="stable_rotation_fraction")
    plt.xlabel("theta")
    plt.ylabel("|eps|")
    plt.title("Phase diagram: stable rotation fraction (Δphi tail variance test)")
    plt.tight_layout()
    plt.savefig(outdir / "phase_diagram.png")
    plt.close()

    # -----------------------
    # Theory overlay plot
    # -----------------------
    overlay_r = float(cfg.get("overlay_r", 10.0))
    eps_bound = dominance_eps_bound_at_r(overlay_r, lam)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        Z,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
        extent=[theta_min, theta_max, eps_abs_min, eps_abs_max]
    )
    plt.colorbar(label="stable_rotation_fraction")
    plt.xlabel("theta")
    plt.ylabel("|eps|")
    plt.title("Phase diagram + theory overlay (dominance bound)")
    # overlay: horizontal line eps_abs = eps_bound
    plt.hlines(
        eps_bound,
        xmin=theta_min,
        xmax=theta_max,
        linestyles="--",
        linewidth=2,
        label=f"dominance bound @ r={overlay_r:g}: |eps| <= {eps_bound:.3f}"
    )
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outdir / "phase_diagram_with_theory_overlay.png")
    plt.close()

    print(f"Saved plots to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outcsv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_phase_diagram(args.config, args.outcsv, args.outdir, args.seed)
