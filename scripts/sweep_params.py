"""
Parameter sweep engine for spiral_fractals.

Run:
    python -m scripts.sweep_params

Outputs:
    figures/sweeps/*.png
    results/spiral_sweep.csv

Sweeps:
- quadratic map: c grid
- exponential map: c grid
- controlled map:
    additive mode: (delta_r, omega, phase_eps)
    power mode:    (alpha, omega, phase_eps)

Controlled map uses a CHAOTIC CORE (quadratic by default) + polar drift,
so the escape boundary is fractal and can form spiral arms.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

# -------- Repo-root relative paths --------
ROOT = Path(__file__).resolve().parents[1]
FIG_SWEEPS = ROOT / "figures" / "sweeps"
RESULTS_DIR = ROOT / "results"
DEFAULT_CSV = RESULTS_DIR / "spiral_sweep.csv"

# -------- Import render machinery --------
_RENDER_GRID = None
def _soft_import():
    global _RENDER_GRID
    candidates = [
        ("src.render", "render_grid"),
        ("src.render", "render_fractal"),
        ("src.engine", "render_grid"),
        ("src.engine", "render_fractal"),
    ]
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                _RENDER_GRID = fn
                break
        except Exception:
            pass
    if _RENDER_GRID is None:
        raise ImportError("Could not find render_grid/render_fractal.")
_soft_import()


# -------- Parameter dataclass --------
@dataclass
class SweepParams:
    id: str
    map_type: str  # "quadratic" | "exponential" | "controlled"
    c_real: float
    c_imag: float

    radial_mode: str   # "additive" | "power" | "nan"
    delta_r: float     # δ for additive else nan
    alpha: float       # α for power else nan
    omega: float       # ω for controlled else nan
    phase_eps: float   # modulation else nan
    k_arms: int        # arm symmetry (controlled only)

    grid_size: int
    max_iter: int
    escape_radius: float
    image_path: str


def generate_parameter_grid(
    map_types: List[str],
    c_real_vals: Iterable[float],
    c_imag_vals: Iterable[float],
    delta_r_vals: Iterable[float],
    alpha_vals: Iterable[float],
    omega_vals: Iterable[float],
    phase_eps_vals: Iterable[float],
    radial_modes: Iterable[str],
    k_arms_vals: Iterable[int],
) -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []

    for mt in map_types:
        if mt in ("quadratic", "exponential"):
            for cr, ci in itertools.product(c_real_vals, c_imag_vals):
                grid.append(dict(
                    map_type=mt,
                    c=complex(cr, ci),
                    c_real=cr, c_imag=ci,
                    radial_mode="nan",
                    delta_r=float("nan"),
                    alpha=float("nan"),
                    omega=float("nan"),
                    phase_eps=float("nan"),
                    k_arms=int("0"),
                ))

        elif mt == "controlled":
            for rm in radial_modes:
                for k in k_arms_vals:
                    if rm == "additive":
                        for dr, om, pe in itertools.product(delta_r_vals, omega_vals, phase_eps_vals):
                            grid.append(dict(
                                map_type=mt,
                                c=0 + 0j,
                                c_real=0.0, c_imag=0.0,
                                radial_mode=rm,
                                delta_r=dr,
                                alpha=float("nan"),
                                omega=om,
                                phase_eps=pe,
                                k_arms=k,
                            ))
                    elif rm == "power":
                        for a, om, pe in itertools.product(alpha_vals, omega_vals, phase_eps_vals):
                            grid.append(dict(
                                map_type=mt,
                                c=0 + 0j,
                                c_real=0.0, c_imag=0.0,
                                radial_mode=rm,
                                delta_r=float("nan"),
                                alpha=a,
                                omega=om,
                                phase_eps=pe,
                                k_arms=k,
                            ))
                    else:
                        raise ValueError(f"Unknown radial_mode: {rm}")
        else:
            raise ValueError(f"Unknown map_type: {mt}")

    return grid


def _render_dispatch(
    map_type: str,
    c: complex,
    radial_mode: str,
    delta_r: float,
    alpha: float,
    omega: float,
    phase_eps: float,
    k_arms: int,
    grid_size: int,
    max_iter: int,
    escape_radius: float,
    out_path: Path,
):
    from PIL import Image

    map_name = map_type if map_type != "exponential" else "exp"

    if map_type == "controlled":
        from src.controlled_map import iterate_controlled, ControlledConfig

        cfg = ControlledConfig(
            core="quadratic",          # keep fractal boundary
            radial_mode=radial_mode,
            delta_r=delta_r,
            alpha=alpha,
            omega=omega,
            phase_eps=phase_eps,
            k_arms=k_arms,
            escape_radius=escape_radius,
        )

        def step_fn(z0):
            traj = iterate_controlled(z0=z0, c=c, cfg=cfg, max_iter=max_iter)
            n = len(traj)
            last = traj[-1] if n > 0 else z0
            return n, last

        rgb = _RENDER_GRID(
            step_fn=step_fn,
            xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
            width=grid_size, height=grid_size,
            max_iter=max_iter,
            escape_radius=escape_radius,
            color_mode="iters",
        )
    else:
        rgb = _RENDER_GRID(
            map_name=map_name,
            c=c,
            xmin=-2.5, xmax=2.5, ymin=-2.5, ymax=2.5,
            width=grid_size, height=grid_size,
            max_iter=max_iter,
            escape_radius=escape_radius,
            color_mode="iters",
        )

    Image.fromarray(rgb).save(str(out_path))


def write_csv(rows: List[SweepParams], csv_path: Path, overwrite: bool):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or (not csv_path.exists()) else "a"
    write_header = (mode == "w")
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--map-types", nargs="*", default=["quadratic", "exponential", "controlled"])
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--max-iter", type=int, default=350)
    p.add_argument("--escape-radius", type=float, default=80.0)
    p.add_argument("--output-dir", type=str, default=str(FIG_SWEEPS))
    p.add_argument("--output-csv", type=str, default=str(DEFAULT_CSV))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--small", action="store_true")
    p.add_argument("--controlled-only", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--radial-modes", nargs="*", default=["additive", "power"])
    p.add_argument("--delta-r-values", nargs="*", type=float, default=None)
    p.add_argument("--alpha-values", nargs="*", type=float, default=None)
    p.add_argument("--omega-values", nargs="*", type=float, default=None)
    p.add_argument("--phase-eps-values", nargs="*", type=float, default=None)
    p.add_argument("--k-arms-values", nargs="*", type=int, default=None)

    p.add_argument("--c-real-values", nargs="*", type=float, default=None)
    p.add_argument("--c-imag-values", nargs="*", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    if args.controlled_only:
        args.map_types = ["controlled"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for fp in out_dir.glob("*.png"):
            fp.unlink()

    if args.small:
        c_real_vals = args.c_real_values or np.linspace(-0.3, 0.3, 3)
        c_imag_vals = args.c_imag_values or np.linspace(-0.3, 0.3, 3)
        delta_r_vals = args.delta_r_values or [0.001, 0.006, 0.015]
        alpha_vals = args.alpha_values or [1.01, 1.04]
        omega_vals = args.omega_values or [0.2*math.pi, 0.3*math.pi, 0.4*math.pi]
        phase_eps_vals = args.phase_eps_values or [0.00, 0.01, 0.02]
        k_arms_vals = args.k_arms_values or [2, 3, 4]
    elif args.full:
        c_real_vals = args.c_real_values or np.linspace(-0.6, 0.6, 9)
        c_imag_vals = args.c_imag_values or np.linspace(-0.6, 0.6, 9)
        delta_r_vals = args.delta_r_values or [0.001, 0.003, 0.006, 0.01, 0.015]
        alpha_vals = args.alpha_values or [1.01, 1.02, 1.04, 1.06]
        omega_vals = args.omega_values or [0.2*math.pi, 0.25*math.pi, 0.3*math.pi, 0.35*math.pi, 0.4*math.pi]
        phase_eps_vals = args.phase_eps_values or [0.00, 0.01, 0.02, 0.03]
        k_arms_vals = args.k_arms_values or [2, 3, 4, 5]
    else:
        c_real_vals = args.c_real_values or np.linspace(-0.5, 0.5, 5)
        c_imag_vals = args.c_imag_values or np.linspace(-0.5, 0.5, 5)
        delta_r_vals = args.delta_r_values or [0.001, 0.003, 0.006, 0.01, 0.015]
        alpha_vals = args.alpha_values or [1.01, 1.02, 1.04, 1.06]
        omega_vals = args.omega_values or [0.2*math.pi, 0.25*math.pi, 0.3*math.pi, 0.35*math.pi, 0.4*math.pi]
        phase_eps_vals = args.phase_eps_values or [0.00, 0.01, 0.02]
        k_arms_vals = args.k_arms_values or [2, 3, 4]

    grid = generate_parameter_grid(
        map_types=args.map_types,
        c_real_vals=c_real_vals,
        c_imag_vals=c_imag_vals,
        delta_r_vals=delta_r_vals,
        alpha_vals=alpha_vals,
        omega_vals=omega_vals,
        phase_eps_vals=phase_eps_vals,
        radial_modes=args.radial_modes,
        k_arms_vals=k_arms_vals,
    )

    rows: List[SweepParams] = []
    for idx, g in enumerate(grid):
        sid = f"sweep_{idx:05d}"
        map_type = g["map_type"]
        c = g["c"]
        cr, ci = g["c_real"], g["c_imag"]
        rm = g["radial_mode"]
        dr = g["delta_r"]
        a  = g["alpha"]
        om = g["omega"]
        pe = g["phase_eps"]
        k  = g["k_arms"]

        img_name = f"{map_type}_{idx:05d}.png"
        img_path = out_dir / img_name

        try:
            _render_dispatch(
                map_type=map_type,
                c=c,
                radial_mode=rm,
                delta_r=dr,
                alpha=a,
                omega=om,
                phase_eps=pe,
                k_arms=k,
                grid_size=args.grid_size,
                max_iter=args.max_iter,
                escape_radius=args.escape_radius,
                out_path=img_path,
            )
        except Exception as e:
            print(f"[ERROR] {img_name}: {e}")
            img_path = Path("")

        rows.append(SweepParams(
            id=sid, map_type=map_type,
            c_real=float(cr), c_imag=float(ci),
            radial_mode=str(rm),
            delta_r=float(dr), alpha=float(a),
            omega=float(om), phase_eps=float(pe),
            k_arms=int(k),
            grid_size=args.grid_size,
            max_iter=args.max_iter,
            escape_radius=args.escape_radius,
            image_path=str(img_path.relative_to(ROOT)) if img_path != Path("") else "",
        ))

        if (idx + 1) % 10 == 0:
            print(f"[sweep] rendered {idx+1}/{len(grid)}")

    write_csv(rows, Path(args.output_csv), overwrite=args.overwrite)
    print(f"[done] wrote {len(rows)} rows to {args.output_csv}")
    print(f"[done] images in {out_dir}")


if __name__ == "__main__":
    main()
