# src/controlled_map.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ControlledConfig:
    # core chaotic map to keep a fractal boundary
    core: str = "quadratic"   # "quadratic" | "exp"

    # controlled spiral params
    radial_mode: str = "additive"  # "additive" | "power"
    delta_r: float = 0.005         # additive → interpreted as small % growth
    alpha: float = 1.02            # power
    omega: float = 0.25 * np.pi    # constant phase drift
    phase_eps: float = 0.02        # modulation strength
    k_arms: int = 3                # number of arms / symmetry channels

    # if sweep passes c=0, use this to create a real Julia boundary
    c_bias: complex = complex(-0.12, 0.74)

    # guardrails
    min_radius: float = 1e-6
    escape_radius: float = 80.0


def core_step(z: complex, c: complex, core: str) -> complex:
    if core == "quadratic":
        return z * z + c
    if core == "exp":
        return np.exp(z) + c
    raise ValueError(f"Unknown core map: {core}")


def iterate_controlled(
    z0: complex,
    c: complex,
    cfg: ControlledConfig,
    max_iter: int = 350,
) -> np.ndarray:
    """
    Controlled polar drift (creates spiral arm channels)
    + chaotic core (creates fractal escape boundary).

    Order matters: polar-update FIRST, chaos SECOND.
    """
    # If sweep gives c=0, inject bias so the boundary isn't degenerate.
    c_eff = cfg.c_bias if abs(c) == 0.0 else c

    z = z0
    traj = []

    for _ in range(max_iter):
        # --- 1) polar of current state ---
        r = max(np.abs(z), cfg.min_radius)
        th = np.angle(z)

        # --- 2) symmetry breaking into k arms ---
        # arm-aligned angles get slightly boosted growth
        arm_mod = 1.0 + cfg.phase_eps * np.cos(cfg.k_arms * th)

        # --- 3) radial update ---
        if cfg.radial_mode == "additive":
            # interpret delta_r as *percentage* growth to avoid washout
            # r_next = r * (1 + delta_r * arm_mod)
            r_next = r * (1.0 + cfg.delta_r * arm_mod)
        elif cfg.radial_mode == "power":
            r_next = (r ** cfg.alpha) * arm_mod
        else:
            raise ValueError(f"Unknown radial_mode: {cfg.radial_mode}")

        # --- 4) angular update ---
        # constant drift + periodic perturbation + tiny r-coupling
        th_next = th + cfg.omega \
                  + cfg.phase_eps * np.sin(cfg.k_arms * th + np.angle(c_eff)) \
                  + 0.15 * cfg.phase_eps * np.sin(0.5 * r)

        # --- 5) rebuild intermediate spiral state ---
        z_spiral = r_next * np.exp(1j * th_next)

        # --- 6) chaotic core applied AFTER spiral shaping ---
        z = core_step(z_spiral, c_eff, cfg.core)

        traj.append(z)

        # escape
        if np.abs(z) > cfg.escape_radius or np.isnan(z.real) or np.isinf(z.real):
            break

    return np.array(traj, dtype=np.complex128)
