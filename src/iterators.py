from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# -----------------------------
# Controlled spiral parameters
# -----------------------------
@dataclass
class ControlledConfig:
    radial_mode: str = "additive"   # "additive" (Option A) or "power" (Option B)
    delta: float = 0.01             # δ for additive
    alpha: float = 1.05             # α for power
    omega: float = 0.2              # ω base rotation
    phase_eps: float = 0.0          # strength of nonlinear φ(z)

def default_phi(z: complex, phase_eps: float = 0.0):
    """Small nonlinear phase term φ(z). Keep tiny for coherent arms."""
    if phase_eps == 0.0:
        return 0.0
    # Smooth, bounded perturbation using angle + magnitude
    return phase_eps * np.sin(np.angle(z)) * np.tanh(np.abs(z))

def radial_add(delta: float):
    return lambda r: r + delta

def radial_pow(alpha: float):
    # protect against r ~ 0
    return lambda r: np.maximum(r, 1e-8) ** alpha


def iterate_map(
    z0: complex,
    c: complex,
    max_iter: int = 200,
    mode: str = "quadratic",
    radial_update=None,
    omega: float = 0.2,
    phi=None,
    escape_radius: float = 1e6,
    stop_on_escape: bool = True,  # NEW: if False, keep iterating even after crossing escape_radius
    stop_radius: float = None,  # NEW: alternative stop condition - stop when |z| > stop_radius
    # ---- controlled spiral args (all optional) ----
    radial_mode: str = "additive",
    delta: float = 0.01,
    alpha: float = 1.05,
    phase_eps: float = 0.0,
    # ---- theorem map args ----
    theta: float = 0.0,
    lam: complex = 1.0 + 0j,
    eps: complex = 0.0 + 0j,
    crash_radius: float = 1e-6,
):
    """
    Iterate a complex map starting from z0.

    mode:
        "quadratic"   -> z_{n+1} = z_n^2 + c
        "exp"         -> z_{n+1} = exp(z_n) + c
        "controlled"  -> polar update with controlled radius + angle
        "theorem_map" -> z_{n+1} = e^{iθ} z + λ z^2 + ε z^{-2}

    Controlled radius options:
        Option A (additive): r_{n+1} = r_n + δ
        Option B (power):    r_{n+1} = r_n^α

    Controlled angle:
        θ_{n+1} = θ_n + ω + φ(z_n)

    escape behavior:
        - if stop_on_escape=True: stop when |z| > escape_radius (classic escape-time)
        - if stop_on_escape=False: continue iterating (useful for asymptotic tail fitting)
    """

    # Back-compat: if caller provides radial_update/phi explicitly, use them.
    if phi is None:
        phi = lambda z: default_phi(z, phase_eps)
    if radial_update is None:
        if radial_mode == "additive":
            radial_update = radial_add(delta)
        elif radial_mode == "power":
            radial_update = radial_pow(alpha)
        else:
            radial_update = radial_add(delta)  # safe fallback

    z = z0
    traj = []

    for _ in range(max_iter):
        if mode == "quadratic":
            z = z * z + c

        elif mode == "exp":
            z = np.exp(z) + c

        elif mode == "controlled":
            r = np.abs(z)
            th = np.angle(z)
            r = radial_update(r)
            th = th + omega + phi(z)
            z = r * np.exp(1j * th) + c

        elif mode == "theorem_map":
            # Crash: stop orbit if we enter the singular region
            if np.abs(z) < crash_radius:
                break

            # z_next = e^{iθ} z + λ z^2 + ε z^{-2}
            term1 = np.exp(1j * theta) * z
            term2 = lam * (z * z)
            term3 = eps * (1.0 / (z * z))  # z^{-2}
            z = term1 + term2 + term3

        else:
            raise ValueError(f"Unknown mode: {mode}")

        traj.append(z)

        # Numerical safety stop
        if (
            np.isnan(z.real) or np.isnan(z.imag) or
            np.isinf(z.real) or np.isinf(z.imag)
        ):
            break

        # Escape stop (optional)
        if stop_on_escape and (np.abs(z) > escape_radius):
            break

        # Stop radius check (hard stop regardless of stop_on_escape)
        if stop_radius is not None and (np.abs(z) > stop_radius):
            break

    return np.array(traj, dtype=np.complex128)


def pick_iterator(map_name: str, controlled_cfg: ControlledConfig | None = None):
    """
    Return an iterator function that matches the renderer's expected signature:
        iterator(z0, c, max_iter, escape_radius) -> (n_iters, last_z)

    controlled_cfg lets the sweep pass parameters into controlled mode.
    """
    name = map_name.lower()

    if name == "quadratic":
        def iterator(z0, c, max_iter, escape_radius):
            traj = iterate_map(z0=z0, c=c, max_iter=max_iter, mode="quadratic",
                              escape_radius=escape_radius, stop_on_escape=True)
            n = len(traj)
            last = traj[-1] if n > 0 else z0
            return n, last
        return iterator

    if name == "exp":
        def iterator(z0, c, max_iter, escape_radius):
            traj = iterate_map(z0=z0, c=c, max_iter=max_iter, mode="exp",
                              escape_radius=escape_radius, stop_on_escape=True)
            n = len(traj)
            last = traj[-1] if n > 0 else z0
            return n, last
        return iterator

    if name == "controlled":
        if controlled_cfg is None:
            controlled_cfg = ControlledConfig()

        def iterator(z0, c, max_iter, escape_radius):
            traj = iterate_map(
                z0=z0,
                c=c,
                max_iter=max_iter,
                mode="controlled",
                radial_mode=controlled_cfg.radial_mode,
                delta=controlled_cfg.delta,
                alpha=controlled_cfg.alpha,
                omega=controlled_cfg.omega,
                phase_eps=controlled_cfg.phase_eps,
                escape_radius=escape_radius,
                stop_on_escape=True,
            )
            n = len(traj)
            last = traj[-1] if n > 0 else z0
            return n, last

        return iterator

    raise ValueError(f"Unknown map name: {map_name}")
