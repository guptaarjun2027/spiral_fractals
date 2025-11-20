import numpy as np

def iterate_map(
    z0: complex,
    c: complex,
    max_iter: int = 200,
    mode: str = "quadratic",
    radial_update=None,
    omega: float = 0.2,
    phi=None,
    escape_radius: float = 1e6,
):
    """
    Iterate a complex map starting from z0.

    mode:
        "quadratic"  -> z_{n+1} = z_n^2 + c
        "exp"        -> z_{n+1} = exp(z_n) + c
        "controlled" -> polar update with custom radial and angular rules
    """
    if phi is None:
        phi = lambda z: 0.0
    if radial_update is None:
        radial_update = lambda r: r

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
            z = r * np.exp(1j * th)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        traj.append(z)

        if np.abs(z) > escape_radius or np.isnan(z.real) or np.isinf(z.real):
            break

    return np.array(traj, dtype=np.complex128)


def radial_add(delta: float):
    return lambda r: r + delta


def radial_pow(alpha: float):
    return lambda r: r ** alpha


def pick_iterator(map_name: str):
    """Return an iterator function that matches the renderer's expected signature:
    iterator(z0, c, max_iter, escape_radius) -> (n_iters, last_z)
    """
    name = map_name.lower()

    if name == "quadratic":
        def iterator(z0, c, max_iter, escape_radius):
            traj = iterate_map(z0=z0, c=c, max_iter=max_iter, mode="quadratic", escape_radius=escape_radius)
            n = len(traj)
            last = traj[-1] if n > 0 else z0
            return n, last
        return iterator

    if name == "exp":
        def iterator(z0, c, max_iter, escape_radius):
            traj = iterate_map(z0=z0, c=c, max_iter=max_iter, mode="exp", escape_radius=escape_radius)
            n = len(traj)
            last = traj[-1] if n > 0 else z0
            return n, last
        return iterator

    if name == "controlled":
        def iterator(z0, c, max_iter, escape_radius):
            radial = radial_add(0.05)
            traj = iterate_map(z0=z0, c=c, max_iter=max_iter, mode="controlled", radial_update=radial, omega=0.25, escape_radius=escape_radius)
            n = len(traj)
            last = traj[-1] if n > 0 else z0
            return n, last
        return iterator

    raise ValueError(f"Unknown map name: {map_name}")
