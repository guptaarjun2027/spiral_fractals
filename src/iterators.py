# src/iterators.py
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def iterate_quadratic(z0: complex, c: complex, max_iter: int, escape_radius: float):
    """
    z_{n+1} = z_n^2 + c
    Returns (n_iter, z_last)
    """
    z = z0
    r2 = escape_radius * escape_radius
    for n in range(max_iter):
        # z = z*z + c
        zr = z.real
        zi = z.imag
        # (a+bi)^2 = (a^2 - b^2) + 2ab i
        zr2 = zr*zr - zi*zi + c.real
        zi2 = 2.0*zr*zi + c.imag
        z = complex(zr2, zi2)
        if (z.real*z.real + z.imag*z.imag) > r2:
            return n + 1, z
    return max_iter, z

@njit(cache=True, fastmath=True)
def iterate_exponential(z0: complex, c: complex, max_iter: int, escape_radius: float):
    """
    z_{n+1} = exp(z_n) + c
    """
    z = z0
    r2 = escape_radius * escape_radius
    for n in range(max_iter):
        # exp(a+bi) = exp(a) * (cos b + i sin b)
        a = z.real
        b = z.imag
        ea = np.exp(a)
        z = complex(ea * np.cos(b) + c.real, ea * np.sin(b) + c.imag)
        if (z.real*z.real + z.imag*z.imag) > r2:
            return n + 1, z
    return max_iter, z

def pick_iterator(name: str):
    name = name.lower()
    if name in ("quad", "quadratic"):
        return iterate_quadratic
    if name in ("exp", "exponential"):
        return iterate_exponential
    raise ValueError(f"Unknown map '{name}'. Use 'quadratic' or 'exponential'.")
