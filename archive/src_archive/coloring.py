# src/coloring.py
import numpy as np

def normalize(arr):
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    if amax <= amin:
        return np.zeros_like(arr)
    return (arr - amin) / (amax - amin)

def color_by_iters(iters, max_iter):
    """
    Grayscale by iteration count.
    """
    x = iters.astype(np.float64) / max_iter
    x = np.clip(x, 0.0, 1.0)
    rgb = np.stack([x, x, x], axis=-1)
    return (rgb * 255).astype(np.uint8)

def color_by_angle(final_z):
    """
    HSV-like mapping from angle to color. Value is fixed at 1, saturation by radius.
    """
    ang = np.angle(final_z)
    hue = (ang + np.pi) / (2.0 * np.pi)  # [0,1)
    sat = np.tanh(np.sqrt(np.real(final_z)**2 + np.imag(final_z)**2) / 5.0)
    val = np.ones_like(hue)
    return hsv_to_rgb(hue, sat, val)

def color_by_smooth(iters, last_z, power_hint=2.0, max_iter=200):
    """
    Smoothed escape time. Works best for power maps. For exp, power_hint can be > 2.
    """
    # Smoothed count: n + 1 - log log |z| / log power
    absz = np.abs(last_z)
    absz[absz == 0] = 1e-12
    sm = iters.astype(np.float64) + 1.0 - np.log(np.log(absz + 1e-12) + 1e-12) / np.log(power_hint)
    sm = np.where(iters >= max_iter, np.nan, sm)
    smn = normalize(np.nan_to_num(sm, nan=np.nanmax(sm[np.isfinite(sm)]) if np.any(np.isfinite(sm)) else 0.0))
    # Map to a simple gradient
    r = smn
    g = np.sqrt(smn)
    b = 1.0 - smn
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def hsv_to_rgb(h, s, v):
    """
    h,s,v in [0,1]. Returns uint8 RGB image with shape (..., 3)
    """
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i_mod = np.mod(i, 6)
    choices = np.stack([
        np.stack([v, t, p], axis=-1),
        np.stack([q, v, p], axis=-1),
        np.stack([p, v, t], axis=-1),
        np.stack([p, q, v], axis=-1),
        np.stack([t, p, v], axis=-1),
        np.stack([v, p, q], axis=-1),
    ], axis=0)
    rgb = choices[i_mod, np.arange(h.size).reshape(h.shape), :]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
