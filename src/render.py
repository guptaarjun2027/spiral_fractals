# src/render.py
import numpy as np
from PIL import Image
from .iterators import pick_iterator
from .coloring import color_by_iters, color_by_angle, color_by_smooth

def make_grid(xmin, xmax, ymin, ymax, width, height):
    """
    Returns a complex grid Z0 with shape (height, width).
    """
    xs = np.linspace(xmin, xmax, width, dtype=np.float64)
    ys = np.linspace(ymin, ymax, height, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    return X + 1j * Y

def render_image(map_name: str,
                 c: complex,
                 xmin: float, xmax: float, ymin: float, ymax: float,
                 width: int, height: int,
                 max_iter: int = 200,
                 escape_radius: float = 1e6,
                 color_mode: str = "iters",
                 power_hint: float = 2.0):
    """
    Render an image for the selected map.
    Important: choose [xmin,xmax] and [ymin,ymax] with radii > 1 to seed |z0| > 1.
    """
    iterator = pick_iterator(map_name)
    Z0 = make_grid(xmin, xmax, ymin, ymax, width, height)
    iters = np.zeros((height, width), dtype=np.int32)
    lastz = np.zeros((height, width), dtype=np.complex128)

    # Iterate per pixel (vectorized loop over rows to keep memory reasonable)
    for i in range(height):
        for j in range(width):
            n, zlast = iterator(Z0[i, j], c, max_iter, escape_radius)
            iters[i, j] = n
            lastz[i, j] = zlast

    # Color
    mode = color_mode.lower()
    if mode == "iters":
        rgb = color_by_iters(iters, max_iter)
    elif mode == "angle":
        rgb = color_by_angle(lastz)
    elif mode == "smooth":
        rgb = color_by_smooth(iters, lastz, power_hint=power_hint, max_iter=max_iter)
    else:
        raise ValueError("color_mode must be one of: iters, angle, smooth")

    return Image.fromarray(rgb)
