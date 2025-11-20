# src/render.py
import numpy as np
from PIL import Image
try:
    from .iterators import pick_iterator
except Exception:
    # Fallback: define pick_iterator here using iterate_map/radial_add in case
    # the symbol isn't available due to import ordering or older files.
    from .iterators import iterate_map, radial_add

    def pick_iterator(map_name: str):
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


def render_grid(*,
                step_fn=None,
                map_name: str = None,
                c: complex = None,
                xmin: float = None, xmax: float = None, ymin: float = None, ymax: float = None,
                width: int = None, height: int = None,
                max_iter: int = 200,
                escape_radius: float = 1e6,
                color_mode: str = "iters",
                power_hint: float = 2.0):
    """Return an RGB NumPy array (height, width, 3).

    Two modes supported:
    - If `step_fn` is provided: it should be a callable taking `z0` and returning either
      a trajectory array (np.array of complex) or a tuple `(n_iters, last_z)`.
    - Otherwise, `map_name` must be provided and the function will use `pick_iterator`
      to obtain a per-pixel iterator.
    """
    if step_fn is None:
        if map_name is None:
            raise ValueError("Either step_fn or map_name must be provided")
        iterator = pick_iterator(map_name)
        use_callable = False
    else:
        iterator = step_fn
        use_callable = True

    Z0 = make_grid(xmin, xmax, ymin, ymax, width, height)
    iters = np.zeros((height, width), dtype=np.int32)
    lastz = np.zeros((height, width), dtype=np.complex128)

    for i in range(height):
        for j in range(width):
            z0 = Z0[i, j]
            if use_callable:
                out = iterator(z0)
                # step_fn from scripts returns trajectory arrays, but other callables
                # may return (n, lastz). Handle both.
                if isinstance(out, tuple) and len(out) == 2:
                    n, zlast = out
                else:
                    traj = np.asarray(out)
                    n = len(traj)
                    zlast = traj[-1] if n > 0 else z0
            else:
                n, zlast = iterator(z0, c, max_iter, escape_radius)

            iters[i, j] = n
            lastz[i, j] = zlast

    mode = color_mode.lower()
    if mode == "iters":
        rgb = color_by_iters(iters, max_iter)
    elif mode == "angle":
        rgb = color_by_angle(lastz)
    elif mode == "smooth":
        rgb = color_by_smooth(iters, lastz, power_hint=power_hint, max_iter=max_iter)
    else:
        raise ValueError("color_mode must be one of: iters, angle, smooth")

    return rgb
