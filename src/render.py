import numpy as np

try:
    from src.iterators import pick_iterator
except Exception:
    # fallback if running as script without package context
    from iterators import pick_iterator


def _colorize_iters(iters, max_iter):
    """Map iteration counts to grayscale RGB."""
    iters = np.asarray(iters, dtype=np.float32)
    norm = iters / max_iter
    # invert so interior is dark, exterior bright (looks nicer)
    norm = 1.0 - norm
    gray = (255 * norm).clip(0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _colorize_continuous(last_z):
    """
    Continuous fallback coloring based on magnitude and angle of last_z.
    Produces contrast even if iters are constant.
    """
    last_z = np.asarray(last_z, dtype=np.complex128)

    mag = np.abs(last_z)
    ang = np.angle(last_z)

    # log magnitude for dynamic range
    mag_log = np.log1p(mag)
    mag_norm = mag_log / (mag_log.max() + 1e-9)

    # angle in [0,1]
    ang_norm = (ang + np.pi) / (2 * np.pi)

    # simple HSV-ish mapping without extra deps
    r = (mag_norm * 255).astype(np.uint8)
    g = (ang_norm * 255).astype(np.uint8)
    b = ((1 - mag_norm) * 255).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


def render_grid(
    *,
    # either give map_name+c OR a step_fn
    map_name=None,
    c=0 + 0j,
    step_fn=None,

    xmin=-2.5, xmax=2.5,
    ymin=-2.5, ymax=2.5,
    width=512, height=512,
    max_iter=300,
    escape_radius=10.0,
    color_mode="iters",
):
    """
    Render a fractal grid.

    Compatible with your existing calls:
      render_grid(map_name="quadratic", c=..., ...)
      render_grid(step_fn=..., ...)

    color_mode:
      "iters"       -> grayscale by escape iters (default)
      "continuous"  -> magnitude/angle based
      "auto"        -> uses iters if varied; otherwise continuous fallback
    """

    # choose iterator
    if step_fn is None:
        if map_name is None:
            raise ValueError("Provide either map_name or step_fn.")
        iterator = pick_iterator(map_name)
        def _step(z0):
            return iterator(z0, c, max_iter, escape_radius)
    else:
        # step_fn should be a callable that takes (z0, c, max_iter, escape_radius)
        # or a pre-bound function that only takes z0
        import inspect
        sig = inspect.signature(step_fn)
        if len(sig.parameters) == 1:
            # Already bound or only takes z0
            def _step(z0):
                return step_fn(z0)
        else:
            # Expects full signature
            def _step(z0):
                return step_fn(z0, c, max_iter, escape_radius)

    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)

    iters = np.zeros((height, width), dtype=np.int32)
    lastz = np.zeros((height, width), dtype=np.complex128)

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            z0 = complex(x, y)
            n, last = _step(z0)
            iters[j, i] = n
            lastz[j, i] = last

    # decide coloring
    if color_mode == "iters":
        return _colorize_iters(iters, max_iter)

    if color_mode == "continuous":
        return _colorize_continuous(lastz)

    if color_mode == "auto":
        if np.std(iters) < 1e-6:
            return _colorize_continuous(lastz)
        return _colorize_iters(iters, max_iter)

    # fallback: auto
    if np.std(iters) < 1e-6:
        return _colorize_continuous(lastz)
    return _colorize_iters(iters, max_iter)


# Some repos used render_fractal instead of render_grid
def render_fractal(**kwargs):
    return render_grid(**kwargs)
