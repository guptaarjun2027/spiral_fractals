import argparse
import numpy as np
from pathlib import Path
import os
import sys

# Ensure repository root is on sys.path so `from src...` works when running
# this script directly (e.g. `python scripts/make_image.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.iterators import iterate_map, radial_add
from src.render import render_grid
from src.utils import parse_complex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="quadratic",
                        choices=["quadratic", "exp", "controlled"])
    parser.add_argument("--c", type=str, required=True)
    parser.add_argument("--xmin", type=float, required=True)
    parser.add_argument("--xmax", type=float, required=True)
    parser.add_argument("--ymin", type=float, required=True)
    parser.add_argument("--ymax", type=float, required=True)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--escape", type=float, default=1e6)
    parser.add_argument("--outfile", type=str, required=True)

    # controlled map parameters
    parser.add_argument("--delta_r", type=float, default=0.05)
    parser.add_argument("--omega", type=float, default=0.25)

    args = parser.parse_args()

    c = parse_complex(args.c)
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run] map={args.map}, c={c}, saving to {out_path}")

    if args.map == "controlled":
        radial = radial_add(args.delta_r)

        def step_fn(z0):
            return iterate_map(
                z0=z0,
                c=c,
                max_iter=args.max_iter,
                mode="controlled",
                radial_update=radial,
                omega=args.omega,
                escape_radius=args.escape,
            )
    else:
        def step_fn(z0):
            return iterate_map(
                z0=z0,
                c=c,
                max_iter=args.max_iter,
                mode=args.map,
                escape_radius=args.escape,
            )

    img = render_grid(
        step_fn=step_fn,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        width=args.width,
        height=args.height,
        max_iter=args.max_iter,
    )

    from PIL import Image
    im = Image.fromarray(img)
    im.save(out_path)
    print("[run] done.")


if __name__ == "__main__":
    main()
