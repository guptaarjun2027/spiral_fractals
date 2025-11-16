import argparse
import os
import numpy as np
from PIL import Image

def iterate_quadratic(z0: complex, c: complex, max_iter: int, escape: float) -> int:
    z = z0
    esc2 = escape * escape
    for i in range(max_iter):
        zr = z.real; zi = z.imag
        if zr*zr + zi*zi > esc2:
            return i
        z = z*z + c
    return max_iter

def parse_c(args) -> complex:
    # Priority: (cre,cim) ? c-string
    if args.cre is not None and args.cim is not None:
        return complex(float(args.cre), float(args.cim))
    if args.c is not None:
        s = args.c.strip()
        if "," in s:
            re, im = s.split(",", 1)
            return complex(float(re), float(im))
        return complex(s)
    # default if nothing provided
    return complex(0.0, 0.0)

def render_quadratic(c, xmin, xmax, ymin, ymax, width, height, max_iter, escape):
    xs = np.linspace(xmin, xmax, width, dtype=np.float64)
    ys = np.linspace(ymin, ymax, height, dtype=np.float64)
    img = np.zeros((height, width), dtype=np.uint16)
    for iy, y in enumerate(ys):
        if height >= 10 and iy % (height // 10 or 1) == 0:
            print(f"[render] row {iy+1}/{height}")
        for ix, x in enumerate(xs):
            iters = iterate_quadratic(complex(x, y), c, max_iter, escape)
            img[iy, ix] = iters
    m = img.max()
    if m > 0:
        img8 = (img.astype(np.float64) * (255.0/m)).astype(np.uint8)
    else:
        img8 = img.astype(np.uint8)
    return img8

def main():
    print("[make_image] starting")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cre", type=float, help="real part of c (preferred)")
    ap.add_argument("--cim", type=float, help="imag part of c (preferred)")
    ap.add_argument("--c", help='optional combined complex: "a,b" or "a+bj"')
    ap.add_argument("--xmin", type=float, required=True)
    ap.add_argument("--xmax", type=float, required=True)
    ap.add_argument("--ymin", type=float, required=True)
    ap.add_argument("--ymax", type=float, required=True)
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--escape", type=float, default=1e6)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    c = parse_c(args)
    print(f"[make_image] c={c} bounds=({args.xmin},{args.xmax})x({args.ymin},{args.ymax}) size={args.width}x{args.height}")

    img = render_quadratic(c, args.xmin, args.xmax, args.ymin, args.ymax,
                           args.width, args.height, args.max_iter, args.escape)

    outpath = os.path.normpath(args.outfile)
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    Image.fromarray(img, mode="L").save(outpath)
    print(f"[make_image] saved -> {outpath}")

if __name__ == "__main__":
    main()
