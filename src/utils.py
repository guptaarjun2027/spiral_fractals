# src/utils.py
import re

def parse_complex(s: str) -> complex:
    """
    Parse strings like '0.3+0.5j' or '-0.4-0.6j' into a complex number.
    """
    s = s.strip().lower().replace(" ", "")
    if s.endswith("j") and ("+" in s[1:] or "-" in s[1:]):
        return complex(s)
    # allow plain real numbers too
    return complex(float(s), 0.0)

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))
