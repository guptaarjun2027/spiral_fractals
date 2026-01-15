# Rigor Modules Documentation

## Overview

The **rigor modules** (Step 4) provide judge-proof numerical validation of spiral fractal behavior through:

1. **Rigor Sensitivity (4.1)**: Tests numerical convergence of κ (pitch) and β (scaling exponent) across computational tiers
2. **Phase Diagram (4.2)**: Maps spiral existence in parameter space with theoretical boundary overlay

## Repository Structure

**Git Root**: `/Users/rohannagaram/Documents/spiral_fractals`

All scripts should be run from the git root directory. The repository contains:

```
/Users/rohannagaram/Documents/spiral_fractals/
├── .git/                      # Git repository (this is the ONLY .git)
├── configs/
│   └── rigor_modules.yaml     # Configuration for Step 4
├── experiments/
│   ├── rigor_modules.py       # Unified runner (NEW)
│   ├── rigor_sensitivity.py   # Step 4.1
│   ├── phase_diagram.py       # Step 4.2
│   └── analysis_utils.py      # Growth-valid orbit logic
├── results/
│   ├── rigor_sensitivity.csv  # Convergence data
│   └── phase_map.csv          # Phase map data
└── figures/rigor/
    ├── convergence_strict_tiers.png
    └── phase_diagram.png
```

> **Note**: There may be an inner `spiral_fractals/` directory (NOT a git repo). Ignore it. Always work from the top-level directory containing `.git/`.

---

## Running the Rigor Modules

### Quick Start

```bash
cd /Users/rohannagaram/Documents/spiral_fractals
export PYTHONPATH=/Users/rohannagaram/Documents/spiral_fractals:$PYTHONPATH
python3 experiments/rigor_modules.py --config configs/rigor_modules.yaml
```

### Individual Modules

**Step 4.1 only:**
```bash
python3 experiments/rigor_sensitivity.py --config configs/rigor_modules.yaml
```

**Step 4.2 only:**
```bash
python3 experiments/phase_diagram.py --config configs/rigor_modules.yaml
```

---

## Growth-Valid Orbit Logic (Judge-Proof Contract)

**Why previous implementation failed:**
- Required orbits to reach `escape_radius` within `max_iter` (most didn't)
- Tail filtering used "last N iterations before escape" (fails if no escape)
- Result: `n_valid_orbits=0` → blank κ/β

**Growth-valid orbit solution:**

An orbit is **growth-valid** if:

1. **Reaches growth window**: `r >= r_growth_min` at some iteration
2. **Has sufficient tail**: At least `min_tail_points` points with `r >= r_growth_min`
3. **Shows outward growth**: Tail is monotonically increasing (or nearly so)

**Tail window defined by RADII, not iterations:**
- Tail points: all `z(t)` with `r_growth_min <= |z(t)| <= r_growth_max`
- No requirement for orbit to "escape" to `escape_radius`

**κ (pitch) estimation:**
- For each growth-valid orbit: fit `φ vs log(r)` on tail points
- Aggregate across orbits: median κ with bootstrap CI
- Reject extreme values: `|κ| > kappa_clip`

**Rejection tracking:**
- Each orbit records why it failed (if it did)
- Tier failure reasons include primary rejection cause

---

## Configuration Knobs

All parameters in [`configs/rigor_modules.yaml`](file:///Users/rohannagaram/Documents/spiral_fractals/configs/rigor_modules.yaml):

### Growth-Valid Orbit Settings

```yaml
orbit_validation:
  r_growth_min: 20.0      # Orbit must reach this radius
  r_growth_max: null      # Upper limit for tail (null = no limit)
  min_tail_points: 30     # Minimum points in tail window
  min_valid_orbits: 8     # Minimum successful orbits for passing tier
```

**Tuning guidance:**
- **`r_growth_min`**: Lower = more orbits qualify, but noisier κ. Higher = fewer orbits, cleaner signal.
- **`min_tail_points`**: Higher = more robust fits, but rejects more orbits.
- **`min_valid_orbits`**: Minimum for statistical significance.

### Kappa Estimation

```yaml
pitch:
  bootstrap_B: 200        # Bootstrap iterations for CI
  kappa_clip: 10.0        # Reject |κ| > this value
```

### Beta (Scaling) Detection

```yaml
scaling:
  min_points: 5           # Minimum points in scaling window
  min_span_decade: 0.5    # Minimum log₁₀(r) span
  r_sq_thresh: 0.95       # Minimum R² for fit
```

**Note**: β detection is strict. If no asymptotic scaling regime exists, β is blank with `fail_reason="beta_scaling_regime_not_found"`.

---

## Output Files

### `results/rigor_sensitivity.csv`

**Columns (exact order):**
```
tier, grid_n, max_iters, escape_radius, param_id, theta, lam, eps,
wedge_found, wedge_tau, wedge_eta, wedge_valid_fraction,
kappa_hat, kappa_ci_low, kappa_ci_high, n_orbits_total, n_orbits_valid,
scaling_found, beta_hat, beta_ci_low, beta_ci_high, n_valid_points,
fail_reason
```

**Interpretation:**
- **`fail_reason=None`**: Tier succeeded (both κ and β estimated)
- **`fail_reason="insufficient_valid_orbits(...)"`**: Too few orbits passed growth-valid check
- **`fail_reason="beta_scaling_regime_not_found"`**: κ succeeded, but no asymptotic β

### `results/phase_map.csv`

**Columns:**
```
theta, eps, lam, spiral_exists, wedge_found, wedge_valid_fraction,
kappa_hat, n_orbits_valid, escape_fraction, fail_reason
```

**Spiral classification:**
```python
spiral_exists = (wedge_found == 1) AND
                (kappa_estimated == 1) AND  
                (n_orbits_valid >= min_valid_orbits)
```

**Note**: β is NOT required for spiral existence.

---

## Troubleshooting

### All tiers show `n_orbits_valid=0`

**Diagnosis**: `r_growth_min` too high, or `max_iters` too low.

**Solutions:**
1. Lower `r_growth_min` (try 10.0 or 15.0)
2. Increase `max_iters` in tier configs
3. Check `fail_reason` in CSV for primary rejection cause

### Phase diagram is blank

**Diagnosis**: Either no spiral points found, or matrix orientation wrong.

**Check:**
1. Look at `results/phase_map.csv`: any rows with `spiral_exists=1`?
2. If yes but plot blank: matrix orientation bug (should be fixed now)
3. If no: parameters may be in non-spiral regime, or criteria too strict

**Solutions:**
1. Check `fail_reason` column for most common failure
2. Try parameter from `configs/theo rem_check.yaml` (known to have spirals)
3. Lower `min_valid_orbits` temporarily to test

### Convergence plot shows "Insufficient tiers"

**Expected** if most tiers fail. Check `fail_reason` in CSV.

If tiers have valid κ but plot says insufficient:
- Bug in convergence check logic (should be fixed now)

---

## Theory References

**Wedge condition**: See [`src/theorem_conditions.py`](file:///Users/rohannagaram/Documents/spiral_fractals/src/theorem_conditions.py)

**Theoretical sufficient boundary** (phase diagram):
```
λ >= 4/R₀ + 4ε/R₀⁴
```
where R₀ is conservative radius (default 5.0).

---

## Questions?

- **Config**: [`configs/rigor_modules.yaml`](file:///Users/rohannagaram/Documents/spiral_fractals/configs/rigor_modules.yaml)
- **Code**: [`experiments/rigor_modules.py`](file:///Users/rohannagaram/Documents/spiral_fractals/experiments/rigor_modules.py)
- **Utils**: [`experiments/analysis_utils.py`](file:///Users/rohannagaram/Documents/spiral_fractals/experiments/analysis_utils.py)
