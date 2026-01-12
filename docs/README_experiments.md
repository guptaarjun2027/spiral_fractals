# Spiral Fractals Experiments

This directory contains the computational experiments for Step 3.

## 1. Theorem Check (Step 3.1)
Verifies the geometric properties of escaping orbits compared to theoretical predictions.

**Run command:**
```bash
python -m experiments.theorem_check --config configs/theorem_check.yaml --outcsv results/theorem_check.csv --outdir figures/
```

**Output:**
- `results/theorem_check.csv`: Statistics on pitch scaling ($k$) and escape fractions.
- `figures/theorem_phi_logr_*.png`: Plots of $\phi$ vs $\ln(r)$ for escaping trajectories.
- `figures/kappa_agreement.png`: Summary of agreement between estimated $k$.

## 2. Scaling Exponents (Step 3.2)
Computes the escape fraction $\rho(r)$ and fits the power law $\rho(r) \sim r^{-\beta}$.

**Run command:**
```bash
python -m experiments.scaling_exponents --config configs/scaling_exponents.yaml --outcsv results/scaling_exponents.csv --outdir figures/
```

**Output:**
- `results/scaling_exponents.csv`: Estimated exponents $\beta$ and pitch scalings.
- `figures/rho_loglog_*.png`: Log-log plots of escape fraction vs radius.
- `figures/pitch_vs_r_*.png`: Pitch scaling analysis.
