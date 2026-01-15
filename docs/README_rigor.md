# Rigor Modules Documentation

This directory contains experiments for **Step 4** of the Spiral Fractals project, focusing on numerical rigor and theoretical validation.

## 4.1 Numerical Convergence (Rigor Sensitivity)

Tests the stability of our measured spiral properties (Pitch $\kappa$, Scaling Exponent $\beta$) against computational parameters.

### Running the Experiment
```bash
python3 -m experiments.rigor_sensitivity \
  --config configs/rigor_sensitivity.yaml \
  --outcsv results/rigor_sensitivity.csv \
  --outdir figures/rigor_sensitivity
```

### Configuration
Edit `configs/rigor_sensitivity.yaml` to adjust:
- `tiers`: Define LOW/MED/HIGH settings for `num_angles`, `num_radii`, `max_iter`, etc.
- `params`: List of spiral parameter sets ($\theta, \lambda, \varepsilon$) to test.

### Output
- **CSV**: `results/rigor_sensitivity.csv`
    - Contains `median_kappa` and `beta` with confidence intervals for each Tier.
    - Look for consistency across Tiers (overlapping error bars).
- **Plot**: `figures/rigor_sensitivity/rigor_sensitivity.png`
    - Visual comparison of metrics across tiers.

## 4.2 Phase Diagram

Maps the "Spiral Phase" in the ($\lambda, \varepsilon$) plane and compares it with the theoretical sufficient condition.

### Running the Experiment
```bash
python3 -m experiments.phase_diagram \
  --config configs/phase_diagram.yaml \
  --outcsv results/phase_map.csv \
  --outdir figures/phase_diagram
```

### Configuration
Edit `configs/phase_diagram.yaml`:
- `sweep`: Range and steps for `lam` and `eps`.
- `theory`: Constants for the overlay curve ($R_0$).

### Output
- **CSV**: `results/phase_map.csv`
    - Raw classification data (1 = Spiral, 0 = Unbounded/No-Wedge).
- **Plot**: `figures/phase_diagram/phase_diagram.png`
    - Heatmap of the spiral phase.
    - Red line indicates the theoretical sufficient boundary ($\lambda \ge 4/R_0 + 4\varepsilon/R_0^4$).
    - Regions **below** the line that are still classified as spirals represent the gap between our conservative sufficient condition and the true boundary.
