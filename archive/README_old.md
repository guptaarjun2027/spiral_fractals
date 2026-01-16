# Beyond the Mandelbrot: Modeling Real-World Spiral Growth Using Expanding Complex Dynamics

**ISEF 2026 Mathematics Research Project**

This repository contains the complete research pipeline for studying expanding complex fractals and their relationship to real-world spiral structures.

We investigate what happens when complex numbers with |z| > 1 are iterated under nonlinear maps such as
\\( z_{n+1} = z_n^2 + c \\) and \\( z_{n+1} = e^{z_n} + c \\).

Unlike traditional bounded fractals (e.g., the Mandelbrot set), these systems **expand outward** and often form striking spiral structures that resemble natural patterns—galaxies, hurricanes, shells, and plant growth.

Our goal is to **quantify** these spirals and determine whether simple iterative rules can reproduce the geometric properties of real-world spirals.

---

## 🔬 Research Question

Can expanding complex maps reproduce measurable spiral characteristics—slope \\(b\\), arm count, arm spacing \\(\Delta \theta\\), and fractal dimension \\(D\\)—found in natural spiral images better than baseline logarithmic or Fourier models?

---

## 🧠 Project Objectives

- Implement iterative complex maps (quadratic, exponential, controlled-spiral variants)
- Visualize outward-growing fractals and analyze how parameters affect spiral formation
- Fit generated patterns to the logarithmic-spiral equation \\( r = a e^{b\theta} \\)
- Compute box-counting fractal dimension \\(D\\)
- Validate theoretical scaling laws: \\( b \approx \delta/\omega \\) for additive radial growth
- Compare quantitative metrics with real spiral data (galaxies, hurricanes, shells)

---

## 🗂️ Repository Structure

```
spiral_fractals/
├── src/                      # Core modules
│   ├── iterators.py          # Complex map iterations (quadratic, exp, controlled)
│   ├── render.py             # Fractal rendering engine
│   ├── geometry.py           # Spiral geometry analysis (NEW: complete pipeline)
│   └── controlled_map.py     # Controlled polar dynamics
├── scripts/                  # Command-line tools
│   ├── sweep_params.py       # Parameter sweep engine (UPDATED: research-grade)
│   ├── quick_sanity.py       # Thumbnail sanity checker (NEW)
│   ├── real_compare.py       # Real-world spiral comparison (NEW)
│   └── run_all.py            # One-click reproducibility pipeline (NEW)
├── notebooks/                # Analysis notebooks
│   ├── analysis.ipynb        # Complete geometry analysis (NEW)
│   └── theory.ipynb          # Theoretical scaling law validation (NEW)
├── configs/                  # Parameter configurations (NEW)
│   ├── controlled_best.json  # Best spiral parameters
│   ├── sweep_small.json      # Quick test sweep
│   └── sweep_full.json       # Comprehensive sweep
├── tests/                    # Unit tests (NEW)
│   └── test_geometry.py      # Geometry pipeline tests
├── figures/                  # Auto-generated visualizations
│   ├── sweeps/               # Parameter sweep images
│   ├── analysis/             # Distribution plots, phase diagrams
│   └── real_matches/         # Real-world comparison panels
├── results/                  # Computed data
│   ├── spiral_sweep.csv      # All sweep parameters and paths
│   ├── spiral_metrics.csv    # Geometry metrics for all spirals
│   └── top_spirals.csv       # Best spirals ranked by quality
└── data/                     # External data
    └── real/                 # Real-world spiral images (user-provided)
```

---

## Geometry Analysis & Stability

New tooling for Week 3 analysis:

### 1. Geometry Quality Analysis
Run the notebook `notebooks/analysis_geometry.ipynb` to:
- Load metrics from `results/spiral_metrics.csv`
- Label spirals as "clean", "borderline", or "messy"
- Visualize distributions and example grids
- Save labeled metrics to `results/spiral_metrics_labeled.csv`

### 2. Stability Experiment
Run the stability experiment script:
```bash
python -m scripts.run_stability_experiment
```
This will:
- Pick 3 clean baseline spirals
- Perturb parameters (delta_r, omega)
- Generate new images in `figures/stability/`
- Analyze geometry and save results to `results/stability_metrics.csv`
- Generate summary plots in `figures/stability/`

## 🚀 Quick Start: ISEF Reproducibility

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- numpy, scipy, pandas
- matplotlib, seaborn
- Pillow, scikit-image
- jupyter, tqdm (optional but recommended)
- pytest (for tests)

### Complete Pipeline (One Command)

```bash
# Small sweep (quick test, ~5 minutes)
python -m scripts.run_all --sweep-mode small

# Full sweep (comprehensive, ~30-60 minutes)
python -m scripts.run_all --sweep-mode full
```

This will:
1. Generate controlled spiral images across parameter space
2. Compute geometry metrics (arm count, \\(b\\), \\(R^2\\), fractal dimension)
3. Export analysis figures and CSV results

### Step-by-Step Manual Reproduction

If you prefer to run each step individually:

#### Step 1: Generate Spiral Images

```bash
# Small controlled-only sweep (for testing)
python -m scripts.sweep_params --controlled-only --small

# Full controlled sweep (research-grade)
python -m scripts.sweep_params --controlled-only --full

# Quick sanity check (creates thumbnail grid)
python -m scripts.quick_sanity
```

**Output**:
- `figures/sweeps/*.png` - Generated spiral images
- `results/spiral_sweep.csv` - Parameter records
- `figures/analysis/sanity_grid.png` - First 9 spirals

#### Step 2: Analyze Geometry

Open and run:
```bash
jupyter notebook notebooks/analysis.ipynb
```

This notebook:
- Loads all sweep images
- Computes geometric features (arms, \\(b\\), \\(R^2\\), dimension)
- Generates distributions and parameter relationship plots
- Identifies "clean spiral" stability regions
- Exports top 20 spirals by quality

**Output**:
- `results/spiral_metrics.csv` - Full metrics table
- `results/top_spirals.csv` - Best spirals
- `figures/analysis/distributions.png`
- `figures/analysis/parameter_relationships.png`
- `figures/analysis/phase_diagrams.png`
- `figures/analysis/top_spirals_grid.png`

#### Step 3: Validate Theory

```bash
jupyter notebook notebooks/theory.ipynb
```

This notebook:
- Validates theoretical scaling law \\( b \approx \delta/\omega \\)
- Plots empirical vs. predicted \\(b\\)
- Analyzes divergence boundaries
- Provides mathematical derivations

**Output**:
- `figures/analysis/scaling_law_additive.png`
- `figures/analysis/divergence_boundary.png`
- `figures/analysis/theory_lemma_visualization.png`

#### Step 4: Compare with Real-World Spirals (Optional)

1. Add real spiral images to `data/real/*.png`
2. Run comparison:

```bash
python -m scripts.real_compare
```

**Output**:
- `results/real_matches.csv` - Top 5 matches per real image
- `figures/real_matches/*.png` - Side-by-side comparisons

---

## 📊 Key Results

### Spiral-Safe Parameter Ranges (Controlled Mode)

**Additive Mode** (\\( r_{n+1} = r_n + \delta \\)):
- \\(\delta\\): [0.001, 0.003, 0.006, 0.01, 0.015]
- \\(\omega\\): [0.2π, 0.25π, 0.3π, 0.35π, 0.4π]
- \\(\phi_{\text{eps}}\\): [0.0, 0.01, 0.02]

**Power Mode** (\\( r_{n+1} = r_n^\alpha \\)):
- \\(\alpha\\): [1.01, 1.02, 1.04, 1.06]
- \\(\omega\\): [0.2π, 0.25π, 0.3π, 0.35π, 0.4π]
- \\(\phi_{\text{eps}}\\): [0.0, 0.01, 0.02]

Default: `max_iter=350`, `escape_radius=80`, `grid_size=256`

### Theoretical Scaling Law

For additive radial growth:
$$b = \frac{\ln(1 + \delta/r_0)}{\omega} \approx \frac{\delta}{\omega r_0}$$

Validated empirically with \\(R^2 > 0.5\\) correlation.

### Clean Spiral Criteria

A spiral is "clean" if:
- \\(\text{arm\_count} \geq 2\\)
- \\(R^2 \geq 0.7\\) (good log-spiral fit)
- \\(b > 0\\) (outward spiral)

Stability region: \\(\delta/\omega < \text{threshold}\\) (prevents divergence)

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run geometry tests only
pytest tests/test_geometry.py -v
```

Tests verify:
- Image loading and preprocessing
- Polar coordinate conversion
- Log-spiral fitting
- Fractal dimension computation
- Full `analyze_spiral_image` pipeline

---

## 📝 Citation

If you use this code or methodology, please cite:

```
[Your Name], [Co-authors]. "Beyond the Mandelbrot: Modeling Real-World Spiral Growth
Using Expanding Complex Dynamics." ISEF 2026 Mathematics.
GitHub: https://github.com/guptaarjun2027/spiral_fractals
```

---

## 🛠️ Advanced Usage

### Custom Parameter Sweeps

```bash
# Custom delta values
python -m scripts.sweep_params \
    --controlled-only \
    --delta-r-values 0.005 0.01 0.015 \
    --omega-values 0.628 0.942 1.257 \
    --radial-modes additive

# Specify grid size and iterations
python -m scripts.sweep_params \
    --controlled-only \
    --grid-size 512 \
    --max-iter 500 \
    --escape-radius 100
```

### Programmatic API

```python
from pathlib import Path
from src.geometry import analyze_spiral_image, GeometryConfig

# Analyze a single spiral
cfg = GeometryConfig(
    threshold="otsu",
    min_arm_length=50,
    box_sizes=[2, 4, 8, 16, 32, 64],
    n_subsamples=10
)

result = analyze_spiral_image("figures/sweeps/controlled_00042.png", cfg)

print(f"Arms: {result['arm_count']}")
print(f"b: {result['b_mean']:.4f} ± {result['b_std']:.4f}")
print(f"R²: {result['r2_mean']:.4f}")
print(f"Fractal dimension: {result['fractal_dimension']:.3f}")
```

---

## 📚 Documentation

### Geometry Metrics

Each analyzed spiral returns:
- `arm_count`: Number of detected spiral arms
- `b_mean`, `b_std`: Log-spiral slope \\(b\\) and variation
- `r2_mean`: Average \\(R^2\\) goodness-of-fit
- `arm_spacing_mean`, `arm_spacing_std`: Angular spacing statistics
- `fractal_dimension`: Box-counting dimension
- `fractal_dimension_ci_low`, `fractal_dimension_ci_high`: 90% confidence interval

### Iterator Modes

**Quadratic**: \\( z_{n+1} = z_n^2 + c \\)
**Exponential**: \\( z_{n+1} = e^{z_n} + c \\)
**Controlled**: Polar update with separate radius and angle rules

Controlled polar dynamics:
- Additive: \\( r_{n+1} = r_n + \delta \\)
- Power: \\( r_{n+1} = r_n^\alpha \\)
- Angle: \\( \theta_{n+1} = \theta_n + \omega + \phi(z_n) \\)

---

## ⚠️ Notes

### Complex Argument Quoting

When passing complex values with minus signs to CLI:

```bash
# Use quotes or equals form
python scripts/make_image.py --c='-0.4+0.6j' --map quadratic

# Or quote the entire value
python scripts/make_image.py --c "-0.4+0.6j" --map quadratic
```

### Computational Requirements

- **Small sweep**: ~50-100 images, ~5 min
- **Full sweep**: ~300-500 images, ~30-60 min
- **Memory**: ~2-4 GB for full analysis
- **Disk**: ~100-500 MB for image output

---

## 🤝 Contributing

This is an ISEF research project. For questions or collaboration:
- Open an issue on GitHub
- Contact: [your email or school]

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

Special thanks to:
- Dr. [Advisor Name] for guidance on complex dynamics
- [School/Institution] for computational resources
- The fractal geometry and complex systems research community

**Last updated**: November 2025
