\# Expanding Complex Fractals



This repository contains the code and data for our \*\*ISEF 2026 Mathematics\*\* project on \*Expanding Complex Fractals\*.



We study what happens when complex numbers with |z| > 1 are iterated under nonlinear maps such as  

\\( z\_{n+1} = z\_n^2 + c \\) and \\( z\_{n+1} = e^{z\_n} + c \\).  

Unlike traditional bounded fractals (e.g., the Mandelbrot set), these systems \*\*expand outward\*\* and often form striking spiral structures that resemble natural patterns—galaxies, hurricanes, shells, and plant growth.  

Our goal is to \*\*quantify\*\* these spirals and determine whether simple iterative rules can reproduce the geometric properties of real-world spirals.



---



\## 🔬 Research Question

Can expanding complex maps reproduce measurable spiral characteristics—slope \\(b\\), mean arm spacing \\(\\Delta \\theta\\), and fractal dimension \\(D\\)—found in natural spiral images better than baseline logarithmic or Fourier models?



---



\## 🧠 Project Objectives

\- Implement iterative complex maps (quadratic, exponential, controlled-spiral variants).  

\- Visualize outward-growing fractals and analyze how parameters affect spiral formation.  

\- Fit generated patterns to the logarithmic-spiral equation \\( r = a e^{b\\theta} \\).  

\- Compute box-counting fractal dimension \\(D\\).  

\- Compare quantitative metrics with real spiral data (galaxies, hurricanes, shells).  



---



\## 🗂️ Repository Structure

src/            # Core modules: iteration, rendering, coloring, geometry

scripts/        # Command-line utilities for image generation and sweeps

notebooks/      # Jupyter experiments and analysis

figures/        # Auto-generated images

data/           # Reference or comparison images

results/        # Computed tables and fits



