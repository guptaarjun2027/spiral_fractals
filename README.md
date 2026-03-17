Escape Dynamics and Spiral Geometry in Complex Systems

SEFMD Regional Science & Engineering Fair Project
Category: Mathematics / Dynamical Systems

Authors:
Rohan Nagaram
Arjun Gupta
Himanish Pasula

Abstract

Many nonlinear dynamical systems exhibit spiral escape trajectories.
Classical theory predicts that once trajectories reach sufficiently large radii, they should approach a logarithmic spiral structure.

This project investigates whether that prediction holds under perturbation.

We analyze the perturbed quadratic complex map

F(z) = e^(iθ)z + λz² + εz⁻²

which contains:

• rotational dynamics
• quadratic radial growth
• an inverse perturbation term

Through theoretical analysis and large-scale simulations, we find that:

Radial escape occurs, but spiral geometry does not converge.

This demonstrates a structural decoupling between escape dynamics and geometric stabilization in perturbed nonlinear systems.

Key Result

The central discovery of this project is:

Radial escape and spiral stabilization are structurally decoupled phenomena.

Even after trajectories enter regions where quadratic growth dominates, the predicted logarithmic spiral relation

φ ≈ κ log r

fails to converge.

This indicates that early perturbations can permanently alter trajectory geometry even when asymptotic dynamics appear stable.

Research Question

Does radial escape in a perturbed quadratic complex map guarantee convergence to a logarithmic spiral geometry?

Specifically:

• Do trajectories converge to a stable spiral pitch κ?
• Does the relation φ ≈ κ log r hold asymptotically?
• Can perturbations permanently alter escape geometry?

Mathematical Model

We analyze the complex dynamical system

F(z) = e^(iθ)z + λz² + εz⁻²

where

Term	Role
e^(iθ)z	rotational component
λz²	quadratic radial growth
εz⁻²	inverse perturbation

The perturbation introduces nonlinear instability near the origin, potentially affecting asymptotic geometry.

Theoretical Insight

We prove the existence of an outer escape region.

Lemma: Outer Escape Wedge

There exists a radius R₀ such that when |z| > R₀, the quadratic term λz² dominates the dynamics.

In this regime

|zₙ₊₁| ≈ |λ| |zₙ|²

which produces monotonic radial growth and guarantees escape.

However, although escape is guaranteed, angular structure does not stabilize.

Dynamical Regimes Observed

Simulations reveal three regions:

Stable Interior

The perturbation term εz⁻² dominates and prevents escape.

Transition Boundary

Trajectories experience competing effects from perturbation and quadratic growth.

Outer Escape Region

Quadratic growth dominates and trajectories escape outward.

Despite entering this regime, spiral pitch does not converge.

Computational Experiments

We performed large-scale numerical experiments to measure trajectory geometry.

Methods

• escape-time fractal generation
• spiral pitch estimation
• phase-space vector field visualization
• angular stability diagnostics

Diagnostic Metric

To test logarithmic spiral convergence we analyze

Δφ / log r vs log r

If spiral geometry stabilizes, this value should converge to zero.

Instead we observe persistent variability across trajectories.

Example Visualizations

The repository generates several key figures:

Escape-Time Fractal
Shows how quickly points escape beyond the escape radius.

Phase-Space Flow
Vector field visualization illustrating local trajectory direction.

Angular Stability Diagnostic
Scatter plot measuring convergence to logarithmic spiral geometry.

Repository Structure
spiral_fractals/

├── fractal_generator.py
├── trajectory_simulation.py
├── angular_stability_analysis.py
├── phase_space_visualization.py
├── figures/
│   ├── escape_fractal.png
│   ├── phase_space_flow.png
│   ├── angular_stability.png
└── README.md
Installation

Clone the repository

git clone https://github.com/guptaarjun2027/spiral_fractals
cd spiral_fractals

Install dependencies

pip install numpy matplotlib scipy
Running the Experiments

Generate the escape-time fractal

python fractal_generator.py

Simulate trajectory dynamics

python trajectory_simulation.py

Compute angular stability diagnostics

python angular_stability_analysis.py
Reproducing Poster Figures

Running the scripts above will generate the figures used in the SEFMD science fair poster, including

• escape-time fractals
• phase-space flow diagrams
• spiral pitch diagnostics

All plots are saved automatically to the figures/ directory.

Why This Research Matters

Understanding escape dynamics in nonlinear systems has implications for

• chaotic dynamical systems
• nonlinear stability analysis
• complex systems modeling

This work highlights how small perturbations can permanently affect geometric structure, even when overall system behavior appears stable.

Future Work

Possible extensions include:

Parameter bifurcation analysis
Investigating how varying ε changes escape geometry.

Higher-order perturbations
Testing cubic and quartic perturbation terms.

Connections to rational map theory
Relating these results to external ray theory in complex dynamics.

References

Milnor, J. Dynamics in One Complex Variable.

Devaney, R. An Introduction to Chaotic Dynamical Systems.

Peitgen, Jürgens, Saupe. Chaos and Fractals.

Carleson & Gamelin. Complex Dynamics.

Douady & Hubbard. Étude Dynamique des Polynômes Complexes.

Acknowledgments

We thank the SEFMD science fair organizers and mentors who supported this research.
