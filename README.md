Structural Decoupling in Perturbed Quadratic Maps

SEFMD Science & Engineering Fair Project
Category: Mathematics / Dynamical Systems

Authors:
Rohan Nagaram
Arjun Gupta
Himanish Pasula

Overview

This project studies escape dynamics in a perturbed quadratic complex map:

F(z) = e^(iθ)z + λz² + εz⁻²

The system contains:

rotational dynamics

quadratic radial growth

an inverse perturbation term

We investigate whether escaping trajectories converge to a logarithmic spiral geometry.

Research Question

Does radial escape in a perturbed quadratic complex map guarantee convergence to a logarithmic spiral spiral pitch?

Key Finding

Radial escape does not guarantee geometric stabilization.

Even after trajectories reach regions where quadratic growth dominates, spiral pitch does not converge to the predicted logarithmic relation:

φ ≈ κ log r

This indicates a structural decoupling between escape dynamics and spiral geometry.

Repository Contents
spiral_fractals/

fractal_generator.py
trajectory_simulation.py
angular_stability_analysis.py
phase_space_visualization.py

These scripts generate:

escape-time fractals

trajectory simulations

angular stability diagnostics

phase-space flow visualizations

Running the Code

Install dependencies:

pip install numpy matplotlib scipy

Run simulations:

python fractal_generator.py
python angular_stability_analysis.py

Figures will be saved automatically.

References

Milnor — Dynamics in One Complex Variable
Devaney — An Introduction to Chaotic Dynamical Systems
Peitgen et al. — Chaos and Fractals
Carleson & Gamelin — Complex Dynamics
