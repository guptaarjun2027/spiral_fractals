import numpy as np
import matplotlib.pyplot as plt
import time
import os

## ----------------------------------------------------
## 1. Parameters & Setup
## ----------------------------------------------------

# --- Resolution and Iterations ---
WIDTH = 800
HEIGHT = 800
MAX_ITERS = 50   # Use more iterations for better Julia boundary detail

# --- Julia Set Constant (C) ---
# This is the single, fixed constant that defines the shape of the fractal.
# Choose a value that produces spirals (like this one).
C_REAL = -0.5
C_IMAG = 0.5
C_CONST = complex(C_REAL, C_IMAG)

# --- Complex Plane Coordinates (Z₀ Grid) ---
# CRITICAL: We define the viewing window so that ALL initial seeds (Z₀)
# are outside the unit circle (r₀ > 1). The closest point in this window 
# has a radius of at least r0_min = 1.2.
Z_MIN_R, Z_MAX_R = 1.2, 3.2  # Real part 
Z_MIN_I, Z_MAX_I = -1.0, 1.0  # Imaginary part
r0_min = 1.2 # The closest distance to the origin is 1.2, satisfying r₀ > 1

# --- Output Paths ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_FIGURE = os.path.join('figures', f'exponential_julia_{C_REAL}_{C_IMAG}_{timestamp}.png')

os.makedirs('figures', exist_ok=True)

print("--- Python Exponential Julia Set Generator ---")
print(f"Constant C = {C_CONST} | Resolution: {WIDTH}x{HEIGHT}")
print(f"Viewing Z₀ seeds where r₀ > {r0_min}")

## ----------------------------------------------------
## 2. Generate the Initial Seed Grid (Z₀)
## ----------------------------------------------------

# Z is now the initial seeds Z₀, spanning the view window
# This grid represents the complex plane where z₀ = r₀ * e^(iθ₀)
X = np.linspace(Z_MIN_R, Z_MAX_R, WIDTH)
Y = np.linspace(Z_MAX_I, Z_MIN_I, HEIGHT)
Z = X + Y[:, None] * 1j # Z holds all the initial z₀ values

# M (Mask) holds the iteration count when the point escapes
M = np.full(Z.shape, MAX_ITERS, dtype=int) 

## ----------------------------------------------------
## 3. The Core Exponential Iterator (eᶻ + C)
## ----------------------------------------------------

print("\nStarting vectorized calculation...")
start_time = time.time()

# Iteratively apply Z = e^Z + C
for i in range(MAX_ITERS):
    # Find points that haven't escaped yet (|Z| is below the boundary)
    not_escaped = np.abs(Z) <= 20.0 
    
    # Check if there are any points left to iterate
    if not np.any(not_escaped):
        break
    
    # Update the Z values ONLY for the points that haven't escaped
    # CORE FORMULA: Z_next = e^Z + C (C is the fixed constant defined in Section 1)
    Z[not_escaped] = np.exp(Z[not_escaped]) + C_CONST
    
    # Update the iteration count (M) for points that ESCAPED in this step
    escaped_now = (np.abs(Z) > 20.0) & (M == MAX_ITERS)
    M[escaped_now] = i

end_time = time.time()
print(f"Calculation finished in {end_time - start_time:.2f} seconds.")


## ----------------------------------------------------
## 4. Visualization and Saving
## ----------------------------------------------------

plt.figure(figsize=(10, 10))

# Use 'twilight' colormap for good visual contrast
plt.imshow(M, cmap='twilight', 
           extent=[Z_MIN_R, Z_MAX_R, Z_MIN_I, Z_MAX_I])

# Title shows the fixed constant C and the starting r₀ condition.
plt.title(f'Exponential Julia Set (C={C_CONST}, r₀ > {r0_min}) - {end_time - start_time:.2f}s')
plt.xlabel('Re(z₀)')
plt.ylabel('Im(z₀)')
plt.colorbar(label='Iterations to Escape (MAX=50)')
plt.gca().set_aspect('equal', adjustable='box') 

# Save the figure
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
print(f"\nImage saved successfully to {OUTPUT_FIGURE}")