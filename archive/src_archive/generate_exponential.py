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
MAX_ITERS = 20  # Exponential maps require fewer iterations
r0 = 1.1        # Initial seed radius (r0 > 1) as required by project

# --- Complex Plane Coordinates (C-Plane for Exponential Map) ---
# This range captures the interesting central features of the set
X_MIN, X_MAX = -2.5, 2.5
Y_MIN, Y_MAX = -2.5, 2.5 

# --- Output Paths ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_FIGURE = os.path.join('figures', f'exponential_python_{timestamp}.png')

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

print("--- Python Exponential Fractal Generator (Vectorized) ---")
print(f"Resolution: {WIDTH}x{HEIGHT} | Max Iterations: {MAX_ITERS}")

## ----------------------------------------------------
## 2. Generate the Complex Grid and Initial Seed Z₀
## ----------------------------------------------------

# C is the constant (grid of points we are testing in the C-plane)
X = np.linspace(X_MIN, X_MAX, WIDTH)
Y = np.linspace(Y_MAX, Y_MIN, HEIGHT)
C = X + Y[:, None] * 1j

# Z₀ is the initial seed. We set Z₀ to be the constant r₀ + 0j across the entire grid.
Z_seed = np.full(C.shape, r0 + 0j, dtype=complex)
Z = Z_seed.copy() 

# M (Mask) holds the iteration count when the point escapes
M = np.full(C.shape, MAX_ITERS, dtype=int) 

## ----------------------------------------------------
## 3. The Core Exponential Iterator (Vectorized eᶻ + C)
## ----------------------------------------------------

print("\nStarting vectorized calculation...")
start_time = time.time()

# Iteratively apply Z = e^Z + C
for i in range(MAX_ITERS):
    # Find points that haven't escaped yet (|Z| is below the boundary)
    # A generous boundary is used here for the chaotic exponential map.
    not_escaped = np.abs(Z) <= 20.0 
    
    # Update the Z values ONLY for the points that haven't escaped
    # CORE FORMULA: Z_next = e^Z + C
    Z[not_escaped] = np.exp(Z[not_escaped]) + C[not_escaped]
    
    # Update the iteration count (M) for points that ESCAPED in this step
    escaped_now = (np.abs(Z) > 20.0) & (M == MAX_ITERS)
    M[escaped_now] = i

end_time = time.time()
print(f"Calculation finished in {end_time - start_time:.2f} seconds.")


## ----------------------------------------------------
## 4. Visualization and Saving
## ----------------------------------------------------

plt.figure(figsize=(10, 10))

# Use 'twilight' colormap for good visual contrast in exponential fractals
plt.imshow(M, cmap='twilight', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])

plt.title(f'Exponential Map (Z₀={r0}, Iters={MAX_ITERS}) - {end_time - start_time:.2f}s')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.colorbar(label='Iterations to Escape (MAX=20)')
plt.gca().set_aspect('equal', adjustable='box') 

# Save the figure
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
print(f"\nImage saved successfully to {OUTPUT_FIGURE}")