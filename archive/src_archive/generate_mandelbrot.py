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
MAX_ITERS = 100

# --- Complex Plane Coordinates (Mandelbrot Region) ---
X_MIN, X_MAX = -2.0, 1.0
Y_MIN, Y_MAX = -1.5, 1.5

# --- Output Paths ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_FIGURE = os.path.join('figures', f'mandelbrot_python_{timestamp}.png')

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

print("--- Python Fractal Generator (Vectorized) ---")
print(f"Resolution: {WIDTH}x{HEIGHT} | Max Iterations: {MAX_ITERS}")

## ----------------------------------------------------
## 2. Generate the Complex Grid
## ----------------------------------------------------

# Create a 2D array of complex numbers (C = Real + Imag * i)
X = np.linspace(X_MIN, X_MAX, WIDTH)
Y = np.linspace(Y_MAX, Y_MIN, HEIGHT) # Y is inverted for image plotting
C = X + Y[:, None] * 1j

# Initialize Z to 0 and the array to hold iteration counts
Z = np.zeros_like(C)
M = np.full(C.shape, MAX_ITERS, dtype=int) # M = Mask (Initial: all are MAX_ITERS)

## ----------------------------------------------------
## 3. The Core Iterator Function (Vectorized Z² + C)
## ----------------------------------------------------

print("\nStarting vectorized calculation...")
start_time = time.time()

# Iteratively apply Z = Z² + C
for i in range(MAX_ITERS):
    # Find points that haven't escaped yet
    # Use a boolean mask where |Z| <= 2 (or |Z|² <= 4)
    not_escaped = np.abs(Z) <= 2.0
    
    # Update the Z values ONLY for the points that haven't escaped
    Z[not_escaped] = Z[not_escaped]**2 + C[not_escaped]
    
    # Update the iteration count (M) for points that ESCAPED in this step
    # An escaped point has (|Z| > 2) AND (it hasn't been set yet, i.e., M == MAX_ITERS)
    escaped_now = (np.abs(Z) > 2.0) & (M == MAX_ITERS)
    M[escaped_now] = i

end_time = time.time()
print(f"Calculation finished in {end_time - start_time:.2f} seconds.")


## ----------------------------------------------------
## 4. Visualization and Saving
## ----------------------------------------------------

plt.figure(figsize=(10, 10))

# Use 'magma' colormap for a glowing, high-contrast effect
plt.imshow(M, cmap='magma', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])

plt.title(f'Mandelbrot Set ({WIDTH}x{HEIGHT}) - {end_time - start_time:.2f}s')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.colorbar(label='Iterations to Escape (0=In Set)')
plt.gca().set_aspect('equal', adjustable='box') 

# Save the figure
plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
print(f"\nImage saved successfully to {OUTPUT_FIGURE}")
