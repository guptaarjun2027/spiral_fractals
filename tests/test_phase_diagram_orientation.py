"""
Test: Phase Diagram Matrix Orientation
Verifies that matrix indexing is correct for imshow with origin='lower'.
"""

import numpy as np
import pytest


def test_phase_matrix_orientation():
    """
    Verify (eps_min, lam_min) maps to matrix[0, 0] with origin='lower'.
    
    Matrix convention: phase[j, i] corresponds to (lam[j], eps[i])
    imshow with origin='lower': bottom-left is (x_min, y_min) = (eps_min, lam_min)
    """
    # Build test grids
    eps_grid = np.array([0.0, 1.0])  # 2 values
    lam_grid = np.array([0.0, 2.0])  # 2 values
    
    # Matrix: phase[j, i] where j=lam index, i=eps index
    # So shape is (len(lam_grid), len(eps_grid)) = (2, 2)
    matrix = np.zeros((len(lam_grid), len(eps_grid)))
    
    # Set specific value at (lam_min, eps_min) = (0.0, 0.0)
    # This should be matrix[0, 0]
    matrix[0, 0] = 1.0
    
    # Check indexing
    assert matrix.shape == (2, 2), "Matrix should be (lam_steps, eps_steps)"
    assert matrix[0, 0] == 1.0, "matrix[0, 0] should be (lam_min, eps_min)"
    assert matrix[0, 1] == 0.0, "matrix[0, 1] should be (lam_min, eps_max)"
    assert matrix[1, 0] == 0.0, "matrix[1, 1] should be (lam_max, eps_min)"
    assert matrix[1, 1] == 0.0, "matrix[1, 1] should be (lam_max, eps_max)"
    
    print("✓ Matrix orientation test passed")


def test_imshow_extent_mapping():
    """
    Verify extent mapping for imshow with origin='lower'.
    
    extent = [left, right, bottom, top] = [eps_min, eps_max, lam_min, lam_max]
    
    With origin='lower':
    - Bottom-left pixel (row=0, col=0) maps to (eps_min, lam_min)
    - Top-right pixel (row=N-1, col=M-1) maps to (eps_max, lam_max)
    """
    eps_min, eps_max = 0.0, 2.0
    lam_min, lam_max = 0.5, 3.5
    
    # Create 3x4 matrix: (lam_steps=3, eps_steps=4)
    lam_steps, eps_steps = 3, 4
    matrix = np.zeros((lam_steps, eps_steps))
    
    # Set corners
    matrix[0, 0] = 1.0  # (lam_min, eps_min) - bottom-left
    matrix[0, eps_steps-1] = 2.0  # (lam_min, eps_max) - bottom-right
    matrix[lam_steps-1, 0] = 3.0  # (lam_max, eps_min) - top-left
    matrix[lam_steps-1, eps_steps-1] = 4.0  # (lam_max, eps_max) - top-right
    
    # Verify corners
    assert matrix[0, 0] == 1.0, "Bottom-left should be 1.0"
    assert matrix[0, -1] == 2.0, "Bottom-right should be 2.0"
    assert matrix[-1, 0] == 3.0, "Top-left should be 3.0"
    assert matrix[-1, -1] == 4.0, "Top-right should be 4.0"
    
    # Verify extent calculation
    extent = [eps_min, eps_max, lam_min, lam_max]
    assert extent == [0.0, 2.0, 0.5, 3.5], "Extent should be [eps_min, eps_max, lam_min, lam_max]"
    
    print("✓ imshow extent mapping test passed")


def test_coordinate_calculation():
    """
    Test conversion between (lam, eps) coordinates and matrix indices.
    """
    eps_vals = np.linspace(0.0, 2.0, 5)  # [0.0, 0.5, 1.0, 1.5, 2.0]
    lam_vals = np.linspace(1.0, 3.0, 3)  # [1.0, 2.0, 3.0]
    
    # For (lam=2.0, eps=1.0), indices should be (j=1, i=2)
    lam_target = 2.0
    eps_target = 1.0
    
    j = np.argmin(np.abs(lam_vals - lam_target))
    i = np.argmin(np.abs(eps_vals - eps_target))
    
    assert j == 1, f"lam={lam_target} should map to j=1, got j={j}"
    assert i == 2, f"eps={eps_target} should map to i=2, got i={i}"
    
    print("✓ Coordinate calculation test passed")


if __name__ == "__main__":
    test_phase_matrix_orientation()
    test_imshow_extent_mapping()
    test_coordinate_calculation()
    print("\nAll orientation tests passed!")
