
import numpy as np
import pytest
from src.iterators import iterate_map

def test_theorem_map_iteration():
    """
    Test that iterate_map works with mode="theorem_map".
    z_next = e^(i*theta)*z + lam*z^2 + eps*z^-2
    """
    theta = 0.5
    lam = 1.0 + 0j
    eps = 0.0 + 0j
    z0 = 2.0 + 0j
    
    # One step manual calculation
    # z1 = exp(0.5j)*2 + 1*4 + 0
    z1_expected = np.exp(1j * theta) * z0 + lam * (z0**2)
    
    traj = iterate_map(
        z0=z0, c=0j, max_iter=1,
        mode="theorem_map",
        theta=theta, lam=lam, eps=eps,
        crash_radius=1e-6, escape_radius=100.0
    )
    
    assert len(traj) == 1 # Initial point is not stored in traj usually? 
    # Wait, iterate_controlled stores z0? No, iterate_map usually appends after step.
    # Let's check iterate_map implementation.
    # It does `z = z0; for ... z = ...; traj.append(z)`.
    # So traj[0] is z1.
    
    z1_actual = traj[0]
    
    np.testing.assert_allclose(z1_actual, z1_expected, rtol=1e-6)

def test_theorem_map_crash():
    """Test that orbit stops if z becomes too small."""
    z0 = 0.5 + 0j
    crash_radius = 1.0
    # z0 < crash_radius, so it should break immediately in loop if checked after update?
    # No, check is at start of loop: if abs(z) < crash_radius: break.
    # So if z0 < crash_radius, it breaks before appending anything?
    # Or does it append?
    # "z = z0; for _: if abs(z) < crash: break ..."
    # So if z0 is small, loop breaks, traj is empty.
    
    traj = iterate_map(
        z0=z0, c=0j, max_iter=10,
        mode="theorem_map",
        crash_radius=crash_radius
    )
    
    assert len(traj) == 0

def test_theorem_map_eps_term():
    """Test effect of eps term."""
    z0 = 2.0 + 0j
    theta = 0.0
    lam = 0.0
    eps = 1.0 # z_next = z + 0 + 1/z^2
    
    # z_next = 2 + 1/4 = 2.25
    z1_expected = z0 + (1.0 / (z0**2))
    
    traj = iterate_map(
        z0=z0, c=0j, max_iter=1,
        mode="theorem_map",
        theta=theta, lam=lam, eps=eps
    )
    
    np.testing.assert_allclose(traj[0], z1_expected, rtol=1e-6)
