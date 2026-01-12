# src/utils.py
import re
import numpy as np
from scipy import stats

def parse_complex(s: str) -> complex:
    """
    Parse strings like '0.3+0.5j' or '-0.4-0.6j' into a complex number.
    """
    s = s.strip().lower().replace(" ", "")
    if s.endswith("j") and ("+" in s[1:] or "-" in s[1:]):
        return complex(s)
    # allow plain real numbers too
    return complex(float(s), 0.0)

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def extract_monotone_branch(phi: np.ndarray, log_r: np.ndarray, min_len: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the longest contiguous segment where phi is monotonic with respect to log_r.
    """
    if len(phi) < min_len:
        return np.array([]), np.array([])
        
    dphi = np.diff(phi)
    # Signs of slope. ignore zeros.
    signs = np.sign(dphi)
    signs[signs == 0] = 1 
    
    # Identify changes in sign
    change_indices = np.where(signs[:-1] != signs[1:])[0] + 1
    split_indices = np.concatenate(([0], change_indices, [len(signs)]))
    
    longest_start = 0
    longest_end = 0
    max_len = 0
    
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i+1]
        length = end - start
        if length > max_len:
            max_len = length
            longest_start = start
            longest_end = end
            
    if (max_len + 1) < min_len:
        return np.array([]), np.array([])
        
    # extract segment
    # dphi indices [start, end-1] corresponds to phi indices [start, end]
    # actually dphi[i] = phi[i+1] - phi[i].
    # if signs[start:end] are constant, it means phi[start]...phi[end] are monotone.
    # length of signs chunk is 'length'.
    # so we have length+1 points.
    return phi[longest_start : longest_end + 1], log_r[longest_start : longest_end + 1]

def filter_dominant_arm(log_r: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Select points belonging to the dominant spiral arm.
    Strategy:
    1. Estimate rough slope k.
    2. Compute residuals: psi = phi - k * log_r.
    3. Cluster psi modulo 2*pi.
    4. Keep points in the largest cluster.
    """
    if len(log_r) < 10:
        return log_r, phi
        
    # 1. Estimate slope. Use Theil-Sen (robust) or just median of pair slopes?
    # Simple linear regression might be biased by 2 parallel lines, but slope should be ok-ish
    # if they are balanced.
    # Let's use scipy.stats.linregress, it's fast.
    res = stats.linregress(log_r, phi)
    k_approx = res.slope
    
    # 2. Residuals
    psi = phi - k_approx * log_r
    
    # 3. Cluster psi mod 2pi
    # We want to find the offset.
    psi_mod = np.mod(psi, 2 * np.pi)
    
    # Histogram
    hist, bin_edges = np.histogram(psi_mod, bins=30, range=(0, 2*np.pi))
    best_bin_idx = np.argmax(hist)
    
    # Define circular window around peak
    bin_center = (bin_edges[best_bin_idx] + bin_edges[best_bin_idx+1]) / 2.0
    half_width = (bin_edges[1] - bin_edges[0]) * 2.5 # somewhat loose, catch neighbors
    
    # Distance on circle
    dist = np.abs(psi_mod - bin_center)
    dist = np.minimum(dist, 2*np.pi - dist)
    
    mask = dist < half_width
    
    # Return filtered
    if np.sum(mask) < 5:
        # Fallback: if filtering removes almost everything, return original
        return log_r, phi
        
    return log_r[mask], phi[mask]
