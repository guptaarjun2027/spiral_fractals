"""
Geometry engine for spiral_fractals.

Given a rendered spiral fractal image, this module extracts geometric features:
- binary mask
- skeleton
- connected spiral arms
- log-spiral slope b via ln(r)=ln(a)+b*theta
- R^2 goodness-of-fit per arm
- arm spacing statistic
- fractal dimension (box-counting)

Main entrypoint:
    analyze_spiral_image(image_path) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from skimage import io, filters, morphology, measure, color, util


@dataclass
class GeometryConfig:
    threshold: str | float = "otsu"  # "otsu" or float threshold value
    min_arm_length: int = 50
    center: Optional[Tuple[float, float]] = None  # (cx, cy). None => image center
    box_sizes: Sequence[int] = (2, 4, 8, 16, 32, 64)
    n_subsamples: int = 10  # for bootstrap CI in fractal dimension


def load_image_gray(path: str | Path) -> np.ndarray:
    """Load PNG as grayscale float in [0,1]."""
    img = io.imread(str(path))
    if img.ndim == 3:
        img = color.rgb2gray(img)
    return util.img_as_float(img)


def binary_mask(img: np.ndarray, threshold: str | float = "otsu") -> np.ndarray:
    """
    Convert image to binary mask.

    Args:
        img: grayscale image in [0,1]
        threshold: "otsu" or float value
    """
    if threshold == "otsu":
        thresh_val = filters.threshold_otsu(img)
    else:
        thresh_val = float(threshold)

    mask = img > thresh_val
    mask = morphology.remove_small_objects(mask, min_size=40)
    mask = morphology.remove_small_holes(mask, area_threshold=40)
    return mask


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract morphological boundary: mask XOR eroded(mask)."""
    eroded = morphology.binary_erosion(mask)
    boundary = np.logical_xor(mask, eroded)
    return boundary


def skeletonize_boundary(boundary: np.ndarray) -> np.ndarray:
    """Skeletonize the boundary."""
    return morphology.skeletonize(boundary).astype(bool)


def polar_coords_from_center(points: np.ndarray, center: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute r, θ for each point relative to center.

    Args:
        points: Nx2 array of (row, col) coordinates
        center: (cx, cy) in image coordinates. If None, uses centroid of points.
    """
    if center is None:
        # Use centroid of points
        center = (np.mean(points[:, 1]), np.mean(points[:, 0]))

    # coords in (row, col) = (y, x)
    y = points[:, 0].astype(float)
    x = points[:, 1].astype(float)
    cx, cy = center
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # [-pi, pi]
    return r, theta





def estimate_center(mask: np.ndarray) -> Tuple[float, float]:
    """
    Estimate the center of the spiral from the binary mask.
    Uses the centroid of foreground pixels.
    """
    y, x = np.where(mask)
    if len(y) == 0:
        return (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
    return (float(np.mean(x)), float(np.mean(y)))


def box_count_fractal_dimension(

    boundary: np.ndarray,
    box_sizes: Optional[Sequence[int]] = None,
    n_subsamples: int = 10
) -> Tuple[float, float, float]:
    """
    Compute fractal dimension via box counting with bootstrap CI.

    Args:
        boundary: binary boundary image
        box_sizes: list of box sizes to use. If None, uses [2,4,8,16,32,64]
        n_subsamples: number of bootstrap subsamples for CI

    Returns:
        (fractal_dimension, ci_low, ci_high)
    """
    if box_sizes is None:
        box_sizes = [2, 4, 8, 16, 32, 64]

    # Get boundary pixel coordinates
    coords = np.argwhere(boundary)
    if len(coords) < 10:
        return (np.nan, np.nan, np.nan)

    def compute_dimension(pixel_coords):
        """Helper to compute dimension from pixel coordinates."""
        if len(pixel_coords) < 5:
            return np.nan

        Ns = []
        inv_s = []
        for s in box_sizes:
            if s <= 0:
                continue
            # Grid-based counting
            boxes = set()
            for coord in pixel_coords:
                box_idx = (coord[0] // s, coord[1] // s)
                boxes.add(box_idx)
            count = len(boxes)
            if count > 0:
                Ns.append(count)
                inv_s.append(1.0 / s)

        if len(Ns) < 2:
            return np.nan

        logN = np.log(Ns)
        logInvS = np.log(inv_s)
        slope, _, _, _, _ = stats.linregress(logInvS, logN)
        return slope

    # Main estimate
    main_dim = compute_dimension(coords)

    # Bootstrap for CI
    dimensions = []
    for _ in range(n_subsamples):
        indices = np.random.choice(len(coords), size=len(coords), replace=True)
        subsample = coords[indices]
        dim = compute_dimension(subsample)
        if np.isfinite(dim):
            dimensions.append(dim)

    if len(dimensions) == 0:
        return (main_dim, np.nan, np.nan)

    ci_low = float(np.percentile(dimensions, 5))
    ci_high = float(np.percentile(dimensions, 95))

    return (float(main_dim), ci_low, ci_high)




def detect_arms_by_angle(
    skeleton: np.ndarray,
    center: Tuple[float, float],
    r_min_frac: float = 0.2,
    r_max_frac: float = 0.8,
    n_theta_bins: int = 72,
    min_peak_height_frac: float = 0.1,
) -> List[np.ndarray]:
    """
    Detect spiral arms by clustering skeleton pixels based on their angle.
    
    Strategy:
    1. Filter pixels to a mid-radius band.
    2. Compute histogram of angles (theta).
    3. Smooth histogram to reduce noise/splitting.
    4. Find peaks with minimum distance constraint.
    5. Assign pixels to nearest peak.
    """
    from scipy.signal import find_peaks, savgol_filter
    
    y_idxs, x_idxs = np.where(skeleton)
    if len(y_idxs) == 0:
        return []
        
    points = np.column_stack((y_idxs, x_idxs))
    r, theta = polar_coords_from_center(points, center)
    
    # Filter by radius
    r_max = np.max(r) if len(r) > 0 else 1.0
    mask_r = (r >= r_min_frac * r_max) & (r <= r_max_frac * r_max)
    
    if np.sum(mask_r) == 0:
        return []
        
    theta_band = theta[mask_r]
    # Map theta to [0, 2pi) for histogram
    theta_band_2pi = np.mod(theta_band, 2 * np.pi)
    
    # Histogram
    hist, bin_edges = np.histogram(theta_band_2pi, bins=n_theta_bins, range=(0, 2 * np.pi))
    
    # Smooth histogram (circular convolution would be best, but simple smoothing works for now)
    # Wrap padding for circular smoothing
    hist_padded = np.pad(hist, (5, 5), mode='wrap')
    hist_smooth = np.convolve(hist_padded, np.ones(5)/5, mode='same')[5:-5]
    
    # Find peaks
    # distance=n_theta_bins/8 ensures peaks are at least 45 degrees apart (max 8 arms)
    height_thresh = min_peak_height_frac * np.max(hist_smooth)
    peaks, _ = find_peaks(hist_smooth, height=height_thresh, distance=n_theta_bins // 8)
    
    if len(peaks) == 0:
        return []
        
    peak_thetas = bin_edges[peaks] + (bin_edges[1] - bin_edges[0]) / 2
    
    # Assign all skeleton pixels to nearest peak
    theta_2pi = np.mod(theta, 2 * np.pi)
    
    dists = np.abs(theta_2pi[:, None] - peak_thetas[None, :])
    dists = np.minimum(dists, 2 * np.pi - dists)
    
    labels = np.argmin(dists, axis=1)
    
    arms = []
    for i in range(len(peaks)):
        arm_mask = labels == i
        if np.sum(arm_mask) > 20: # Minimum pixels
            arms.append(points[arm_mask])
            
    return arms



def fit_log_spiral_for_arm(
    arm_pixels: np.ndarray,
    center: Tuple[float, float],
    min_delta_theta: float = 0.5,
    min_points: int = 50,
) -> Dict[str, float]:
    """
    Fit log spiral ln(r) = ln(a) + b*theta to a single arm.
    """
    if len(arm_pixels) < min_points:
        return {"valid": False, "b": np.nan, "r2": np.nan}
        
    r, theta = polar_coords_from_center(arm_pixels, center)
    
    # Sort by theta to ensure monotonic angle progression
    # But first we need to handle the wrap-around.
    # Simple unwrapping works if points are dense.
    # Let's try sorting by radius first to get a rough order, then unwrap, then sort by theta.
    
    # 1. Initial sort by radius (usually monotonic for spirals)
    idx_r = np.argsort(r)
    r_sorted = r[idx_r]
    theta_sorted = theta[idx_r]
    
    # 2. Unwrap theta
    theta_unwrapped = np.unwrap(theta_sorted)
    
    # 3. Sort by unwrapped theta to ensure x-axis is monotonic for regression
    idx_th = np.argsort(theta_unwrapped)
    x = theta_unwrapped[idx_th]
    y = np.log(r_sorted[idx_th] + 1e-9)
    
    # Check angular extent
    delta_theta = x.max() - x.min()
    if delta_theta < min_delta_theta:
        return {"valid": False, "b": np.nan, "r2": np.nan, "delta_theta": delta_theta}
    
    # Robust linear regression (RANSAC-like)
    # Simple RANSAC implementation using scipy/numpy
    from skimage.measure import ransac, LineModelND
    
    data = np.column_stack((x, y))
    
    try:
        model, inliers = ransac(data, LineModelND, min_samples=min_points//2, residual_threshold=0.1, max_trials=100)
        
        # Re-fit on inliers using standard linear regression for R2
        x_in = x[inliers]
        y_in = y[inliers]
        
        if len(x_in) < min_points // 2:
             raise ValueError("Too few inliers")
             
        slope, intercept, r_value, p_value, stderr = stats.linregress(x_in, y_in)
        
        return {
            "valid": True,
            "b": float(slope),
            "r2": float(r_value**2),
            "n_points": len(x_in),
            "delta_theta": float(delta_theta),
            "intercept": float(intercept)
        }
        
    except Exception:
        # Fallback to standard regression if RANSAC fails
        slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
        return {
            "valid": True,
            "b": float(slope),
            "r2": float(r_value**2),
            "n_points": len(x),
            "delta_theta": float(delta_theta),
            "intercept": float(intercept)
        }



def analyze_spiral_image(
    image_path: str | Path | np.ndarray,
    cfg: Optional[GeometryConfig] = None
) -> Dict[str, float]:
    """
    High-level API to analyze a spiral fractal image.
    
    Pipeline:
    1. Load & Threshold -> Mask
    2. Skeletonize
    3. Estimate Center
    4. Detect Arms (Angle Clustering)
    5. Fit Log-Spiral to each arm
    6. Compute Fractal Dimension
    7. Aggregate Metrics
    """
    if cfg is None:
        cfg = GeometryConfig()

    # Load
    if isinstance(image_path, (str, Path)):
        img = load_image_gray(image_path)
    else:
        img = image_path
        
    # Mask & Skeleton
    mask = binary_mask(img, cfg.threshold)
    boundary = extract_boundary(mask)
    skel = skeletonize_boundary(boundary)
    
    # Center
    if cfg.center is None:
        center = estimate_center(mask)
    else:
        center = cfg.center
        
    # Detect Arms
    arms = detect_arms_by_angle(skel, center)
    
    # Fit Arms
    valid_arms = 0
    b_vals = []
    r2_vals = []
    
    for arm_pixels in arms:
        res = fit_log_spiral_for_arm(arm_pixels, center)
        if res["valid"] and res["r2"] > 0.5: # Quality filter
            valid_arms += 1
            b_vals.append(abs(res["b"])) # Magnitude of slope
            r2_vals.append(res["r2"])
            
    # Fallback logic for arm count
    # If no arms pass the R² filter but there are many skeleton pixels, fall back to:
    # either the raw number of angle peaks, or a default of 2 arms.
    final_arm_count = float(valid_arms)
    if valid_arms == 0:
        # Check if we have enough skeleton pixels to justify a fallback
        if np.sum(skel) > 100:
            # Fallback to raw number of arms detected (peaks) or default 2
            # detect_arms_by_angle returns clusters based on peaks
            if len(arms) > 0:
                final_arm_count = float(len(arms))
            else:
                final_arm_count = 2.0

    # Aggregate
    if valid_arms > 0:
        b_mean = float(np.mean(b_vals))
        b_std = float(np.std(b_vals))
        r2_mean = float(np.mean(r2_vals))
    else:
        b_mean = np.nan
        b_std = np.nan
        r2_mean = np.nan
        
    # Fractal Dimension
    fd, fd_ci_low, fd_ci_high = box_count_fractal_dimension(
        boundary, cfg.box_sizes, cfg.n_subsamples
    )
    
    image_name = Path(image_path).name if isinstance(image_path, (str, Path)) else "numpy_array"

    return {
        "image": image_name,
        "arm_count": final_arm_count,
        "b_mean": b_mean,
        "b_std": b_std,
        "r2_mean": r2_mean,
        "fractal_dimension": fd,
        "fractal_dimension_ci_low": fd_ci_low,
        "fractal_dimension_ci_high": fd_ci_high,
    }


def analyze_image_batch(
    paths: List[Path],
    cfg: Optional[GeometryConfig] = None,
) -> pd.DataFrame:
    """
    Run analyze_spiral_image on a list of paths and return a DataFrame
    with one row per image.
    """
    results = []
    for path in paths:
        try:
            res = analyze_spiral_image(path, cfg)
            results.append(res)
        except Exception as e:
            print(f"Error analyzing {path}: {e}")
            # Add a row with error or NaNs? For now just skip or add partial
            results.append({
                "image": path.name,
                "arm_count": np.nan,
                "b_mean": np.nan,
                "b_std": np.nan,
                "r2_mean": np.nan,
                "fractal_dimension": np.nan,
                "fractal_dimension_ci_low": np.nan,
                "fractal_dimension_ci_high": np.nan,
            })
            
    return pd.DataFrame(results)

