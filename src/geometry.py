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


def trace_arms(skel: np.ndarray, center: Tuple[float, float], min_length: int = 50) -> List[np.ndarray]:
    """
    Convert skeleton to graph of connected components ("arms").

    Args:
        skel: binary skeleton image
        center: center point for ordering by radius
        min_length: minimum number of pixels to keep an arm

    Returns:
        List of arms, where each arm is an Nx2 array of (row, col) coords ordered by increasing radius
    """
    labeled = measure.label(skel, connectivity=2)
    arms = []

    for region in measure.regionprops(labeled):
        if region.area < min_length:
            continue

        coords = region.coords  # (row, col)
        # Order by increasing radius from center
        r, _ = polar_coords_from_center(coords, center)
        order = np.argsort(r)
        coords_ordered = coords[order]
        arms.append(coords_ordered)

    return arms


def fit_log_spiral(r: np.ndarray, theta: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit log spiral: ln(r) = ln(a) + b*theta

    Args:
        r: radial distances
        theta: angles

    Returns:
        (b, a, r2, residual_std)
    """
    # clean
    m = np.isfinite(r) & np.isfinite(theta) & (r > 1e-6)
    r_clean = r[m]
    theta_clean = theta[m]

    if len(r_clean) < 10:
        return (np.nan, np.nan, np.nan, np.nan)

    # unwrap theta to be monotone
    theta_unwrap = np.unwrap(theta_clean)
    y = np.log(r_clean)
    x = theta_unwrap

    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
    b = slope
    a = float(np.exp(intercept))
    r2 = r_value**2

    # compute residual std
    y_pred = slope * x + intercept
    residuals = y - y_pred
    residual_std = float(np.std(residuals))

    return (float(b), float(a), float(r2), residual_std)


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


def compute_arm_spacing(arms: List[np.ndarray], center: Tuple[float, float]) -> Tuple[float, float]:
    """
    Compute mean and std of angular spacing between arms.

    Strategy:
    - For each arm, compute median angle
    - Sort angles and compute angular differences
    - Return mean and std of differences

    Returns:
        (arm_spacing_mean, arm_spacing_std)
    """
    if len(arms) < 2:
        return (np.nan, np.nan)

    med_angles = []
    for arm_coords in arms:
        _, theta = polar_coords_from_center(arm_coords, center)
        if len(theta) == 0:
            continue
        theta_unwrap = np.unwrap(theta)
        med_angles.append(np.median(theta_unwrap))

    if len(med_angles) < 2:
        return (np.nan, np.nan)

    med_angles = np.sort(med_angles)
    diffs = np.diff(med_angles)
    # include wrap-around
    wrap_diff = (med_angles[0] + 2*np.pi) - med_angles[-1]
    diffs = np.append(diffs, wrap_diff)

    return (float(np.mean(diffs)), float(np.std(diffs)))


def analyze_spiral_image(
    image_path: str | Path,
    cfg: Optional[GeometryConfig] = None
) -> Dict[str, float]:
    """
    High-level API to analyze a spiral fractal image.

    Args:
        image_path: path to spiral image
        cfg: GeometryConfig with analysis parameters

    Returns:
        dict with keys:
            - arm_count
            - b_mean, b_std
            - r2_mean
            - arm_spacing_mean, arm_spacing_std
            - fractal_dimension, fractal_dimension_ci_low, fractal_dimension_ci_high
    """
    if cfg is None:
        cfg = GeometryConfig()

    # Load and preprocess
    img = load_image_gray(image_path)
    mask = binary_mask(img, cfg.threshold)
    boundary = extract_boundary(mask)
    skel = skeletonize_boundary(boundary)

    # Determine center
    h, w = img.shape
    if cfg.center is None:
        center = (w / 2.0, h / 2.0)
    else:
        center = cfg.center

    # Trace arms
    arms = trace_arms(skel, center, min_length=cfg.min_arm_length)

    if len(arms) == 0:
        # No arms found - return NaNs
        return {
            "arm_count": 0.0,
            "b_mean": np.nan,
            "b_std": np.nan,
            "r2_mean": np.nan,
            "arm_spacing_mean": np.nan,
            "arm_spacing_std": np.nan,
            "fractal_dimension": np.nan,
            "fractal_dimension_ci_low": np.nan,
            "fractal_dimension_ci_high": np.nan,
        }

    # Fit log spiral to each arm
    b_vals = []
    r2_vals = []
    for arm_coords in arms:
        r, theta = polar_coords_from_center(arm_coords, center)
        b, a, r2, residual_std = fit_log_spiral(r, theta)
        if np.isfinite(b) and np.isfinite(r2):
            b_vals.append(b)
            r2_vals.append(r2)

    # Aggregate statistics
    if len(b_vals) > 0:
        b_mean = float(np.mean(b_vals))
        b_std = float(np.std(b_vals))
        r2_mean = float(np.mean(r2_vals))
    else:
        b_mean = np.nan
        b_std = np.nan
        r2_mean = np.nan

    # Arm spacing
    arm_spacing_mean, arm_spacing_std = compute_arm_spacing(arms, center)

    # Fractal dimension
    fd, fd_ci_low, fd_ci_high = box_count_fractal_dimension(
        boundary, cfg.box_sizes, cfg.n_subsamples
    )

    return {
        "arm_count": float(len(arms)),
        "b_mean": b_mean,
        "b_std": b_std,
        "r2_mean": r2_mean,
        "arm_spacing_mean": arm_spacing_mean,
        "arm_spacing_std": arm_spacing_std,
        "fractal_dimension": fd,
        "fractal_dimension_ci_low": fd_ci_low,
        "fractal_dimension_ci_high": fd_ci_high,
    }
