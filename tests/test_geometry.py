"""
Unit tests for geometry module.

Run:
    pytest tests/test_geometry.py
    or
    python -m pytest tests/test_geometry.py -v
"""

import numpy as np
import pytest
from pathlib import Path

try:
    from src.geometry import (
        load_image_gray,
        binary_mask,
        extract_boundary,
        skeletonize_boundary,
        polar_coords_from_center,
        trace_arms,
        fit_log_spiral,
        box_count_fractal_dimension,
        compute_arm_spacing,
        analyze_spiral_image,
        GeometryConfig,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.geometry import (
        load_image_gray,
        binary_mask,
        extract_boundary,
        skeletonize_boundary,
        polar_coords_from_center,
        trace_arms,
        fit_log_spiral,
        box_count_fractal_dimension,
        compute_arm_spacing,
        analyze_spiral_image,
        GeometryConfig,
    )


def test_load_image_gray():
    """Test that we can create and load a simple synthetic image."""
    # Create a simple test image
    from PIL import Image
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (100, 100), color="white")
        img.save(f.name)
        temp_path = f.name

    # Load it
    gray = load_image_gray(temp_path)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.float64 or gray.dtype == np.float32
    assert 0 <= gray.min() <= gray.max() <= 1

    Path(temp_path).unlink()


def test_binary_mask():
    """Test binary mask creation."""
    img = np.random.rand(50, 50)
    mask = binary_mask(img, threshold=0.5)
    assert mask.dtype == bool
    assert mask.shape == img.shape


def test_extract_boundary():
    """Test boundary extraction."""
    # Create a simple filled circle
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True

    boundary = extract_boundary(mask)
    assert boundary.dtype == bool
    assert boundary.shape == mask.shape
    # Boundary should have fewer pixels than filled mask
    assert np.sum(boundary) < np.sum(mask)


def test_skeletonize_boundary():
    """Test skeletonization."""
    boundary = np.zeros((50, 50), dtype=bool)
    boundary[10:40, 25] = True  # vertical line

    skel = skeletonize_boundary(boundary)
    assert skel.dtype == bool
    assert skel.shape == boundary.shape


def test_polar_coords_from_center():
    """Test polar coordinate conversion."""
    # Simple test: 4 points in cardinal directions from center
    points = np.array([
        [25, 35],  # right of center (row=25, col=35)
        [25, 15],  # left of center
        [15, 25],  # above center
        [35, 25],  # below center
    ])
    center = (25, 25)  # (cx, cy)

    r, theta = polar_coords_from_center(points, center)

    assert len(r) == 4
    assert len(theta) == 4
    # All should be distance 10 from center
    np.testing.assert_array_almost_equal(r, [10, 10, 10, 10])


def test_fit_log_spiral():
    """Test log spiral fitting with synthetic data."""
    # Create a perfect log spiral: r = a * exp(b * theta)
    a_true = 1.0
    b_true = 0.1
    theta = np.linspace(0, 4*np.pi, 100)
    r = a_true * np.exp(b_true * theta)

    b_fit, a_fit, r2, residual_std = fit_log_spiral(r, theta)

    assert np.isfinite(b_fit)
    assert np.isfinite(a_fit)
    assert np.isfinite(r2)
    # Should be a good fit
    assert r2 > 0.99
    # b should be close to true value
    assert abs(b_fit - b_true) < 0.01


def test_box_count_fractal_dimension():
    """Test box counting fractal dimension."""
    # Create a simple boundary (horizontal line)
    boundary = np.zeros((100, 100), dtype=bool)
    boundary[50, 10:90] = True

    fd, ci_low, ci_high = box_count_fractal_dimension(
        boundary, box_sizes=[2, 4, 8, 16], n_subsamples=5
    )

    # Should return finite values
    assert np.isfinite(fd)
    # For a 1D line, dimension should be close to 1
    assert 0.5 < fd < 1.5


def test_compute_arm_spacing():
    """Test arm spacing computation."""
    # Create 3 arms evenly spaced at 120 degrees
    center = (50, 50)

    # Arm 1: angle 0
    arm1 = np.array([[50, i] for i in range(50, 70)])
    # Arm 2: angle 120 deg
    arm2 = np.array([[50 - int(i*0.866), 50 + int(i*0.5)] for i in range(20)])
    # Arm 3: angle 240 deg
    arm3 = np.array([[50 + int(i*0.866), 50 + int(i*0.5)] for i in range(20)])

    arms = [arm1, arm2, arm3]

    spacing_mean, spacing_std = compute_arm_spacing(arms, center)

    assert np.isfinite(spacing_mean)
    # Should be roughly 2*pi/3
    expected = 2 * np.pi / 3
    # Allow generous tolerance since our synthetic arms are approximate
    assert abs(spacing_mean - expected) < 1.0


def test_analyze_spiral_image_no_crash():
    """Test that analyze_spiral_image runs without crashing on a simple image."""
    from PIL import Image
    import tempfile

    # Create a simple synthetic spiral-like image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (256, 256), color="black")
        # Draw some white pixels in a rough spiral pattern
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for i in range(0, 360, 10):
            angle = np.radians(i)
            r = 50 + i * 0.2
            x = 128 + int(r * np.cos(angle))
            y = 128 + int(r * np.sin(angle))
            draw.ellipse([x-2, y-2, x+2, y+2], fill="white")

        img.save(f.name)
        temp_path = f.name

    # Analyze it
    cfg = GeometryConfig(threshold=0.5, min_arm_length=10)
    result = analyze_spiral_image(temp_path, cfg)

    # Check all required keys are present
    required_keys = [
        "arm_count",
        "b_mean",
        "b_std",
        "r2_mean",
        "arm_spacing_mean",
        "arm_spacing_std",
        "fractal_dimension",
        "fractal_dimension_ci_low",
        "fractal_dimension_ci_high",
    ]

    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    # All values should be floats (possibly NaN)
    for key, val in result.items():
        assert isinstance(val, (float, np.floating)), f"{key} is not float: {type(val)}"

    Path(temp_path).unlink()


def test_analyze_spiral_image_on_real_controlled_image():
    """
    Test analyze_spiral_image on an actual controlled spiral if one exists.
    This test is skipped if no images are found.
    """
    # Look for a controlled spiral image
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sweep_dir = root / "figures" / "sweeps"

    if not sweep_dir.exists():
        pytest.skip("No sweep directory found")

    images = list(sweep_dir.glob("controlled_*.png"))
    if len(images) == 0:
        pytest.skip("No controlled images found")

    # Test on first image
    img_path = images[0]
    result = analyze_spiral_image(img_path)

    # Check all required keys
    required_keys = [
        "arm_count",
        "b_mean",
        "b_std",
        "r2_mean",
        "arm_spacing_mean",
        "arm_spacing_std",
        "fractal_dimension",
        "fractal_dimension_ci_low",
        "fractal_dimension_ci_high",
    ]

    for key in required_keys:
        assert key in result

    print(f"\nAnalyzed {img_path.name}:")
    print(f"  arm_count: {result['arm_count']}")
    print(f"  b_mean: {result['b_mean']:.4f}")
    print(f"  r2_mean: {result['r2_mean']:.4f}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
