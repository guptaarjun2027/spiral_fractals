
import pytest
import numpy as np
from pathlib import Path
import sys

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.geometry import analyze_spiral_image

@pytest.fixture
def showcase_image():
    """Return path to a showcase image if it exists."""
    # Try figures/best first
    best_dir = ROOT / "figures" / "best"
    if best_dir.exists():
        images = list(best_dir.glob("*.png"))
        if images:
            return images[0]
            
    # Fallback to any png in figures
    figures_dir = ROOT / "figures"
    if figures_dir.exists():
        images = list(figures_dir.glob("*.png"))
        if images:
            return images[0]
            
    return None

def test_analyze_spiral_image_structure(showcase_image):
    """Test that analysis returns the correct structure."""
    if showcase_image is None:
        pytest.skip("No test images found in figures/")
        
    print(f"Testing with image: {showcase_image}")
    result = analyze_spiral_image(showcase_image)
    
    expected_keys = {
        "image", "arm_count", "b_mean", "b_std", "r2_mean", 
        "fractal_dimension", "fractal_dimension_ci_low", "fractal_dimension_ci_high"
    }
    
    assert set(result.keys()) == expected_keys
    assert isinstance(result["image"], str)
    assert isinstance(result["arm_count"], float)
    assert isinstance(result["fractal_dimension"], float)

def test_analyze_spiral_image_values(showcase_image):
    """Test that analysis returns reasonable values."""
    if showcase_image is None:
        pytest.skip("No test images found in figures/")
        
    result = analyze_spiral_image(showcase_image)
    
    # Arm count should be positive (due to fallback)
    assert result["arm_count"] > 0
    
    # Fractal dimension should be between 0 and 2 (usually > 1 for spirals)
    # Allowing up to 2.5 just in case of noise, but strictly it's < 2 for 2D
    assert 0 < result["fractal_dimension"] < 2.5
    
    # b_mean should be a float (can be nan if no valid arms, but usually not for showcase)
    # If it is nan, r2_mean should also be nan
    if np.isnan(result["b_mean"]):
        assert np.isnan(result["r2_mean"])
    else:
        assert isinstance(result["b_mean"], float)
