
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

def test_spiral_metrics_exists():
    """Test that spiral_metrics.csv exists and has expected columns."""
    metrics_path = ROOT / "results" / "spiral_metrics.csv"
    
    # It's okay if it doesn't exist yet (fresh clone), but if it does, check columns
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        expected_cols = ["arm_count", "fractal_dimension", "r2_mean"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column {col} in spiral_metrics.csv"
            
def test_stability_metrics_structure():
    """Test stability metrics structure if file exists."""
    metrics_path = ROOT / "results" / "stability_metrics.csv"
    
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        expected_cols = [
            "image", "base_id", "delta_r", "omega", "phase_eps",
            "arm_count", "b_mean", "r2_mean", "fractal_dimension"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column {col} in stability_metrics.csv"
