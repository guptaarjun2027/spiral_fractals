#!/usr/bin/env python
"""
Script to run geometry analysis on a batch of spiral images.
"""
import argparse
import sys
from pathlib import Path

# Add repo root to path to allow imports from src
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.geometry import analyze_image_batch

def main():
    parser = argparse.ArgumentParser(
        description="Run geometry analysis on sweep images."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="figures/sweeps",
        help="Directory containing .png spiral images",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/spiral_metrics.csv",
        help="Path to output CSV file",
    )
    args = parser.parse_args()

    input_dir = ROOT / args.input_dir
    output_csv = ROOT / args.output_csv

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return 1

    print(f"Looking for images in {input_dir}...")
    # Support both .png and .jpg
    paths = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    
    if not paths:
        print(f"No images found in {input_dir}")
        return 0

    print(f"Found {len(paths)} images. Starting analysis...")
    
    df = analyze_image_batch(paths)
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n[geometry] Analysis complete.")
    print(f"[geometry] Wrote {len(df)} rows to {output_csv}")
    
    # Print a quick summary if we have data
    if not df.empty and "arm_count" in df.columns:
        print("\nQuick Summary:")
        print(df[["arm_count", "fractal_dimension", "r2_mean"]].describe().to_string())

    return 0

if __name__ == "__main__":
    sys.exit(main())
