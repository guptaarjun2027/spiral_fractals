"""
Real-world spiral comparison pipeline.

This script:
1. Loads real spiral images from data/real/*.png
2. Preprocesses them (grayscale → edges → skeleton)
3. Runs geometry analysis
4. Compares features to simulated spirals
5. Finds best matches

Run:
    python -m scripts.real_compare

Outputs:
    results/real_matches.csv
    figures/real_matches/*.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters, feature, color

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry import analyze_spiral_image, GeometryConfig


def preprocess_real_image(img_path: Path, output_path: Path) -> None:
    """
    Preprocess a real-world image for analysis.

    Steps:
    1. Convert to grayscale
    2. Apply Canny edge detection
    3. Save preprocessed image

    Args:
        img_path: path to original image
        output_path: path to save preprocessed image
    """
    from skimage import io, morphology

    # Load image
    img = io.imread(str(img_path))

    # Convert to grayscale
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img

    # Apply Canny edge detection
    edges = feature.canny(gray, sigma=2.0)

    # Optional: thin the edges
    skel = morphology.skeletonize(edges)

    # Convert to uint8 for saving
    output = (skel * 255).astype(np.uint8)

    # Save
    Image.fromarray(output).save(str(output_path))
    print(f"  Preprocessed {img_path.name} → {output_path.name}")


def compute_feature_distance(
    real_features: Dict[str, float],
    sim_features: Dict[str, float],
    feature_names: List[str],
    feature_weights: Dict[str, float],
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
) -> float:
    """
    Compute weighted Euclidean distance between feature vectors.

    Features are z-score normalized before distance computation.

    Args:
        real_features: feature dict for real image
        sim_features: feature dict for simulated image
        feature_names: list of feature keys to use
        feature_weights: weight for each feature
        feature_means: mean of each feature across simulated dataset
        feature_stds: std of each feature across simulated dataset

    Returns:
        weighted distance
    """
    distance_sq = 0.0

    for fname in feature_names:
        real_val = real_features.get(fname, np.nan)
        sim_val = sim_features.get(fname, np.nan)

        # Skip if either is NaN
        if not np.isfinite(real_val) or not np.isfinite(sim_val):
            continue

        # Z-score normalization
        mean = feature_means.get(fname, 0.0)
        std = feature_stds.get(fname, 1.0)

        if std < 1e-9:
            std = 1.0  # avoid division by zero

        real_z = (real_val - mean) / std
        sim_z = (sim_val - mean) / std

        weight = feature_weights.get(fname, 1.0)

        distance_sq += weight * (real_z - sim_z)**2

    return np.sqrt(distance_sq)


def main():
    # Paths
    real_dir = ROOT / "data" / "real"
    real_preprocessed_dir = ROOT / "data" / "real_preprocessed"
    sim_metrics_path = ROOT / "results" / "spiral_metrics.csv"
    output_csv = ROOT / "results" / "real_matches.csv"
    output_fig_dir = ROOT / "figures" / "real_matches"

    # Create directories
    real_preprocessed_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    # Check if real images exist
    if not real_dir.exists():
        print(f"WARNING: Real image directory not found: {real_dir}")
        print("Creating placeholder directory. Add real spiral images to data/real/*.png")
        real_dir.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {real_dir}")
        print("Exiting - no images to process.")
        return

    real_images = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))

    if len(real_images) == 0:
        print(f"WARNING: No images found in {real_dir}")
        print("Add real spiral images (*.png or *.jpg) to this directory.")
        print("Exiting - no images to process.")
        return

    print(f"Found {len(real_images)} real images")

    # Load simulated metrics
    if not sim_metrics_path.exists():
        print(f"ERROR: Simulated metrics not found at {sim_metrics_path}")
        print("Run analysis.ipynb first to generate metrics.")
        return

    sim_df = pd.read_csv(sim_metrics_path)
    print(f"Loaded {len(sim_df)} simulated spiral metrics")

    # Filter to valid simulated spirals
    sim_df = sim_df[
        sim_df['b_mean'].notna() &
        np.isfinite(sim_df['b_mean'])
    ].copy()
    print(f"Filtered to {len(sim_df)} valid simulated spirals")

    # Define features to compare
    feature_names = ['b_mean', 'arm_count', 'arm_spacing_mean', 'fractal_dimension']
    feature_weights = {
        'b_mean': 2.0,  # log-spiral slope is most important
        'arm_count': 1.5,
        'arm_spacing_mean': 1.0,
        'fractal_dimension': 1.0,
    }

    # Compute feature statistics for normalization
    feature_means = {}
    feature_stds = {}
    for fname in feature_names:
        if fname in sim_df.columns:
            vals = sim_df[fname].dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                feature_means[fname] = float(vals.mean())
                feature_stds[fname] = float(vals.std())
            else:
                feature_means[fname] = 0.0
                feature_stds[fname] = 1.0

    print("\nFeature statistics (for normalization):")
    for fname in feature_names:
        print(f"  {fname}: mean={feature_means[fname]:.3f}, std={feature_stds[fname]:.3f}")

    # Analyze each real image
    cfg = GeometryConfig(threshold="otsu", min_arm_length=30)

    real_results = []

    for real_img_path in real_images:
        print(f"\n=== Processing {real_img_path.name} ===")

        # Preprocess
        preprocessed_path = real_preprocessed_dir / f"preprocessed_{real_img_path.name}"
        try:
            preprocess_real_image(real_img_path, preprocessed_path)
        except Exception as e:
            print(f"  ERROR during preprocessing: {e}")
            continue

        # Analyze
        try:
            real_features = analyze_spiral_image(preprocessed_path, cfg)
            print(f"  Features: arm_count={real_features['arm_count']:.0f}, "
                  f"b_mean={real_features['b_mean']:.3f}, "
                  f"r2_mean={real_features['r2_mean']:.3f}")
        except Exception as e:
            print(f"  ERROR during analysis: {e}")
            continue

        # Compute distances to all simulated spirals
        distances = []
        for idx, sim_row in sim_df.iterrows():
            sim_features = sim_row.to_dict()
            dist = compute_feature_distance(
                real_features, sim_features,
                feature_names, feature_weights,
                feature_means, feature_stds
            )
            distances.append((idx, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Get top 5 matches
        top_matches = distances[:5]

        print(f"  Top 5 matches:")
        for rank, (idx, dist) in enumerate(top_matches, 1):
            sim_id = sim_df.loc[idx, 'id']
            sim_b = sim_df.loc[idx, 'b_mean']
            sim_arms = sim_df.loc[idx, 'arm_count']
            print(f"    {rank}. {sim_id} (dist={dist:.3f}, b={sim_b:.3f}, arms={sim_arms:.0f})")

        # Save results
        for rank, (idx, dist) in enumerate(top_matches, 1):
            real_results.append({
                'real_image': real_img_path.name,
                'rank': rank,
                'sim_id': sim_df.loc[idx, 'id'],
                'distance': dist,
                'sim_image_path': sim_df.loc[idx, 'image_path'],
                'real_b_mean': real_features['b_mean'],
                'real_arm_count': real_features['arm_count'],
                'sim_b_mean': sim_df.loc[idx, 'b_mean'],
                'sim_arm_count': sim_df.loc[idx, 'arm_count'],
            })

        # Create side-by-side comparison figure for top 3 matches
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Real image (preprocessed)
        axes[0].imshow(Image.open(preprocessed_path), cmap='gray')
        axes[0].set_title(f"Real: {real_img_path.name}\n"
                         f"b={real_features['b_mean']:.3f}, arms={real_features['arm_count']:.0f}",
                         fontsize=9)
        axes[0].axis('off')

        # Top 3 simulated matches
        for i, (idx, dist) in enumerate(top_matches[:3]):
            sim_img_path = ROOT / sim_df.loc[idx, 'image_path']
            if sim_img_path.exists():
                axes[i+1].imshow(Image.open(sim_img_path))
                axes[i+1].set_title(
                    f"Match #{i+1}: {sim_df.loc[idx, 'id']}\n"
                    f"dist={dist:.2f}, b={sim_df.loc[idx, 'b_mean']:.3f}",
                    fontsize=9
                )
            else:
                axes[i+1].text(0.5, 0.5, 'Image\nMissing', ha='center', va='center')
            axes[i+1].axis('off')

        plt.suptitle(f"Real Image Matching: {real_img_path.name}", fontsize=12)
        plt.tight_layout()

        fig_path = output_fig_dir / f"match_{real_img_path.stem}.png"
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close()

        print(f"  Saved comparison to {fig_path}")

    # Save all results to CSV
    if len(real_results) > 0:
        results_df = pd.DataFrame(real_results)
        results_df.to_csv(output_csv, index=False)
        print(f"\n=== Done ===")
        print(f"Saved {len(real_results)} match results to {output_csv}")
        print(f"Comparison figures in {output_fig_dir}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
