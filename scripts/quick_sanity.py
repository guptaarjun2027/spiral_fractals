"""
Quick sanity check: load first 9 controlled spirals from CSV and make thumbnail grid.

Run:
    python -m scripts.quick_sanity

Output:
    figures/analysis/sanity_grid.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "spiral_sweep.csv"
OUTPUT_PATH = ROOT / "figures" / "analysis" / "sanity_grid.png"


def main():
    # Create output directory
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load CSV
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV not found at {CSV_PATH}")
        print("Run sweep first: python -m scripts.sweep_params --controlled-only")
        return

    rows = []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["map_type"] == "controlled" and row["image_path"]:
                rows.append(row)
            if len(rows) >= 9:
                break

    if len(rows) == 0:
        print("[ERROR] No controlled images found in CSV")
        return

    print(f"[sanity] found {len(rows)} controlled spirals")

    # Load images
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()

    for i, row in enumerate(rows):
        img_rel_path = row["image_path"]
        img_path = ROOT / img_rel_path

        if not img_path.exists():
            axes[i].text(0.5, 0.5, "Missing", ha="center", va="center")
            axes[i].set_title(f"ID: {row['id']}")
            axes[i].axis("off")
            continue

        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(
            f"{row['radial_mode'][:3]}\n"
            f"δ={row.get('delta_r', 'nan')[:6]} α={row.get('alpha', 'nan')[:5]}\n"
            f"ω={float(row['omega']):.2f}",
            fontsize=8,
        )
        axes[i].axis("off")

    # Hide any unused subplots
    for i in range(len(rows), 9):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    print(f"[done] saved sanity grid to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
