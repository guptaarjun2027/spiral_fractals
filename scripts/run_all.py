"""
One-click reproduction pipeline for ISEF research project.

This script runs the entire analysis pipeline:
1. Parameter sweep (generates spiral images)
2. Geometry analysis (computes metrics)
3. Exports figures and results

Run:
    python -m scripts.run_all

Options:
    --skip-sweep    Skip the parameter sweep (use existing images)
    --sweep-mode    Sweep mode: small (default) | full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,
            text=True,
            capture_output=False,
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete ISEF spiral fractal analysis pipeline"
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip parameter sweep (use existing images)",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=["small", "full"],
        default="small",
        help="Sweep size: small (quick test) or full (comprehensive)",
    )
    args = parser.parse_args()

    print(f"""
{'='*60}
  ISEF Spiral Fractal Analysis Pipeline
  Beyond the Mandelbrot: Modeling Real-World Spiral Growth
{'='*60}

Repository: {ROOT}
Mode: {args.sweep_mode}
Skip sweep: {args.skip_sweep}
""")

    # Step 1: Parameter sweep
    if not args.skip_sweep:
        sweep_flags = ["--controlled-only", "--overwrite"]
        if args.sweep_mode == "full":
            sweep_flags.append("--full")
        else:
            sweep_flags.append("--small")

        success = run_command(
            [sys.executable, "-m", "scripts.sweep_params"] + sweep_flags,
            f"Parameter Sweep ({args.sweep_mode} mode)",
        )

        if not success:
            print("\nERROR: Sweep failed. Stopping pipeline.")
            return 1

        # Quick sanity check
        run_command(
            [sys.executable, "-m", "scripts.quick_sanity"],
            "Sanity Check (thumbnail grid)",
        )
    else:
        print("\n[SKIPPED] Parameter sweep (using existing images)")

    # Step 1.5: Geometry Extraction
    print(f"\n{'='*60}")
    print("STEP: Geometry Extraction")
    print(f"{'='*60}\n")
    
    success = run_command(
        [sys.executable, "-m", "scripts.run_geometry_on_sweeps"],
        "Geometry Analysis",
    )
    
    if not success:
        print("\n[WARNING] Geometry analysis failed. Continuing pipeline...")


    # Step 2: Run analysis notebook programmatically
    # Check if jupyter/nbconvert is available
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        print(f"\n{'='*60}")
        print("STEP: Analysis Notebook Execution")
        print(f"{'='*60}\n")

        notebook_path = ROOT / "notebooks" / "analysis.ipynb"

        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

        try:
            ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
            print("\n✓ Analysis notebook executed successfully")
        except Exception as e:
            print(f"\n✗ Analysis notebook execution failed: {e}")
            print("You can run it manually: jupyter notebook notebooks/analysis.ipynb")

    except ImportError:
        print("\n[INFO] nbconvert not available - skipping automated notebook execution")
        print("To run analysis manually:")
        print("  jupyter notebook notebooks/analysis.ipynb")

    # Step 3: Export summary statistics
    print(f"\n{'='*60}")
    print("STEP: Generate Summary Report")
    print(f"{'='*60}\n")

    metrics_path = ROOT / "results" / "spiral_metrics.csv"
    if metrics_path.exists():
        import pandas as pd

        df = pd.read_csv(metrics_path)
        controlled = df[df["map_type"] == "controlled"]

        print(f"Total spirals analyzed: {len(df)}")
        print(f"Controlled spirals: {len(controlled)}")

        if len(controlled) > 0:
            clean = controlled[
                (controlled["arm_count"] >= 2)
                & (controlled["r2_mean"] >= 0.7)
                & (controlled["b_mean"] > 0)
            ]
            print(f"Clean spirals: {len(clean)} ({100*len(clean)/len(controlled):.1f}%)")

            print("\nGeometry metrics (controlled spirals):")
            print(controlled[["arm_count", "b_mean", "r2_mean", "fractal_dimension"]].describe())

        print(f"\n✓ Summary report generated")
    else:
        print("Metrics file not found. Run analysis.ipynb to generate metrics.")

    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}\n")

    print("Output locations:")
    print(f"  Images:        figures/sweeps/")
    print(f"  Metrics:       results/spiral_metrics.csv")
    print(f"  Top spirals:   results/top_spirals.csv")
    print(f"  Analysis:      figures/analysis/")
    print(f"  Notebooks:     notebooks/analysis.ipynb, notebooks/theory.ipynb")

    print("\nNext steps:")
    print("  1. Review figures in figures/analysis/")
    print("  2. Open notebooks/theory.ipynb for theoretical validation")
    print("  3. Add real-world images to data/real/ and run:")
    print("     python -m scripts.real_compare")

    return 0


if __name__ == "__main__":
    sys.exit(main())
