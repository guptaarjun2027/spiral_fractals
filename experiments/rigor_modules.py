"""
Rigor Modules Runner (Step 4)
Unified entry point for judge-proof numerical experiments.

Runs both:
  - 4.1: Rigor Sensitivity (convergence across tiers)
  - 4.2: Phase Diagram (spiral existence sweep)
"""

import argparse
import yaml
import sys
from pathlib import Path

# Import the individual modules
from experiments.rigor_sensitivity import run_rigor_sensitivity
from experiments.phase_diagram import run_phase_diagram


def main():
    parser = argparse.ArgumentParser(
        description="Rigor Modules Runner (Step 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/rigor_modules.py --config configs/rigor_modules.yaml
  
Outputs:
  results/rigor_sensitivity.csv
  figures/rigor/convergence_strict_tiers.png
  results/phase_map.csv
  figures/rigor/phase_diagram.png
"""
    )
    parser.add_argument('--config', required=True, help="Path to rigor_modules.yaml config")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--skip-sensitivity', action='store_true', 
                       help="Skip rigor sensitivity analysis")
    parser.add_argument('--skip-phase', action='store_true',
                       help="Skip phase diagram")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("RIGOR MODULES RUNNER (Step 4)")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Seed: {args.seed}")
    print()
    
    # Run Step 4.1: Rigor Sensitivity
    if not args.skip_sensitivity:
        print("-" * 70)
        print("STEP 4.1: Rigor Sensitivity (Convergence Analysis)")
        print("-" * 70)
        try:
            run_rigor_sensitivity(
                config, 
                output_csv='results/rigor_sensitivity.csv',
                output_dir='figures/rigor',
                seed=args.seed
            )
            print("\n✓ Rigor Sensitivity complete\n")
        except Exception as e:
            print(f"\n✗ Rigor Sensitivity FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("Skipping Rigor Sensitivity (--skip-sensitivity)\n")
    
    # Run Step 4.2: Phase Diagram
    if not args.skip_phase:
        print("-" * 70)
        print("STEP 4.2: Phase Diagram")
        print("-" * 70)
        try:
            run_phase_diagram(
                config,
                output_csv='results/phase_map.csv',
                output_dir='figures/rigor',
                seed=args.seed
            )
            print("\n✓ Phase Diagram complete\n")
        except Exception as e:
            print(f"\n✗ Phase Diagram FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("Skipping Phase Diagram (--skip-phase)\n")
    
    # Summary
    print("=" * 70)
    print("RIGOR MODULES COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    
    outputs = [
        ("results/rigor_sensitivity.csv", "Convergence data (3 tiers)"),
        ("figures/rigor/convergence_strict_tiers.png", "Convergence plot"),
        ("results/phase_map.csv", "Phase map data"),
        ("figures/rigor/phase_diagram.png", "Phase diagram visualization")
    ]
    
    for path, desc in outputs:
        p = Path(path)
        if p.exists():
            print(f"  ✓ {path:<45} {desc}")
        else:
            print(f"  ✗ {path:<45} NOT FOUND")
    
    print("\nNext steps:")
    print("  1. Review results/rigor_sensitivity.csv for convergence across tiers")
    print("  2. Check figures/rigor/convergence_strict_tiers.png for visual confirmation")
    print("  3. Examine results/phase_map.csv for spiral existence regions")
    print("  4. View figures/rigor/phase_diagram.png for phase space visualization")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
