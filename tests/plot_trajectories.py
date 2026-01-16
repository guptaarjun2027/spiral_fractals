import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from scipy.stats import linregress

def run_final_analysis():
    # 1. Setup Plot
    plt.figure(figsize=(12, 7))
    files = sorted(glob.glob('tests/data/trajectories/*.csv'))
    
    if not files:
        print("No data files found in tests/data/trajectories/")
        return

    print(f"{'Point ID':<12} | {'Slope (Kappa)':<15} | {'Straightness (R^2)':<15}")
    print("-" * 50)

    for file in files:
        df = pd.read_csv(file)
        label = os.path.basename(file).replace('.csv', '')
        
        # 2. Plotting the Trajectory
        plt.plot(df['log_r'], df['phi_unwrapped'], alpha=0.7, label=label)

        # 3. Statistical Analysis (Linearity Check)
        # We focus on the "Stable" part of the escape (log_r > 5)
        stable_df = df[df['log_r'] > 5]
        
        if len(stable_df) >= 3:
            slope, intercept, r_value, p_value, std_err = linregress(stable_df['log_r'], stable_df['phi_unwrapped'])
            r_squared = r_value**2
            print(f"{label:<12} | {slope:<15.6f} | {r_squared:<15.6f}")
        else:
            print(f"{label:<12} | Insufficient data for R^2 calculation")

    # 4. Finalizing the Plot
    plt.title('Logarithmic Spiral Verification (Experiment 3.1)')
    plt.xlabel('Log of Radius (log r)')
    plt.ylabel('Unwrapped Angle (phi)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    os.makedirs('tests/plots', exist_ok=True)
    plt.savefig('tests/plots/final_verification_plot.png', dpi=300)
    print("-" * 50)
    print("Plot saved to: tests/plots/final_verification_plot.png")
    plt.show()

if __name__ == "__main__":
    run_final_analysis()
    
    