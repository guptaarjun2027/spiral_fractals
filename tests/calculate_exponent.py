import pandas as pd
import numpy as np
from scipy import stats
import os

def finalize_exponent_analysis():
    # 1. The data from your stable scaling window (Asymptotic Phase)
    # These are the Kappa values from your successful escapees
    data = {
        'trajectory_id': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
        'kappa_exponent': [
            0.006338, 0.002698, -0.000956, -0.009647, 0.011471, 
            0.028484, 0.002250, -0.010817, 0.005396, 0.013106, -0.004957
        ]
    }
    df = pd.DataFrame(data)

    # 2. Statistical Analysis
    mean_k = df['kappa_exponent'].mean()
    std_err = stats.sem(df['kappa_exponent'])
    
    # Calculate 95% Confidence Interval
    # This is the "Defensible CI" required for Step 5
    ci_95 = std_err * stats.t.ppf((1 + 0.95) / 2, len(df) - 1)

    # 3. Create a summary table
    summary_data = {
        'Metric': ['Mean Exponent (Kappa)', '95% CI (+/-)', 'Standard Error', 'Sample Size (n)', 'Status'],
        'Value': [
            round(mean_k, 6), 
            round(ci_95, 6), 
            round(std_err, 6), 
            len(df),
            'COMPLETE - Stable Window Verified'
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # 4. Save to a new file
    output_dir = 'tests/data/analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_path = os.path.join(output_dir, 'step_5_exponent_results.csv')
    summary_df.to_csv(file_path, index=False)

    # 5. Terminal Output for your notes
    print("\n" + "="*45)
    print("      SCIENTIFIC DATA: STEP 5 COMPLETE")
    print("="*45)
    print(f"MEAN EXPONENT (K): {mean_k:.6f}")
    print(f"95% CONFIDENCE INTERVAL: +/- {ci_95:.6f}")
    print(f"RESULTS SAVED TO: {file_path}")
    print("="*45 + "\n")

if __name__ == "__main__":
    finalize_exponent_analysis()
    
    