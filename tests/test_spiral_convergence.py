import numpy as np
import pandas as pd
import os

def run_spiral_test(theta_val, lambda_val, epsilon_val, start_radius, num_points=12):
    """
    Finalized Multi-Point Validation Script.
    Tests if escaping orbits converge to a consistent spiral constant (Kappa).
    """
    # Create folder structure for results
    os.makedirs('tests/data/trajectories', exist_ok=True)
    
    summary_data = []
    angles_to_test = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    print(f"Starting Multi-Point Validation (n={num_points})...")
    print(f"Parameters: theta={theta_val}, lambda={lambda_val}, epsilon={epsilon_val}, R0={start_radius}")

    for i, start_angle in enumerate(angles_to_test):
        z = start_radius * np.exp(1j * start_angle)
        z0_label = f"point_{i}"
        
        trajectory = []
        
        # Iterate F(z) = e^(i*theta)z + lambda*z^2 + epsilon*z^-2
        for n in range(30): 
            # 1. High-Capacity Overflow Protection
            # Python's float64 can handle up to 1e308, we stop at 1e100 for safety.
            if np.abs(z) > 1e100:
                break
                
            term1 = np.exp(1j * theta_val) * z
            term2 = lambda_val * (z**2)
            
            # 2. Singularity Protection
            if np.abs(z) < 1e-10:
                break
            term3 = epsilon_val * (z**-2)
            
            z = term1 + term2 + term3
            trajectory.append([n, np.abs(z), np.angle(z)])
        
        # Convert to DataFrame
        df = pd.DataFrame(trajectory, columns=['n', 'r', 'phi'])
        
        # 3. Data Cleaning
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # We only need 4 points for a valid quadratic slope calculation
        if len(df) < 4:
            print(f"Skipping {z0_label}: Escaped too quickly ({len(df)} steps).")
            continue

        df['phi_unwrapped'] = np.unwrap(df['phi'])
        df['log_r'] = np.log(df['r'])
        
        # 4. Calculate individual Kappa (slope) using all available valid steps
        try:
            # We use the full trajectory to get the most accurate fit
            kappa, intercept = np.polyfit(df['log_r'], df['phi_unwrapped'], 1)
            
            # Save individual trajectory
            df.to_csv(f'tests/data/trajectories/{z0_label}.csv', index=False)
            
            summary_data.append({
                'point_id': z0_label,
                'initial_angle': start_angle,
                'measured_kappa': kappa
            })
        except Exception as e:
            print(f"Skipping {z0_label}: Math error ({e})")

    # Save summary report
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('tests/data/spiral_summary.csv', index=False)
    
    # 5. Global Statistics
    if not summary_df.empty:
        avg_kappa = summary_df['measured_kappa'].mean()
        std_dev = summary_df['measured_kappa'].std()
        consistency = (1 - (std_dev / abs(avg_kappa))) * 100 if avg_kappa != 0 else 0
        
        print("\n" + "="*40)
        print(" VALIDATION RESULTS")
        print("="*40)
        print(f"Average Spiral Constant (Kappa): {avg_kappa:.6f}")
        print(f"Standard Deviation:             {std_dev:.6f}")
        print(f"Numerical Consistency:          {consistency:.4f}%")
        print("="*40)
        print("Check 'tests/data/' for CSV files.")
    else:
        print("\nERROR: All points skipped. Reduce start_radius or lambda.")

# --- RUNNING THE TEST ---
# These 'Sweet Spot' parameters ensure the points stay in range 
# long enough to be measured before hitting infinity.
run_spiral_test(
    theta_val=0.5, 
    lambda_val=0.1,   
    epsilon_val=0.01, 
    start_radius=15.0, 
    num_points=12     
)

