import numpy as np
import pandas as pd
import os

def run_spiral_test(theta_val, lambda_val, epsilon_val, start_radius, num_points=12):
    """
    Tests multiple starting points to see if they converge to the same 
    Spiral Constant (Kappa) without crashing from numerical overflow.
    """
    # Create folder structure for results
    os.makedirs('tests/data/trajectories', exist_ok=True)
    
    summary_data = []
    angles_to_test = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    print(f"Starting Multi-Point Validation (n={num_points})...")

    for i, start_angle in enumerate(angles_to_test):
        z = start_radius * np.exp(1j * start_angle)
        z0_label = f"point_{i}"
        
        trajectory = []
        
        # Iterate F(z) = e^(i*theta)z + lambda*z^2 + epsilon*z^-2
        for n in range(25): 
            # 1. Overflow Protection: Stop if numbers get too big for Python to handle
            if np.abs(z) > 1e15:
                break
                
            term1 = np.exp(1j * theta_val) * z
            term2 = lambda_val * (z**2)
            
            # 2. Singularity Protection: Avoid division by zero
            if np.abs(z) < 1e-6:
                break
            term3 = epsilon_val * (z**-2)
            
            z = term1 + term2 + term3
            trajectory.append([n, np.abs(z), np.angle(z)])
        
        # Convert to DataFrame
        df = pd.DataFrame(trajectory, columns=['n', 'r', 'phi'])
        
        # 3. Data Cleaning: Remove any infinities that slipped through
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # We need at least 3 points to calculate a trend
        if len(df) < 5:
            print(f"Skipping {z0_label}: Escaped too quickly to measure.")
            continue

        df['phi_unwrapped'] = np.unwrap(df['phi'])
        df['log_r'] = np.log(df['r'])
        
        # Calculate individual Kappa (slope) using the last 10 valid steps
        stable_df = df.tail(10)
        try:
            kappa, intercept = np.polyfit(stable_df['log_r'], stable_df['phi_unwrapped'], 1)
            
            # Save individual trajectory
            df.to_csv(f'tests/data/trajectories/{z0_label}.csv', index=False)
            
            summary_data.append({
                'point_id': z0_label,
                'initial_angle': start_angle,
                'measured_kappa': kappa
            })
        except np.linalg.LinAlgError:
            print(f"Skipping {z0_label}: Math error during slope calculation.")

    # Save summary report
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('tests/data/spiral_summary.csv', index=False)
    
    # Calculate global stats
    if not summary_df.empty:
        avg_kappa = summary_df['measured_kappa'].mean()
        std_dev = summary_df['measured_kappa'].std()
        
        print("-" * 40)
        print(f"Validation Summary")
        print(f"Average Kappa: {avg_kappa:.6f}")
        print(f"Standard Deviation: {std_dev:.6f}")
        print(f"Consistency: {(1 - (std_dev/abs(avg_kappa)))*100:.2f}%")
        print("-" * 40)
    else:
        print("Error: No valid points were measured. Try increasing start_radius.")

# --- RUNNING THE TEST ---
# Note: Lowered lambda and radius slightly to prevent immediate overflow
run_spiral_test(
    theta_val=0.5, 
    lambda_val=0.2,   # Quadratic growth speed
    epsilon_val=0.1,  # Inverse power strength
    start_radius=3.0, # Distance from origin
    num_points=12     # Number of test points
)