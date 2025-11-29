import os
import pandas as pd
import numpy as np

SEED_FILE = 'data/metadata/seed_masses.csv'
FINAL_FILE = 'data/metadata/blackhole_masses.csv'
IMAGE_DIR = 'data/real_images/'

# Ensure directories exist
os.makedirs(os.path.dirname(FINAL_FILE), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(SEED_FILE)

# Add columns
df['RA'] = np.nan
df['Dec'] = np.nan
df['image_path'] = np.nan

# Generate placeholder images
for idx, row in df.iterrows():
    filepath = os.path.join(IMAGE_DIR, f"galaxy_{idx+1}.jpg")
    with open(filepath, 'w') as f:
        f.write(f"Placeholder for {row['galaxy_name']} (log_mbh={row['log_mbh']})\n")
    df.at[idx, 'image_path'] = filepath

# Save final CSV
df.to_csv(FINAL_FILE, index=False)
print(f"Curation complete! CSV saved to {FINAL_FILE} and {len(df)} placeholder images created.")
