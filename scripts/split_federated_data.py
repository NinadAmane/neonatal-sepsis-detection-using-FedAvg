# scripts/split_federated_data.py
import pandas as pd
from pathlib import Path
import numpy as np



# Input and output paths
input_path = Path("data/processed/cleaned_dataset.csv")
output_dir = Path("data/federated")
output_dir.mkdir(parents=True, exist_ok=True)

# Number of hospitals (clients)
NUM_HOSPITALS = 3

# Read the cleaned dataset
df = pd.read_csv(input_path)

# Shuffle before splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split roughly equally
splits = np.array_split(df, NUM_HOSPITALS)

# Save each hospital’s data
for i, split_df in enumerate(splits, start=1):
    split_path = output_dir / f"hospital_{i}.csv"
    split_df.to_csv(split_path, index=False)
    print(f"✅ Saved {split_path} with {len(split_df)} records.")
