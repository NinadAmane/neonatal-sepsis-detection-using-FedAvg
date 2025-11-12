# scripts/rebuild_scaler.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

print("🔧 Rebuilding global scaler based on cleaned 31-feature dataset...")

# Load the processed dataset
df = pd.read_csv("data/processed/cleaned_dataset.csv")

# Drop constant columns (same ones removed during cleaning)
const_cols = [
    'AST', 'Alkalinephos', 'Bilirubin_direct',
    'Bilirubin_total', 'TroponinI', 'Fibrinogen'
]
df = df.drop(columns=const_cols, errors='ignore')

# Separate features and labels
features = [c for c in df.columns if c != "SepsisLabel"]
X = df[features]

# Fit a new scaler on the final features
scaler = StandardScaler()
scaler.fit(X)

# Save it
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

print(f"✅ New scaler trained and saved at models/scaler.pkl")
print(f"✅ Final number of features used: {len(features)} ({features[:5]}...)")
