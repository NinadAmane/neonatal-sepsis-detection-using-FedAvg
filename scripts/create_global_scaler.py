# scripts/create_global_scaler.py
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Path to your processed dataset
DATA_PATH = "data/processed/cleaned_dataset.csv"
SCALER_PATH = "models/scaler.pkl"

# --- Load data ---
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: {DATA_PATH} not found. Please make sure your preprocessed data exists.")
    exit()

print(f"📂 Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Drop label column
if "SepsisLabel" not in df.columns:
    print("❌ Error: Missing 'SepsisLabel' column.")
    exit()

features = df.drop(columns=["SepsisLabel"])
print(f"✅ Found {len(features.columns)} feature columns.")

# --- Fit a global scaler ---
scaler = StandardScaler()
scaler.fit(features)

# --- Save the scaler ---
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, SCALER_PATH)
print(f"💾 Saved global scaler to: {SCALER_PATH}")
