# scripts/clean_federated_data.py
import pandas as pd
import glob
import os
import joblib
from sklearn.utils import resample

# --- Load the global scaler ---
try:
    scaler = joblib.load("models/scaler.pkl")
    print("✅ Loaded global scaler from models/scaler.pkl")
except FileNotFoundError:
    print("❌ models/scaler.pkl not found. Run preprocess_data.py first.")
    exit()

os.makedirs("data/federated_cleaned", exist_ok=True)
print("=== Cleaning and Balancing Federated Datasets (using GLOBAL scaler) ===")

# Columns that must be removed (constant across hospitals)
CONST_COLS = [
    "AST", "Alkalinephos", "Bilirubin_direct",
    "Bilirubin_total", "TroponinI", "Fibrinogen"
]

for file in glob.glob("data/federated/hospital_*.csv"):
    print(f"\n--- Processing {file} ---")
    df = pd.read_csv(file)

    # 1️⃣ Remove constant columns explicitly
    found_consts = [c for c in CONST_COLS if c in df.columns]
    if found_consts:
        print(f"Removing constant columns: {found_consts}")
        df = df.drop(columns=found_consts)

    # 2️⃣ Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicates")

    # 3️⃣ Balance dataset (oversample minority)
    neg = df[df["SepsisLabel"] == 0]
    pos = df[df["SepsisLabel"] == 1]
    if len(pos) > 0:
        pos_up = resample(pos, replace=True, n_samples=len(neg), random_state=42)
        df_balanced = pd.concat([neg, pos_up])
        print(f"Balanced dataset → {len(neg)} neg / {len(pos_up)} pos")
    else:
        df_balanced = df

    # 4️⃣ Scale using global scaler (.transform only)
    features = [c for c in df_balanced.columns if c != "SepsisLabel"]
    try:
        df_balanced[features] = scaler.transform(df_balanced[features])
        print("✅ Scaled features using global scaler.")
    except Exception as e:
        print(f"⚠️ Scaling failed: {e}")
        print("Continuing without scaling for this file.")

    # 5️⃣ Save cleaned version
    out_path = file.replace("data/federated", "data/federated_cleaned")
    df_balanced.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned file: {out_path} ({len(df_balanced.columns)} columns)")
