import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# -------------------- CONFIG --------------------
PROJECT_ROOT = os.getcwd()
RAW_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "Dataset.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
PROCESSED_CSV = os.path.join(PROCESSED_DIR, "cleaned_dataset.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# -------------------- LOAD --------------------
print("Loading raw dataset...")
df = pd.read_csv(RAW_CSV)
print("Original dataset shape:", df.shape)

# -------------------- KEEP COLUMNS --------------------
columns_to_keep = [
    # Vitals
    "HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2",
    # Labs
    "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN",
    "Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
    "Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total",
    "TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets",
    # Demographics
    "Age","Gender",
    # Target
    "SepsisLabel",
    # ID
    "Patient_ID"
]
df = df[columns_to_keep].copy()

# -------------------- CLEANING --------------------
df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)

# Convert all numeric where possible
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Encode Gender if needed
if df["Gender"].dtype == "object":
    df["Gender"] = df["Gender"].map({"M": 1, "F": 0})

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for c in numeric_cols:
    df[c].fillna(df[c].median(), inplace=True)

# Clip outliers (1st–99th percentile)
for c in numeric_cols:
    lower, upper = df[c].quantile(0.01), df[c].quantile(0.99)
    df[c] = np.clip(df[c], lower, upper)
print("✅ Outliers clipped (1st–99th percentile).")

# -------------------- SCALING --------------------
features_to_scale = [c for c in numeric_cols if c not in ["SepsisLabel", "Patient_ID"]]
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler saved to {SCALER_PATH}")

# -------------------- FINALIZE --------------------
df = shuffle(df, random_state=42).reset_index(drop=True)
df.to_csv(PROCESSED_CSV, index=False)
print(f"📁 Saved preprocessed dataset to {PROCESSED_CSV}")
