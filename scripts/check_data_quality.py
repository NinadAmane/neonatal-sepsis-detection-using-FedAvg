import pandas as pd, glob

print("=== Checking Federated Data Quality ===")

for file in glob.glob("data/federated/hospital_*.csv"):
    print(f"\n--- {file} ---")
    df = pd.read_csv(file)
    if "SepsisLabel" not in df.columns:
        print("❌ Missing 'SepsisLabel'")
        continue

    counts = df["SepsisLabel"].value_counts()
    print(f"Label balance:\n{counts}")
    print(f"Positive ratio: {counts.get(1,0)/(counts.sum()):.4f}")

    # Check for NaNs
    nan_counts = df.isna().sum().sum()
    print(f"NaN count: {nan_counts}")

    # Check for constant columns
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        print(f"⚠️ Constant columns: {const_cols}")

    # Check duplicates
    dupes = df.duplicated().sum()
    print(f"Duplicate rows: {dupes}")
