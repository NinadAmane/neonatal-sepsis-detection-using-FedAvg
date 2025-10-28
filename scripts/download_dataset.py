import kagglehub

# Download latest version
path = kagglehub.dataset_download("salikhussaini49/prediction-of-sepsis")

import os
import zipfile
import shutil

# -----------------------------
# Step 1: Define paths
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# -----------------------------
# Step 2: Download dataset
# -----------------------------
print("Downloading Kaggle dataset...")
dataset_path = kagglehub.dataset_download("salikhussaini49/prediction-of-sepsis")
print(f"Dataset downloaded to: {dataset_path}")

# -----------------------------
# Step 3: Unzip into data/raw/
# -----------------------------
if dataset_path.endswith(".zip"):
    print("Unzipping dataset...")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
    print(f"Dataset extracted to: {RAW_DATA_DIR}")
else:
    print("Dataset is not a zip file. Copying to raw data folder...")
    # Use copytree if it's a folder
    if os.path.isdir(dataset_path):
        dest_path = os.path.join(RAW_DATA_DIR, os.path.basename(dataset_path))
        shutil.copytree(dataset_path, dest_path)
        print(f"Dataset copied to: {dest_path}")
    else:
        # If it's a single file
        shutil.copy(dataset_path, RAW_DATA_DIR)
        print(f"Dataset copied to: {RAW_DATA_DIR}")
