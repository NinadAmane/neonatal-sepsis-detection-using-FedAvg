import pandas as pd
import joblib
import os
from pathlib import Path

# ---------------- DATA ----------------
def load_data(path):
    """Load CSV dataset"""
    return pd.read_csv(path)

# ---------------- MODELS ----------------
def save_model(model, path):
    """Save a model using joblib"""
    os.makedirs(Path(path).parent, exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    """Load a model"""
    return joblib.load(path)
