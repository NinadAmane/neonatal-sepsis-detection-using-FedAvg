# scripts/fedavg_aggregate.py
import torch
import pandas as pd
import glob
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.model import SepsisMLP

# -----------------------------
# Federated Averaging
# -----------------------------
def fedavg(state_dicts):
    """Averages the weights from a list of state_dicts."""
    avg = {}
    keys = state_dicts[0].keys()
    for k in keys:
        avg[k] = sum([sd[k].float() for sd in state_dicts]) / len(state_dicts)
    return avg

# -----------------------------
# Prepare Test Data with Global Scaler
# -----------------------------
def prepare_test_data(csv_path="data/processed/cleaned_dataset.csv", scaler_path="models/scaler.pkl"):
    print(f"📂 Loading test data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"❌ Test data not found at {csv_path}")
        return None, None
        
    df = pd.read_csv(csv_path)

    # Drop same constant columns removed during training
    const_cols = [
        'AST', 'Alkalinephos', 'Bilirubin_direct', 
        'Bilirubin_total', 'TroponinI', 'Fibrinogen'
    ]
    df = df.drop(columns=const_cols, errors='ignore')
    print(f"✅ Removed constant columns from test set for compatibility.")

    # Load global scaler
    try:
        scaler = joblib.load(scaler_path)
        print(f"✅ Loaded global scaler from {scaler_path}")
    except FileNotFoundError:
        print(f"❌ Scaler not found at {scaler_path}")
        return None, None

    # Split data
    _, test = train_test_split(df, test_size=0.2, random_state=42)

    y = torch.tensor(test["SepsisLabel"].values.astype("float32")).view(-1, 1)
    features = [c for c in test.columns if c != "SepsisLabel"]

    try:
        X_scaled = scaler.transform(test[features])
        X = torch.tensor(X_scaled.astype("float32"))
        print("✅ Scaled test features using global scaler.")
    except Exception as e:
        print(f"❌ Scaler error: {e}")
        return None, None

    return X, y

# -----------------------------
# Evaluate Global Model
# -----------------------------
def eval_model(state_dict, X, y):
    model = SepsisMLP(X.shape[1])
    model.load_state_dict(state_dict)
    model.eval()
    
    with torch.no_grad():
        preds_logits = model(X)
        preds = torch.sigmoid(preds_logits).numpy().ravel()
        
    y_true = y.numpy().ravel()
    preds_bin = (preds > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, preds_bin),
        "precision": precision_score(y_true, preds_bin, zero_division=0),
        "recall": recall_score(y_true, preds_bin, zero_division=0),
        "f1": f1_score(y_true, preds_bin, zero_division=0),
        "auc": roc_auc_score(y_true, preds),
        "confusion_matrix": confusion_matrix(y_true, preds_bin).tolist()
    }
    return metrics

# -----------------------------
# Main
# -----------------------------
def main():
    model_paths = glob.glob("models/local_model_hospital_*.pt")
    if not model_paths:
        print("❌ No local models found. Train them first with scripts/train_local_torch.py")
        return

    print(f"🔗 Loading {len(model_paths)} models...")

    state_dicts = []
    for path in model_paths:
        state = torch.load(path, map_location="cpu")

        # ✅ Debug: show the first linear layer shape
        for k, v in state.items():
            if v.ndim == 2:
                print(f"🧠 {os.path.basename(path)} → first layer shape = {tuple(v.shape)}")
                if v.shape[1] == 37:
                    print(f"⚠️ Model {path} still has 37 input features! It was trained on old (un-cleaned) data.")
                    print("👉 Retrain this hospital using data/federated_cleaned/... before aggregation.")
                    return
                break

        state_dicts.append(state)

    print("⚖️ Performing Federated Averaging...")
    avg_state = fedavg(state_dicts)

    global_model_path = "models/global_federated_model.pt"
    torch.save(avg_state, global_model_path)
    print(f"💾 Saved global model: {global_model_path}")

    X_test, y_test = prepare_test_data()
    if X_test is None:
        print("❌ Could not prepare test data. Exiting.")
        return

    metrics = eval_model(avg_state, X_test, y_test)

    print("\n📈 --- Global Model Evaluation ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k.capitalize()}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
