import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score
)
from pathlib import Path

# --- Configuration ---
MODEL_DIR = Path("models")
DATA_PATH = Path("data/processed/cleaned_dataset.csv")
TARGET_COL = "SepsisLabel"


# --- Function to evaluate a model ---
def evaluate_model(model, X_test, y_test):
    """Calculates all key metrics for a given model."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_pred_proba)
    }

# --- 1. Load the FULL dataset ---
print("Loading full dataset for evaluation...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"❌ Error: Full dataset not found at {DATA_PATH}")
    exit(1)

# --- 2. Create the Master Test Set ---
# This is the most important step for a fair comparison.
# We hold out 20% of the *entire* dataset as a final test set.
print("Creating master train/test split...")
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# --- FIX: Convert continuous y back to discrete 0s and 1s ---
# This is the same fix from the training script
y = (y > 0.5).astype(int) 

# Split into a 'full train' set (80%) and 'test' set (20%)
X_train_full, X_test_master, y_train_full, y_test_master = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. Train "Gold Standard" Centralized Model ---
# This model is trained on ALL the training data (80% of the total)
print("Training 'Centralized' (gold standard) model...")
centralized_model = LogisticRegression(max_iter=1000, class_weight='balanced')
centralized_model.fit(X_train_full, y_train_full)

# --- 4. Load all our trained models ---
print("Loading local and federated models...")
models_to_evaluate = {}
try:
    models_to_evaluate["Local Hospital 1"] = joblib.load(MODEL_DIR / "local_model_hospital_1.pkl")
    models_to_evaluate["Local Hospital 2"] = joblib.load(MODEL_DIR / "local_model_hospital_2.pkl")
    models_to_evaluate["Local Hospital 3"] = joblib.load(MODEL_DIR / "local_model_hospital_3.pkl")
    models_to_evaluate["FEDERATED GLOBAL"] = joblib.load(MODEL_DIR / "global_federated_model.pkl")
    models_to_evaluate["CENTRALIZED (Gold Standard)"] = centralized_model
except FileNotFoundError:
    print("❌ Error: Not all model files were found. Make sure you ran all previous steps.")
    exit(1)

# --- 5. Evaluate all models on the MASTER test set ---
print("\n--- 📊 FINAL MODEL EVALUATION ---")
print("All models are being tested on the same unseen 20% of data.\n")

results = {}
for name, model in models_to_evaluate.items():
    results[name] = evaluate_model(model, X_test_master, y_test_master)

# --- 6. Print the final results table ---
print(f"{'Model':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
print("-" * 86)

for name, metrics in results.items():
    print(f"{name:<30} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1-Score']:<10.4f} | {metrics['AUC-ROC']:<10.4f}")

print("\nEvaluation Complete.")