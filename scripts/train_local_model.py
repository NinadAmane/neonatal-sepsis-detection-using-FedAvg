import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib
from pathlib import Path

#Obtaining dataset from the CLI:
if len(sys.argv) < 2:
    print("❌ Usage: python src/train_local_model.py <path_to_hospital_dataset>")
    sys.exit(1)

data_path = Path(sys.argv[1])
hospital_id = data_path.stem.split('_')[-1]   #Extract numbers from the csv files 1, 2, 3.

#Read the dataset
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"❌ Error: Dataset file not found at {data_path}")
    sys.exit(1)


#Define features and target
target_col = "SepsisLabel"
if target_col not in df.columns:
    print(f"❌ Error: Target column '{target_col}' not found in the dataset.")
    sys.exit(1)

X = df.drop(columns=[target_col])   # X is now a new dataframe that contains all features...that we will use to predict the target_col.
y = df[target_col]   # y is the value that needs to be predicted.

# --- FIX: Convert continuous y back to discrete 0s and 1s ---
# Assumes the scaler made '0' a negative number and '1' a positive number
y = (y > 0.5).astype(int)

#Splitting the data for test and train
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


#Training the Model
model = LogisticRegression(max_iter= 1000,  class_weight='balanced')
model.fit(X_train,y_train)


#Evaluate the model
y_pred =  model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]    # model.predict_proba() gives the model's confidence (e.g., "I'm 85% sure this is sepsis").
                                                    # We need this for the AUC-ROC metric. [:, 1] just selects the probability for the "positive" (sepsis) class.

#Calculate the Metrics
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# --- Print evaluation metrics ---
print(f"--- 🏥 Hospital {hospital_id} Evaluation ---")
print(f"  Accuracy:   {acc:.4f}")
print(f"  Precision:  {precision:.4f}  (How many predicted sepsis cases were real)")
print(f"  Recall:     {recall:.4f}  (How many real sepsis cases did we find)")
print(f"  F1-Score:   {f1:.4f}  (Balance of Precision and Recall)")
print(f"  AUC-ROC:    {auc:.4f}  (Model's ability to discriminate classes)")
print(f"  Confusion Matrix:\n{cm}")
print(f"------------------------------------")


# --- Save model ---
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)
model_path = output_dir / f"local_model_hospital_{hospital_id}.pkl"
joblib.dump(model, model_path)
print(f"✅ Saved model: {model_path}")

