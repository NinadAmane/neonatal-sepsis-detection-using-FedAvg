import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path


#--- config    (You can change these settings later if u have 4th hospital. (change NUM_HOSPITALS = 3 to 4.))
num_hospitals = 3
model_dir = Path("models")
model_name_prefix = "local_model_hospital_"
global_model_path = model_dir / "global_federated_model.pkl"

#--- Load all local models ---
print(f"Federated Averaging: Loading {num_hospitals} local models...")

local_models = []
for i in range(1, num_hospitals + 1):
    model_path = model_dir / f"{model_name_prefix}{i}.pkl"
    try:
        model = joblib.load(model_path)
        local_models.append(model)
        print(f"✅Loaded {model_path}")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please run the local training script for all hospitals first.")
        exit(1)



#--- PERFORM FEDERATED AVERAGING (FedAvg) (IMP) --- 
# 1. Get coefficients (weights) from all models
all_coefs = [model.coef_ for model in local_models]

# 2. Calculate the average of the coefficients
average_coef = np.mean(all_coefs, axis = 0)

# 3. Get intercepts (biases) from all models
all_intercepts = [model.intercept_ for model in local_models]

# 4. Calculate average of all the intercepts
average_intercept = np.mean(all_intercepts, axis=0)

print("\n Successfully averaged model parameters")



# --- Create and save the new Global Model ---
# 1. Create a new, "empty" model instance
# It MUST be the same type (LogisticRegression)
global_model = LogisticRegression(max_iter=1000, class_weight='balanced')

# 2. Manually set its learned parameters to our new averages
# We must .fit() it first to initialize its attributes
# We use a dummy dataset [0, 1] just to enable setting the parameters
global_model.fit(np.array([[0], [1]]), np.array([0, 1]))
global_model.coef_ = average_coef
global_model.intercept_ = average_intercept

# 3. Save the new global model
joblib.dump(global_model, global_model_path)
print(f"\n✅ Federated Global Model saved to: {global_model_path}")