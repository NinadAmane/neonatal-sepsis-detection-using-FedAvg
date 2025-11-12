import syft as sy
import torch
import torch.nn as nn
import sys
import inspect
from syft.service.code.user_code import SubmitUserCode
from syft.service.policy.policy import CustomInputPolicy, CustomOutputPolicy

# -----------------------------------------------------------------------------
# Remote training function
# -----------------------------------------------------------------------------
def train_sepsis_model(context, epochs=3, lr=0.001):
    """
    Runs entirely on the Datasite (server side).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import pandas as pd

    # --- Define model ---
    class SepsisModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    # --- Fetch dataset from context ---
    dataset_pointer = next((ds for ds in context.datasets if ds.name == "hospital_1"), None)
    if dataset_pointer is None:
        return {"error": "Dataset 'hospital_1' not found on server context."}

    df = dataset_pointer.asset_list[0].data

    # --- Convert to tensors ---
    try:
        y = torch.tensor(df["SepsisLabel"].values, dtype=torch.float32).view(-1, 1)
        X = torch.tensor(df.drop("SepsisLabel", axis=1).values, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Failed to process data: {e}"}

    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    model = SepsisModel(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    print(f"✅ Training complete. Final loss: {loss.item():.4f}")
    return model.state_dict()


# -----------------------------------------------------------------------------
# Researcher workflow (local)
# -----------------------------------------------------------------------------
def run_researcher_workflow():
    try:
        client = sy.login(port=8080, email="info@openmined.org", password="changethis")
        print(f"✅ Logged into: {client.name}")

        # ---------------------------------------------------------------------
        # Upload training function as executable code (SubmitUserCode)
        # ---------------------------------------------------------------------
        print("⬆️ Submitting remote training function...")

        func_code = inspect.getsource(train_sepsis_model)
        func_sig = inspect.signature(train_sepsis_model)
        func_kwargs = list(func_sig.parameters.keys())

        code_to_submit = SubmitUserCode(
            code=func_code,
            func_name="train_sepsis_model_v1",
            signature=func_sig,
            input_policy_type=CustomInputPolicy,
            output_policy_type=CustomOutputPolicy,
            input_kwargs=func_kwargs,
        )

        submitted_code = client.code.submit(code=code_to_submit)
        print("✅ Code submission complete.")

        # ---------------------------------------------------------------------
        # Execute the uploaded code remotely
        # ---------------------------------------------------------------------
        print("⏳ Executing remote training job on the Datasite...")
        result_ptr = client.code.call(
            code_id=submitted_code.id,
            kwargs={"epochs": 3, "lr": 0.01},
        )

        print("✅ Remote job started. Waiting for result...")
        trained_model_params = result_ptr.get()
        print("✅ Received trained model parameters from remote site.")

        if isinstance(trained_model_params, dict) and "error" in trained_model_params:
            raise RuntimeError(f"Server error: {trained_model_params['error']}")

        # ---------------------------------------------------------------------
        # Load returned model parameters locally
        # ---------------------------------------------------------------------
        class SepsisModelLocal(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)
            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        local_model = SepsisModelLocal(37)
        local_model.load_state_dict(trained_model_params)

        print("\n🎉 SUCCESS — Model trained remotely and loaded locally!")
        print("First layer weights (preview):")
        print(local_model.linear.weight)

    except Exception as e:
        print(f"❌ Error during training: {e}", file=sys.stderr)
        print("Is the server running?", file=sys.stderr)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_researcher_workflow()
