# scripts/train_local_torch.py
import argparse, os
import pandas as pd, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.model import SepsisMLP

def load_csv(path):
    return pd.read_csv(path)

def prepare_tensors(df):
    y = torch.tensor(df["SepsisLabel"].values.astype(np.float32)).view(-1, 1)
    X = torch.tensor(df.drop(columns=["SepsisLabel"]).values.astype(np.float32))
    return X, y

def get_loss_weight(y):
    c0, c1 = (y == 0).sum(), (y == 1).sum()
    if c0 == 0 or c1 == 0:
        return None
    pos_weight = c0 / c1
    print(f"Data Imbalance: {c0} Neg / {c1} Pos → pos_weight={pos_weight:.2f}")
    return torch.tensor([pos_weight])

def train_model(X, y, epochs=5, batch_size=256, lr=1e-3, device="cpu"):
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    model = SepsisMLP(X.shape[1]).to(device)
    print(f"🧩 Model initialized with input_dim = {X.shape[1]}")
    pos_weight = get_loss_weight(y)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader.dataset):.6f}")
    return model

def eval_model(model, X, y, device="cpu"):
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X.to(device))).cpu().numpy().ravel()
    y_true = y.numpy().ravel()
    preds_bin = (preds > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds_bin),
        "precision": precision_score(y_true, preds_bin, zero_division=0),
        "recall": recall_score(y_true, preds_bin, zero_division=0),
        "f1": f1_score(y_true, preds_bin, zero_division=0),
        "auc": roc_auc_score(y_true, preds),
        "confusion_matrix": confusion_matrix(y_true, preds_bin).tolist()
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"\n🏥 Training local model on {args.csv}")
    df = load_csv(args.csv)
    X, y = prepare_tensors(df)
    model = train_model(X, y, epochs=args.epochs)
    torch.save(model.state_dict(), args.out)
    metrics = eval_model(model, X, y)
    print(f"💾 Saved to {args.out}\n📊 Metrics:\n{metrics}")

if __name__ == "__main__":
    main()
