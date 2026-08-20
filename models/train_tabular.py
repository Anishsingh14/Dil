import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, classification_report


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int = 13, hidden_dims: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x)).squeeze(-1)


def load_and_preprocess_data(csv_path: str):
    df = pd.read_csv(csv_path)
    feature_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    return X, y, feature_cols


def train_model(
    csv_path: str = "dil_heart_disease_uci_cleveland.csv",
    model_path: str = "models/tabular_mlp.pth",
    scaler_path: str = "models/scaler.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
):
    X, y, feature_cols = load_and_preprocess_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    model = TabularMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Training on {device} for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size(0))
        epoch_loss = 0.0
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        train_preds = (model(X_train_tensor) > 0.5).float().cpu().numpy()
        test_preds = (model(X_test_tensor) > 0.5).float().cpu().numpy()

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_recall = recall_score(y_train, train_preds)
    test_recall = recall_score(y_test, test_preds)

    print(f"\nTraining Accuracy: {train_acc:.4f}, Recall: {train_recall:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, Recall: {test_recall:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["No Disease", "Disease"]))

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": X_train.shape[1],
        "hidden_dims": [64, 32],
        "dropout": 0.3,
        "feature_cols": feature_cols,
        "model_version": "tabular-mlp-v1.3.0",
    }, model_path)
    print(f"\nModel saved to {model_path}")

    metadata = {
        "feature_cols": feature_cols,
        "model_version": "tabular-mlp-v1.3.0",
        "test_accuracy": float(test_acc),
        "test_recall": float(test_recall),
        "train_accuracy": float(train_acc),
        "train_recall": float(train_recall),
    }
    with open("models/tabular_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model, scaler, metadata


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(device=device)