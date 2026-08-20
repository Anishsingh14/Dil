import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

from models.train_tabular import TabularMLP


class TabularInferenceEngine:
    def __init__(
        self,
        model_path: str = "models/tabular_mlp.pth",
        scaler_path: str = "models/scaler.joblib",
        metadata_path: str = "models/tabular_metadata.json",
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.metadata_path = Path(metadata_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TabularMLP] = None
        self.scaler = None
        self.feature_cols = None
        self.model_version = "tabular-mlp-v1.3.0"
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.feature_cols = checkpoint.get("feature_cols", [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ])
        self.model_version = checkpoint.get("model_version", "tabular-mlp-v1.3.0")

        self.model = TabularMLP(
            input_dim=checkpoint.get("input_dim", 13),
            hidden_dims=checkpoint.get("hidden_dims", [64, 32]),
            dropout=checkpoint.get("dropout", 0.3),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = joblib.load(self.scaler_path)

        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                metadata = json.load(f)
                self.model_version = metadata.get("model_version", self.model_version)

        self._loaded = True
        print(f"Tabular model loaded on {self.device}: {self.model_version}")

    def _predict_sync(self, features: np.ndarray) -> tuple[float, str]:
        if not self._loaded:
            self.load()

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            risk_score = float(self.model(features_tensor).cpu().item())

        if risk_score < 0.33:
            risk_level = "Low Risk"
        elif risk_score < 0.66:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return risk_score, risk_level

    async def predict(self, features: np.ndarray) -> tuple[float, str, int]:
        start_time = time.perf_counter()
        risk_score, risk_level = await asyncio.to_thread(self._predict_sync, features)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return risk_score, risk_level, latency_ms

    def is_loaded(self) -> bool:
        return self._loaded

    def get_device(self) -> str:
        return str(self.device)


_tabular_engine: Optional[TabularInferenceEngine] = None


def get_tabular_engine() -> TabularInferenceEngine:
    global _tabular_engine
    if _tabular_engine is None:
        _tabular_engine = TabularInferenceEngine()
    return _tabular_engine


async def initialize_tabular_model() -> None:
    engine = get_tabular_engine()
    engine.load()