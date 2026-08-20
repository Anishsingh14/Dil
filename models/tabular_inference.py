import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import xgboost as xgb


class TabularInferenceEngine:
    def __init__(
        self,
        model_path: str = "models/tabular_xgb.pkl",
        scaler_path: str = "models/scaler.joblib",
        metadata_path: str = "models/tabular_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.metadata_path = Path(metadata_path)
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = None
        self.feature_cols = None
        self.model_version = "tabular-xgb-v1.1.0"
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                metadata = json.load(f)
                self.feature_cols = metadata.get("feature_cols", [
                    "age", "sex", "cp", "trestbps", "chol", "fbs",
                    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
                ])
                self.model_version = metadata.get("model_version", self.model_version)
        else:
            self.feature_cols = [
                "age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
            ]

        self._loaded = True
        print(f"Tabular model loaded: {self.model_version}")

    def _predict_sync(self, features: np.ndarray) -> tuple[float, str]:
        if not self._loaded:
            self.load()

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        if hasattr(self.model, 'predict_proba'):
            risk_score = float(self.model.predict_proba(features_scaled)[0, 1])
        else:
            risk_score = float(self.model.predict(features_scaled)[0])

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
        return "cpu"


_tabular_engine: Optional[TabularInferenceEngine] = None


def get_tabular_engine() -> TabularInferenceEngine:
    global _tabular_engine
    if _tabular_engine is None:
        _tabular_engine = TabularInferenceEngine()
    return _tabular_engine


async def initialize_tabular_model() -> None:
    engine = get_tabular_engine()
    engine.load()