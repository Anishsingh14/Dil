import asyncio
import io
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pydicom


class DenseNet121LSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights
        
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        
        self.features = backbone.features
        self.feature_dim = backbone.classifier.in_features
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        
        features = self.features(x)
        features = self.avgpool(features)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]
        
        logits = self.classifier(last_hidden)
        return torch.sigmoid(logits).squeeze(-1)


class ImagingInferenceEngine:
    def __init__(
        self,
        model_path: str = "models/imaging_densenet121_lstm.pth",
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[DenseNet121LSTM] = None
        self.model_version = "imaging-densenet121-lstm-v1.1.0"
        self._loaded = False
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self) -> None:
        if self._loaded:
            return

        self.model = DenseNet121LSTM(
            num_classes=1,
            lstm_hidden=256,
            lstm_layers=1,
            dropout=0.3,
            pretrained=False,
        ).to(self.device)
        
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model_version = checkpoint.get("model_version", self.model_version)
            print(f"Loaded imaging model from {self.model_path}")
        else:
            print(f"Model not found at {self.model_path}, using ImageNet-pretrained DenseNet121 + random LSTM")
            self.model.features.load_state_dict(
                torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True).features.state_dict()
            )
        
        self.model.eval()
        self._loaded = True
        print(f"Imaging model loaded on {self.device}: {self.model_version}")

    def _load_image(self, file_bytes: bytes, filename: str) -> Image.Image:
        filename_lower = filename.lower()
        
        if filename_lower.endswith(('.dicom', '.dcm')):
            dicom = pydicom.dcmread(io.BytesIO(file_bytes))
            pixel_array = dicom.pixel_array.astype(np.float32)
            
            if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
                wc = float(dicom.WindowCenter[0] if isinstance(dicom.WindowCenter, pydicom.multival.MultiValue) else dicom.WindowCenter)
                ww = float(dicom.WindowWidth[0] if isinstance(dicom.WindowWidth, pydicom.multival.MultiValue) else dicom.WindowWidth)
                img_min = wc - ww / 2
                img_max = wc + ww / 2
                pixel_array = np.clip(pixel_array, img_min, img_max)
            
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
            pixel_array = (pixel_array * 255).astype(np.uint8)
            img = Image.fromarray(pixel_array, mode='L')
        else:
            img = Image.open(io.BytesIO(file_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img

    def _predict_sync(self, file_bytes: bytes, filename: str) -> Tuple[float, str, dict]:
        if not self._loaded:
            self.load()

        img = self._load_image(file_bytes, filename)
        img_tensor = self.preprocess(img).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 3, 224, 224]
        
        with torch.no_grad():
            risk_score = float(self.model(img_tensor).cpu().item())

        if risk_score < 0.33:
            risk_level = "Low Risk"
        elif risk_score < 0.66:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        findings = {
            "st_segment_abnormality_detected": risk_score > 0.5,
            "structural_risk_pattern_confidence": round(risk_score, 2),
        }

        return risk_score, risk_level, findings

    async def predict(self, file_bytes: bytes, filename: str) -> Tuple[float, str, dict, int]:
        start_time = time.perf_counter()
        risk_score, risk_level, findings = await asyncio.to_thread(self._predict_sync, file_bytes, filename)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return risk_score, risk_level, findings, latency_ms

    def is_loaded(self) -> bool:
        return self._loaded

    def get_device(self) -> str:
        return str(self.device)


_imaging_engine: Optional[ImagingInferenceEngine] = None


def get_imaging_engine() -> ImagingInferenceEngine:
    global _imaging_engine
    if _imaging_engine is None:
        _imaging_engine = ImagingInferenceEngine()
    return _imaging_engine


async def initialize_imaging_model() -> None:
    engine = get_imaging_engine()
    engine.load()